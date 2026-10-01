import argparse
import dolfin as df
import numpy as np
from addictif.common.utils import mpi_root, Params, create_folder_safely, helpers, xdmf_params, mpi_print, mpi_root, axis2index, mpi_max, mpi_min, Top, Btm, Boundary, SideWalls, Slice 
#from chemistry.react_1.reaction import equilibrium_constants, compute_secondary_spec, compute_primary_spec, compute_conserved_spec, nspec, c_ref
import importlib
from importlib.resources import files

import matplotlib.pyplot as plt
import scipy.interpolate as intp
import os
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Post process complex reaction network with heterogeneous nucleation")
    parser.add_argument("-i", "--input", required=True, type=str, help="Folder with concentration file (required)")
    parser.add_argument("--crn", type=str, default="react3", help="Reaction")
    parser.add_argument("--sols", type=str, default="default", help="End-member solutions")
    parser.add_argument("--swap", action="store_true", help="Swap end members")
    return parser.parse_args()

def load_sols(crn, sols):
    data_text = files("addictif.chemistry." + crn + ".sols").joinpath(sols + ".dat").read_text()
    sols = eval(data_text)
    return sols

def main():
    args = parse_args()

    crn = importlib.import_module(f"addictif.chemistry.{args.crn}")
    sols = load_sols(args.crn, args.sols) #load concentrations of primary species in end members

    c_a = np.zeros(crn.nspec)
    c_b = np.zeros(crn.nspec)

    # End members from the second example of de Simoni et al. WRR 2007
    #c_a[0] = 3.4*10**-4 / c_ref
    #c_a[1] = 10**-7.3 / c_ref

    for i, sol0i in enumerate(sols[0]):
        c_a[i] = sol0i / crn.c_ref

    for i, sol1i in enumerate(sols[1]):
        c_b[i] = sol1i / crn.c_ref

    #c_b[0] = 3.4*10**-5 / c_ref
    #c_b[1] = 10**-7.3  / c_ref

    K_ = crn.equilibrium_constants(crn.c_ref)

    crn.compute_secondary_spec_initial(c_a, K_)
    crn.compute_secondary_spec_initial(c_b, K_)
    u_a = crn.compute_conserved_spec_initial(c_a)
    u_b = crn.compute_conserved_spec_initial(c_b)

    # Solving 5th degree polynomials is expensive.
    # First we make an interpolation scheme to speed things up!

    N_intp = 1000

    alpha = np.linspace(0., 1., N_intp)
    u_ = np.outer(1-alpha, u_a) + np.outer(alpha, u_b)

    c_ = np.zeros((len(alpha), crn.nspec))
    for i in range(len(alpha)):
        crn.compute_primary_spec_initial(c_[i, :], u_[i, :], K_)
        crn.compute_secondary_spec_initial(c_[i, :], K_)

    c_intp = [None for _ in range(crn.nspec)]
    for ispec in range(crn.nspec):
        c_intp[ispec] = intp.InterpolatedUnivariateSpline(alpha, c_[:, ispec])


    if False and mpi_root:
        fig, ax = plt.subplots(1, 6, figsize=(15,3))

        # 1: CO2, 2: H^+, 3: HCO3^-, 4: CO3^2-, 5: Ca^2+, 6: OH^-

        ax[0].plot(alpha, c_ref * c_[:, 0])
        ax[0].plot(alpha, c_ref * c_intp[0](alpha))
        ax[0].plot(alpha, c_ref * c_a[0]*np.ones_like(alpha))
        ax[0].plot(alpha, c_ref * c_b[0]*np.ones_like(alpha))
        ax[0].set_title("CO2")

        ax[1].plot(alpha, -np.log10(c_ref * c_[:, 1]))
        ax[1].plot(alpha, -np.log10(c_ref * c_intp[1](alpha)))
        ax[1].set_title("pH")
        
        ax[2].plot(alpha, c_ref * c_[:, 2])
        ax[2].plot(alpha, c_ref * c_intp[2](alpha))
        ax[2].set_title("HCO3^-")
        
        ax[3].plot(alpha, c_ref * c_[:, 3])
        ax[3].plot(alpha, c_ref * c_intp[3](alpha))
        ax[3].set_title("CO3^2-")
        
        ax[4].plot(alpha, c_ref * c_[:, 4])
        ax[4].plot(alpha, c_ref * c_intp[4](alpha))
        ax[4].set_title("Ca^2+")
        
        ax[5].plot(alpha, c_ref * c_[:, 5])
        ax[5].plot(alpha, c_ref * c_intp[5](alpha))
        ax[5].set_title("OH^-")

        plt.show()

    prm = Params()
    prm.load(os.path.join(args.input, "params.dat"))
    mesh_path = os.path.relpath(os.path.join(args.input, prm["mesh"]), os.getcwd())
    u_path = os.path.relpath(os.path.join(args.input, prm["u"]), os.getcwd())
    D = prm["D"]
    eps = prm["eps"]
    tol = prm["tol"]

    linear_solver = "bicgstab"
    preconditioner = "hypre_euclid"

    prm_u = Params(os.path.join(u_path, "params.dat"), required=True)
    mesh_u_path = os.path.join(u_path, prm_u["mesh"])
    direction = axis2index[prm_u["direction"]]

    # Load velocity mesh
    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), mesh_u_path, "r") as h5f:
        h5f.read(mesh_u, "mesh", False)

    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    u_ = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), os.path.join(u_path, "u.h5"), "r") as h5f:
        h5f.read(u_, "u")

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)
    
    alpha_ = df.Function(S, name="alpha")

    with df.HDF5File(mesh.mpi_comm(), os.path.join(args.input, "delta.h5"), "r") as h5f:
        h5f.read(alpha_, "delta")

    # Translate from delta (-1, 1) to alpha (0, 1)
    if args.swap:
            alpha_.vector()[:] *= -1
    alpha_.vector()[:] = 0.5*(alpha_.vector()[:]+1)
    # Clip for physical reasons
    #alph = alpha_.vector()[:]
    #alpha_.vector()[alph < 0] = 0.0
    #alpha_.vector()[alph > 1] = 1.0
    # Leads to unphysical gradients!

    c_spec_cm = [df.Function(S, name=f"c_{ispec}_cm") for ispec in range(crn.nspec)]
    saturation_ratio_cm = df.Function(S, name="saturation_ratio_cm")

    for ispec in range(crn.nspec):
        c_spec_cm[ispec].vector()[:] = (c_a[ispec]*(1-alpha_.vector()[:]) + c_b[ispec]*alpha_.vector()[:])* crn.c_ref
    saturation_ratio_cm.vector()[:] = c_spec_cm[4].vector()[:] * c_spec_cm[3].vector()[:] / crn.saturation_product_calcite()
    

    c_spec_ = [df.Function(S, name=f"c_{ispec}") for ispec in range(crn.nspec)]
    for ispec in range(crn.nspec):
        c_spec_[ispec].vector()[:] = c_intp[ispec](alpha_.vector()[:]) * crn.c_ref

    u_1 = df.Function(S, name="u_1")
    u_2 = df.Function(S, name="u_2")
    u_3 = df.Function(S, name="u_3")
    

    u_1.vector()[:] = c_spec_[0].vector()[:] + c_spec_[2].vector()[:] + c_spec_[3].vector()[:]
    u_2.vector()[:] = c_spec_[1].vector()[:] - c_spec_[2].vector()[:] - 2*c_spec_[3].vector()[:] - c_spec_[5].vector()[:]
    u_3.vector()[:] = c_spec_[4].vector()[:]

    x = mesh.coordinates()[:]

    x_min = mpi_min(x)
    x_max = mpi_max(x)

    mpi_print("Dimensions:", x_max, x_min)

    # Boundaries
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)

    grains = Boundary()
    sidewall_dims = [0, 1, 2]
    sidewall_dims.remove(direction)
    sidewalls = [SideWalls(x_min, x_max, dim, tol) for dim in sidewall_dims]
    top = Top(x_min, x_max, tol, direction)
    btm = Btm(x_min, x_max, tol, direction)

    grains.mark(subd, 3)
    [sw.mark(subd, 4+index) for index, sw in enumerate(sidewalls)]
    top.mark(subd, 1)
    btm.mark(subd, 2)

    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    
    S_DG0 = df.FunctionSpace(mesh, "DG", 0)
    DG0 = df.VectorFunctionSpace(mesh, "DG", 0)

    xi = df.TrialFunction(S)
    psi = df.TestFunction(S)
    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)
    n = df.FacetNormal(mesh) # point to the outside of the domain?????



    mpi_print("interpolating velocity field...")
    u_proj_ = df.Function(V, name="u")
    df.LagrangeInterpolator.interpolate(u_proj_, u_)
    mpi_print("done.")

    mpi_print("Interpolating norm")
    u_norm_ = df.interpolate(df.CompiledExpression(helpers.AbsVecCell(), u=u_proj_, degree=0), S_DG0)
    u_norm_.rename("u_norm", "u_norm")

    mpi_print("Interpolating cell size")
    h_ = df.interpolate(df.CompiledExpression(helpers.CellSize(), mesh=mesh, degree=0), S_DG0)
    h_.rename("h", "h")

    mpi_print("Computing grid Peclet")
    Pe_el_ = df.Function(S_DG0, name="Pe_el")
    Pe_el_.vector()[:] = u_norm_.vector()[:] * h_.vector()[:] / (2 * D)

    mpi_print("Computing tau")
    tau_ = df.Function(S_DG0, name="tau")
    tau_.vector()[:] = h_.vector()[:] / (2 * u_norm_.vector()[:] + 1e-16)
    arr = 1. - 1. / (Pe_el_.vector()[:] + 1e-16)
    arr[arr < 0] = 0.
    #arr = 1./np.tanh(Pe_el_.vector()[:]) - 1./Pe_el_.vector()[:]
    tau_.vector()[:] *= arr
    mpi_print("Done.")

    r_xi = df.dot(u_proj_, df.grad(xi)) - D * df.div(df.grad(xi))
    a_xi = df.dot(u_proj_, df.grad(xi)) * psi * df.dx \
        + D * df.dot(df.grad(psi), df.grad(xi)) * df.dx
    a_xi += tau_ * r_xi * df.dot(u_proj_, df.grad(psi)) * df.dx

    q_ = df.Constant(0.)
    L_ = q_ * psi * df.dx

    u1_ = df.Function(S, name="u1")
    u2_ = df.Function(S, name="u2")
    u3_ = df.Function(S, name="u3")


    bc_u1_inlet = df.DirichletBC(S, u_1, subd, 1)
    bc_u1_grains = df.DirichletBC(S, u_1, subd, 3)

    bcs_u1 = [bc_u1_inlet, bc_u1_grains]

    bc_u2_inlet = df.DirichletBC(S, u_2, subd, 1)
    bc_u2_grains = df.DirichletBC(S, u_2, subd, 3)

    bcs_u2 = [bc_u2_inlet, bc_u2_grains]

    bc_u3_inlet = df.DirichletBC(S, u_3, subd, 1)
    bc_u3_grains = df.DirichletBC(S, u_3, subd, 3)

    bcs_u3 = [bc_u3_inlet, bc_u3_grains]

    t0 = df.Timer("Assembling system for u1")
    t0.start()

    problem_u1 = df.LinearVariationalProblem(a_xi,L_, u1_, bcs=bcs_u1)
    solver_u1 = df.LinearVariationalSolver(problem_u1)

    t0.stop()

    solver_u1.parameters["linear_solver"] = linear_solver
    solver_u1.parameters["preconditioner"] = preconditioner
    solver_u1.parameters["krylov_solver"]["monitor_convergence"] = True
    solver_u1.parameters["krylov_solver"]["relative_tolerance"] = 1e-9

    t1 = df.Timer("Solving u1")
    t1.start()
    
    solver_u1.solve()

    t1.stop()

    mpi_print("Solving u1 done")

    t2 = df.Timer("Assembling system for u2")
    t2.start()

    problem_u2 = df.LinearVariationalProblem(a_xi,L_, u2_, bcs=bcs_u2)
    solver_u2 = df.LinearVariationalSolver(problem_u2)

    t2.stop()

    solver_u2.parameters["linear_solver"] = linear_solver
    solver_u2.parameters["preconditioner"] = preconditioner
    solver_u2.parameters["krylov_solver"]["monitor_convergence"] = True
    solver_u2.parameters["krylov_solver"]["relative_tolerance"] = 1e-9

    t3 = df.Timer("Solving u2")
    t3.start()
    
    solver_u2.solve()

    t3.stop()

    mpi_print("Solving u2 done")

    t4 = df.Timer("Assembling system for u3")
    t4.start()

    problem_u3 = df.LinearVariationalProblem(a_xi,L_, u3_, bcs=bcs_u3)
    solver_u3 = df.LinearVariationalSolver(problem_u3)

    t4.stop()

    solver_u3.parameters["linear_solver"] = linear_solver
    solver_u3.parameters["preconditioner"] = preconditioner
    solver_u3.parameters["krylov_solver"]["monitor_convergence"] = True
    solver_u3.parameters["krylov_solver"]["relative_tolerance"] = 1e-9

    t5 = df.Timer("Solving u3")
    t5.start()
    
    solver_u3.solve()

    t5.stop()

    mpi_print("Solving u3 done")

    u1 = u1_.vector().get_local()
    u2 = u2_.vector().get_local()
    u3 = u3_.vector().get_local()

    c_new = np.zeros((len(u1), crn.nspec))
    for i in range(len(u1)):
        crn.compute_primary_spec(c_new[i, :], [u1[i],u2[i],u3[i]], K_)
        crn.compute_secondary_spec(c_new[i, :], [u1[i],u2[i],u3[i]], K_)



    c_spec_new = [df.Function(S, name=f"c_{ispec}_new") for ispec in range(crn.nspec)]
    
    for i in range(crn.nspec):
        c_spec_new[i].vector().set_local(c_new[:,i])
        c_spec_new[i].vector().apply("insert") #unit is mol/l

    pH = df.Function(S, name="pH")
    pH.vector()[:] = -np.log10(c_spec_new[1].vector()[:])

    saturation_ratio = df.Function(S, name="saturation_ratio")
    saturation_ratio.vector()[:] = c_spec_new[4].vector()[:] * c_spec_new[3].vector()[:] / crn.saturation_product_calcite()

    grad_c4 = df.project(df.grad(c_spec_new[4]), DG0)
    grad_c4.rename("grad_c4", "grad_c4")
    grad_c4.set_allow_extrapolation(True)

    #reaction_rate = df.Function(S, name="reaction_rate")
    reaction_rate_expr = -df.dot(n, df.grad(c_spec_new[4]))
    total_reaction_rate = df.assemble(reaction_rate_expr * ds(3))
    #non-dimensionalized by D*c_ref*L

    mpi_print(f"Total reaction rate: {total_reaction_rate}")
  
    bmesh = df.BoundaryMesh(mesh, "exterior", True)
    #reaction_rate = df.Function(S_DG0, name = "reaction_rate")
    B_DG0 = df.FunctionSpace(bmesh,"DG", 0)
    reaction_rate = df.Function(B_DG0, name="reaction_rate")
    reaction_rate.set_allow_extrapolation(True)
    
    u_ = df.TrialFunction(DG0)
    v_ = df.TestFunction(DG0)
    a = df.inner(u_,v_)*ds
    l = df.inner(n, v_)*ds
    A = df.assemble(a, keep_diagonal=True)
    L = df.assemble(l)

    A.ident_zeros()
    nh = df.Function(DG0, name="facet_normal")

    df.solve(A, nh.vector(), L)
    nh.set_allow_extrapolation(True)

    class flux(df.UserExpression):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def eval(self, values, x):
            n_eval = nh(x)
            grad_eval = grad_c4(x)
            values[0] = -(grad_eval[0]*n_eval[0] + grad_eval[1]*n_eval[1] + grad_eval[2]*n_eval[2])

        def value_shape(self):
            return ()


    reaction_rate.interpolate(flux())

    
    Flux = df.interpolate(df.CompiledExpression(helpers.Flux(), n=nh, a=c_spec_new[4], degree=0), S_DG0)
    Flux.rename("Flux", "Flux")
    #reaction_rate = df.project(reaction_rate_expr, S_DG0, form_compiler_parameters={'quadrature_degree': 4})
    #non-dimensionalized by D*(c_ref/L), c_ref = 1 mol/l, L is the size of the domain

    mpi_print("Start saving")

    # Dump parameters 



    output_folder = os.path.join(args.input, f"crn_{args.crn}_{args.sols}_surface_reaction")
    if args.swap:
            output_folder += "_swapped"
    create_folder_safely(output_folder)



    
    mpi_print("Saving CRN.")
    prm = Params()
    prm["species"] = ",".join([f"c_{ispec}" for ispec in range(crn.nspec)])
    #prm["reaction_rates"] = ",".join([f"R_{ispec}" for ispec in range(crn.nspec)])
    prm["ade"] = os.path.relpath(args.input, output_folder)
    prm["Total_reaction_rate"] = total_reaction_rate
    prm.dump(os.path.join(output_folder, "params.dat"))

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "subd.xdmf")) as xdmff:
        xdmff.write(subd)

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "conc_show.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        xdmff.write(pH, 0.)
        xdmff.write(saturation_ratio_cm, 0.)
        xdmff.write(saturation_ratio, 0.)
        xdmff.write(grad_c4, 0.)
        xdmff.write(nh, 0.)
        xdmff.write(Flux, 0.)
        #xdmff.write(reaction_rate, 0.)
        for ispec in range(crn.nspec):
            xdmff.write(c_spec_new[ispec], 0.)

 
    with df.XDMFFile(bmesh.mpi_comm(), os.path.join(output_folder, "reaction_rate_show.xdmf")) as xdmff:
        xdmff.write(reaction_rate, 0.)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(output_folder, "conc.h5"), "w") as h5f:
        for ispec in range(crn.nspec):
            h5f.write(c_spec_new[ispec], f"c_{ispec}")
    '''
    with df.HDF5File(bmesh.mpi_comm(), os.path.join(output_folder, "reaction_rate.h5"), "w") as h5f:
        h5f.write(reaction_rate, "reaction_rate")
    '''
    # integrate reaction rate over each slice perpendicular to flow direction
    mpi_print("Integrating reaction rate over slices.")
    subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    subdomains.set_all(0)


    Ns = 20
    L = x_max-x_min
    dslice = L[direction] / Ns
    xs = np.linspace(x_min[direction], x_max[direction], Ns, endpoint=False)+0.5*dslice
        
    for i in range(Ns):
        x_slice = xs[i]
        slice= Slice(direction, x_slice - 0.5*dslice, x_slice + 0.5*dslice, tol)
        slice.mark(subdomains, i+1)

    sidewall_dims = [0, 1, 2]
    sidewalls = [SideWalls(x_min, x_max, dim, tol) for dim in sidewall_dims]
    [sw.mark(subdomains, Ns+1+index) for index, sw in enumerate(sidewalls)]

    ds_sub = df.Measure("ds", domain=mesh, subdomain_data=subdomains)

    reac_integrated = np.zeros(Ns)
    area = np.zeros(Ns)

    for i in range(Ns):
        area[i] = df.assemble(df.Constant(1.0) * ds_sub(i+1))
        reac_integrated[i] = df.assemble(reaction_rate_expr * ds_sub(i+1))


    if mpi_root:
        header = " ".join(["z","area", "reaction_rate_integrated"])
        data = np.vstack([xs, area, reac_integrated]).T
        np.savetxt(os.path.join(output_folder, "reaction_rate_per_slice.dat"), data, header=header)

if __name__ == "__main__":
    main()