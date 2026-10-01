"""Post-process a complex reaction network with a kinetic (Neumann) grain-surface condition.

Unlike postprocess_sr, which pins the grain surface to calcite equilibrium with a
Dirichlet condition, here the surface imposes a diffusive flux

    D * dc/dn = nu * Da * (1 - Omega),     Omega = [Ca2+][CO3^2-] / K_sp

which couples the three conserved components through Omega and therefore requires
a fixed-point iteration.

"""
import argparse
import importlib
import os
from importlib.resources import files

import dolfin as df
import numpy as np
import scipy.interpolate as intp
import mpi4py.MPI as MPI

from addictif.common.utils import (mpi_root, Params, create_folder_safely, helpers,
                                   xdmf_params, mpi_print, axis2index, mpi_max, mpi_min,
                                   Top, Btm, Boundary, SideWalls, Slice)

comm = MPI.COMM_WORLD

# CaCO3 -> Ca2+ + CO3^2- written in the (u1, u2, u3) conserved-component basis
nu_calcite = (1.0, -2.0, 1.0)


"""
Da value in the code is Da = (L/d)(Cca2+/C_ref) Da_p, where Da_p is the Damköhler number based on pore size and Ca2+ concentration.
D value in the code is D = (d/L) Pe^-1, where Pe is the peclet number based on pore size
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Post process complex reaction network with kinetic surface reaction")
    parser.add_argument("-i", "--input", required=True, type=str, help="Folder with concentration file (required)")
    parser.add_argument("--crn", type=str, default="react3", help="Reaction")
    parser.add_argument("--sols", type=str, default="co2mix2", help="End-member solutions")
    parser.add_argument("--Da", type=float, default=1.0, help="Dimensionless surface rate constant")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum fixed-point iterations")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance on the surface flux")
    parser.add_argument("--relax", type=float, default=0.1, help="Under-relaxation factor for the surface flux")
    parser.add_argument("--sr", type=float, default=1, help="Initial surface saturation ratio (1 = equilibrium)")
    return parser.parse_args()


def load_sols(crn, sols):
    data_text = files("addictif.chemistry." + crn + ".sols").joinpath(sols + ".dat").read_text()
    return eval(data_text)


def speciate(crn, K_, u1, u2, u3):
    c = np.zeros((len(u1), crn.nspec))
    for i in range(len(u1)):
        u = [u1[i], u2[i], u3[i]]
        try:
            crn.compute_primary_spec(c[i, :], u, K_)
        except SystemExit:
            raise RuntimeError(f"Speciation found no physical root for u = {u}")
        crn.compute_secondary_spec(c[i, :], u, K_)
    return c


def main():
    args = parse_args()

    crn = importlib.import_module(f"addictif.chemistry.{args.crn}")
    sols = load_sols(args.crn, args.sols)

    c_a = np.zeros(crn.nspec)
    c_b = np.zeros(crn.nspec)

    for i, sol0i in enumerate(sols[0]):
        c_a[i] = sol0i / crn.c_ref
    for i, sol1i in enumerate(sols[1]):
        c_b[i] = sol1i / crn.c_ref

    K_ = crn.equilibrium_constants(crn.c_ref)

    crn.compute_secondary_spec_initial(c_a, K_)
    crn.compute_secondary_spec_initial(c_b, K_)
    u_a = crn.compute_conserved_spec_initial(c_a)
    u_b = crn.compute_conserved_spec_initial(c_b)

    # Solving 5th degree polynomials is expensive.
    # First we make an interpolation scheme to speed things up!
    N_intp = 1000

    alpha = np.linspace(0., 1., N_intp)
    u_mix = np.outer(1 - alpha, u_a) + np.outer(alpha, u_b)

    c_ = np.zeros((len(alpha), crn.nspec))
    for i in range(len(alpha)):
        crn.compute_primary_spec_initial(c_[i, :], u_mix[i, :], K_)
        crn.compute_secondary_spec_initial(c_[i, :], K_)

    c_intp = [intp.InterpolatedUnivariateSpline(alpha, c_[:, ispec]) for ispec in range(crn.nspec)]

    prm = Params()
    prm.load(os.path.join(args.input, "params.dat"))
    mesh_path = os.path.relpath(os.path.join(args.input, prm["mesh"]), os.getcwd())
    u_path = os.path.relpath(os.path.join(args.input, prm["u"]), os.getcwd())
    D = prm["D"]
    tol = prm["tol"]

    linear_solver = "bicgstab"
    preconditioner = "hypre_euclid"

    prm_u = Params(os.path.join(u_path, "params.dat"), required=True)
    mesh_u_path = os.path.join(u_path, prm_u["mesh"])
    direction = axis2index[prm_u["direction"]]

    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), mesh_u_path, "r") as h5f:
        h5f.read(mesh_u, "mesh", False)

    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    u_vel = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), os.path.join(u_path, "u.h5"), "r") as h5f:
        h5f.read(u_vel, "u")

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)

    alpha_ = df.Function(S, name="alpha")
    with df.HDF5File(mesh.mpi_comm(), os.path.join(args.input, "delta.h5"), "r") as h5f:
        h5f.read(alpha_, "delta")

    # Translate from delta (-1, 1) to alpha (0, 1)
    alpha_.vector()[:] = 0.5 * (alpha_.vector()[:] + 1)

    c_spec_cm = [df.Function(S, name=f"c_{ispec}_cm") for ispec in range(crn.nspec)]
    saturation_ratio_cm = df.Function(S, name="saturation_ratio_cm")

    for ispec in range(crn.nspec):
        c_spec_cm[ispec].vector()[:] = (c_a[ispec] * (1 - alpha_.vector()[:])
                                        + c_b[ispec] * alpha_.vector()[:]) * crn.c_ref
    saturation_ratio_cm.vector()[:] = (c_spec_cm[4].vector()[:] * c_spec_cm[3].vector()[:]
                                       / crn.saturation_product_calcite())

    c_spec_ = [df.Function(S, name=f"c_{ispec}") for ispec in range(crn.nspec)]
    for ispec in range(crn.nspec):
        c_spec_[ispec].vector()[:] = c_intp[ispec](alpha_.vector()[:]) * crn.c_ref

    u_1 = df.Function(S, name="u_1")
    u_2 = df.Function(S, name="u_2")
    u_3 = df.Function(S, name="u_3")

    u_1.vector()[:] = c_spec_[0].vector()[:] + c_spec_[2].vector()[:] + c_spec_[3].vector()[:]
    u_2.vector()[:] = (c_spec_[1].vector()[:] - c_spec_[2].vector()[:]
                       - 2 * c_spec_[3].vector()[:] - c_spec_[5].vector()[:])
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
    [sw.mark(subd, 4 + index) for index, sw in enumerate(sidewalls)]
    top.mark(subd, 1)
    btm.mark(subd, 2)

    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    S_DG0 = df.FunctionSpace(mesh, "DG", 0)

    xi = df.TrialFunction(S)
    psi = df.TestFunction(S)
    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)

    mpi_print("interpolating velocity field...")
    u_proj_ = df.Function(V, name="u")
    df.LagrangeInterpolator.interpolate(u_proj_, u_vel)
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
    tau_.vector()[:] *= arr
    mpi_print("Done.")

    r_xi = df.dot(u_proj_, df.grad(xi)) - D * df.div(df.grad(xi))
    a_xi = df.dot(u_proj_, df.grad(xi)) * psi * df.dx \
        + D * df.dot(df.grad(psi), df.grad(xi)) * df.dx
    a_xi += tau_ * r_xi * df.dot(u_proj_, df.grad(psi)) * df.dx

    u1_ = df.Function(S, name="u1")
    u2_ = df.Function(S, name="u2")
    u3_ = df.Function(S, name="u3")

    # Only the inlet is prescribed; the grain surface now carries a flux condition
    bcs = [df.DirichletBC(S, u_1, subd, 1),
           df.DirichletBC(S, u_2, subd, 1),
           df.DirichletBC(S, u_3, subd, 1)]

    # The dof set is identical for all three components, so the matrix is assembled once
    mpi_print("Assembling system")
    A = df.assemble(a_xi)
    bcs[0].apply(A)

    solver = df.KrylovSolver(linear_solver, preconditioner)
    solver.set_operator(A)
    solver.parameters["relative_tolerance"] = 1e-9
    solver.parameters["monitor_convergence"] = False
    solver.parameters["nonzero_initial_guess"] = True

    R_ = df.Function(S, name="reaction_rate")
    R_.vector()[:] = 0.0
    fields = [u1_, u2_, u3_]

    def solve_components():
        for j, (field, bc) in enumerate(zip(fields, bcs)):
            b = df.assemble(nu_calcite[j] * R_ * psi * ds(3))
            bc.apply(b)
            solver.solve(field.vector(), b)

    # Dofs on the grain surface, where the flux law has to be evaluated
    bc_grains = df.DirichletBC(S, df.Constant(0.), subd, 3)
    n_owned = u1_.vector().local_size()
    grain_dofs = np.fromiter(bc_grains.get_boundary_values().keys(), dtype=int)
    grain_dofs = np.sort(grain_dofs[grain_dofs < n_owned])

    R_init = np.zeros(n_owned)
    R_init[grain_dofs] = args.Da * (1.0 - args.sr) # negative means precipitation
    R_.vector().set_local(R_init)
    R_.vector().apply("insert")

    def update_flux():
        u1 = u1_.vector().get_local()
        u2 = u2_.vector().get_local()
        u3 = u3_.vector().get_local()

        c_g = speciate(crn, K_, u1[grain_dofs], u2[grain_dofs], u3[grain_dofs])
        Omega = c_g[:, 4] * c_g[:, 3] / crn.saturation_product_calcite()

        R_old = R_.vector().get_local()
        R_new = np.zeros(n_owned)
        R_new[grain_dofs] = args.Da * (1.0 - Omega) # negative means precipitation
        R_new = (1.0 - args.relax) * R_old + args.relax * R_new

        R_.vector().set_local(R_new)
        R_.vector().apply("insert")

        dR = comm.allreduce(np.max(np.abs(R_new - R_old), initial=0.), op=MPI.MAX)
        R_mag = comm.allreduce(np.max(np.abs(R_new), initial=0.), op=MPI.MAX)
        return dR / max(R_mag, 1e-30)

    mpi_print("Initial solve with 1.01 surface sr")
    solve_components()

    converged = False
    for it in range(1, args.max_iter + 1):
        rel_change = update_flux()
        solve_components()
        mpi_print(f"Iteration {it}: relative flux change = {rel_change:.3e}")
        if rel_change < args.rtol:
            converged = True
            mpi_print(f"Converged after {it} iterations.")
            break

    if not converged:
        mpi_print(f"WARNING: not converged after {args.max_iter} iterations.")

    mpi_print("Reconstructing speciation")
    u1 = u1_.vector().get_local()
    u2 = u2_.vector().get_local()
    u3 = u3_.vector().get_local()
    c_new = speciate(crn, K_, u1, u2, u3)

    c_spec_new = [df.Function(S, name=f"c_{ispec}_new") for ispec in range(crn.nspec)]
    for i in range(crn.nspec):
        c_spec_new[i].vector().set_local(c_new[:, i])
        c_spec_new[i].vector().apply("insert")  # unit is mol/l

    pH = df.Function(S, name="pH")
    pH.vector()[:] = -np.log10(c_spec_new[1].vector()[:])

    saturation_ratio = df.Function(S, name="saturation_ratio")
    saturation_ratio.vector()[:] = (c_spec_new[4].vector()[:] * c_spec_new[3].vector()[:]
                                    / crn.saturation_product_calcite())

    # Output sign convention is opposite to the flux variable R_ driving the solve:
    # positive means precipitation
    R_out = df.Function(S, name="reaction_rate")
    R_out.vector()[:] = -R_.vector()[:]

    total_reaction_rate = df.assemble(R_out * ds(3)) #positive values for precipitation
    # The solved Ca gradient must reproduce the imposed flux; the gap measures Picard convergence
    n = df.FacetNormal(mesh)
    total_flux_check = -df.assemble(df.dot(n, df.grad(u3_)) * ds(3))

    mpi_print(f"Total reaction rate: {total_reaction_rate}")
    mpi_print(f"Total Ca flux from solution: {total_flux_check}")

    mpi_print("Start saving")

    output_folder = os.path.join(args.input, f"crn_{args.crn}_{args.sols}_kinetic_Da{args.Da:g}")
    create_folder_safely(output_folder)

    mpi_print("Saving CRN.")
    prm = Params()
    prm["species"] = ",".join([f"c_{ispec}" for ispec in range(crn.nspec)])
    prm["ade"] = os.path.relpath(args.input, output_folder)
    prm["Da"] = args.Da
    prm["total_reaction_rate"] = total_reaction_rate
    prm["total_Ca_flux"] = total_flux_check
    prm.dump(os.path.join(output_folder, "params.dat"))

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "subd.xdmf")) as xdmff:
        xdmff.write(subd)

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "conc_show.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        xdmff.write(pH, 0.)
        xdmff.write(saturation_ratio_cm, 0.)
        xdmff.write(saturation_ratio, 0.)
        xdmff.write(R_out, 0.)
        for ispec in range(crn.nspec):
            xdmff.write(c_spec_new[ispec], 0.)

    bmesh = df.BoundaryMesh(mesh, "exterior", True)
    B_S = df.FunctionSpace(bmesh, "Lagrange", 1)
    reaction_rate_b = df.Function(B_S, name="reaction_rate")
    saturation_ratio_b = df.Function(B_S, name="saturation_ratio")
    df.LagrangeInterpolator.interpolate(reaction_rate_b, R_out)
    df.LagrangeInterpolator.interpolate(saturation_ratio_b, saturation_ratio)

    with df.XDMFFile(bmesh.mpi_comm(), os.path.join(output_folder, "reaction_rate_show.xdmf")) as xdmff:
        xdmff.write(reaction_rate_b, 0.)
    with df.XDMFFile(bmesh.mpi_comm(), os.path.join(output_folder, "saturation_ratio_show.xdmf")) as xdmff:
        xdmff.write(saturation_ratio_b, 0.)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(output_folder, "conc.h5"), "w") as h5f:
        for ispec in range(crn.nspec):
            h5f.write(c_spec_new[ispec], f"c_{ispec}")
        h5f.write(R_out, "reaction_rate")

    mpi_print("Integrating reaction rate over slices.")
    subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subdomains.set_all(0)

    Ns = 20
    L = x_max - x_min
    dslice = L[direction] / Ns
    xs = np.linspace(x_min[direction], x_max[direction], Ns, endpoint=False) + 0.5 * dslice

    for i in range(Ns):
        x_slice = xs[i]
        sl = Slice(direction, x_slice - 0.5 * dslice, x_slice + 0.5 * dslice, tol)
        sl.mark(subdomains, i + 1)

    # Drop everything that is not grain surface, so inlet/outlet/sidewalls stay out of the slabs
    subdomains.array()[subd.array() != 3] = 0

    ds_sub = df.Measure("ds", domain=mesh, subdomain_data=subdomains)

    reac_integrated = np.zeros(Ns)
    area = np.zeros(Ns)

    for i in range(Ns):
        area[i] = df.assemble(df.Constant(1.0) * ds_sub(i + 1))
        reac_integrated[i] = df.assemble(R_out * ds_sub(i + 1))

    if mpi_root:
        header = " ".join(["z", "area", "reaction_rate_integrated"])
        data = np.vstack([xs, area, reac_integrated]).T
        np.savetxt(os.path.join(output_folder, "reaction_rate_per_slice.dat"), data, header=header)


if __name__ == "__main__":
    main()
