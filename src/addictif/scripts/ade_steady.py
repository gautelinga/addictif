import dolfin as df
import numpy as np
from addictif.common.utils import (
    helpers, mpi_print, mpi_sum, mpi_rank, mpi_max, mpi_min, 
    Top, Btm, axis2index, index2axis, Params, create_folder_safely, compile_cpp_file)
import os
import h5py
import argparse
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Solve steady conservative transport")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the folder with the velocity field")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("--eps", type=float, default=0.01, help="Epsilon")
    parser.add_argument("--inlet_func", choices=["erf", "tanh"], type=str.lower, default="erf", help="Inlet function")
    parser.add_argument("--angle", type=float, default=0.0,
                        help="Angle (degrees) between the inlet interface and the plane x[1]=ymid, "
                             "rotated in the inlet plane about its mid-point")
    #parser.add_argument("--refine", type=bool, default=True, help="Do you want refinement")
    parser.add_argument("-D", type=float, required=True, help="Diffusion coefficient")
    #parser.add_argument("--L", type=float, default=1, help="Pore size (to compute Peclet number)")
    #parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    #parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    #parser.add_argument("--direction", type=str, default='z', help="x, y or z direction of flow")
    parser.add_argument("--invert", action="store_true", help="Invert flow direction")
    parser.add_argument("--tol", type=float, default=df.DOLFIN_EPS_LARGE, help="tol for subdomains")
    return parser.parse_args()

def main():
    args = parse_args()

    D = args.D  # diffusion coefficient
    #pore_size = args.L
    it = args.it 
    eps = args.eps
    inlet_func = args.inlet_func
    # refine_tol = 0.2
    angle = args.angle
    case_name = f"conservative_D{D}_eps{eps}"
    if angle != 0.:
        case_name += f"_angle{angle}"
    output_folder = os.path.join(args.input, case_name, f"it{it}")
    create_folder_safely(output_folder)

    linear_solver = "bicgstab"
    preconditioner = "hypre_euclid"

    prm_u = Params(os.path.join(args.input, "params.dat"), required=True)
    mesh_u_path = os.path.relpath(os.path.join(args.input, prm_u["mesh"]), os.getcwd())
    direction = axis2index[prm_u["direction"]]

    # Load velocity mesh
    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), mesh_u_path, "r") as h5f:
        h5f.read(mesh_u, "mesh", False)

    # Pe__ = pore_size / D
    # mpi_print("Pe = {}".format(Pe__))

    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    u_ = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), os.path.join(args.input, "u.h5"), "r") as h5f:
        h5f.read(u_, "u")

    if args.invert:
        u_.vector()[:] *= -1
   
    if it == 0:
      mesh = mesh_u
      mesh_path = mesh_u_path
    else: 
      mesh = df.Mesh()
      mesh_path = os.path.join(output_folder, "refined_mesh.h5")
      with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f_up:
          h5f_up.read(mesh, "mesh", False)
   
    tol = args.tol

    x = mesh.coordinates()[:]

    x_min = mpi_min(x)
    x_max = mpi_max(x)

    mpi_print("Dimensions:", x_max, x_min)

    # Boundaries
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)

    if not args.invert:
        top = Top(x_min, x_max, tol, direction)
        top.mark(subd, 1)
    else:
        btm = Btm(x_min, x_max, tol, direction)
        btm.mark(subd, 1)

    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    S = df.FunctionSpace(mesh, "Lagrange", 1)
    S_DG0 = df.FunctionSpace(mesh, "DG", 0)

    xi = df.TrialFunction(S)
    psi = df.TestFunction(S)
    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)
    n = df.FacetNormal(mesh)

    # Signed distance to the inlet interface. The interface passes through the mid-point
    # of the inlet and is rotated by `angle` within the inlet plane away from x[1] = ymid,
    # i.e. it follows x[1] - ymid = tan(angle) * (x[k] - kmid), where k is the in-plane axis
    # normal to both the flow direction and x[1].
    x_mid = 0.5 * (x_max + x_min)
    if angle == 0.:
        k = 1
        dist_str = "(x[1]-ymid)"
    else:
        in_plane_axes = [i for i in range(mesh.geometry().dim()) if i not in (direction, 1)]
        if len(in_plane_axes) != 1:
            raise ValueError("--angle requires a 3D mesh with flow direction x or z.")
        k = in_plane_axes[0]
        dist_str = f"((x[1]-ymid)*cos(theta) - (x[{k}]-kmid)*sin(theta))"
        mpi_print(f"Inlet interface inclined by {angle} degrees towards the {index2axis[k]} axis.")

    if inlet_func == "erf":
        mpi_print("Choosing erf as the inlet function.")
        expr_str = f"erf({dist_str}/(eps))"
    else:
        mpi_print("Choosing tanh as the inlet function.")
        expr_str = f"tanh({dist_str}/(sqrt(2)*eps))"

    delta_top_expr = df.Expression(expr_str, eps=eps, ymid=x_mid[1], kmid=x_mid[k],
                                   theta=np.deg2rad(angle), degree=2)
    rho_top_expr = df.Expression("1.0", degree=2)

    delta_top = df.interpolate(delta_top_expr, S)
    rho_top = df.interpolate(rho_top_expr, S)

    mpi_print("interpolating velocity field...")
    u_proj_ = df.Function(V, name="u")
    df.LagrangeInterpolator.interpolate(u_proj_, u_)
    mpi_print("done.")

    #with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "u_proj.xdmf")) as xdmff:
        #xdmff.write(u_proj_)

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
    #a_xi += -D * df.dot(n, df.grad(xi)) * psi * ds(3)

    q_delta = df.Constant(0.)
    L_delta = q_delta * psi * df.dx

    delta_ = df.Function(S, name="delta")

    bc_delta_inlet = df.DirichletBC(S, delta_top, subd, 1)
    bcs_delta = [bc_delta_inlet]

    t0 = df.Timer("Assembling system")
    t0.start()

    problem_delta = df.LinearVariationalProblem(a_xi,L_delta, delta_, bcs=bcs_delta)
    solver_delta = df.LinearVariationalSolver(problem_delta)

    t0.stop()

    solver_delta.parameters["linear_solver"] = linear_solver
    solver_delta.parameters["preconditioner"] = preconditioner
    solver_delta.parameters["krylov_solver"]["monitor_convergence"] = True
    solver_delta.parameters["krylov_solver"]["relative_tolerance"] = 1e-9

    t1 = df.Timer("Solving conservative transport")
    t1.start()
    
    solver_delta.solve()

    t1.stop()

    mpi_print("Solving done")
    mpi_print("Start saving")

    # Dump parameters
    paramsfile = os.path.join(output_folder, "params.dat")

    prm = Params(paramsfile, required=False)
    mesh_relpath = os.path.relpath(mesh_path, output_folder)
    u_relpath = os.path.relpath(args.input, output_folder)
    prm["mesh"] = mesh_relpath
    prm["u"] = u_relpath
    prm["tol"] = args.tol
    prm["D"] = args.D
    prm["eps"] = eps
    prm["angle"] = angle
    prm["it"] = it
    prm["invert"] = args.invert
    prm.dump(paramsfile)

    '''
    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "conc_show.xdmf")) as xdmff:
        xdmff.parameters.update({"functions_share_mesh": True,
                                 "rewrite_function_mesh": False,
                                 "flush_output": True})
        xdmff.write(delta_, 0.)
        xdmff.write(Pe_el_, 0.)
        xdmff.write(tau_, 0.)
        xdmff.write(u_norm_, 0.)
        xdmff.write(h_, 0.)
    '''

    with df.HDF5File(mesh.mpi_comm(), os.path.join(output_folder, "delta.h5"), "w") as h5f:
        h5f.write(delta_, "delta")

    mpi_print("Saving done.")
   
    df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

if __name__ == "__main__":
    main()