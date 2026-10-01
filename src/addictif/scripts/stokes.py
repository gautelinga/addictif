import dolfin as df
from addictif.common.utils import mpi_print, mpi_max, mpi_min, create_folder_safely, Top, Btm, Boundary, SideWalls, Params, axis2index
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Solve Stokes equations in a given mesh")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--direction", type=str, default="z", help="x, y or z direction of flow")
    parser.add_argument("--tol", type=float, default=df.DOLFIN_EPS_LARGE, help="Tolerance for subdomains")
    parser.add_argument("--sidewall_bc", type=str, default="slip", help="Sidewall boundary conditon")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.direction not in axis2index:
        mpi_print(f"Invalid direction: {args.direction}")
        exit()
    direction = axis2index[args.direction]

    # Create the mesh
    mpi_print("Importing mesh")

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), args.mesh, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    
    mpi_print("Mesh read successfully.")

    # Preparing for output
    create_folder_safely(args.output)
    velocity_folder = args.output # os.path.join(args.output, "velocity")
    # create_folder_safely(velocity_folder)
    mesh_relpath = os.path.relpath(args.mesh, velocity_folder)

    tol = args.tol

    x = mesh.coordinates()[:]

    x_min = mpi_min(x)
    x_max = mpi_max(x)

    mpi_print("Dimensions:", x_max, x_min)

    # Define function spaces
    V = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    VP = df.MixedElement([V, P])
    W = df.FunctionSpace(mesh , VP)

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
 
    with df.XDMFFile(mesh.mpi_comm(), os.path.join(velocity_folder, "subd.xdmf")) as xdmff:
        xdmff.write(subd)

    noslip = df.Constant((0.0, 0.0, 0.0))

    bc_porewall = df.DirichletBC(W.sub(0), noslip, subd, 3)
    bc_top = df.DirichletBC(W.sub(1), df.Constant(0.), subd, 1)
    bc_bottom = df.DirichletBC(W.sub(1), df.Constant(0.), subd, 2)
    
    # Choose sidewall boundary conditions
    if args.sidewall_bc == "slip":
        bc_sidewalls = [df.DirichletBC(W.sub(0).sub(sw.dim), df.Constant(0.), subd, 4+index) for index, sw in enumerate(sidewalls)]
    else:
        bc_sidewalls = [df.DirichletBC(W.sub(0), noslip, subd, 4+index) for index in range(len(sidewalls))]
    bcs = [bc_porewall, bc_top, bc_bottom, *bc_sidewalls]

    F = [0., 0., 0.]
    F[direction] = 1.0
    f = df.Constant(F)

    # Define variational problem
    (u, p) = df.TrialFunctions(W)
    (v, q) = df.TestFunctions(W)

    a = df.inner(df.grad(u), df.grad(v))*df.dx + df.div(v)*p*df.dx + q*df.div(u)*df.dx
    L = df.inner(f, v)*df.dx

    # Form for use in constructing preconditioner matrix
    b = df.inner(df.grad(u), df.grad(v))*df.dx + p*q*df.dx

    # Assemble system
    A, bb = df.assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = df.assemble_system(b, L, bcs)

    # Create Krylov solver and AMG preconditioner
    solver = df.KrylovSolver("minres", "hypre_amg")
    solver.parameters["monitor_convergence"] = True
    solver.parameters["relative_tolerance"] = 1e-10
    solver.parameters["maximum_iterations"] = 100000

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    mpi_print("Computing volume")
    vol = df.assemble(df.Constant(1.) * df.dx(domain=mesh))

    mpi_print("Solving system")

    # Solve
    U = df.Function(W)

    solver.solve(U.vector(), bb)
    mpi_print("Solving done.")

    # Get sub-functions
    u_, p_ = U.split(deepcopy=True)

    ui_mean = abs(df.assemble(u_[direction] * df.dx))/vol
    mpi_print(f"Mean velocity in {args.direction} direction: {ui_mean}")
    u_.vector()[:] /= ui_mean

    # Dump parameters
    prm = Params()
    prm["mesh"] = mesh_relpath
    prm["direction"] = args.direction
    prm["tol"] = args.tol
    prm["sidewall_bc"] = args.sidewall_bc
    prm.dump(os.path.join(velocity_folder, "params.dat"))

    # Create XDMF files for visualization output
    with df.XDMFFile(os.path.join(velocity_folder, "u_show.xdmf")) as xdmffile_u:
        xdmffile_u.parameters["flush_output"] = True
        # Save solution to file (XDMF/HDF5)
        xdmffile_u.write(u_)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(velocity_folder, "u.h5"), "w") as h5f:
        h5f.write(u_, "u")

if __name__ == "__main__":
    main()