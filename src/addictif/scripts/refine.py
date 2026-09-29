import argparse
import dolfin as df
from addictif.common.utils import helpers, mpi_print, mpi_root, mpi_sum, Params, create_folder_safely, read_field
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Refine mesh based on conservative field")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the folder with the concentration field (delta)")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("--refine_tol", type=float, default=0.2, help="tolerance for refinement")
    return parser.parse_args()

def main():
    args = parse_args()

    prm = Params()
    prm.load(os.path.join(args.input, "params.dat"), True)
    it = prm["it"]+1

    mesh = df.Mesh()
    mesh_path = os.path.relpath(os.path.join(args.input, prm["mesh"]), os.getcwd())
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)
    S_DG0 = df.FunctionSpace(mesh, "DG", 0)

    h_ = df.interpolate(df.CompiledExpression(helpers.CellSize(), mesh=mesh, degree=0), S_DG0)
    h_.rename("h", "h")

    delta_ = df.Function(S, name="delta")
    read_field(os.path.join(args.input, "delta.h5"), delta_, "delta")

    indicator_ = df.Function(S_DG0, name="indicator")

    mpi_print("computing absgrad")
    absgrad_delta_ = df.interpolate(df.CompiledExpression(helpers.AbsGrad(), a=delta_, degree=0), S_DG0)
    indicator_.vector()[:] = h_.vector()[:] * absgrad_delta_.vector()[:]

    # refinement
    mpi_print("marking for refinement")
    cell_markers = df.MeshFunction("bool", mesh, mesh.topology().dim())
    num_marked = helpers.mark_for_refinement(cell_markers, indicator_.cpp_object(), args.refine_tol)

    output_folder = os.path.join(os.path.dirname(os.path.dirname(args.input)), f"it{it}")
    create_folder_safely(output_folder)
    prm.dump(os.path.join(output_folder, "params.dat"))

    mpi_print("refining")
    new_mesh = df.refine(mesh, cell_markers)
    with df.HDF5File(new_mesh.mpi_comm(), os.path.join(output_folder, "refined_mesh.h5"), "w") as h5f:
        h5f.write(new_mesh, "mesh")
    mpi_print("done")

    prev_size = mpi_sum(mesh.num_cells())
    new_size = mpi_sum(new_mesh.num_cells())
    num_marked = mpi_sum(num_marked)
    if mpi_root:
        mpi_print(
            ("Old mesh size: {}\n"
            "Marked cells:  {}\t({:.3f}%)\n"
            "New mesh size: {}\t({:.3f}x)").format(prev_size, num_marked, float(100*num_marked)/prev_size, new_size, float(new_size)/prev_size))

if __name__ == "__main__":
    main()