import argparse
import numpy as np
import dolfin as df
from addictif.common.utils import mpi_print, Params, create_folder_safely, read_field
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Post-processing")
    parser.add_argument("-i", "--input", type=str, required=True, help="Folder with concentration file (required)")
    #parser.add_argument("--it", type=int, default=0, help="Iteration")
    #parser.add_argument("--D", type=float, default=1e-2, help="Diffusion")
    #parser.add_argument("--L", type=float, default=1, help="Pore size")
    #parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    #parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()

def main():
    args = parse_args()
    
    prm_ade = Params()
    prm_ade.load(os.path.join(args.input, "params.dat"))

    D = prm_ade["D"]
    it = prm_ade["it"]
    mesh_path = os.path.relpath(os.path.join(args.input, prm_ade["mesh"]), os.getcwd())

    output_folder = os.path.join(args.input, "abc")
    create_folder_safely(output_folder)

    mpi_print("Loading mesh")
    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f_mesh:
        h5f_mesh.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)
    delta_ = df.Function(S, name="delta")

    mpi_print("Loading delta")
    read_field(os.path.join(args.input, "delta.h5"), delta_, "delta")

    a_ = df.Function(S, name="a")
    b_ = df.Function(S, name="b") 
    c_ = df.Function(S, name="c") 

    a_.vector()[:] = np.maximum(0, delta_.vector()[:])
    b_.vector()[:] = np.maximum(0, -delta_.vector()[:])
    c_.vector()[:] = (1 - abs(delta_.vector()[:]))/2

    mpi_print("Saving abc.")
    prm = Params()
    prm["species"] = "a,b,c"
    prm["ade"] = os.path.relpath(args.input, output_folder)
    prm.dump(os.path.join(output_folder, "params.dat"))

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "c_spec_show.xdmf")) as xdmff:
        xdmff.parameters.update(
            {"functions_share_mesh": True,
             "rewrite_function_mesh": False,
             "flush_output": True})
        xdmff.write(a_, 0.)
        xdmff.write(b_, 0.)
        xdmff.write(c_, 0.)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(output_folder, "c_spec.h5"), "w") as h5f:
        h5f.write(a_, "a")
        h5f.write(b_, "b")
        h5f.write(c_, "c")

if __name__ == "__main__":
    main()