#!/usr/bin/env python3
"""`utils.read_field` reads a P1 field in either layout to the same Function.

    python3 -m pytest -q tests/test_read_field.py     # serial and on 3 ranks
    mpirun -np 3 python3 tests/test_read_field.py DIR  # one run by hand

The two layouts: dolfin's (`HDF5File.write(f, name)`), and one value per mesh
vertex in the mesh file's vertex order (`<name>/vertex_values`, attrs
layout="vertex", n_vertices, n_cells), as stokes-beadpack writes delta.h5
(`stokes_beadpack/io/vertex_field.py`). The field is a smooth function of
position, so both files hold the same nodal values and the read Functions must
agree to the last bit, and with the interpolant on the read mesh.
"""
import os
import shutil
import subprocess
import sys

import numpy as np


def field(x):
    return x[:, 0] + 2 * x[:, 1] ** 2 - 3 * x[:, 0] * x[:, 2]


def run(folder):
    import dolfin as df
    import h5py
    from mpi4py import MPI
    from addictif.common.utils import read_field

    comm = MPI.COMM_WORLD
    mesh_path = os.path.join(folder, "mesh.h5")
    dolfin_path = os.path.join(folder, "delta_dolfin.h5")
    vertex_path = os.path.join(folder, "delta_vertex.h5")

    # Write the mesh and the dolfin-layout field.
    mesh = df.UnitCubeMesh(df.MPI.comm_world, 5, 4, 3)
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "w") as h5f:
        h5f.write(mesh, "mesh")
    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    S = df.FunctionSpace(mesh, "CG", 1)
    expr = df.Expression("x[0] + 2*x[1]*x[1] - 3*x[0]*x[2]", degree=2)
    f_ref = df.interpolate(expr, S)
    with df.HDF5File(mesh.mpi_comm(), dolfin_path, "w") as h5f:
        h5f.write(f_ref, "delta")

    # Write the vertex layout from the mesh file's coordinates, serially.
    if comm.Get_rank() == 0:
        with h5py.File(mesh_path, "r") as h5:
            x = h5["mesh/coordinates"][:]
            n_cells = h5["mesh/topology"].shape[0]
        with h5py.File(vertex_path, "w") as h5:
            g = h5.create_group("delta")
            g.create_dataset("vertex_values", data=field(x).astype(np.float64))
            g.attrs["layout"] = "vertex"
            g.attrs["n_vertices"] = len(x)
            g.attrs["n_cells"] = n_cells
    comm.Barrier()

    # Read each into a fresh mesh read from the file, as the scripts do.
    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    S = df.FunctionSpace(mesh, "CG", 1)
    f_dolfin = read_field(dolfin_path, df.Function(S), "delta")
    f_vertex = read_field(vertex_path, df.Function(S), "delta")
    f_exact = df.interpolate(expr, S)

    a = f_dolfin.vector().get_local()
    b = f_vertex.vector().get_local()
    c = f_exact.vector().get_local()
    err_layouts = comm.allreduce(np.max(np.abs(a - b), initial=0.0), op=MPI.MAX)
    err_exact = comm.allreduce(np.max(np.abs(b - c), initial=0.0), op=MPI.MAX)
    norm = comm.allreduce(np.max(np.abs(c), initial=0.0), op=MPI.MAX)
    # Ghost values come from apply("insert"): check through evaluation too.
    err_norm = abs(df.assemble((f_dolfin - f_vertex) ** 2 * df.dx))
    if comm.Get_rank() == 0:
        print("ranks={} |dolfin-vertex|={:.3e} |vertex-exact|={:.3e} "
              "L2^2={:.3e} max={:.3e}".format(comm.Get_size(), err_layouts,
                                              err_exact, err_norm, norm))
    assert norm > 0.5
    assert err_layouts == 0.0, err_layouts
    assert err_exact < 1e-14, err_exact
    assert err_norm < 1e-28, err_norm

    # A field on another mesh is refused.
    if comm.Get_rank() == 0:
        with h5py.File(vertex_path, "a") as h5:
            h5["delta"].attrs["n_cells"] = n_cells + 1
    comm.Barrier()
    try:
        read_field(vertex_path, df.Function(S), "delta")
    except ValueError:
        pass
    else:
        raise AssertionError("a field on another mesh was read")


def _launch(tmp_path, ranks):
    if ranks > 1 and shutil.which("mpirun") is None:
        import pytest
        pytest.skip("no mpirun")
    cmd = [sys.executable, os.path.abspath(__file__), str(tmp_path)]
    if ranks > 1:
        # mpirun, not mpiexec: on some machines the latter is ParaView's.
        cmd = ["mpirun", "-np", str(ranks)] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ranks={} ".format(ranks) in out.stdout, out.stdout


def test_read_field_serial(tmp_path):
    _launch(tmp_path, 1)


def test_read_field_3_ranks(tmp_path):
    _launch(tmp_path, 3)


if __name__ == "__main__":
    run(sys.argv[1])
