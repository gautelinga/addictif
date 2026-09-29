import dolfin as df
import numpy as np
import h5py
import argparse
import os
import mpi4py.MPI as MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

axis2index = dict(x=0, y=1, z=2)
index2axis = ["x", "y", "z"]

df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True

xdmf_params = dict(
        functions_share_mesh=True,
        rewrite_function_mesh=False,
        flush_output=True)

# Code for C++ evaluation of absolute gradients of CG1
def compile_cpp_file(filename):
    with open(filename, "r") as infile:
        code = infile.read()
    return df.compile_cpp_code(code)

helpers = compile_cpp_file(os.path.join(os.path.dirname(__file__), "helper_code.cpp"))

def mpi_print(*args):
    if mpi_rank == 0:
        print(*args)

def mpi_max(x):
    x_max_loc = x.max(axis=0)
    x_max = np.zeros_like(x_max_loc)
    mpi_comm.Allreduce(x_max_loc, x_max, op=MPI.MAX)
    return x_max

def mpi_min(x):
    x_min_loc = x.min(axis=0)
    x_min = np.zeros_like(x_min_loc)
    mpi_comm.Allreduce(x_min_loc, x_min, op=MPI.MIN)
    return x_min

def mpi_sum(data):
    data = mpi_comm.gather(data, root=0)
    if mpi_root:
        data = sum(data)
    else:
        data = 0    
    return data

def create_folder_safely(dirname):
    if mpi_root and not os.path.exists(dirname):
        os.makedirs(dirname)

def fetch_rows(path, dataset, wanted, comm):
    """Entries `wanted` of a 1D dataset. Each rank reads one block, then they trade:
    no rank holds the whole, and no h5py point selection (~1000x slower)."""
    size, rank = comm.Get_size(), comm.Get_rank()
    with h5py.File(path, "r") as h5:
        n = h5[dataset].shape[0]
        edges = np.array([n * r // size for r in range(size + 1)], dtype=np.int64)
        block = h5[dataset][edges[rank]:edges[rank + 1]]
    wanted = np.asarray(wanted, dtype=np.int64)
    owner = np.searchsorted(edges, wanted, side="right") - 1
    order = np.argsort(owner, kind="stable")
    counts = np.bincount(owner, minlength=size).astype(np.int64)
    recv = np.empty(size, dtype=np.int64)
    comm.Alltoall(counts, recv)
    asked = np.empty(recv.sum(), dtype=np.int64)
    comm.Alltoallv([wanted[order], counts.tolist()], [asked, recv.tolist()])
    back = np.empty(len(wanted), dtype=block.dtype)
    comm.Alltoallv([np.ascontiguousarray(block[asked - edges[rank]]), recv.tolist()],
                   [back, counts.tolist()])
    out = np.empty_like(back)
    out[order] = back
    return out

def read_field(path, f, name):
    """Read field `name` in `path` into the P1 function `f`.

    Either dolfin's layout, or one value per mesh vertex in the mesh file's
    vertex order (`<name>/vertex_values`), as stokes-beadpack stores delta.
    """
    mesh = f.function_space().mesh()
    with h5py.File(path, "r") as h5:
        by_vertex = "vertex_values" in h5[name]
        sizes = (h5[name].attrs.get("n_vertices"), h5[name].attrs.get("n_cells"))
    if not by_vertex:
        with df.HDF5File(mesh.mpi_comm(), path, "r") as h5f:
            h5f.read(f, name)
        return f
    if sizes != (mesh.num_entities_global(0), mesh.num_entities_global(3)):
        raise ValueError("{} holds {} on {} vertices and {} cells; the mesh has {} and {}".format(
            path, name, *sizes, mesh.num_entities_global(0), mesh.num_entities_global(3)))
    gvi = np.asarray(mesh.topology().global_indices(0), dtype=np.int64)
    values = fetch_rows(path, name + "/vertex_values", gvi, mesh.mpi_comm())
    v2d = df.vertex_to_dof_map(f.function_space())
    x = f.vector().get_local()
    owned = (v2d >= 0) & (v2d < len(x))  # ghost vertices map past the local range
    x[v2d[owned]] = values[owned]
    f.vector().set_local(x)
    f.vector().apply("insert")
    return f

def fetch_intp_data(input):
    x = []
    u = []
    conc = dict()
    grad = dict()
    with h5py.File(input, "r") as h5f:
        phi = h5f["phi"][:]
        for axis in index2axis:
            x.append(h5f[axis][:])
            u.append(h5f[f"u{axis}"][:])

        for key in h5f:
            grp = h5f[key]
            if isinstance(grp, h5py.Group):
                conc[key] = grp["conc"][:]
                grad[key] = []
                for axis in index2axis:
                    grad[key].append(grp[f"grad{axis}"][:])

        prm = dict([(key, h5f.attrs[key]) for key in h5f.attrs])
    return x, phi, u, conc, grad, prm

class Params():
    def __init__(self, input_file=None, required=False):
        self.prm = dict()
        if input_file is not None:
            self.load(input_file, required=required)

    def load(self, input_file, required=False):
        if not os.path.exists(input_file) and required:
            mpi_print(f"No such parameters file: {input_file}")
            exit()
        if os.path.exists(input_file):
            with open(input_file, "r") as infile:
                for el in infile.read().split("\n"):
                    if "=" in el:
                        key, val = el.split("=")
                        if val in ["true", "TRUE"]:
                            val = "True"
                        elif val in ["false", "FALSE"]:
                            val = "False"
                        try:
                            self.prm[key] = eval(val)
                        except:
                            self.prm[key] = val

    def dump(self, output_file):
        if mpi_root:
            with open(output_file, "w") as ofile:      
                ofile.write("\n".join([f"{key}={val}" for key, val in self.prm.items()]))

    def __getitem__(self, key):
        if key in self.prm:
            return self.prm[key]
        else:
            mpi_print("No such parameter: {}".format(key))
            exit()
            #return None

    def __setitem__(self, key, val):
        self.prm[key] = val

    def __str__(self):
        string = "\n".join(["{}: {}".format(key, val) for key, val in self.prm.items()])
        return string
    
    def __contains__(self, key):
        return key in self.prm

class GenSubDomain(df.SubDomain):
    def __init__(self, x_min, x_max, tol=df.DOLFIN_EPS_LARGE, direction=2):
        self.x_min = x_min
        self.x_max = x_max
        self.tol = tol
        self.direction = direction
        super().__init__()

class Top(GenSubDomain):
    def inside(self, x, on_boundary):
      return on_boundary and x[self.direction] < self.x_min[self.direction] + self.tol
    
class Btm(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[self.direction] > self.x_max[self.direction] - self.tol

class Boundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class SideWalls(df.SubDomain):
    def __init__(self, x_min, x_max, dim, tol=df.DOLFIN_EPS_LARGE):
        self.x_min = x_min
        self.x_max = x_max
        self.tol = tol
        self.dim = dim
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and bool( 
            x[self.dim] < self.x_min[self.dim] + self.tol or x[self.dim] > self.x_max[self.dim] - self.tol)

class SideWallsY(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and bool( 
            x[1] < self.x_min[1] + self.tol or x[1] > self.x_max[1] - self.tol)
    
class SideWallsZ(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and bool(
            x[2] < self.x_min[2] + self.tol or x[2] > self.x_max[2] - self.tol)

class SideWallsX(GenSubDomain):
    def inside(self, x, on_boundary): 
        return on_boundary and bool(
            x[0] < self.x_min[0] + self.tol or x[0] > self.x_max[0] - self.tol)