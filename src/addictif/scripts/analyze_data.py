import dolfin as df
#from fenicstools import Probes
import h5py
import os
import numpy as np
import argparse
from addictif.common.utils import mpi_root, mpi_print, axis2index, index2axis, Params, mpi_max, mpi_min, helpers, read_field
from addictif.common.fenicstools.Probes import Probes, GradProbes

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze steady ADR")
    parser.add_argument("-i", "--input", type=str, required=True, help="Concentration file (required)")
    parser.add_argument("-Nt", type=int, default=200, help="Number of interpolation points transverse")
    parser.add_argument("-Nn", type=int, default=50, help="Number of interpolation points longitudinal")
    # parser.add_argument("--it", type=int, default=0, help="Iteration")
    # parser.add_argument("--D", type=float, default=1e-2, help="Diffusion")
    # parser.add_argument("--L", type=float, default=1, help="Pore size")
    # parser.add_argument("--Lx", type=float, default=1, help="Lx")
    # parser.add_argument("--Ly", type=float, default=1, help="Ly")
    # parser.add_argument("--Lz", type=float, default=1, help="Lz")
    # parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    # parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    # parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    # parser.add_argument("--direction", type=str, default='z', help="x or z direction of flow")
    return parser.parse_args()

def main():
    args = parse_args()

    prm_spec = Params(os.path.join(args.input, "params.dat"), required=True)

    # specii = ["a", "b", "c", 'delta']
    specii = []
    reaction_rates = []
    if "ade" in prm_spec or "u" in prm_spec:
        if "ade" in prm_spec:
            specii = prm_spec["species"].split(",")
            ade_path = os.path.join(args.input, prm_spec["ade"])
            conc_key = "c_spec.h5"
        elif "u" in prm_spec:
            prm_ade = prm_spec
            ade_path = args.input
            specii = ["delta"]
            conc_key = "delta.h5"
        prm_ade = Params(os.path.join(ade_path, "params.dat"))
        u_path = os.path.join(ade_path, prm_ade["u"])
        mesh_path = os.path.join(ade_path, prm_ade["mesh"])
        D = prm_ade["D"]
    else:
        u_path = args.input

    if len(specii) > 1 and "reaction_rates" in prm_spec:
        reaction_rates = prm_spec["reaction_rates"].split(",")
    

    prm_u = Params(os.path.join(u_path, "params.dat"))

    mesh_u_path = os.path.join(u_path, prm_u["mesh"])

    direction = axis2index[prm_u["direction"]]
    
    #pore_size = args.L
    #it = args.it
    #Lx = args.Lx
    #Ly = args.Ly
    #Lz = args.Lz

    # Create the mesh
    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), mesh_u_path, "r") as h5f:
        h5f.read(mesh_u, "mesh", False)

    coords = mesh_u.coordinates()[:]
    x_max = mpi_max(coords)
    x_min = mpi_min(coords)
    L = x_max-x_min
    mpi_print("Dimensions:", x_min, x_max)

    eps = 1e-8

    N = np.array([0, 0, 0])
    tdims = [0, 1, 2]
    tdims.remove(direction)

    dxt = np.sqrt(L[tdims[0]]*L[tdims[1]]) / args.Nt
    for dim in tdims:
        N[dim] = int(np.ceil(L[dim]/dxt))
    N[direction] = args.Nn
    dx = L / N

    # Pe__ = pore_size / D
    #N = [200, 200, 50]

    x = [np.linspace(x_min[i], x_max[i], N[i], endpoint=False)+0.5*dx[i] for i in range(3)]
    X, Y, Z = np.meshgrid(*x, indexing='ij')
    pts = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T

    Jname = ["J_adv", "J_diff", "Iz"]

    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    S_u = df.FunctionSpace(mesh_u, "Lagrange", 2)
    u_ = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), os.path.join(u_path, "u.h5"), "r") as h5f:
        h5f.read(u_, "u")

    mpi_print("Initial assignment.")
    #u_x_ = df.project(u_[0], S_u, solver_type="gmres", preconditioner_type="petsc_amg") #, bcs=uwall)
    #u_y_ = df.project(u_[1], S_u, solver_type="gmres", preconditioner_type="petsc_amg") #, bcs=uwall)
    #u_z_ = df.project(u_[2], S_u, solver_type="gmres", preconditioner_type="petsc_amg") #, bcs=uwall)
    ux_ = [df.Function(S_u, name=f"u_{dim}") for dim in range(3)]
    [df.assign(uxi_, u_.sub(dim)) for dim, uxi_ in enumerate(ux_)]
    phi_ = df.Function(S_u, name="phi")
    phi_.vector()[:] = 1.0
    
    mpi_print("Probing u.")
    prob_u = Probes(pts.flatten(), S_u)
    ux_data = []
    for i in range(3):
        prob_u(ux_[i])
        ux_data.append(prob_u.array(0))
        prob_u.clear()
    prob_u(phi_)
    phi_data = prob_u.array(0)
    prob_u.clear()

    if mpi_root:
        ofilename = os.path.join(args.input, "intpdata.h5")
        ofile = h5py.File(ofilename, "w")
        for dim in range(3):
            ofile.create_dataset(f"u{index2axis[dim]}", data=ux_data[dim].reshape(N))
            ofile.create_dataset(index2axis[dim], data=x[dim])
        ofile.create_dataset("phi", data=phi_data.reshape(N))
        ofile.attrs["direction"] = direction
        ofile.attrs["x_max"] = x_max.tolist()
        ofile.attrs["x_min"] = x_min.tolist()
    
    if len(specii) > 0:
        if mpi_root:
            ofile.attrs["D"] = D

        mesh = df.Mesh()
        with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f:
            h5f.read(mesh, "mesh", False)

        mpi_print("Setting up function spaces.")
        S = df.FunctionSpace(mesh, "Lagrange", 1)
        #V_DG0 = df.VectorFunctionSpace(mesh, "DG", 0)
        S_DG0 = df.FunctionSpace(mesh, "DG", 0)

        conc_ = dict()
        for species in specii:
            conc_[species] = df.Function(S, name=species)

        reac_ = dict()
        for species, reaction_rate in zip(specii, reaction_rates):
            reac_[species] = df.Function(S, name=reaction_rate)
        
        for species in specii:
            read_field(os.path.join(args.input, conc_key), conc_[species], species)

        for species, reaction_rate in zip(specii, reaction_rates):
            read_field(os.path.join(args.input, conc_key), reac_[species], reaction_rate)

        """
        mpi_print("Interpolating gradients.")
        dcdx_ = [dict(), dict(), dict()]
        for species in specii:
            #Jz_loc = [u_proj_z_ * conc_[species], - D * conc_[species].dx(2)]
            #Jz_[species] = [df.project(ufl_expr, S, solver_type="gmres", preconditioner_type="amg")
            #                for ufl_expr in Jz_loc]
            #ufl_expr_Jz_diff = - D * conc_[species].dx(direction)
                
            #ufl_Iz = conc_[species].dx(tdims[0])**2 + conc_[species].dx(tdims[1])**2
            #ufl_expr_Iz = sum([conc_[species].dx(dim)**2 for dim in tdims])

            #Jz_diff_[species] = df.project(ufl_expr_Jz_diff, S, solver_type="minres", preconditioner_type="petsc_amg")
            #Iz_[species] = df.project(ufl_expr_Iz, S, solver_type="minres", preconditioner_type="petsc_amg")
            gradc = df.interpolate(df.CompiledExpression(helpers.Grad(), a=conc_[species], degree=0), V_DG0)
            for dim in range(3):
                #dcdx_[dim][species] = df.project(conc_[species].dx(dim), solver_type="minres", preconditioner_type="petsc_amg")
                dcdx_[dim][species] = df.Function(S_DG0)
                df.assign(dcdx_[dim][species], gradc.sub(dim))
        """

        #sg = StructuredGrid(S_u, N, origin, vectors, dL)

        #xgrid, ygrid, zgrid = sg.create_coordinate_vectors()
        #xgrid = xgrid[0]
        #ygrid = ygrid[1]
        #zgrid = zgrid[2]

        #sg = StructuredGrid(S, N, origin, vectors, dL)

        prob_conc = GradProbes(pts.flatten(), S)
        prob_reac = Probes(pts.flatten(), S_DG0)
        #prob_grad = Probes(pts.flatten(), S)

        for species in conc_:
            mpi_print(f"Species: {species}")
            prob_conc.grad(conc_[species])
            data = prob_conc.array(0)
            data_grad = prob_conc.array_grad(0)
            prob_conc.clear()
            prob_conc.clear_grad()

            if mpi_root:
                ofile.create_dataset(f"{species}/conc", data=data.reshape(N))
                for dim in range(3):
                    #prob_grad(dcdx_[dim][species])
                    #data = prob_grad.array(0)

                    #err = data - data_grad[:, dim]
                    #print(err.max())

                    ofile.create_dataset(f"{species}/grad{index2axis[dim]}", data=data_grad[:, dim].reshape(N))
                    #prob_grad.clear()
       
        for species in reac_:
            mpi_print(f"Reaction rate: {species}")
            prob_reac(reac_[species])
            data = prob_reac.array(0)
            prob_reac.clear()
            if mpi_root:
                ofile.create_dataset(f"{species}/rate", data=data.reshape(N))
        
        

    if mpi_root:
        ofile.close()

if __name__ == "__main__":
    main()