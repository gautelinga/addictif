import argparse
import dolfin as df
import numpy as np
from addictif.common.utils import mpi_root, Params, create_folder_safely, helpers, xdmf_params, mpi_print, mpi_root, read_field, axis2index
#from chemistry.react_1.reaction import equilibrium_constants, compute_secondary_spec, compute_primary_spec, compute_conserved_spec, nspec, c_ref
import importlib
from importlib.resources import files

import matplotlib.pyplot as plt
import scipy.interpolate as intp
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Post process complex reaction network")
    parser.add_argument("-i", "--input", required=True, type=str, help="Folder with concentration file (required)")
    parser.add_argument("--crn", type=str, default="react1", help="Reaction")
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

    crn.compute_secondary_spec(c_a, K_)
    crn.compute_secondary_spec(c_b, K_)
    u_a = crn.compute_conserved_spec(c_a)
    u_b = crn.compute_conserved_spec(c_b)

    # Solving 5th degree polynomials is expensive.
    # First we make an interpolation scheme to speed things up!

    N_intp = 1000

    alpha = np.linspace(0., 1., N_intp)
    u_ = np.outer(1-alpha, u_a) + np.outer(alpha, u_b)

    c_ = np.zeros((len(alpha), crn.nspec))
    for i in range(len(alpha)):
        crn.compute_primary_spec(c_[i, :], u_[i, :], K_)
        crn.compute_secondary_spec(c_[i, :], K_)

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
    D = prm["D"]

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)
    S_DG0 = df.FunctionSpace(mesh, "DG", 0)
    alpha_ = df.Function(S, name="alpha")

    read_field(os.path.join(args.input, "delta.h5"), alpha_, "delta")

    # Translate from delta (-1, 1) to alpha (0, 1)
    if args.swap:
        alpha_.vector()[:] *= -1

    alpha_.vector()[:] = 0.5*(alpha_.vector()[:]+1)
    # Clip for physical reasons
    #alph = alpha_.vector()[:]
    #alpha_.vector()[alph < 0] = 0.0
    #alpha_.vector()[alph > 1] = 1.0
    # Leads to unphysical gradients!

    logalpha_ = df.Function(S, name="logalpha")
    logalpha_.vector()[:] = alpha_.vector()[:]  # np.log(alpha_.vector()[:])

    output_folder = os.path.join(args.input, f"crn_{args.crn}_{args.sols}")
    if args.swap:
        output_folder += "_swapped"
    create_folder_safely(output_folder)

    c_spec_ = [df.Function(S, name=f"c_{ispec}") for ispec in range(crn.nspec)]
    for ispec in range(crn.nspec):
        c_spec_[ispec].vector()[:] = c_intp[ispec](alpha_.vector()[:]) * crn.c_ref

    pH = df.Function(S, name="pH")
    pH.vector()[:] = -np.log10(c_spec_[1].vector()[:])

    if False:
        solver_type = "gmres"
        params = dict(
            relative_tolerance=1e-9
        )

        dlogalphadx_ = df.project(logalpha_.dx(0), S, solver_type=solver_type, form_compiler_parameters=params)
        dlogalphady_ = df.project(logalpha_.dx(1), S, solver_type=solver_type, form_compiler_parameters=params)
        dlogalphadz_ = df.project(logalpha_.dx(2), S, solver_type=solver_type, form_compiler_parameters=params)

        gradalpha2_ = df.Function(S, name="sqGradAlpha")
        gradalpha2_.vector()[:] = dlogalphadx_.vector()[:]**2 + dlogalphady_.vector()[:]**2 + dlogalphadz_.vector()[:]**2
    else:
        absgrad_alpha = df.interpolate(df.CompiledExpression(helpers.AbsGrad(), a=alpha_, degree=0), S_DG0)

        gradalpha2_ = df.Function(S_DG0, name="sqGradAlpha")
        gradalpha2_.vector()[:] = absgrad_alpha.vector()[:]**2

        alpha_DG0_ = df.interpolate(df.CompiledExpression(helpers.ScalarDG0(), a=alpha_, degree=0), S_DG0)

    #R_spec_ = [df.Function(S_DG0, name=f"R_{ispec}") for ispec in range(crn.nspec)]
    #for ispec in range(crn.nspec):
        #d2c_intp = c_intp[ispec].derivative(2)
        #R_spec_[ispec].vector()[:] = D * d2c_intp(alpha_DG0_.vector()[:]) * gradalpha2_.vector()[:] * crn.c_ref

    #only calculate precipitation rate for Ca^2+ (spec 4), non-dimensionalized by c_ref/(L^2/D) and L is the size of the domain
    R_spec_ = df.Function(S_DG0, name=f"R_4")
    d2c_intp = c_intp[4].derivative(2)
    R_spec_.vector()[:] = d2c_intp(alpha_DG0_.vector()[:]) * gradalpha2_.vector()[:]

    total_reaction_rate = df.assemble(R_spec_ * df.dx)

    #from ca2+ transport equation
    u_path = os.path.relpath(os.path.join(args.input, prm["u"]), os.getcwd())
    prm_u = Params(os.path.join(u_path, "params.dat"), required=True)
    mesh_u_path = os.path.join(u_path, prm_u["mesh"])
    
    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), mesh_u_path, "r") as h5f:
        h5f.read(mesh_u, "mesh", False)
    
    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    u_vel = df.Function(V_u)
    
    with df.HDF5File(mesh_u.mpi_comm(), os.path.join(u_path, "u.h5"), "r") as h5f:
        h5f.read(u_vel, "u")

    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    mpi_print("interpolating velocity field...")
    u_proj_ = df.Function(V, name="u")
    df.LagrangeInterpolator.interpolate(u_proj_, u_vel)
    mpi_print("done.")
    n = df.FacetNormal(mesh)
    net_ca_transport = (df.assemble(df.dot(u_proj_, df.grad(c_spec_[4])) * df.dx)
                            - D * df.assemble(df.dot(n, df.grad(c_spec_[4])) * df.ds)) / D

    mpi_print(f"Total reaction rate: {total_reaction_rate}")
    mpi_print(f"Net Ca transport: {net_ca_transport}")

    # Output
    mpi_print("Saving CRN.")
    prm = Params()
    prm["species"] = ",".join([f"c_{ispec}" for ispec in range(crn.nspec)])
    prm["reaction_rates"] = "R_4"
    #prm["reaction_rates"] = ",".join([f"R_{ispec}" for ispec in range(crn.nspec)])
    prm["ade"] = os.path.relpath(args.input, output_folder)
    prm["Total reaction rate"] = total_reaction_rate
    prm["Net Ca transport"] = net_ca_transport
    prm.dump(os.path.join(output_folder, "params.dat"))

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "alpha_show.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        #xdmff.write(gradalpha2_, 0.)
        #xdmff.write(pH, 0.)
        xdmff.write(alpha_, 0.)
        #for ispec in range(crn.nspec):
            #xdmff.write(R_spec_[ispec], 0.)
            #xdmff.write(c_spec_[ispec], 0.)

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "c_spec_show.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        #xdmff.write(gradalpha2_, 0.)
        xdmff.write(pH, 0.)
        for ispec in range(crn.nspec):
            #xdmff.write(R_spec_[ispec], 0.)
            xdmff.write(c_spec_[ispec], 0.)

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "R_4_show.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        xdmff.write(R_spec_, 0.)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(output_folder, "c_spec.h5"), "w") as h5f:
        for ispec in range(crn.nspec):
            h5f.write(c_spec_[ispec], f"c_{ispec}")
            #h5f.write(R_spec_[ispec], f"R_{ispec}")
        h5f.write(R_spec_, f"R_4")

if __name__ == "__main__":
    main()