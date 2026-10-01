import argparse
import dolfin as df
import numpy as np
from addictif.common.utils import mpi_root, Params, create_folder_safely, helpers, xdmf_params, mpi_print, Top, Btm, Boundary, SideWalls, mpi_max, mpi_min, Slice, axis2index
#from chemistry.react_1.reaction import equilibrium_constants, compute_secondary_spec, compute_primary_spec, compute_conserved_spec, nspec, c_ref
import importlib
from importlib.resources import files

import matplotlib.pyplot as plt
import scipy.interpolate as intp
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Post process complex reaction network")
    parser.add_argument("-i", "--input", required=True, type=str, help="Folder with concentration file (required)")
    parser.add_argument("--crn", type=str, default="react3", help="Reaction")
    parser.add_argument("--sols", type=str, default="default", help="End-member solutions")
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
        crn.compute_secondary_spec(c_[i, :], u_[i, :], K_)

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

    with df.HDF5File(mesh.mpi_comm(), os.path.join(args.input, "delta.h5"), "r") as h5f:
        h5f.read(alpha_, "delta")

    # Translate from delta (-1, 1) to alpha (0, 1)
    alpha_.vector()[:] = 0.5*(alpha_.vector()[:]+1)
    # Clip for physical reasons
    #alph = alpha_.vector()[:]
    #alpha_.vector()[alph < 0] = 0.0
    #alpha_.vector()[alph > 1] = 1.0
    # Leads to unphysical gradients!

    #logalpha_ = df.Function(S, name="logalpha")
    #logalpha_.vector()[:] = alpha_.vector()[:]  # np.log(alpha_.vector()[:])

    output_folder = os.path.join(args.input, f"crn_{args.crn}_{args.sols}")
    create_folder_safely(output_folder)

    c_spec_ = [df.Function(S, name=f"c_{ispec}") for ispec in range(crn.nspec)]
    for ispec in range(crn.nspec):
        c_spec_[ispec].vector()[:] = c_intp[ispec](alpha_.vector()[:]) * crn.c_ref

    pH = df.Function(S, name="pH")
    pH.vector()[:] = -np.log10(c_spec_[1].vector()[:])

    saturation_ratio = df.Function(S, name="saturation_ratio")
    saturation_ratio.vector()[:] = c_spec_[4].vector()[:] * c_spec_[3].vector()[:] * crn.c_ref**2 / crn.saturation_product_calcite()
    reaction_rate = df.Function(S, name="reaction_rate")
    k = crn.rate_constants()
    reaction_rate.vector()[:] = -(k[0]*c_spec_[1].vector()[:] + k[1]*c_spec_[0].vector()[:] +k[2])*(1-saturation_ratio.vector()[:])

    c_spec_cm = [df.Function(S, name=f"c_{ispec}_cm") for ispec in range(crn.nspec)]
    pH_cm = df.Function(S, name="pH_cm")
    saturation_ratio_cm = df.Function(S, name="saturation_ratio_cm")
    reaction_rate_cm = df.Function(S, name="reaction_rate_cm")

    for ispec in range(crn.nspec):
        c_spec_cm[ispec].vector()[:] = (c_a[ispec]*(1-alpha_.vector()[:]) + c_b[ispec]*alpha_.vector()[:])* crn.c_ref
    pH_cm.vector()[:] = -np.log10(c_spec_cm[1].vector()[:])
    saturation_ratio_cm.vector()[:] = c_spec_cm[4].vector()[:] * c_spec_cm[3].vector()[:] * crn.c_ref**2 / crn.saturation_product_calcite()
    reaction_rate_cm.vector()[:] = -(k[0]*c_spec_cm[1].vector()[:] + k[1]*c_spec_cm[0].vector()[:] +k[2])*(1-saturation_ratio_cm.vector()[:])
    
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)
                
    grains = Boundary()
    grains.mark(subd, 1)

    sidewall_dims = [0, 1, 2]
    x = mesh.coordinates()[:]

    x_min = mpi_min(x)
    x_max = mpi_max(x)
    L = x_max - x_min
    Ns = 20
    dslice = L[2] / Ns
    tol = df.DOLFIN_EPS_LARGE
    xs = np.linspace(x_min[2], x_max[2], Ns, endpoint=False)+0.5*dslice

    sidewalls = [SideWalls(x_min, x_max, dim, tol) for dim in sidewall_dims]
    x_min[2] += 1.1/30.0
    x_max[2] -= 1.1/30.0
    sidewalls.append(SideWalls(x_min, x_max, 2, tol))

    [sw.mark(subd, 0) for index, sw in enumerate(sidewalls)]
    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "subd.xdmf")) as xdmff:
        xdmff.write(subd)

    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)
    total_reaction_rate = df.assemble(reaction_rate * ds(1))
    total_reaction_rate_cm = df.assemble(reaction_rate_cm * ds(1))

    for i in range(Ns):
        x_slice = xs[i]
        slice= Slice(2, x_slice - 0.5*dslice, x_slice + 0.5*dslice, tol)
        slice.mark(subd, i+2)

    x_min = mpi_min(x)
    x_max = mpi_max(x)

    sidewalls = [SideWalls(x_min, x_max, dim, tol) for dim in sidewall_dims]
    x_min[2] += 1.1/30.0
    x_max[2] -= 1.1/30.0
    sidewalls.append(SideWalls(x_min, x_max, 2, tol))

    [sw.mark(subd, 0) for index, sw in enumerate(sidewalls)]

    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)

    reac_integrated = np.zeros(Ns)
    reac_integrated_cm = np.zeros(Ns)
    area = np.zeros(Ns)

    for i in range(Ns):
        area[i] = df.assemble(df.Constant(1.0) * ds(i+2))
        reac_integrated[i] = df.assemble(reaction_rate * ds(i+2))
        reac_integrated_cm[i] = df.assemble(reaction_rate_cm * ds(i+2))


    if mpi_root:
        header = " ".join(["z","area", "reaction_rate_integrated", "reaction_rate_integrated_cm"])
        data = np.vstack([xs, area, reac_integrated, reac_integrated_cm]).T
        np.savetxt(os.path.join(output_folder, "reaction_rate_per_slice.dat"), data, header=header)

    

    

    
    


    

    #R_spec_ = [df.Function(S_DG0, name=f"R_{ispec}") for ispec in range(crn.nspec)]
    #for ispec in range(crn.nspec):
        #d2c_intp = c_intp[ispec].derivative(2)
        #R_spec_[ispec].vector()[:] = D * d2c_intp(alpha_DG0_.vector()[:]) * gradalpha2_.vector()[:] * crn.c_ref

    #only calculate precipitation rate for Ca^2+ (spec 4), non-dimensionalized by c_ref/(L^2/D) and L is the size of the domain
    

    # Output
    mpi_print("Saving CRN.")
    prm = Params()
    prm["species"] = ",".join([f"c_{ispec}" for ispec in range(crn.nspec)])
    prm["reaction_rates"] = "R_4"
    #prm["reaction_rates"] = ",".join([f"R_{ispec}" for ispec in range(crn.nspec)])
    prm["ade"] = os.path.relpath(args.input, output_folder)
    prm.dump(os.path.join(output_folder, "params.dat"))

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "saturation_ratio.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        #xdmff.write(gradalpha2_, 0.)
        #xdmff.write(pH, 0.)
        xdmff.write(saturation_ratio, 0.)
        #for ispec in range(crn.nspec):
            #xdmff.write(R_spec_[ispec], 0.)
            #xdmff.write(c_spec_[ispec], 0.)

    #with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "c_spec_show.xdmf")) as xdmff:
     #   xdmff.parameters.update(xdmf_params)
        #xdmff.write(gradalpha2_, 0.)
      #  xdmff.write(pH, 0.)
       # for ispec in range(crn.nspec):
            #xdmff.write(R_spec_[ispec], 0.)
        #    xdmff.write(c_spec_[ispec], 0.)

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "R_show.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        xdmff.write(reaction_rate, 0.)

   # with df.HDF5File(mesh.mpi_comm(), os.path.join(output_folder, "c_spec.h5"), "w") as h5f:
        #for ispec in range(crn.nspec):
            #h5f.write(c_spec_[ispec], f"c_{ispec}")
            #h5f.write(R_spec_[ispec], f"R_{ispec}")
    #    h5f.write(reaction_rate, f"R_4")
     #   h5f.write(saturation_ratio, f"saturation_ratio")

    mpi_print(f"Total reaction rate: {total_reaction_rate}")

    

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "saturation_ratio_cm.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        #xdmff.write(gradalpha2_, 0.)
        #xdmff.write(pH, 0.)
        xdmff.write(saturation_ratio_cm, 0.)
        #for ispec in range(crn.nspec):
            #xdmff.write(R_spec_[ispec], 0.)
            #xdmff.write(c_spec_[ispec], 0.)

    

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "R_show_cm.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        xdmff.write(reaction_rate_cm, 0.)


    mpi_print(f"Total reaction rate_cm: {total_reaction_rate_cm}")


if __name__ == "__main__":
    main()