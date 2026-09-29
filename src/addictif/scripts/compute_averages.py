'''
#command: 
'''
import numpy as np
import argparse
from addictif.common.utils import fetch_intp_data, index2axis, axis2index, create_folder_safely, mpi_root, mpi_size, mpi_print
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Compute average concentration and fluxes")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to interpolated data (e.g. intpdata.h5)")
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--invert", action="store_true", help="Invert flow direction")
    parser.add_argument("--direction", type=str, default=None, help="Override direction (x, y or z)")
    return parser.parse_args()

def main():
    args = parse_args()

    if mpi_size > 1:
        mpi_print("MPI is not yet supported for compute_averages. Please run in serial.")
        exit()

    x, phi, u, conc, grad, prm = fetch_intp_data(args.input)
    

    direction = prm["direction"]

    if args.invert:
        flip=-1
    else:
        flip=1

    if args.direction is not None:
        direction = axis2index[args.direction]
    x_min = np.array(prm["x_min"])
    x_max = np.array(prm["x_max"])
    if len(conc) > 0:
        D = prm["D"]

    analysisfolder = os.path.join(os.path.dirname(args.input), "Analysis")
    create_folder_safely(analysisfolder)

    tdims = [0, 1, 2]
    tdims.remove(direction)

    conc_mean = dict()
    Jz_mean = dict()
    I_mean = dict()
    Rz_mean = dict()
    Ixy_mean = dict()
    for species in conc:
        conc_mean[species] = conc[species].mean(axis=tuple(tdims))
        Jz = flip * conc[species] * u[direction] - D * grad[species][direction]
        Jz_mean[species] = Jz.mean(axis=tuple(tdims))
        I = D * np.sum([grad[species][dim]**2 for dim in range(3)], axis=0)
        I_mean[species] = I.mean(axis=tuple(tdims))
        
        laplacian_c = np.zeros_like(grad[species])
        for dim in range(3):
            laplacian_c += np.gradient(grad[species][dim], axis=dim, edge_order=2)
        
        Rz = np.sum([flip * grad[species][dim] * u[dim] - D * laplacian_c[dim] for dim in range(3)], axis=0)
        #Rz = np.sum([- grad[species][dim] * u[dim] for dim in range(3)], axis=0)
        Rz_mean[species] = Rz.mean(axis=tuple(tdims))
        
    uz_mean = u[direction].mean(axis=tuple(tdims))
    por_mean = phi.mean(axis=tuple(tdims))

    # Mean and variance over the fluid in each cross-section, one plane at a time
    fluid_mean = dict()
    fluid_var = dict()
    for species in conc:
        fluid_mean[species] = np.zeros(len(x[direction]))
        fluid_var[species] = np.zeros(len(x[direction]))
        for k in range(len(x[direction])):
            ind = [slice(None)] * 3
            ind[direction] = k
            ind = tuple(ind)
            c_k = conc[species][ind][phi[ind] > 0.5].astype(np.float64)
            if len(c_k):
                fluid_mean[species][k] = c_k.mean()
                fluid_var[species][k] = ((c_k - c_k.mean())**2).mean()

    if mpi_root:
        header = " ".join([index2axis[direction], 
                        "porosity", f"u_{index2axis[direction]}", *conc_mean.keys(), 
                        *[f"J_{key}_{index2axis[direction]}" for key in Jz_mean.keys()],
                        *[f"I_{key}" for key in Jz_mean.keys()],
                        *[f"R_{key}" for key in Jz_mean.keys()],
                        *[f"mean_{key}_fluid" for key in fluid_mean.keys()],
                        *[f"var_{key}_fluid" for key in fluid_var.keys()]]) 
        data = np.vstack([x[direction], por_mean, uz_mean, *conc_mean.values(), *Jz_mean.values(), *I_mean.values(), *Rz_mean.values(),
                          *fluid_mean.values(), *fluid_var.values()]).T
        #data = np.vstack([x[direction], por_mean, uz_mean, *conc_mean.values(), *Jz_mean.values(), *I_mean.values()]).T
        np.savetxt(os.path.join(analysisfolder, f"averages_{index2axis[direction]}.dat"), data, header=header)

    if mpi_root and args.show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2 + int(len(conc) > 0))
        for species_idx, species in enumerate(conc):
            ax[0].plot(x[direction], conc_mean[species]/por_mean, label=f"${species}$")
            ax[1].plot(x[direction], Jz_mean[species], label=f"$J_{{{species},{index2axis[direction]}}}$")
            ax[2].plot(x[direction], I_mean[species], label=f"$I_{{{species}}}$")
        ax[0].plot(x[direction], por_mean, label=f"$\phi$")
        ax[1].plot(x[direction], uz_mean, label=f"$u_{index2axis[direction]}$")

        for _ax in ax:    
            _ax.set_xlabel(f"${index2axis[direction]}$")
            _ax.legend()

        ax[0].set_ylabel("Average concentration")
        ax[1].set_ylabel("Total flux")
        if len(conc) > 0:
            ax[2].set_ylabel("Scalar dissipation rate")

        # ax[2].loglog()

        fig.tight_layout()
        plt.show()
        #plt.savefig(args.con + 'plots/average_conc/conc_Pe{}_it{}_.png'.format(Pe__, it), dpi=300)

if __name__ == "__main__":
    main()