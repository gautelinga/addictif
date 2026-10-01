import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from addictif.common.utils import index2axis, axis2index, create_folder_safely, fetch_intp_data, mpi_rank, mpi_size

class Parser():
    def __init__(self):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Plot imshow")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to interpolated data (e.g. intpdata.h5)")
    parser.add_argument("--arrows", action="store_true", help="Plot arrows")
    parser.add_argument("--direction", type=str, default=None, help="Override direction (x, y or z)")
    parser.add_argument("--vmin", type=float, default=None, help="Lower end of the colour range (default: the data's minimum)")
    parser.add_argument("--vmax", type=float, default=None, help="Upper end of the colour range (default: the data's maximum)")
    return parser.parse_args()

def main():
    args = parse_args()

    x, phi, u, conc, grad, rate, prm = fetch_intp_data(args.input)

    direction = prm["direction"]
    if args.direction is not None:
        direction = axis2index[args.direction]
    x_min = np.array(prm["x_min"])
    x_max = np.array(prm["x_max"])

    if len(conc) > 0:
        D = prm["D"]

    imgfolder = os.path.join(os.path.dirname(args.input), "Images")
    create_folder_safely(imgfolder)

    tdims = [0, 1, 2]
    tdims.remove(direction)

    ma = np.array(np.logical_not(phi), dtype=bool)
    
    X, Y = np.meshgrid(x[tdims[0]], x[tdims[1]], indexing='ij')

    # Colour range from --vmin/--vmax, else the data; extend the colorbar where the fluid exceeds it
    vmin = dict()
    vmax = dict()
    extend = dict()
    for species in conc:
        vmin[species] = conc[species].min() if args.vmin is None else args.vmin
        vmax[species] = conc[species].max() if args.vmax is None else args.vmax
        fluid = conc[species][np.logical_not(ma)]
        below = fluid.size > 0 and fluid.min() < vmin[species]
        above = fluid.size > 0 and fluid.max() > vmax[species]
        extend[species] = {(False, False): "neither", (True, False): "min",
                           (False, True): "max", (True, True): "both"}[(below, above)]

    zvals = list(enumerate(x[direction]))
    for iz, z in zvals[mpi_rank::mpi_size]:
        ind = [slice(len(x[0])), slice(len(x[1])), slice(len(x[2]))]
        ind[direction] = iz
        ind = tuple(ind)
        for species in conc:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            if species == "delta":
                cmap = plt.get_cmap("coolwarm")
            else:
                cmap = plt.get_cmap("viridis")
            pcm = ax.pcolormesh(X, Y, np.ma.masked_where(ma[ind], conc[species][ind]), cmap=cmap, shading="nearest", vmin=vmin[species], vmax=vmax[species])
            if args.arrows:
                ax.streamplot(X.T, Y.T,
                              np.ma.masked_where(ma[ind], u[tdims[0]][ind]).T,
                              np.ma.masked_where(ma[ind], u[tdims[1]][ind]).T, color='r', density=1.0)
            ax.set_aspect("equal")
            ax.set_xlabel(f"${index2axis[tdims[0]]}$", fontsize=16)
            ax.set_ylabel(f"${index2axis[tdims[1]]}$", fontsize=16)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            ax.tick_params(labelsize=16)
            cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, extend=extend[species]) #, label=f"Concentration of ${species}$") # 
            cbar.ax.tick_params(labelsize=16) 
            cbar.ax.set_ylabel(f"${species}$", rotation=90, labelpad=20, fontsize=16)

            ax.set_title(f"${index2axis[direction]}$ = {x[direction][iz]:.2f}", fontsize=16)
            fig.tight_layout()
            plt.savefig(os.path.join(imgfolder, f"scan_{species}_{index2axis[direction]}_step{iz:06d}.png"))
            plt.close()

if __name__ == "__main__":
    main()