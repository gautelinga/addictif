"""Post-process a complex reaction network with a bulk (volumetric) kinetic reaction.

Where postprocess_sr_kinetic places the calcite reaction on the grain surface as a
Neumann flux, here the reaction is distributed through the pore fluid and the grain
surface is inert:

    u . grad(u_j) - D * div(grad(u_j)) = nu_j * Da * (1 - Omega)   in the pore space
    D * du_j/dn                        = 0                         on the grain surface

with Omega = [Ca2+][CO3^2-] / K_sp and (u1, u2, u3) the conserved-component basis.
The no-flux condition on the grains is the natural boundary condition of the weak
form, so it is imposed simply by leaving the grain facets out of the linear form;
only the inlet carries a Dirichlet condition.

Omega is a function of the unknowns, so the source is nonlinear and is resolved by an
under-relaxed fixed-point (Picard) iteration.  Unlike the surface version, the source
acts everywhere, so each sweep has to re-speciate the whole mesh rather than the grain
dofs alone.  That is the dominant cost, so a vectorised speciation path is used when
the reaction network is react3; it is checked against the per-node reference routine
the first time it runs.

"""
import argparse
import importlib
import os
from importlib.resources import files

import dolfin as df
import numpy as np
import scipy.interpolate as intp
import mpi4py.MPI as MPI

from addictif.common.utils import (mpi_root, Params, create_folder_safely, helpers,
                                   xdmf_params, mpi_print, axis2index, mpi_max, mpi_min,
                                   Top, Btm, Boundary, SideWalls)

comm = MPI.COMM_WORLD

# CaCO3 -> Ca2+ + CO3^2- written in the (u1, u2, u3) conserved-component basis
nu_calcite = (1.0, -2.0, 1.0)


"""
Da value in the code is Da = (L/d)(Cca2+/C_ref) (Pe^-1) Da_p, where Da_p is the Damköhler number based on pore size and Ca2+ concentration.
D value in the code is D = (d/L) (Pe^-1), where Pe is the peclet number based on pore size
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Post process complex reaction network with bulk reaction and inert grains")
    parser.add_argument("-i", "--input", required=True, type=str, help="Folder with concentration file (required)")
    parser.add_argument("--crn", type=str, default="react3", help="Reaction")
    parser.add_argument("--sols", type=str, default="co2mix2", help="End-member solutions")
    parser.add_argument("--Da", type=float, default=1.0, help="Dimensionless bulk rate constant")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "precip", "diss"],
                        help="Which branch of the rate law is active: both, precipitation only "
                             "(supersaturated nodes), or dissolution only (undersaturated nodes)")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum fixed-point iterations")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance on the reaction rate")
    parser.add_argument("--relax", type=float, default=0.1, help="Under-relaxation factor for the reaction rate")
    parser.add_argument("--sr", type=float, default=None,
                        help="Uniform initial saturation ratio (1 = equilibrium). "
                             "Default: start from the local equilibrium mixing field")
    parser.add_argument("--reference_speciation", action="store_true",
                        help="Force the per-node speciation routine instead of the vectorised one")
    return parser.parse_args()


def load_sols(crn, sols):
    data_text = files("addictif.chemistry." + crn + ".sols").joinpath(sols + ".dat").read_text()
    return eval(data_text)


def speciate_reference(crn, K_, u1, u2, u3):
    """Per-node speciation, valid for any reaction network.

    Returns the concentrations and the indices where the root finder found no
    physical solution; crn.compute_primary_spec signals that by calling exit().
    """
    c = np.zeros((len(u1), crn.nspec))
    bad = []
    for i in range(len(u1)):
        u = [u1[i], u2[i], u3[i]]
        try:
            crn.compute_primary_spec(c[i, :], u, K_)
            crn.compute_secondary_spec(c[i, :], u, K_)
        except SystemExit:
            bad.append(i)
    return c, np.array(bad, dtype=int)


def speciate_react3(crn, K_, u1, u2, u3):
    """Vectorised speciation for react3.

    react3 reduces to a quartic in [H+] whose coefficients depend only on u1 and u2,
    so the whole mesh is solved with one stacked companion-matrix eigenvalue problem
    instead of one polyroots call per node.  Root selection mirrors
    crn.compute_primary_spec: a root is physical if it is real, positive, and yields a
    positive CO2 concentration, and the node is rejected unless exactly one such root
    exists (or two that coincide).
    """
    K1, K2, K3, K4 = K_
    n = len(u1)

    # -c1^4 + (u2 - K1) c1^3 + (K4 - K2 + K1 u1 + K1 u2) c1^2
    #   + (K1 K4 + 2 K2 u1 + K2 u2) c1 + K2 K4 = 0, made monic by dividing through by -1
    b = np.empty((n, 4))
    b[:, 0] = -(K2 * K4)
    b[:, 1] = -(K1 * K4 + 2 * K2 * u1 + K2 * u2)
    b[:, 2] = -(K4 - K2 + K1 * u1 + K1 * u2)
    b[:, 3] = -(u2 - K1)

    companion = np.zeros((n, 4, 4))
    companion[:, 1, 0] = 1.0
    companion[:, 2, 1] = 1.0
    companion[:, 3, 2] = 1.0
    companion[:, :, 3] = -b

    roots = np.linalg.eigvals(companion)

    with np.errstate(invalid="ignore", divide="ignore"):
        accept = (np.abs(roots.imag) < 1e-8) & (roots.real > 0)
        c1_cand = np.where(accept, roots.real, 1.0)
        c0_cand = c1_cand**2 * u1[:, None] / (c1_cand**2 + c1_cand * K1 + K2)
        accept &= c0_cand > 0

    n_accept = accept.sum(axis=1)
    c1_lo = np.min(np.where(accept, roots.real, np.inf), axis=1)
    c1_hi = np.max(np.where(accept, roots.real, -np.inf), axis=1)
    ok = (n_accept == 1) | ((n_accept == 2) & (c1_hi - c1_lo < 1e-7))
    bad = np.flatnonzero(~ok)

    c1 = np.where(ok, c1_lo, 1.0)
    c = np.zeros((n, crn.nspec))
    c[:, 1] = c1
    c[:, 0] = c1**2 * u1 / (c1**2 + c1 * K1 + K2)
    c[:, 2] = c[:, 0] * c1**-1 * K1
    c[:, 3] = c[:, 0] * c1**-2 * K2
    c[:, 4] = u3
    c[:, 5] = c1**-1 * K4
    c[bad, :] = 0.0
    return c, bad


def make_speciator(crn, crn_name, K_, force_reference):
    """Pick a speciation routine and, for the fast path, self-check it once."""
    if force_reference or crn_name != "react3":
        reason = "forced" if force_reference else f"no vectorised path for {crn_name}"
        mpi_print(f"Speciation: per-node reference routine ({reason}).")
        return lambda u1, u2, u3: speciate_reference(crn, K_, u1, u2, u3)

    mpi_print("Speciation: vectorised react3 path (checked against the reference on first use).")
    state = {"checked": False}

    def speciate(u1, u2, u3):
        c, bad = speciate_react3(crn, K_, u1, u2, u3)
        if not state["checked"]:
            state["checked"] = True
            verify_speciation(crn, K_, u1, u2, u3, c, bad)
        return c, bad

    return speciate


def verify_speciation(crn, K_, u1, u2, u3, c, bad, n_sample=512):
    """Compare the vectorised result against the reference on a sample of nodes."""
    good = np.setdiff1d(np.arange(len(u1)), bad, assume_unique=False)
    if len(good) == 0:
        return
    sample = good[np.linspace(0, len(good) - 1, min(n_sample, len(good))).astype(int)]
    c_ref, bad_ref = speciate_reference(crn, K_, u1[sample], u2[sample], u3[sample])
    if len(bad_ref):
        raise RuntimeError("Vectorised speciation accepted nodes the reference routine rejected; "
                           "rerun with --reference_speciation")
    err = np.max(np.abs(c_ref - c[sample]) / (np.abs(c_ref) + 1e-300))
    err = comm.allreduce(err, op=MPI.MAX)
    mpi_print(f"Speciation self-check: max relative deviation from reference = {err:.3e}")
    if err > 1e-8:
        raise RuntimeError(f"Vectorised speciation disagrees with the reference by {err:.3e}; "
                           "rerun with --reference_speciation")


def rate_from_omega(Omega, Da, mode):
    """Bulk rate law. Positive means dissolution, negative means precipitation."""
    R = Da * (1.0 - Omega)
    if mode == "precip":
        # No solid in the pore fluid to dissolve, so only the supersaturated branch acts
        return np.minimum(R, 0.0)
    if mode == "diss":
        return np.maximum(R, 0.0)
    return R


def main():
    args = parse_args()

    crn = importlib.import_module(f"addictif.chemistry.{args.crn}")
    sols = load_sols(args.crn, args.sols)

    c_a = np.zeros(crn.nspec)
    c_b = np.zeros(crn.nspec)

    for i, sol0i in enumerate(sols[0]):
        c_a[i] = sol0i / crn.c_ref
    for i, sol1i in enumerate(sols[1]):
        c_b[i] = sol1i / crn.c_ref

    K_ = crn.equilibrium_constants(crn.c_ref)

    crn.compute_secondary_spec_initial(c_a, K_)
    crn.compute_secondary_spec_initial(c_b, K_)
    u_a = crn.compute_conserved_spec_initial(c_a)
    u_b = crn.compute_conserved_spec_initial(c_b)

    # Solving 5th degree polynomials is expensive.
    # First we make an interpolation scheme to speed things up!
    N_intp = 1000

    alpha = np.linspace(0., 1., N_intp)
    u_mix = np.outer(1 - alpha, u_a) + np.outer(alpha, u_b)

    c_ = np.zeros((len(alpha), crn.nspec))
    for i in range(len(alpha)):
        crn.compute_primary_spec_initial(c_[i, :], u_mix[i, :], K_)
        crn.compute_secondary_spec_initial(c_[i, :], K_)

    c_intp = [intp.InterpolatedUnivariateSpline(alpha, c_[:, ispec]) for ispec in range(crn.nspec)]

    prm = Params()
    prm.load(os.path.join(args.input, "params.dat"))
    mesh_path = os.path.relpath(os.path.join(args.input, prm["mesh"]), os.getcwd())
    u_path = os.path.relpath(os.path.join(args.input, prm["u"]), os.getcwd())
    D = prm["D"]
    tol = prm["tol"]

    linear_solver = "bicgstab"
    preconditioner = "hypre_euclid"

    prm_u = Params(os.path.join(u_path, "params.dat"), required=True)
    mesh_u_path = os.path.join(u_path, prm_u["mesh"])
    direction = axis2index[prm_u["direction"]]

    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), mesh_u_path, "r") as h5f:
        h5f.read(mesh_u, "mesh", False)

    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    u_vel = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), os.path.join(u_path, "u.h5"), "r") as h5f:
        h5f.read(u_vel, "u")

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), mesh_path, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)

    alpha_ = df.Function(S, name="alpha")
    with df.HDF5File(mesh.mpi_comm(), os.path.join(args.input, "delta.h5"), "r") as h5f:
        h5f.read(alpha_, "delta")

    # Translate from delta (-1, 1) to alpha (0, 1)
    alpha_.vector()[:] = 0.5 * (alpha_.vector()[:] + 1)

    c_spec_cm = [df.Function(S, name=f"c_{ispec}_cm") for ispec in range(crn.nspec)]
    saturation_ratio_cm = df.Function(S, name="saturation_ratio_cm")

    for ispec in range(crn.nspec):
        c_spec_cm[ispec].vector()[:] = (c_a[ispec] * (1 - alpha_.vector()[:])
                                        + c_b[ispec] * alpha_.vector()[:]) * crn.c_ref
    saturation_ratio_cm.vector()[:] = (c_spec_cm[4].vector()[:] * c_spec_cm[3].vector()[:]
                                       / crn.saturation_product_calcite())

    c_spec_ = [df.Function(S, name=f"c_{ispec}") for ispec in range(crn.nspec)]
    for ispec in range(crn.nspec):
        c_spec_[ispec].vector()[:] = c_intp[ispec](alpha_.vector()[:]) * crn.c_ref

    u_1 = df.Function(S, name="u_1")
    u_2 = df.Function(S, name="u_2")
    u_3 = df.Function(S, name="u_3")

    u_1.vector()[:] = c_spec_[0].vector()[:] + c_spec_[2].vector()[:] + c_spec_[3].vector()[:]
    u_2.vector()[:] = (c_spec_[1].vector()[:] - c_spec_[2].vector()[:]
                       - 2 * c_spec_[3].vector()[:] - c_spec_[5].vector()[:])
    u_3.vector()[:] = c_spec_[4].vector()[:]

    x = mesh.coordinates()[:]
    x_min = mpi_min(x)
    x_max = mpi_max(x)

    mpi_print("Dimensions:", x_max, x_min)

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
    [sw.mark(subd, 4 + index) for index, sw in enumerate(sidewalls)]
    top.mark(subd, 1)
    btm.mark(subd, 2)

    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    S_DG0 = df.FunctionSpace(mesh, "DG", 0)

    xi = df.TrialFunction(S)
    psi = df.TestFunction(S)

    mpi_print("interpolating velocity field...")
    u_proj_ = df.Function(V, name="u")
    df.LagrangeInterpolator.interpolate(u_proj_, u_vel)
    mpi_print("done.")

    mpi_print("Interpolating norm")
    u_norm_ = df.interpolate(df.CompiledExpression(helpers.AbsVecCell(), u=u_proj_, degree=0), S_DG0)
    u_norm_.rename("u_norm", "u_norm")

    mpi_print("Interpolating cell size")
    h_ = df.interpolate(df.CompiledExpression(helpers.CellSize(), mesh=mesh, degree=0), S_DG0)
    h_.rename("h", "h")

    mpi_print("Computing grid Peclet")
    Pe_el_ = df.Function(S_DG0, name="Pe_el")
    Pe_el_.vector()[:] = u_norm_.vector()[:] * h_.vector()[:] / (2 * D)

    mpi_print("Computing tau")
    tau_ = df.Function(S_DG0, name="tau")
    tau_.vector()[:] = h_.vector()[:] / (2 * u_norm_.vector()[:] + 1e-16)
    arr = 1. - 1. / (Pe_el_.vector()[:] + 1e-16)
    arr[arr < 0] = 0.
    tau_.vector()[:] *= arr
    mpi_print("Done.")

    r_xi = df.dot(u_proj_, df.grad(xi)) - D * df.div(df.grad(xi))
    a_xi = df.dot(u_proj_, df.grad(xi)) * psi * df.dx \
        + D * df.dot(df.grad(psi), df.grad(xi)) * df.dx
    a_xi += tau_ * r_xi * df.dot(u_proj_, df.grad(psi)) * df.dx

    u1_ = df.Function(S, name="u1")
    u2_ = df.Function(S, name="u2")
    u3_ = df.Function(S, name="u3")

    # Only the inlet is prescribed. The grain surface is inert, and no-flux is the
    # natural condition of the weak form, so no grain term appears anywhere below.
    bcs = [df.DirichletBC(S, u_1, subd, 1),
           df.DirichletBC(S, u_2, subd, 1),
           df.DirichletBC(S, u_3, subd, 1)]

    # The dof set is identical for all three components, so the matrix is assembled once
    mpi_print("Assembling system")
    A = df.assemble(a_xi)
    bcs[0].apply(A)

    solver = df.KrylovSolver(linear_solver, preconditioner)
    solver.set_operator(A)
    solver.parameters["relative_tolerance"] = 1e-9
    solver.parameters["monitor_convergence"] = False
    solver.parameters["nonzero_initial_guess"] = True

    R_ = df.Function(S, name="reaction_rate")

    # Volumetric source, with the SUPG-consistent contribution that matches the
    # stabilisation already built into a_xi. The three components share this form up to
    # the stoichiometric factor, so it is assembled once per sweep and then scaled.
    L_unit = R_ * psi * df.dx + tau_ * R_ * df.dot(u_proj_, df.grad(psi)) * df.dx

    fields = [u1_, u2_, u3_]

    def solve_components():
        b_unit = df.assemble(L_unit)
        for j, (field, bc) in enumerate(zip(fields, bcs)):
            b = b_unit.copy()
            b *= nu_calcite[j]
            bc.apply(b)
            solver.solve(field.vector(), b)

    n_owned = u1_.vector().local_size()
    speciate = make_speciator(crn, args.crn, K_, args.reference_speciation)

    def speciate_checked(u1, u2, u3, stage):
        c, bad = speciate(u1, u2, u3)
        n_bad = comm.allreduce(len(bad), op=MPI.SUM)
        if n_bad:
            u1_lo = comm.allreduce(float(np.min(u1)), op=MPI.MIN)
            raise RuntimeError(
                f"Speciation found no physical root at {n_bad} node(s) during {stage}. "
                f"Smallest total dissolved carbon u1 = {u1_lo:.3e}; the bulk sink drives u1 "
                f"negative when the reaction outruns transport. Reduce --Da or --relax.")
        return c

    # Start from the local equilibrium mixing field unless a uniform value is requested
    if args.sr is None:
        Omega_0 = (c_spec_[4].vector().get_local() * c_spec_[3].vector().get_local()
                   / crn.saturation_product_calcite())
        mpi_print("Initial saturation ratio from equilibrium mixing: "
                  f"[{comm.allreduce(float(Omega_0.min()), op=MPI.MIN):.6f}, "
                  f"{comm.allreduce(float(Omega_0.max()), op=MPI.MAX):.6f}]")
    else:
        Omega_0 = np.full(n_owned, args.sr)
        mpi_print(f"Initial saturation ratio set uniformly to {args.sr}")

    R_.vector().set_local(rate_from_omega(Omega_0, args.Da, args.mode))
    R_.vector().apply("insert")

    def update_reaction():
        u1 = u1_.vector().get_local()
        u2 = u2_.vector().get_local()
        u3 = u3_.vector().get_local()

        c_loc = speciate_checked(u1, u2, u3, "the fixed-point iteration")
        Omega = c_loc[:, 4] * c_loc[:, 3] / crn.saturation_product_calcite()

        R_old = R_.vector().get_local()
        R_raw = rate_from_omega(Omega, args.Da, args.mode)
        R_new = (1.0 - args.relax) * R_old + args.relax * R_raw

        R_.vector().set_local(R_new)
        R_.vector().apply("insert")

        # Measured on the un-relaxed update, so that --relax does not silently loosen --rtol
        dR = comm.allreduce(np.max(np.abs(R_raw - R_old), initial=0.), op=MPI.MAX)
        R_mag = comm.allreduce(np.max(np.abs(R_raw), initial=0.), op=MPI.MAX)
        return dR / max(R_mag, 1e-30)

    mpi_print("Initial solve")
    solve_components()

    converged = False
    for it in range(1, args.max_iter + 1):
        rel_change = update_reaction()
        solve_components()
        mpi_print(f"Iteration {it}: relative rate change = {rel_change:.3e}")
        if rel_change < args.rtol:
            converged = True
            mpi_print(f"Converged after {it} iterations.")
            break

    if not converged:
        mpi_print(f"WARNING: not converged after {args.max_iter} iterations.")

    mpi_print("Reconstructing speciation")
    u1 = u1_.vector().get_local()
    u2 = u2_.vector().get_local()
    u3 = u3_.vector().get_local()
    c_new = speciate_checked(u1, u2, u3, "the final reconstruction")

    c_spec_new = [df.Function(S, name=f"c_{ispec}_new") for ispec in range(crn.nspec)]
    for i in range(crn.nspec):
        c_spec_new[i].vector().set_local(c_new[:, i])
        c_spec_new[i].vector().apply("insert")  # unit is mol/l

    pH = df.Function(S, name="pH")
    pH.vector()[:] = -np.log10(c_spec_new[1].vector()[:])

    saturation_ratio = df.Function(S, name="saturation_ratio")
    saturation_ratio.vector()[:] = (c_spec_new[4].vector()[:] * c_spec_new[3].vector()[:]
                                    / crn.saturation_product_calcite())

    # Rate recomputed from the final fields without relaxation, so the written rate and
    # the written concentrations are consistent with each other.
    # Output sign convention is opposite to the source variable R_ driving the solve:
    # positive means precipitation
    Omega_final = saturation_ratio.vector().get_local()
    R_out = df.Function(S, name="reaction_rate")
    R_out.vector().set_local(-rate_from_omega(Omega_final, args.Da, args.mode) / D)
    R_out.vector().apply("insert")

    total_reaction_rate = df.assemble(R_out * df.dx)  # positive values for precipitation

    # Source is the left-hand side of the u3 equation, u . grad(u3) - D div(grad(u3)), on the
    # converged field. div(grad(u3)) vanishes inside every P1 cell, so the diffusive part is
    # recovered weakly from the interior flux jumps; the exterior-facet flux is subtracted so
    # no spurious layer appears at the inlet and outlet. A lumped mass matrix gives the nodal
    # values, being far less oscillatory than the consistent one on unstructured meshes.
    # Source equals the calcium source that drove the solve, so dissolution is positive.
    n = df.FacetNormal(mesh)
    Source = df.Function(S, name="Source")
    Source = df.project((df.dot(u_proj_, df.grad(u3_)) - D * df.div(df.grad(u3_))) / D, S, solver_type="cg", preconditioner_type="hypre_euclid")

    # Global calcium balance: integrating the steady equation over the pore space gives
    # advection through the open ends plus diffusion out of the boundary plus net
    # precipitation = 0. The residual is not machine zero, because SUPG perturbs the
    # discrete equation and the inlet rows are replaced by the Dirichlet condition, but
    # it should be small compared with the total rate once the iteration has converged.
    net_ca_transport = (df.assemble(df.dot(u_proj_, df.grad(u3_)) * df.dx)
                        - D * df.assemble(df.dot(n, df.grad(u3_)) * df.ds)) / D
    balance_residual = net_ca_transport + total_reaction_rate
    balance_scale = max(abs(net_ca_transport), abs(total_reaction_rate), 1e-30)

    mpi_print(f"Total reaction rate: {total_reaction_rate}")
    mpi_print(f"Net Ca transport: {net_ca_transport}")
    mpi_print(f"Balance residual: {balance_residual} (relative {balance_residual / balance_scale:.3e})")

    mpi_print("Start saving")

    output_folder = os.path.join(args.input, f"crn_{args.crn}_{args.sols}_bulk_{args.mode}_Da{args.Da:g}")
    create_folder_safely(output_folder)

    mpi_print("Saving CRN.")
    prm = Params()
    prm["species"] = ",".join([f"c_{ispec}" for ispec in range(crn.nspec)])
    prm["ade"] = os.path.relpath(args.input, output_folder)
    prm["Da"] = args.Da
    prm["mode"] = args.mode
    prm["total_reaction_rate"] = total_reaction_rate
    prm["net_Ca_transport"] = net_ca_transport
    prm["balance_residual"] = balance_residual
    prm["converged"] = int(converged)
    prm.dump(os.path.join(output_folder, "params.dat"))

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "subd.xdmf")) as xdmff:
        xdmff.write(subd)

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(output_folder, "conc_show.xdmf")) as xdmff:
        xdmff.parameters.update(xdmf_params)
        xdmff.write(pH, 0.)
        xdmff.write(saturation_ratio_cm, 0.)
        xdmff.write(saturation_ratio, 0.)
        xdmff.write(R_out, 0.)
        xdmff.write(Source, 0.)
        for ispec in range(crn.nspec):
            xdmff.write(c_spec_new[ispec], 0.)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(output_folder, "conc.h5"), "w") as h5f:
        for ispec in range(crn.nspec):
            h5f.write(c_spec_new[ispec], f"c_{ispec}")
        h5f.write(R_out, "reaction_rate")
        h5f.write(Source, "source")

    # The reaction now fills the pore volume, so the slabs are cell subdomains integrated
    # with dx, not facet subdomains integrated with ds. Cells are binned by their
    # midpoint, which assigns every cell to exactly one slab.
    mpi_print("Integrating reaction rate over slices.")
    Ns = 20
    L = x_max - x_min
    dslice = L[direction] / Ns
    xs = np.linspace(x_min[direction], x_max[direction], Ns, endpoint=False) + 0.5 * dslice

    midpoints = mesh.coordinates()[mesh.cells()].mean(axis=1)[:, direction]
    islice = np.clip(((midpoints - x_min[direction]) / dslice).astype(int), 0, Ns - 1)

    subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    subdomains.set_all(0)
    subdomains.array()[:] = islice + 1

    dx_sub = df.Measure("dx", domain=mesh, subdomain_data=subdomains)

    reac_integrated = np.zeros(Ns)
    volume = np.zeros(Ns)

    for i in range(Ns):
        volume[i] = df.assemble(df.Constant(1.0) * dx_sub(i + 1))
        reac_integrated[i] = df.assemble(R_out * dx_sub(i + 1))

    if mpi_root:
        header = " ".join(["z", "volume", "reaction_rate_integrated"])
        data = np.vstack([xs, volume, reac_integrated]).T
        np.savetxt(os.path.join(output_folder, "reaction_rate_per_slice.dat"), data, header=header)


if __name__ == "__main__":
    main()
