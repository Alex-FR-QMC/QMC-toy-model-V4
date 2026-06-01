"""
Inspection des résidus r_i = R_i(∞) - R_base du contact 11.
Pas d'alignement. Juste regarder la morphologie spatiale des résidus.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from mcq_v4.factorial_6d import N_AXIS, DX, cfl_dt_max
from mcq_v4.factorial_6d.engine import compute_diffusion_flux, compute_divergence
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero


def rhs(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi = compute_divergence(Jx, Jy, Jz)
    dh = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi, dh


def step(psi, h, D, beta, gamma, h0, dt):
    dpsi, dh = rhs(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi, h + dt * dh


def evolve(psi, h, D, beta, gamma, h0, dt, n_steps):
    for _ in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return psi, h


def make_psi_centered(sigma=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                psi[i,j,k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()


def P1prime_plateau(psi, strength=0.05, radius=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= radius:
                    factor[i,j,k] += strength
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P2prime_corona(psi, strength, radius_inner=2.0):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r >= radius_inner:
                    factor[i,j,k] += strength
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P3prime_anisotropic_z(psi, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for k in range(N_AXIS):
        dist_z = abs(k - c)
        factor[:,:,k] *= 1.0 + strength * (2.0 * dist_z / c - 1.0)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def main():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0
    strength_P2pp = 0.012789

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))

    # État de base
    n_stab = int(50.0 / dt)
    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)

    perturbations = {
        "P1''": lambda p: P1prime_plateau(p),
        "P2''": lambda p: P2prime_corona(p, strength=strength_P2pp),
        "P3''": lambda p: P3prime_anisotropic_z(p),
    }

    # R_i après court relax (10), puis prolongation jusqu'à t=200 (stationnarité)
    n_short = int(10.0 / dt)
    n_long = int(200.0 / dt)

    R_states = {}
    for name, P in perturbations.items():
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        # Court relax pour former R_i (comme contact 11)
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        # Puis relaxation prolongée jusqu'à stationnarité
        psi_R, h_R = evolve(psi_R, h_R, D, beta, gamma, h0, dt, n_long)
        R_states[name] = (psi_R, h_R)

    # Calculer les résidus
    print(f"État de base : ψ_inhomo={psi_base.max()/max(psi_base.min(),1e-30):.3f}, h_min={h_base.min():.3e}\n")

    # Pour chaque résidu, examiner :
    # 1) où il est concentré (argmax|r|, distance au centre)
    # 2) sa structure spatiale (concentré central ? périphérique ? anisotrope ?)
    # 3) son amplitude

    center = (N_AXIS - 1) / 2.0

    for name in perturbations:
        psi_R, h_R = R_states[name]
        r_psi = psi_R - psi_base
        r_h = h_R - h_base

        print(f"=== Résidu {name} ===")
        print(f"  Amplitudes : ||r_ψ||={np.linalg.norm(r_psi):.4e}, "
              f"||r_h||={np.linalg.norm(r_h):.4e}")

        # Position du max de |r_ψ|
        argmax_psi = np.unravel_index(int(np.argmax(np.abs(r_psi))), r_psi.shape)
        argmax_h = np.unravel_index(int(np.argmax(np.abs(r_h))), r_h.shape)
        print(f"  argmax|r_ψ| = {argmax_psi}, valeur = {r_psi[argmax_psi]:+.4e}")
        print(f"  argmax|r_h| = {argmax_h}, valeur = {r_h[argmax_h]:+.4e}")

        # Distance moyenne au centre, pondérée par |r_ψ|
        coords = np.arange(N_AXIS)
        I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
        dist_from_center = np.sqrt((I - center)**2 + (J - center)**2 + (K - center)**2)
        weights = np.abs(r_psi)
        if weights.sum() > 0:
            mean_dist_psi = (weights * dist_from_center).sum() / weights.sum()
        else:
            mean_dist_psi = float('nan')
        weights_h = np.abs(r_h)
        if weights_h.sum() > 0:
            mean_dist_h = (weights_h * dist_from_center).sum() / weights_h.sum()
        else:
            mean_dist_h = float('nan')
        print(f"  distance moyenne au centre (pondérée |r_ψ|) = {mean_dist_psi:.3f}")
        print(f"  distance moyenne au centre (pondérée |r_h|) = {mean_dist_h:.3f}")

        # Anisotropie : variance des distances aux 3 axes
        # |r_ψ| projeté sur chaque axe (somme sur 2 autres axes)
        proj_x = np.abs(r_psi).sum(axis=(1,2))
        proj_y = np.abs(r_psi).sum(axis=(0,2))
        proj_z = np.abs(r_psi).sum(axis=(0,1))
        print(f"  projections |r_ψ| sur axes :")
        print(f"    x : {proj_x}")
        print(f"    y : {proj_y}")
        print(f"    z : {proj_z}")

        # Signe : le résidu est-il positif au centre, négatif à la périphérie, ou autre ?
        psi_center = r_psi[2,2,2]
        psi_corner = r_psi[0,0,0]
        psi_face = r_psi[0,2,2]
        print(f"  r_ψ au centre (2,2,2) = {psi_center:+.4e}")
        print(f"  r_ψ au coin (0,0,0) = {psi_corner:+.4e}")
        print(f"  r_ψ au milieu de face (0,2,2) = {psi_face:+.4e}")
        print()


if __name__ == "__main__":
    main()
