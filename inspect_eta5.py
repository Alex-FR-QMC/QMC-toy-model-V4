"""
6d-γ — Inspection de η₅ = ε₅ - c·v.

Le test P5 a montré que w_center est le seul proxy survivant.
Mais 52% de ε₅ reste non capturé par c·v.

Question (Alex) : quelle morphologie porte η₅ ?
- Si diffus/incohérent → w_center est proche de la bonne coordonnée
- Si organisé → P5 a révélé une seconde direction survivante

Méthode : regarder η₅ AVANT de construire un nouveau proxy.
- Morphologie spatiale brute
- Symétrie (cubique ou non)
- Localisation (centre, couronne, périphérie)
- Comparer à η_i pour i=1,2,3,4 pour voir si η est une signature
  spécifique à P5 ou une propriété générique
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from scipy.optimize import brentq

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


def P1prime(psi, strength=0.05, radius=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= radius: factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

def P2prime(psi, strength, radius_inner=2.0):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r >= radius_inner: factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

def P3prime(psi, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for k in range(N_AXIS):
        d = abs(k - c)
        factor[:,:,k] *= 1.0 + strength * (2.0 * d / c - 1.0)
    p = psi * factor
    return p / p.sum()

def P4(psi, strength, r_inner=1.0, r_outer=2.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= r_inner: factor[i,j,k] += strength
                elif r >= r_outer: factor[i,j,k] -= strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()

def P5_neighbors_only(psi, strength):
    factor = np.ones_like(psi)
    neighbors = [(1,2,2), (3,2,2), (2,1,2), (2,3,2), (2,2,1), (2,2,3)]
    for (i,j,k) in neighbors:
        factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()


def main():
    gamma, D, h0, beta = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)

    # Calibrer P5
    def input_amp(P, s):
        return float(np.linalg.norm(P(psi_base, s) - psi_base))
    amp_target = input_amp(P1prime, 0.05)
    s_P5 = brentq(lambda s: input_amp(P5_neighbors_only, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)

    perturbations = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
        ("P5", lambda p: P5_neighbors_only(p, strength=s_P5)),
    ]

    # Calcul de tous les résidus et Δψ_centre
    r_psi_list = []
    dpsi_center = []
    for name, P in perturbations:
        delta = P(psi_base) - psi_base
        dpsi_center.append(float(delta[2,2,2]))
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta, gamma, h0, dt, n_long)
        r_psi_list.append(psi_R - psi_base)

    # u_ref signe-harmonisé
    r_1_n = r_psi_list[0] / np.linalg.norm(r_psi_list[0])
    aligned = []
    for r in r_psi_list:
        if (r * r_1_n).sum() < 0:
            aligned.append(-r / np.linalg.norm(r))
        else:
            aligned.append(r / np.linalg.norm(r))
    u_ref = np.mean(aligned, axis=0)
    u_ref = u_ref / np.linalg.norm(u_ref)

    # ε_i = r_i - α_i · u_ref
    eps_list = []
    alphas = []
    for r in r_psi_list:
        a = float((r * u_ref).sum())
        alphas.append(a)
        eps_list.append(r - a * u_ref)

    # v ajusté sur P1-P4 uniquement (comme dans le test précédent)
    c_vec_4 = np.array(dpsi_center[:4])
    v_fit = sum(c_vec_4[i] * eps_list[i] for i in range(4)) / np.sum(c_vec_4**2)

    # η_i = ε_i - c_i · v pour chaque cas
    print(f"=== Calcul de η_i = ε_i - c_i · v ===\n")
    print(f"  v ajusté sur P1-P4 (P5 NON inclus dans l'ajustement de v)")
    print(f"  ||v|| = {np.linalg.norm(v_fit):.4e}")
    print(f"\n  {'cas':<6} {'c_i':>14} {'||r_i||':>14} {'||ε_i||':>14} "
          f"{'||η_i||':>14} {'||η||/||ε||':>14}")
    eta_list = []
    for i, (name, _) in enumerate(perturbations):
        eta = eps_list[i] - dpsi_center[i] * v_fit
        eta_list.append(eta)
        eps_norm = np.linalg.norm(eps_list[i])
        eta_norm = np.linalg.norm(eta)
        ratio = eta_norm / eps_norm if eps_norm > 1e-30 else 0
        print(f"  {name:<6} {dpsi_center[i]:>+14.4e} "
              f"{np.linalg.norm(r_psi_list[i]):>14.4e} "
              f"{eps_norm:>14.4e} "
              f"{eta_norm:>14.4e} {ratio:>14.4f}")

    # Pour P5 spécifiquement, inspecter η₅
    print(f"\n=== Morphologie spatiale de η₅ ===")
    eta_5 = eta_list[4]
    print(f"  ||η₅|| = {np.linalg.norm(eta_5):.4e}")

    argmax_eta = np.unravel_index(int(np.argmax(np.abs(eta_5))), eta_5.shape)
    print(f"  argmax|η₅| = {argmax_eta}, valeur = {eta_5[argmax_eta]:+.4e}")

    # Projections sur axes
    proj_x = np.abs(eta_5).sum(axis=(1,2))
    proj_y = np.abs(eta_5).sum(axis=(0,2))
    proj_z = np.abs(eta_5).sum(axis=(0,1))
    print(f"  projections |η₅| sur axes :")
    print(f"    x : {proj_x}")
    print(f"    y : {proj_y}")
    print(f"    z : {proj_z}")
    same_xyz = (np.allclose(proj_x, proj_y, atol=1e-10) and
                np.allclose(proj_x, proj_z, atol=1e-10))
    print(f"  Cubiquement symétrique ? {same_xyz}")

    # Profil radial signé de η₅
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    cc = (N_AXIS - 1) / 2.0
    dist_center = np.sqrt((I-cc)**2 + (J-cc)**2 + (K-cc)**2)
    unique_dists = sorted(set(np.round(dist_center.flatten(), 4)))

    print(f"\n  Profil radial signé de η₅ :")
    for d in unique_dists:
        mask = np.isclose(dist_center, d, atol=1e-4)
        mean_v = eta_5[mask].mean()
        std_v = eta_5[mask].std()
        n = mask.sum()
        print(f"    r={d:.3f} : <η> = {mean_v:+.4e}  σ = {std_v:.4e}  ({n} cellules)")

    # Comparaison avec η₁, η₂, η₃, η₄
    print(f"\n=== η₅ est-il particulier ou similaire à η₁-η₄ ? ===")
    print(f"  Profils radiaux signés normalisés (η_i / ||η_i||) :")

    eta_normalized = []
    for i, eta in enumerate(eta_list):
        n = np.linalg.norm(eta)
        if n > 1e-30:
            eta_normalized.append(eta / n)
        else:
            eta_normalized.append(None)

    print(f"  {'r':>8}" + "".join(f"{perturbations[i][0]+' norm':>14}"
                                   for i in range(5)))
    for d in unique_dists:
        mask = np.isclose(dist_center, d, atol=1e-4)
        row = f"  {d:>8.3f}"
        for en in eta_normalized:
            if en is not None:
                row += f"{en[mask].mean():>+14.4e}"
            else:
                row += f"{'N/A':>14}"
        print(row)

    # Distances entre les profils normalisés
    print(f"\n  Différences entre η normalisés (test : η₅ est-il dans la")
    print(f"  même direction que η₁-η₄, ou une nouvelle ?) :")
    for i in range(5):
        for j in range(i+1, 5):
            if eta_normalized[i] is not None and eta_normalized[j] is not None:
                # ce qu'on veut savoir : sont-ils colinéaires (même direction)
                # cos angle = <a,b>
                cos_angle = float((eta_normalized[i] * eta_normalized[j]).sum())
                n_i = perturbations[i][0]
                n_j = perturbations[j][0]
                print(f"    {n_i}-{n_j} : cos(angle) = {cos_angle:+.4f}  "
                      f"(1 = même direction, 0 = orthogonal, -1 = opposé)")


if __name__ == "__main__":
    main()
