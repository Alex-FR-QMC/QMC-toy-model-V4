"""
Test interne de la réduction r_i = F(Δψ_centre).

P1 et P4 ont Δψ_centre quasi identiques :
- P1 : Δψ(2,2,2) = +8.41e-4
- P4 : Δψ(2,2,2) = +8.32e-4 (différence 1%)

Mais leurs structures spatiales d'entrée sont radicalement différentes :
- P1 : plateau positif uniforme dans une boule centrale
- P4 : bipôle radial (+ au centre, - en périphérie)

Si la réduction r_i = F(Δψ_centre) tient :
- r_1 ≈ r_4 dans TOUTE leur composante (ψ et h), pas seulement ε
- les différences r_1 - r_4 doivent être de l'ordre de la différence
  Δψ_centre, soit ~1%

Si la réduction est partielle :
- r_1 - r_4 a une structure spatiale non négligeable
- on aura identifié une seconde variable qui survit dans le résidu
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

    # Comparer les structures spatiales d'ENTRÉE de P1 et P4
    delta_P1 = P1prime(psi_base) - psi_base
    delta_P4 = P4(psi_base, strength=0.03187) - psi_base

    print(f"=== Comparaison des perturbations d'ENTRÉE ===")
    print(f"  ||ΔP1|| = {np.linalg.norm(delta_P1):.4e}")
    print(f"  ||ΔP4|| = {np.linalg.norm(delta_P4):.4e}")
    print(f"  ΔP1 au centre (2,2,2) : {delta_P1[2,2,2]:+.4e}")
    print(f"  ΔP4 au centre (2,2,2) : {delta_P4[2,2,2]:+.4e}")
    print(f"  Différence relative Δψ_centre : "
          f"{abs(delta_P1[2,2,2] - delta_P4[2,2,2])/abs(delta_P1[2,2,2]):.4f}")

    # Différence des perturbations d'entrée
    diff_input = delta_P1 - delta_P4
    print(f"\n  Différence d'entrée :")
    print(f"  ||ΔP1 - ΔP4|| = {np.linalg.norm(diff_input):.4e}")
    print(f"  ||ΔP1 - ΔP4|| / ||ΔP1|| = "
          f"{np.linalg.norm(diff_input)/np.linalg.norm(delta_P1):.4f}")
    # Donc les entrées diffèrent significativement bien que Δψ_centre soit proche

    # Calcul des résidus
    perturbations = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
    ]
    R = {}
    for name, P in perturbations:
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta, gamma, h0, dt, n_long)
        R[name] = (psi_R - psi_base, h_R - h_base)

    print(f"\n=== Comparaison des résidus FINAUX ===")
    r_psi_1, r_h_1 = R["P1"]
    r_psi_4, r_h_4 = R["P4"]

    diff_r_psi = r_psi_1 - r_psi_4
    diff_r_h = r_h_1 - r_h_4

    print(f"  ||r_psi_1|| = {np.linalg.norm(r_psi_1):.4e}")
    print(f"  ||r_psi_4|| = {np.linalg.norm(r_psi_4):.4e}")
    print(f"  ||r_psi_1 - r_psi_4|| = {np.linalg.norm(diff_r_psi):.4e}")
    print(f"  ||r_psi_1 - r_psi_4|| / ||r_psi_1|| = "
          f"{np.linalg.norm(diff_r_psi)/np.linalg.norm(r_psi_1):.4f}")

    print(f"\n  ||r_h_1|| = {np.linalg.norm(r_h_1):.4e}")
    print(f"  ||r_h_4|| = {np.linalg.norm(r_h_4):.4e}")
    print(f"  ||r_h_1 - r_h_4|| = {np.linalg.norm(diff_r_h):.4e}")
    print(f"  ||r_h_1 - r_h_4|| / ||r_h_1|| = "
          f"{np.linalg.norm(diff_r_h)/np.linalg.norm(r_h_1):.4f}")

    # Détail spatial de la différence
    print(f"\n=== Structure spatiale de r_1 - r_4 ===")
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    cc = (N_AXIS - 1) / 2.0
    dist_center = np.sqrt((I-cc)**2 + (J-cc)**2 + (K-cc)**2)

    proj_x = np.abs(diff_r_psi).sum(axis=(1,2))
    proj_y = np.abs(diff_r_psi).sum(axis=(0,2))
    proj_z = np.abs(diff_r_psi).sum(axis=(0,1))
    print(f"  Projections de |r_psi_1 - r_psi_4| sur axes :")
    print(f"    x : {proj_x}")
    print(f"    y : {proj_y}")
    print(f"    z : {proj_z}")
    same_xyz = (np.allclose(proj_x, proj_y, atol=1e-12) and
                np.allclose(proj_x, proj_z, atol=1e-12))
    print(f"  Cubiquement symétrique ? {same_xyz}")

    argmax_diff = np.unravel_index(int(np.argmax(np.abs(diff_r_psi))), diff_r_psi.shape)
    print(f"  argmax|diff_r_psi| = {argmax_diff}, valeur = "
          f"{diff_r_psi[argmax_diff]:+.4e}")

    # Profil radial signé de la différence
    unique_dists = sorted(set(np.round(dist_center.flatten(), 4)))
    print(f"\n  Profil radial signé de (r_psi_1 - r_psi_4) :")
    for d in unique_dists:
        mask = np.isclose(dist_center, d, atol=1e-4)
        mean_diff = diff_r_psi[mask].mean()
        n_cells = mask.sum()
        print(f"    r={d:.3f} : <diff> = {mean_diff:+.4e}  ({n_cells} cellules)")

    # Test : la différence des résidus est-elle prédite par la différence
    # des Δψ_centre, ou y a-t-il quelque chose d'autre ?
    pred_diff_norm_if_reduction = (
        abs(delta_P1[2,2,2] - delta_P4[2,2,2]) / abs(delta_P1[2,2,2])
        * np.linalg.norm(r_psi_1)
    )
    print(f"\n=== Test de la réduction r_i = F(Δψ_centre) ===")
    print(f"  Si la réduction tient et que F est ~linéaire :")
    print(f"  ||r_1 - r_4|| devrait être de l'ordre de "
          f"{abs(delta_P1[2,2,2] - delta_P4[2,2,2])/abs(delta_P1[2,2,2]):.4f} × ||r_1||")
    print(f"  = {pred_diff_norm_if_reduction:.4e}")
    print(f"  Mesuré : ||r_1 - r_4|| = {np.linalg.norm(diff_r_psi):.4e}")
    ratio = np.linalg.norm(diff_r_psi) / max(pred_diff_norm_if_reduction, 1e-30)
    print(f"  Ratio mesuré/prédit = {ratio:.4f}")
    if ratio < 2.0:
        print(f"  → Compatible avec la réduction (différence des résidus")
        print(f"    explicable par la différence des Δψ_centre)")
    else:
        print(f"  → La différence des résidus est >> prédite par Δψ_centre.")
        print(f"    Une autre variable que Δψ_centre survit.")


if __name__ == "__main__":
    main()
