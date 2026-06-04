"""
6d-δ — Extension de Δt pour distinguer divergence inachevée vs
structure stable.

Test minimal : reprendre exactement le calcul du résidu r_τ=0 aux
fenêtres Δt ∈ {100, 200, 400, 800} et regarder trois choses :

1. ||r_τ=0|| : sature-t-elle ou continue à croître ?
2. La direction de r_τ=0 normalisée : est-elle stable d'un Δt à l'autre ?
3. La brisure de symétrie x : stable d'un Δt à l'autre ?

- Si la direction normalisée converge entre Δt=200 et Δt=800 alors
  qu'on a constance des projections x/y/z : structure stable, divergence
  d'amplitude.
- Si la direction continue à tourner : transitoire pas convergé,
  interprétation de l'analyse précédente à reconsidérer.

Aucun nouveau calcul de référence Δ_3000 ; on garde la définition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import json
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

def evolve_with_trajectory(psi, h, D, beta, gamma, h0, dt, n_steps):
    psis = np.zeros((n_steps + 1,) + psi.shape)
    hs = np.zeros((n_steps + 1,) + h.shape)
    psis[0] = psi
    hs[0] = h
    for n in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        psis[n + 1] = psi
        hs[n + 1] = h
    return psis, hs

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


def P6_face_dipole(psi, strength):
    factor = np.ones_like(psi)
    factor[0, :, :] += strength
    factor[4, :, :] -= strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()

def P_prime(psi, strength, sigma_p=0.8):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                factor[i,j,k] += strength * np.exp(-0.5 * r2 / sigma_p**2)
    p = psi * factor
    return p / p.sum()


def P1prime_std(psi, strength=0.05):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= 1.5: factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()


def compute_delta(psi_start, h_start, s_P_prime, D, beta, gamma_v, h0, dt, n_dt):
    psis_ref, hs_ref = evolve_with_trajectory(
        psi_start.copy(), h_start.copy(),
        D, beta, gamma_v, h0, dt, n_dt)
    psi_pp = P_prime(psi_start.copy(), s_P_prime)
    h_pp = h_start.copy()
    psis_with, hs_with = evolve_with_trajectory(
        psi_pp, h_pp, D, beta, gamma_v, h0, dt, n_dt)
    return psis_with - psis_ref, hs_with - hs_ref


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    amp_P1prime_std = float(np.linalg.norm(P1prime_std(psi_base) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    s_P_prime = 0.008385

    # États : τ=0 et τ=3000 (référence)
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6,
                                D, beta_lock, gamma_v, h0, dt, n_short)
    # État τ=3000 = relaxation prolongée
    n_3000 = int(3000.0 / dt)
    psi_tau3000, h_tau3000 = evolve(psi_tau0.copy(), h_tau0.copy(),
                                      D, beta_lock, gamma_v, h0, dt, n_3000)

    # Tester Δt ∈ {100, 200, 400, 800}
    dt_targets = [100, 200, 400, 800]
    print(f"=== Test extension Δt : structure de r_τ=0 ===\n")
    print(f"  s_P6 = {s_P6:.5f}, s_P' = {s_P_prime:.6f}")
    print(f"  Pour chaque Δt, on calcule :")
    print(f"  - ||r_τ=0|| à la fin de la fenêtre")
    print(f"  - direction de r_τ=0 normalisée")
    print(f"  - projections spatiales selon x, y, z\n")

    # On va calculer une seule fois pour le plus grand Δt et extraire
    # les sous-séquences correspondantes
    max_dt_target = max(dt_targets)
    n_dt_max = int(max_dt_target / dt)

    print(f"  Calcul des trajectoires pour Δt = {max_dt_target} (le max)...")
    d_psi_tau0, d_h_tau0 = compute_delta(psi_tau0, h_tau0, s_P_prime,
                                          D, beta_lock, gamma_v, h0, dt, n_dt_max)
    d_psi_ref, d_h_ref = compute_delta(psi_tau3000, h_tau3000, s_P_prime,
                                        D, beta_lock, gamma_v, h0, dt, n_dt_max)

    # Pour chaque Δt, prendre le résidu à la fin de la fenêtre
    print(f"\n  {'Δt':>5} {'||r_τ=0||':>14} {'a_τ=0':>10} {'||r_resid||':>14} "
          f"{'cos→prev':>10} {'dev_x→4-x':>10}")

    results = []
    prev_resid_norm_vec = None
    for dt_target in dt_targets:
        n_dt = int(dt_target / dt)
        # Calculer r_τ=0 = Δ_τ=0 - a · Δ_3000 sur la trajectoire tronquée à n_dt
        d_psi_sub = d_psi_tau0[:n_dt + 1]
        d_h_sub = d_h_tau0[:n_dt + 1]
        d_psi_ref_sub = d_psi_ref[:n_dt + 1]
        d_h_ref_sub = d_h_ref[:n_dt + 1]

        d_flat = np.concatenate([d_psi_sub.flatten(), d_h_sub.flatten()])
        ref_flat = np.concatenate([d_psi_ref_sub.flatten(), d_h_ref_sub.flatten()])
        ref_norm_sq = float(np.dot(ref_flat, ref_flat))
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq) if ref_norm_sq > 1e-30 else 0.0

        # Résidu vectoriel global
        resid_flat = d_flat - a_tau * ref_flat
        resid_norm = float(np.linalg.norm(resid_flat))
        resid_normed = resid_flat / max(resid_norm, 1e-30)

        # cos avec le résidu du Δt précédent (alignés sur la partie commune)
        if prev_resid_norm_vec is not None:
            # Pour comparer, il faut un vecteur de même taille. On compare
            # la portion commune (le résidu précédent vit sur n_dt_prev,
            # le nouveau sur n_dt). Cela ne fonctionne pas directement.
            # À la place : prendre le RÉSIDU À L'INSTANT FINAL t=Δt-final
            cos_with_prev = "see snapshot"
        else:
            cos_with_prev = "—"

        # Snapshot du résidu à l'instant final
        # r_psi à l'instant final (du sous-Δt)
        r_psi_final = d_psi_sub[-1] - a_tau * d_psi_ref_sub[-1]
        r_psi_final_norm = float(np.linalg.norm(r_psi_final))
        # Projections
        proj_x = np.abs(r_psi_final).sum(axis=(1,2))
        proj_y = np.abs(r_psi_final).sum(axis=(0,2))
        proj_z = np.abs(r_psi_final).sum(axis=(0,1))
        # Déviation à symétrie x→4-x
        if np.linalg.norm(proj_x) > 1e-30:
            dev_x = float(np.linalg.norm(proj_x - proj_x[::-1]) / np.linalg.norm(proj_x))
        else:
            dev_x = 0.0

        results.append({
            "dt_target": dt_target,
            "n_dt": n_dt,
            "norm_delta_tau0": float(np.linalg.norm(d_flat)),
            "a_tau": a_tau,
            "resid_norm_global": resid_norm,
            "r_psi_at_final_t_norm": r_psi_final_norm,
            "proj_x": proj_x.tolist(),
            "proj_y": proj_y.tolist(),
            "proj_z": proj_z.tolist(),
            "deviation_x_to_4_minus_x": dev_x,
            "_r_psi_final": r_psi_final,  # pour comparaison ci-dessous
        })
        print(f"  {dt_target:>5} {float(np.linalg.norm(d_flat)):>14.4e} {a_tau:>10.6f} "
              f"{resid_norm:>14.4e} {'see below':>10} {dev_x:>10.4f}")

    # Comparer les directions des résidus FINALS entre Δt successifs
    print(f"\n  Stabilité de la direction du résidu (r_psi à l'instant final) :")
    print(f"  {'pair':>20} {'cos':>10}")
    for i in range(1, len(results)):
        r_prev = results[i-1]["_r_psi_final"]
        r_curr = results[i]["_r_psi_final"]
        n_prev = np.linalg.norm(r_prev)
        n_curr = np.linalg.norm(r_curr)
        if n_prev > 1e-30 and n_curr > 1e-30:
            cos_val = float(np.sum(r_prev * r_curr) / (n_prev * n_curr))
        else:
            cos_val = 0.0
        label = f"Δt={results[i-1]['dt_target']}→{results[i]['dt_target']}"
        print(f"  {label:>20} {cos_val:>+10.4f}")

    # Vérifier aussi si proj_x est stable entre Δt
    print(f"\n  Stabilité des projections x (forme normalisée à somme 1) :")
    for r in results:
        px = np.array(r["proj_x"])
        if px.sum() > 1e-30:
            px_norm = px / px.sum()
        else:
            px_norm = px
        print(f"  Δt={r['dt_target']:>4} : {px_norm}")

    # Verdict
    print(f"\n=== Lecture ===")
    norms_global = [r["resid_norm_global"] for r in results]
    growth = norms_global[-1] / norms_global[0]
    print(f"  ||r_resid_global|| à Δt=100  : {norms_global[0]:.4e}")
    print(f"  ||r_resid_global|| à Δt=800  : {norms_global[-1]:.4e}")
    print(f"  Croissance : ×{growth:.2f}")

    # Stabilité de direction finale
    cos_finals = []
    for i in range(1, len(results)):
        r_prev = results[i-1]["_r_psi_final"]
        r_curr = results[i]["_r_psi_final"]
        n_prev = np.linalg.norm(r_prev)
        n_curr = np.linalg.norm(r_curr)
        if n_prev > 1e-30 and n_curr > 1e-30:
            cos_finals.append(float(np.sum(r_prev * r_curr) / (n_prev * n_curr)))
    if cos_finals:
        print(f"  cos directions finales successives : {cos_finals}")
        avg_cos = float(np.mean(cos_finals))
        print(f"  cos moyen : {avg_cos:.4f}")

    # Nettoyer le _r_psi_final avant export JSON
    for r in results:
        del r["_r_psi_final"]

    output = {
        "s_P6": float(s_P6),
        "s_P_prime": float(s_P_prime),
        "dt_targets": dt_targets,
        "results_by_dt": results,
        "cos_directions_finales_successives": cos_finals,
        "growth_factor_norm": float(growth),
    }
    with open("/home/claude/mcq_v4/6d_delta_dt_extension.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
