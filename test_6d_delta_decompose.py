"""
6d-δ — Décomposition amplitude / résidu vectoriel.

Sur les données déjà produites par test_6d_delta_main_v2.py.

Pour chaque τ, on décompose :

    Δ_τ(t) = a_τ · Δ_3000(t) + r_τ(t)

où a_τ est le scalaire optimal au sens des moindres carrés sur toute la
trajectoire (toutes les cellules, tous les pas de temps) et r_τ est le
résidu vectoriel orthogonal à Δ_3000.

Mesures :
- a_τ : "facteur d'amplitude" relatif à la référence
- ||r_τ|| : norme du résidu vectoriel
- ||Δ_τ|| : norme totale de la réponse
- ratio_amplitude = (a_τ - 1) · ||Δ_3000|| : contribution attendue
  d'une pure modulation d'amplitude à χ_slow
- ratio_resid = ||r_τ|| : contribution du résidu vectoriel

Si ratio_resid >> ratio_amplitude, alors χ_slow mesure principalement
autre chose qu'une modulation d'intensité globale.
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
    n_dt_response = int(100.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    def P1prime_std(psi, strength=0.05):
        factor = np.ones_like(psi)
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                    if r <= 1.5: factor[i,j,k] += strength
        p = psi * factor
        return p / p.sum()
    amp_P1prime_std = float(np.linalg.norm(P1prime_std(psi_base) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    s_P_prime = 0.008385

    # États P6(τ)
    delays_target = [0, 50, 100, 200, 400, 800, 1500, 3000]
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_R6_init, h_R6_init = evolve(psi_P6, h_P6,
                                     D, beta_lock, gamma_v, h0, dt, n_short)
    states_at_tau = {0: (psi_R6_init.copy(), h_R6_init.copy())}
    psi_curr, h_curr = psi_R6_init.copy(), h_R6_init.copy()
    t_current = 0.0
    for tau in delays_target[1:]:
        dt_seg = tau - t_current
        n_seg = int(dt_seg / dt)
        if n_seg > 0:
            psi_curr, h_curr = evolve(psi_curr, h_curr,
                                       D, beta_lock, gamma_v, h0, dt, n_seg)
        t_current = tau
        states_at_tau[tau] = (psi_curr.copy(), h_curr.copy())

    # Δ vectoriels
    deltas = {}
    for tau in delays_target:
        psi_tau, h_tau = states_at_tau[tau]
        d_psi, d_h = compute_delta(psi_tau, h_tau, s_P_prime,
                                    D, beta_lock, gamma_v, h0, dt, n_dt_response)
        deltas[tau] = (d_psi, d_h)

    # Référence : Δ_3000
    d_psi_ref, d_h_ref = deltas[3000]
    # Aplatir en un vecteur unique (toutes les cellules, tous les pas de temps)
    ref_flat = np.concatenate([d_psi_ref.flatten(), d_h_ref.flatten()])
    ref_norm_sq = float(np.dot(ref_flat, ref_flat))

    print(f"=== Décomposition Δ_τ = a_τ · Δ_3000 + r_τ ===\n")
    print(f"  ||Δ_3000|| (vecteur global) = {np.sqrt(ref_norm_sq):.4e}")
    print(f"\n  {'τ':>6} {'a_τ':>10} {'||r_τ||':>14} {'(a_τ-1)·||Δ_ref||':>20} "
          f"{'χ_slow':>14}")

    results = []
    for tau in delays_target:
        d_psi, d_h = deltas[tau]
        # Vecteur global
        d_flat = np.concatenate([d_psi.flatten(), d_h.flatten()])
        # Projection : a_τ = <d, ref> / <ref, ref>
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq)
        # Résidu vectoriel
        residual_flat = d_flat - a_tau * ref_flat
        residual_norm = float(np.linalg.norm(residual_flat))
        # Pour comparaison : contribution attendue d'une pure modulation d'amplitude
        amplitude_contribution = abs(a_tau - 1.0) * np.sqrt(ref_norm_sq)
        # Vrai χ_slow (différence vectorielle complète)
        diff_flat = d_flat - ref_flat
        chi_slow_global = float(np.linalg.norm(diff_flat))

        # Vérification : ||diff||² = (a-1)²·||ref||² + ||r||²  (orthogonalité)
        check = abs(chi_slow_global**2 - (amplitude_contribution**2 + residual_norm**2))
        
        results.append({
            "tau": tau,
            "a_tau": a_tau,
            "residual_norm": residual_norm,
            "amplitude_contribution": amplitude_contribution,
            "chi_slow_norm": chi_slow_global,
            "orthogonality_check": check,
        })
        print(f"  {tau:>6} {a_tau:>10.6f} {residual_norm:>14.4e} "
              f"{amplitude_contribution:>20.4e} {chi_slow_global:>14.4e}")

    print(f"\n=== Lecture comparative ===\n")
    print(f"  Pour chaque τ, comparer la part amplitude vs la part résidu :")
    print(f"  {'τ':>6} {'amplitude':>14} {'résidu':>14} {'ratio R/A':>12} "
          f"{'dominant':>14}")
    for r in results:
        a_part = r["amplitude_contribution"]
        r_part = r["residual_norm"]
        if a_part > 1e-30:
            ratio = r_part / a_part
        else:
            ratio = float('inf')
        if ratio > 3:
            dom = "résidu"
        elif ratio < 1/3:
            dom = "amplitude"
        else:
            dom = "comparable"
        print(f"  {r['tau']:>6} {a_part:>14.4e} {r_part:>14.4e} {ratio:>12.2f} "
              f"{dom:>14}")

    print(f"\n=== Interprétation ===\n")
    # Sur la fenêtre lente (τ=0 à 200)
    window = [r for r in results if r["tau"] in [0, 50, 100, 200]]
    avg_ratio = np.mean([r["residual_norm"] /
                          max(r["amplitude_contribution"], 1e-30)
                          for r in window])
    print(f"  Ratio moyen résidu/amplitude sur fenêtre lente : {avg_ratio:.2f}")
    if avg_ratio > 3:
        interpretation = (
            "RÉSIDU DOMINE : χ_slow mesure principalement quelque chose qui "
            "n'est pas une modulation d'amplitude globale. La lecture "
            "'effet non réductible à l'amplitude' est étayée."
        )
    elif avg_ratio < 1/3:
        interpretation = (
            "AMPLITUDE DOMINE : χ_slow mesure principalement une modulation "
            "d'amplitude. La lecture 'effet structurel' n'est pas étayée."
        )
    else:
        interpretation = (
            f"PART COMPARABLE : amplitude et résidu sont du même ordre "
            f"(ratio {avg_ratio:.2f}). Les deux composantes contribuent."
        )
    print(f"  {interpretation}")

    output = {
        "delays": delays_target,
        "ref_norm": float(np.sqrt(ref_norm_sq)),
        "results_by_tau": results,
        "avg_residual_over_amplitude_on_window": float(avg_ratio),
        "interpretation": interpretation,
    }
    with open("/home/claude/mcq_v4/6d_delta_decompose.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
