"""
6d-δ — Test principal v2 : χ_slow(τ) avec référence τ=3000.

Correction de v1 sur la base du diagnostic d'Alex :
- La référence E₀ confondait deux effets : (i) effet transitoire du mode
  lent et (ii) écart résiduel asymptotique P6(∞) ≠ E₀ dans le span dominant.
- §12.2.a disait que le mode EXTRA est dissipé, pas que P6 revient à E₀.
- Nouvelle référence : P6(3000), qui est P6 une fois le mode lent dissipé
  mais toujours dans son état asymptotique propre.

Définitions :
- Δ_E(t) = état_avec_P'(t) − état_sans_P'(t)  [vecteur, pas norme]
  trajectoire de réponse à P' depuis l'état E
- χ_slow(τ) = AUC sur Δt = 100 de ||Δ_{P6(τ)}(t) − Δ_{P6(3000)}(t)||
- χ_E0(τ) = AUC de ||Δ_{P6(τ)}(t) − Δ_{E0}(t)||  [mesure secondaire]

Verdict ternaire préinscrit (sur χ_slow) :
- H0 (passif) : χ_slow(τ) ≈ 0 pour tout τ
- HA1 (réactivant) : χ_slow(τ) non trivial dans fenêtre lente, AUC de
  ||Δ_{P6(τ)}|| > AUC de ||Δ_{P6(3000)}||
- HA2 (verrouillant) : χ_slow(τ) non trivial dans fenêtre lente, AUC
  de ||Δ_{P6(τ)}|| < AUC de ||Δ_{P6(3000)}||

Contrôle interne : χ_slow(3000) = 0 par construction (référence elle-même).
χ_E0(3000) mesure l'écart asymptotique déplacé — informatif mais pas
diagnostique pour le mode lent.
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
    """Évolution en enregistrant la trajectoire complète."""
    psis = [psi.copy()]
    hs = [h.copy()]
    for _ in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        psis.append(psi.copy())
        hs.append(h.copy())
    return np.array(psis), np.array(hs)

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


def response_trajectory(psi_E, h_E, D, beta, gamma, h0, dt, n_steps,
                         s_P_prime):
    """Calcule Δ_E(t) = état_avec_P'(t) - état_sans_P'(t) sur n_steps.
    Renvoie deux arrays (delta_psi, delta_h) de shape (n_steps+1, 5,5,5)."""
    # Sans P'
    psis_no, hs_no = evolve_with_trajectory(psi_E.copy(), h_E.copy(),
                                              D, beta, gamma, h0, dt, n_steps)
    # Avec P' (appliquée à t=0)
    psi_pp = P_prime(psi_E.copy(), s_P_prime)
    h_pp = h_E.copy()
    psis_pp, hs_pp = evolve_with_trajectory(psi_pp, h_pp,
                                              D, beta, gamma, h0, dt, n_steps)
    delta_psi = psis_pp - psis_no
    delta_h = hs_pp - hs_no
    return delta_psi, delta_h


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

    # Calibrages identiques à v1
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

    print(f"=== Paramètres ===")
    print(f"  dt = {dt:.5f}, n_dt_response = {n_dt_response} (Δt = {n_dt_response*dt:.1f})")
    print(f"  s_P6 = {s_P6:.5f}, s_P' = {s_P_prime:.6f}")

    # === ÉTAPE 1 : Préparer les états P6(τ) pour tous les τ ===
    print(f"\n=== Étape 1 : Préparation des états P6(τ) ===")

    delays_target = [0, 50, 100, 200, 400, 800, 1500, 3000]

    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_R6_init, h_R6_init = evolve(psi_P6, h_P6,
                                     D, beta_lock, gamma_v, h0, dt, n_short)
    states_at_tau = {}
    states_at_tau[0] = (psi_R6_init.copy(), h_R6_init.copy())

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

    print(f"  États P6(τ) préparés pour τ ∈ {delays_target}")

    # === ÉTAPE 2 : Calculer les trajectoires Δ_E(t) pour chaque τ + E0 ===
    print(f"\n=== Étape 2 : Trajectoires de réponse Δ_E(t) ===")

    # Δ_E0
    delta_psi_E0, delta_h_E0 = response_trajectory(
        psi_base, h_base, D, beta_lock, gamma_v, h0, dt, n_dt_response, s_P_prime)

    # Δ_{P6(τ)} pour chaque τ
    delta_trajectories = {}
    for tau in delays_target:
        psi_tau, h_tau = states_at_tau[tau]
        delta_psi_tau, delta_h_tau = response_trajectory(
            psi_tau, h_tau, D, beta_lock, gamma_v, h0, dt, n_dt_response, s_P_prime)
        delta_trajectories[tau] = (delta_psi_tau, delta_h_tau)
        # Norme de la trajectoire
        norms = np.array([np.sqrt(np.linalg.norm(dp)**2 + np.linalg.norm(dh)**2)
                          for dp, dh in zip(delta_psi_tau, delta_h_tau)])
        AUC = float(norms.sum() * dt)
        print(f"  τ={tau:>5} : AUC ||Δ|| = {AUC:.4e}")

    # === ÉTAPE 3 : Calculer χ_slow et χ_E0 ===
    print(f"\n=== Étape 3 : Mesures χ ===")

    delta_psi_ref, delta_h_ref = delta_trajectories[3000]  # référence τ=3000

    results = []
    print(f"\n  {'τ':>8} {'AUC ||Δ||':>14} {'χ_slow':>14} {'χ_E0':>14} "
          f"{'AUC/AUC(3000)':>16}")

    # AUC de la référence
    norms_ref = np.array([np.sqrt(np.linalg.norm(dp)**2 + np.linalg.norm(dh)**2)
                          for dp, dh in zip(delta_psi_ref, delta_h_ref)])
    AUC_ref_3000 = float(norms_ref.sum() * dt)

    norms_E0 = np.array([np.sqrt(np.linalg.norm(dp)**2 + np.linalg.norm(dh)**2)
                         for dp, dh in zip(delta_psi_E0, delta_h_E0)])
    AUC_E0 = float(norms_E0.sum() * dt)

    for tau in delays_target:
        delta_psi_tau, delta_h_tau = delta_trajectories[tau]

        # χ_slow(τ) : différence vectorielle vs référence τ=3000
        diff_psi_slow = delta_psi_tau - delta_psi_ref
        diff_h_slow = delta_h_tau - delta_h_ref
        norms_diff_slow = np.array([
            np.sqrt(np.linalg.norm(dp)**2 + np.linalg.norm(dh)**2)
            for dp, dh in zip(diff_psi_slow, diff_h_slow)])
        chi_slow = float(norms_diff_slow.sum() * dt)

        # χ_E0(τ) : différence vectorielle vs E0 (mesure secondaire)
        diff_psi_E0 = delta_psi_tau - delta_psi_E0
        diff_h_E0 = delta_h_tau - delta_h_E0
        norms_diff_E0 = np.array([
            np.sqrt(np.linalg.norm(dp)**2 + np.linalg.norm(dh)**2)
            for dp, dh in zip(diff_psi_E0, diff_h_E0)])
        chi_E0 = float(norms_diff_E0.sum() * dt)

        # AUC de la réponse elle-même
        norms_tau = np.array([
            np.sqrt(np.linalg.norm(dp)**2 + np.linalg.norm(dh)**2)
            for dp, dh in zip(delta_psi_tau, delta_h_tau)])
        AUC_tau = float(norms_tau.sum() * dt)
        ratio_AUC = AUC_tau / AUC_ref_3000

        results.append({
            "tau": tau,
            "AUC_response": AUC_tau,
            "chi_slow": chi_slow,
            "chi_E0": chi_E0,
            "ratio_AUC_over_ref3000": ratio_AUC,
        })
        print(f"  {tau:>8} {AUC_tau:>14.4e} {chi_slow:>14.4e} {chi_E0:>14.4e} "
              f"{ratio_AUC:>16.4f}")

    # === ÉTAPE 4 : Verdict préinscrit ===
    print(f"\n=== Étape 4 : Verdict ===")

    chis_slow = np.array([r["chi_slow"] for r in results])
    ratios_AUC = np.array([r["ratio_AUC_over_ref3000"] for r in results])

    # Contrôle interne : χ_slow(3000) doit être ≈ 0 par construction
    chi_slow_3000 = chis_slow[-1]
    print(f"\n  Contrôle interne (par construction) :")
    print(f"  χ_slow(τ=3000) = {chi_slow_3000:.4e}")
    print(f"  (doit être strictement 0 — c'est la référence d'elle-même)")
    if chi_slow_3000 > 1e-15:
        print(f"  → ANOMALIE : référence non égale à elle-même.")

    # Seuil de trivialité
    max_chi_slow = float(chis_slow.max())
    if max_chi_slow > 1e-15:
        chi_threshold = 0.01 * max_chi_slow
    else:
        chi_threshold = 1e-15

    # Classification sur fenêtre lente
    fenetre_lente = [50, 100, 200, 400]
    chis_in_window = [results[i]["chi_slow"] for i in range(len(results))
                      if results[i]["tau"] in fenetre_lente]
    ratios_in_window = [results[i]["ratio_AUC_over_ref3000"]
                        for i in range(len(results))
                        if results[i]["tau"] in fenetre_lente]

    any_non_trivial_in_window = any(c > chi_threshold for c in chis_in_window)
    any_richer = any(r > 1.1 for r in ratios_in_window)
    any_weaker = any(r < 0.9 for r in ratios_in_window)

    print(f"\n  Sur la fenêtre lente {fenetre_lente} :")
    print(f"  χ_slow > seuil ({chi_threshold:.4e}) ? {any_non_trivial_in_window}")
    print(f"  Au moins un AUC > 110% de référence ? {any_richer}")
    print(f"  Au moins un AUC < 90% de référence ? {any_weaker}")

    if not any_non_trivial_in_window:
        verdict = "H0 — Mode lent passif"
        details = "χ_slow trivial sur toute la fenêtre lente"
    elif any_richer and not any_weaker:
        verdict = "HA1 — Mode lent réactivant"
        details = "χ_slow non trivial dans fenêtre lente, réponse plus riche"
    elif any_weaker and not any_richer:
        verdict = "HA2 — Mode lent verrouillant"
        details = "χ_slow non trivial dans fenêtre lente, réponse plus faible"
    elif any_richer and any_weaker:
        verdict = "PATTERN MIXTE"
        details = "Réponses dans les deux directions selon τ"
    else:
        verdict = "INTERMÉDIAIRE"
        details = "χ_slow non trivial mais réponses comparables (ni plus riches ni plus faibles)"

    print(f"\n  VERDICT : {verdict}")
    print(f"  {details}")

    # Mesure secondaire : χ_E0 — susceptibilité asymptotique déplacée
    chi_E0_3000 = results[-1]["chi_E0"]
    print(f"\n  Mesure secondaire (susceptibilité asymptotique déplacée) :")
    print(f"  χ_E0(τ=3000) = {chi_E0_3000:.4e}")
    print(f"  AUC ||Δ_{{E0}}|| = {AUC_E0:.4e}")
    print(f"  AUC ||Δ_{{P6(3000)}}|| = {AUC_ref_3000:.4e}")
    print(f"  Ratio AUC P6(3000)/E0 = {AUC_ref_3000/AUC_E0:.4f}")
    print(f"  → l'écart résiduel P6(∞) ≠ E0 dans le span dominant produit")
    print(f"    une légère différence de susceptibilité (non diagnostique")
    print(f"    pour le mode lent).")

    output = {
        "dt": float(dt),
        "n_dt_response": n_dt_response,
        "delta_t": float(n_dt_response * dt),
        "s_P6": float(s_P6),
        "s_P_prime": float(s_P_prime),
        "AUC_E0": AUC_E0,
        "AUC_ref_3000": AUC_ref_3000,
        "results_by_tau": results,
        "chi_slow_3000_control": float(chi_slow_3000),
        "chi_threshold_trivial": chi_threshold,
        "fenetre_lente_taus": fenetre_lente,
        "any_non_trivial_in_window": bool(any_non_trivial_in_window),
        "any_richer_in_window": bool(any_richer),
        "any_weaker_in_window": bool(any_weaker),
        "verdict": verdict,
        "verdict_details": details,
    }
    with open("/home/claude/mcq_v4/6d_delta_chi_slow.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
