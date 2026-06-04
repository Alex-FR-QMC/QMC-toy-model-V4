"""
6d-δ — Test principal : χ(τ).

Cadrage : 6d-delta-cadrage.md

Test : la présence transitoire du mode lent produit par P6 modifie-t-elle
la réponse du régime verrouillé β=60 à une seconde perturbation P′ ?

Quantité mesurée :
χ(τ) = AUC trajectorielle sur Δt = 100 de
       ‖ R(P6(τ), P′, t) − R(E₀, P′, t) ‖
       
où R(E, P′, t) est la trajectoire après application de P′ depuis l'état E.

Délais τ : {0_nominal, 50, 100, 200, 400, 800, 1500, 3000}
- τ=0_nominal : R6 formé après n_short=10, avant relaxation prolongée
- τ=3000 : proxy τ=∞, contrôle de cohérence (devrait donner χ ≈ 0)

Verdict ternaire préinscrit :
- H0 : χ(τ) ≈ 0 pour tous les τ > 0_nominal → mode lent passif
- HA1 : χ(τ) non trivial pendant fenêtre lente, réponse PLUS RICHE
       → mode lent réactivant
- HA2 : χ(τ) non trivial pendant fenêtre lente, réponse PLUS FAIBLE
       → mode lent verrouillant

Seuils préinscrits AVANT lecture des résultats :
- "χ trivial" : |χ(τ)| < 1% × max|χ(τ)| sur l'ensemble des délais
  (relatif, pas absolu, parce qu'on ne sait pas a priori l'échelle)
- "réponse plus riche" : norme intégrée de la réponse depuis P6(τ)
  > 110% de celle depuis E₀
- "réponse plus faible" : norme intégrée < 90%

Paramètres :
- strength_P' = 0.008385 (calibré au test précédent)
- Δt = 100 unités de temps
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

def evolve_with_trajectory(psi, h, D, beta, gamma, h0, dt, n_steps,
                            sample_every=1):
    """Évolution en enregistrant la trajectoire."""
    psis = [psi.copy()]
    hs = [h.copy()]
    for n in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        if (n + 1) % sample_every == 0:
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
    """Gaussienne centrale atténuée — P' du test 6d-δ."""
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


def trajectory_distance_to_reference(psis_test, hs_test,
                                      psis_ref, hs_ref):
    """Distance instantanée entre deux trajectoires.
    Renvoie une série temporelle de ||état_test(t) - état_ref(t)||."""
    diffs = []
    n_steps = min(len(psis_test), len(psis_ref))
    for k in range(n_steps):
        dpsi = psis_test[k] - psis_ref[k]
        dh = hs_test[k] - hs_ref[k]
        d = np.sqrt(np.linalg.norm(dpsi)**2 + np.linalg.norm(dh)**2)
        diffs.append(float(d))
    return np.array(diffs)


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_dt_response = int(100.0 / dt)  # Δt = 100 pour mesurer la réponse

    # État de référence stabilisé
    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    # Calibrages (refait pour s'assurer de la cohérence)
    def input_amp(P, s, **kw):
        return float(np.linalg.norm(P(psi_base, s, **kw) - psi_base))

    # On utilise P1' standard comme amp de référence pour P6
    # (cf. cycle 6d-γ : amp_target_base = ||P1'||)
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
    s_P6 = brentq(lambda s: input_amp(P6_face_dipole, s) - amp_P1prime_std,
                  1e-4, 0.99, xtol=1e-6)
    s_P_prime = 0.008385  # calibré au test précédent

    amp_P6 = input_amp(P6_face_dipole, s_P6)
    amp_P_prime = input_amp(P_prime, s_P_prime)

    print(f"=== Paramètres ===")
    print(f"  dt = {dt:.5f}")
    print(f"  Δt response = {n_dt_response} steps = {n_dt_response*dt:.1f} unités")
    print(f"  s_P6 = {s_P6:.5f}, ||P6|| = {amp_P6:.4e}")
    print(f"  s_P' = {s_P_prime:.6f}, ||P'|| = {amp_P_prime:.4e}")
    print(f"  Ratio ||P'||/||P6|| = {amp_P_prime/amp_P6:.4f}")

    # === ÉTAPE 1 : Référence — réponse à P' depuis E₀ ===
    print(f"\n=== Étape 1 : Réponse de référence R(E₀, P′) ===")

    # Trajectoire "sans P'" depuis E0 (utilisée pour comparer)
    psi_E0_no_pp, h_E0_no_pp = evolve_with_trajectory(
        psi_base.copy(), h_base.copy(),
        D, beta_lock, gamma_v, h0, dt, n_dt_response)

    # Trajectoire "avec P'" depuis E0
    psi_E0_after_pp = P_prime(psi_base.copy(), s_P_prime)
    h_E0_after_pp = h_base.copy()
    psis_E0_resp, hs_E0_resp = evolve_with_trajectory(
        psi_E0_after_pp, h_E0_after_pp,
        D, beta_lock, gamma_v, h0, dt, n_dt_response)

    # R(E0, P') = distance instantanée entre les deux trajectoires
    R_E0 = trajectory_distance_to_reference(
        psis_E0_resp, hs_E0_resp, psi_E0_no_pp, h_E0_no_pp)
    print(f"  R(E₀, P') : min={R_E0.min():.4e}, max={R_E0.max():.4e}")
    print(f"  AUC ref (∫R(E₀)) = {R_E0.sum()*dt:.4e}")

    # === ÉTAPE 2 : Pour chaque τ, calculer R(P6(τ), P′) puis χ(τ) ===
    print(f"\n=== Étape 2 : Réponse depuis P6(τ) pour chaque τ ===")

    delays_target = [0, 50, 100, 200, 400, 800, 1500, 3000]
    # τ=0 nominal : R6 formé après n_short, avant relaxation prolongée
    
    # Préparer P6(τ=0_nominal)
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_R6_init, h_R6_init = evolve(psi_P6, h_P6,
                                     D, beta_lock, gamma_v, h0, dt, n_short)

    # On va parcourir les délais en accumulant la relaxation post-formation
    # Mais pour τ=0, c'est juste R6_init
    states_at_tau = {}  # τ -> (psi, h)
    states_at_tau[0] = (psi_R6_init.copy(), h_R6_init.copy())

    # Relaxer progressivement pour atteindre chaque τ
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

    # Pour chaque τ, mesurer la réponse à P' et calculer χ(τ)
    results = []
    print(f"\n  {'τ':>8} {'||resp_P6||':>14} {'||resp_E0||':>14} "
          f"{'ratio':>10} {'χ(τ)':>14}")

    for tau in delays_target:
        psi_tau, h_tau = states_at_tau[tau]

        # Trajectoire "sans P'" depuis P6(τ) — pour comparer
        psis_tau_no_pp, hs_tau_no_pp = evolve_with_trajectory(
            psi_tau.copy(), h_tau.copy(),
            D, beta_lock, gamma_v, h0, dt, n_dt_response)

        # Trajectoire "avec P'" depuis P6(τ)
        psi_tau_pp = P_prime(psi_tau.copy(), s_P_prime)
        h_tau_pp = h_tau.copy()
        psis_tau_resp, hs_tau_resp = evolve_with_trajectory(
            psi_tau_pp, h_tau_pp,
            D, beta_lock, gamma_v, h0, dt, n_dt_response)

        # R(P6(τ), P') = écart trajectoriel entre les deux trajectoires
        R_tau = trajectory_distance_to_reference(
            psis_tau_resp, hs_tau_resp, psis_tau_no_pp, hs_tau_no_pp)

        # χ(τ) = AUC de la différence avec R(E₀, P')
        diff = np.abs(R_tau - R_E0)
        chi_tau = float(diff.sum() * dt)

        # AUC de la réponse elle-même (pour comparer "plus riche" / "plus faible")
        auc_resp = float(R_tau.sum() * dt)
        auc_resp_E0 = float(R_E0.sum() * dt)
        ratio_resp = auc_resp / auc_resp_E0

        results.append({
            "tau": tau,
            "AUC_response_from_P6_tau": auc_resp,
            "AUC_response_from_E0": auc_resp_E0,
            "ratio_resp_over_ref": ratio_resp,
            "chi_tau": chi_tau,
            "R_E0_max": float(R_E0.max()),
            "R_tau_max": float(R_tau.max()),
        })
        print(f"  {tau:>8} {auc_resp:>14.4e} {auc_resp_E0:>14.4e} "
              f"{ratio_resp:>10.4f} {chi_tau:>14.4e}")

    # === ÉTAPE 3 : Verdict préinscrit ===
    print(f"\n=== Étape 3 : Verdict ===")
    chis = np.array([r["chi_tau"] for r in results])
    ratios = np.array([r["ratio_resp_over_ref"] for r in results])

    max_chi = float(chis.max())
    chi_threshold = 0.01 * max_chi  # 1% du max comme seuil "trivial"

    # τ=3000 (contrôle de cohérence)
    chi_3000 = chis[-1]
    print(f"\n  Contrôle interne :")
    print(f"  χ(τ=3000) = {chi_3000:.4e}")
    print(f"  Devrait être proche de 0 si §12.2.a est juste.")
    print(f"  Ratio χ(τ=3000) / max(χ) = {chi_3000/max_chi:.4f}")

    # Identifier les τ avec χ non trivial (> 1% du max)
    non_trivial_taus = [results[i]["tau"] for i in range(len(results))
                        if chis[i] > chi_threshold]
    print(f"\n  τ avec χ(τ) non trivial (> 1% du max) : {non_trivial_taus}")
    print(f"  Seuil de trivialité : {chi_threshold:.4e}")

    # Pour les τ dans la fenêtre lente, regarder ratio_resp
    print(f"\n  Direction des réponses (ratio AUC résp / AUC référence) :")
    for r in results:
        if r["chi_tau"] > chi_threshold:
            direction = "plus riche" if r["ratio_resp_over_ref"] > 1.1 else \
                        "plus faible" if r["ratio_resp_over_ref"] < 0.9 else \
                        "comparable"
            print(f"    τ={r['tau']:>5} : χ non trivial, ratio={r['ratio_resp_over_ref']:.4f} → {direction}")
        else:
            print(f"    τ={r['tau']:>5} : χ trivial")

    # Classification ternaire
    # H0 : tous les chi triviaux (sauf éventuellement τ=0)
    # HA1 : χ non trivial dans fenêtre lente ET ratio > 1.1
    # HA2 : χ non trivial dans fenêtre lente ET ratio < 0.9
    fenetre_lente = [50, 100, 200, 400]
    chis_in_window = [results[i]["chi_tau"] for i in range(len(results))
                      if results[i]["tau"] in fenetre_lente]
    ratios_in_window = [results[i]["ratio_resp_over_ref"]
                        for i in range(len(results))
                        if results[i]["tau"] in fenetre_lente]
    any_non_trivial_in_window = any(c > chi_threshold for c in chis_in_window)
    any_richer = any(r > 1.1 for r in ratios_in_window)
    any_weaker = any(r < 0.9 for r in ratios_in_window)

    print(f"\n  Classification :")
    if not any_non_trivial_in_window:
        verdict = "H0 — Mode lent passif"
        details = "χ trivial sur toute la fenêtre lente"
    elif any_richer and not any_weaker:
        verdict = "HA1 — Mode lent réactivant"
        details = "χ non trivial dans fenêtre lente, réponse plus riche"
    elif any_weaker and not any_richer:
        verdict = "HA2 — Mode lent verrouillant"
        details = "χ non trivial dans fenêtre lente, réponse plus faible"
    elif any_richer and any_weaker:
        verdict = "PATTERN MIXTE"
        details = "χ non trivial avec réponses dans les deux directions selon τ"
    else:
        verdict = "INTERMÉDIAIRE"
        details = "χ non trivial mais réponses comparables (ni plus riches ni plus faibles)"

    print(f"\n  VERDICT : {verdict}")
    print(f"  {details}")

    output = {
        "dt": float(dt),
        "n_dt_response": n_dt_response,
        "delta_t": n_dt_response * dt,
        "s_P6": float(s_P6),
        "amp_P6": amp_P6,
        "s_P_prime": float(s_P_prime),
        "amp_P_prime": amp_P_prime,
        "ratio_amp_P_prime_over_P6": amp_P_prime / amp_P6,
        "AUC_R_E0": float(R_E0.sum() * dt),
        "results_by_tau": results,
        "max_chi": max_chi,
        "chi_threshold_trivial": chi_threshold,
        "chi_3000": float(chi_3000),
        "chi_3000_ratio_to_max": float(chi_3000 / max_chi),
        "non_trivial_taus": non_trivial_taus,
        "fenetre_lente_taus": fenetre_lente,
        "any_richer_in_window": bool(any_richer),
        "any_weaker_in_window": bool(any_weaker),
        "verdict": verdict,
        "verdict_details": details,
    }
    with open("/home/claude/mcq_v4/6d_delta_chi_tau.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
