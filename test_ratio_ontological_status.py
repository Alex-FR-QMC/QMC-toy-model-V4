"""
6d-β cycle 2 — statut ontologique du ratio Δh/Δψ ≈ 5.3666.

Question : sous quelle FORME le ratio survit-il à des transformations
contrôlées ? (pas "survit-il exactement ?")

Trois tests, trois hypothèses ontologiques, dans cet ordre :
1. Heun (RK2) vs Euler  → robustesse au schéma temporel
2. β=58 / β=62          → invariant ponctuel vs corridor
3. N_AXIS=7 vs 5        → support discret (asymétrie interprétative :
                          structure de dérive > proximité brute)

Critères de survie FIXÉS AVANT MESURE (ratio réf = 5.3666) :

Heun :
  - survie forte : [5.36, 5.37]
  - survie faible : [5.2, 5.5]
  - rupture : hors [5.2, 5.5]

β± :
  - corridor : ratio ∈ [5.2, 5.5] à β=58 ET β=62
  - ponctuel : 5.3666 à β=60 seulement, dérive nette ailleurs
  - dérive continue : variation monotone lisse avec β

N_AXIS (asymétrie interprétative) :
  - survie : [5.2, 5.5] à N_AXIS=7
  - renormalisation : dérive STRUCTURÉE (loi d'échelle)
  - rupture : dérive non interprétable

Protocole de mesure du ratio (identique aux tests précédents) :
- init gaussienne sigma=1.5
- perturbation psi[centre] *= 1.01 à t=0, renormalisée
- évolution jusqu'à t=500
- ratio = ||Δh_final|| / ||Δψ_final|| vs trajectoire de référence
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mcq_v4.factorial_6d import DX, cfl_dt_max  # noqa: E402
from mcq_v4.factorial_6d.engine import (  # noqa: E402
    compute_diffusion_flux,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero  # noqa: E402


def compute_divergence_local(Jx_face, Jy_face, Jz_face, n_axis):
    """Version locale de compute_divergence, agnostique à la taille.

    Identique à mcq_v4.factorial_6d.engine.compute_divergence mais
    dérive n_axis de l'argument au lieu de la constante hardcodée.
    Le moteur n'est PAS modifié — ceci est une fonction utilitaire de
    test uniquement, pour permettre le test N_AXIS=7.
    """
    dpsi_dt = np.zeros((n_axis, n_axis, n_axis), dtype=float)
    dpsi_dt[:-1, :, :] -= Jx_face / DX
    dpsi_dt[1:, :, :] += Jx_face / DX
    dpsi_dt[:, :-1, :] -= Jy_face / DX
    dpsi_dt[:, 1:, :] += Jy_face / DX
    dpsi_dt[:, :, :-1] -= Jz_face / DX
    dpsi_dt[:, :, 1:] += Jz_face / DX
    return dpsi_dt


def rhs(psi, h, D, beta, gamma, h0):
    n_axis = psi.shape[0]
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi = compute_divergence_local(Jx, Jy, Jz, n_axis)
    dh = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi, dh


def step_euler(psi, h, D, beta, gamma, h0, dt):
    dpsi, dh = rhs(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi, h + dt * dh


def step_heun(psi, h, D, beta, gamma, h0, dt):
    """Heun (RK2) : prédicteur Euler + correction par moyenne des pentes."""
    dpsi1, dh1 = rhs(psi, h, D, beta, gamma, h0)
    psi_pred = psi + dt * dpsi1
    h_pred = h + dt * dh1
    dpsi2, dh2 = rhs(psi_pred, h_pred, D, beta, gamma, h0)
    psi_new = psi + dt * 0.5 * (dpsi1 + dpsi2)
    h_new = h + dt * 0.5 * (dh1 + dh2)
    return psi_new, h_new


def make_psi_centered(n_axis, sigma=1.5):
    coords = np.arange(n_axis) * DX
    c = (n_axis - 1) * DX / 2.0
    psi = np.zeros((n_axis, n_axis, n_axis))
    for i in range(n_axis):
        for j in range(n_axis):
            for k in range(n_axis):
                r2 = ((coords[i] - c)**2 + (coords[j] - c)**2 +
                      (coords[k] - c)**2)
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()


def measure_ratio(n_axis, beta, step_fn, sigma=1.5,
                  gamma=1.0, D=0.1, h0=1.0, t_total=500.0):
    """Mesure le ratio ||Δh||/||Δψ|| pour une config donnée."""
    center = (n_axis - 1) // 2

    psi_ref = make_psi_centered(n_axis, sigma)
    h_ref = np.full((n_axis, n_axis, n_axis), h0)
    psi_max_init = float(psi_ref.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max_init + gamma))
    n_total = int(t_total / dt)

    # Branche perturbée (perturbation à t=0)
    psi_pert = psi_ref.copy()
    psi_pert[center, center, center] *= 1.01
    psi_pert /= psi_pert.sum()
    h_pert = np.full((n_axis, n_axis, n_axis), h0)

    # Évolution parallèle
    for k in range(1, n_total + 1):
        psi_ref, h_ref = step_fn(psi_ref, h_ref, D, beta, gamma, h0, dt)
        psi_pert, h_pert = step_fn(psi_pert, h_pert, D, beta, gamma, h0, dt)

    delta_psi = float(np.linalg.norm(psi_pert - psi_ref))
    delta_h = float(np.linalg.norm(h_pert - h_ref))
    ratio = delta_h / max(delta_psi, 1e-30)

    return {
        "n_axis": n_axis,
        "beta": beta,
        "dt": dt,
        "delta_psi": delta_psi,
        "delta_h": delta_h,
        "ratio": ratio,
        "h_min_ref": float(h_ref.min()),
        "psi_inhomo_ref": float(psi_ref.max() / max(psi_ref.min(), 1e-30)),
    }


def classify_survival(ratio, ref=5.3666):
    if 5.36 <= ratio <= 5.37:
        return "SURVIE FORTE"
    elif 5.2 <= ratio <= 5.5:
        return "survie faible"
    else:
        return "RUPTURE"


if __name__ == "__main__":
    REF = 5.3666
    print(f"{'='*70}")
    print(f"Statut ontologique du ratio Δh/Δψ ≈ {REF}")
    print(f"  Critères fixés avant mesure.")
    print(f"{'='*70}")

    all_results = {}

    # ===== TEST 1 : Heun vs Euler =====
    print(f"\n{'─'*70}")
    print(f"TEST 1 — Heun (RK2) vs Euler (N_AXIS=5, β=60)")
    print(f"{'─'*70}")

    r_euler = measure_ratio(5, 60.0, step_euler)
    r_heun = measure_ratio(5, 60.0, step_heun)
    print(f"  Euler : ratio = {r_euler['ratio']:.4f} "
          f"(dt={r_euler['dt']:.5f})")
    print(f"  Heun  : ratio = {r_heun['ratio']:.4f} "
          f"(dt={r_heun['dt']:.5f})")
    print(f"  Classification Heun : {classify_survival(r_heun['ratio'])}")
    all_results["test_1_heun"] = {
        "euler": r_euler, "heun": r_heun,
        "classification": classify_survival(r_heun['ratio']),
    }

    # ===== TEST 2 : β=58 / β=62 =====
    print(f"\n{'─'*70}")
    print(f"TEST 2 — β=58 / β=60 / β=62 (N_AXIS=5, Euler)")
    print(f"{'─'*70}")

    r_b58 = measure_ratio(5, 58.0, step_euler)
    r_b60 = r_euler  # déjà calculé
    r_b62 = measure_ratio(5, 62.0, step_euler)
    print(f"  β=58 : ratio = {r_b58['ratio']:.4f}, "
          f"h_min={r_b58['h_min_ref']:.2e}, "
          f"ψ_inhomo={r_b58['psi_inhomo_ref']:.2f}")
    print(f"  β=60 : ratio = {r_b60['ratio']:.4f}, "
          f"h_min={r_b60['h_min_ref']:.2e}, "
          f"ψ_inhomo={r_b60['psi_inhomo_ref']:.2f}")
    print(f"  β=62 : ratio = {r_b62['ratio']:.4f}, "
          f"h_min={r_b62['h_min_ref']:.2e}, "
          f"ψ_inhomo={r_b62['psi_inhomo_ref']:.2f}")

    in_corridor = (5.2 <= r_b58['ratio'] <= 5.5 and
                   5.2 <= r_b62['ratio'] <= 5.5)
    # Test de monotonie/dérive
    ratios_beta = [r_b58['ratio'], r_b60['ratio'], r_b62['ratio']]
    monotone = (ratios_beta[0] < ratios_beta[1] < ratios_beta[2] or
                ratios_beta[0] > ratios_beta[1] > ratios_beta[2])
    spread = max(ratios_beta) - min(ratios_beta)

    if in_corridor and spread < 0.1:
        beta_verdict = "INVARIANT DE CORRIDOR (stable dans voisinage β=60)"
    elif in_corridor and monotone:
        beta_verdict = "DÉRIVE CONTINUE LÉGÈRE (corridor + tendance)"
    elif not in_corridor and monotone:
        beta_verdict = "DÉRIVE CONTINUE (coordonnée de régime paramétré)"
    elif spread > 0.5:
        beta_verdict = "INVARIANT PONCTUEL (β=60 singulier)"
    else:
        beta_verdict = "AMBIGU"
    print(f"  Spread = {spread:.4f}")
    print(f"  Verdict β± : {beta_verdict}")
    all_results["test_2_beta"] = {
        "beta_58": r_b58, "beta_60": r_b60, "beta_62": r_b62,
        "spread": spread, "in_corridor": in_corridor,
        "verdict": beta_verdict,
    }

    # ===== TEST 3 : N_AXIS=7 =====
    print(f"\n{'─'*70}")
    print(f"TEST 3 — N_AXIS=7 vs N_AXIS=5 (β=60, Euler)")
    print(f"  [asymétrie interprétative : structure de dérive > proximité]")
    print(f"{'─'*70}")

    r_n5 = r_euler  # déjà calculé
    r_n7 = measure_ratio(7, 60.0, step_euler)
    print(f"  N_AXIS=5 : ratio = {r_n5['ratio']:.4f}, "
          f"h_min={r_n5['h_min_ref']:.2e}, "
          f"ψ_inhomo={r_n5['psi_inhomo_ref']:.2f}")
    print(f"  N_AXIS=7 : ratio = {r_n7['ratio']:.4f}, "
          f"h_min={r_n7['h_min_ref']:.2e}, "
          f"ψ_inhomo={r_n7['psi_inhomo_ref']:.2f}")

    # Note : N_AXIS change le support. On regarde aussi le rapport des ratios
    ratio_of_ratios = r_n7['ratio'] / r_n5['ratio']
    print(f"  Rapport des ratios (N7/N5) = {ratio_of_ratios:.4f}")

    if 5.2 <= r_n7['ratio'] <= 5.5:
        n_verdict = "SURVIE (invariant indépendant de la résolution)"
    else:
        # Asymétrie interprétative : regarder si c'est structuré
        # Si proche d'un facteur simple, possible renormalisation
        print(f"    Le ratio dérive. Analyse de structure :")
        print(f"      ratio N5 = {r_n5['ratio']:.4f}")
        print(f"      ratio N7 = {r_n7['ratio']:.4f}")
        print(f"      N7/N5 cellules : {7**3}/{5**3} = {7**3/5**3:.3f}")
        print(f"      rapport ratios : {ratio_of_ratios:.4f}")
        n_verdict = ("DÉRIVE — à analyser comme renormalisation possible, "
                     "PAS rupture automatique (support modifié)")
    print(f"  Verdict N_AXIS : {n_verdict}")
    all_results["test_3_n_axis"] = {
        "n_axis_5": r_n5, "n_axis_7": r_n7,
        "ratio_of_ratios": ratio_of_ratios,
        "verdict": n_verdict,
    }

    # ===== SYNTHÈSE =====
    print(f"\n{'='*70}")
    print(f"SYNTHÈSE — statut ontologique du 5.3666")
    print(f"{'='*70}")
    print(f"\n  Test 1 (Heun)   : {all_results['test_1_heun']['classification']}")
    print(f"  Test 2 (β±)     : {all_results['test_2_beta']['verdict']}")
    print(f"  Test 3 (N_AXIS) : {all_results['test_3_n_axis']['verdict']}")

    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_ratio_ontological_status.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRésultats : {output_path}")
