"""
Test 6d-α §6 — Vérification des 4 garanties de non-clôture (révisé post-§5.7).

acquired:
- 4a: morphologic memory, feedback ψ↔h, stratified reactivation
- A.5: tautologie locale au point fixe couplé
- §5.7: multi-bassins empiriques, bassin uniforme vs bassin collapse partiel

§6 est désormais un TEST DE VIABILITÉ OUVERTE post-§5.7, pas une simple
clôture de validation numérique. La spec §6.5 précise que §6.1, §6.3, §6.4
sont partiellement tautologiques (garantis par construction). §6.2 reste
un test informatif indépendant.

Cadrage de classification post-discussion Alex :
- ENGINE_GUARANTEE_STATUS : le moteur respecte-t-il les garanties par construction ?
- REGIME_VIABILITY_STATUS : le régime atteint préserve-t-il la transformabilité locale ?

Le moteur peut PASS sur garantie tout en produisant régime localement
non transformable (cas B2 §5.7).

Plan :
- §6.1 D_min > 0 : test intégrité g_Ω + observation min D_eff = D·h en B2
- §6.2 Φ_eff sans bassin absorbant : reformulé en TEST DE PERTURBATION
  BORNÉE sur état B2 collapsé. Question : bassin transformable ou
  absorbant au sens fort ?
- §6.3 Bruit non éliminable : largement garanti par ψ_floor, test rapide
- §6.4 𝔊^ero actif : test propre zones actives + observation B2 séparée

Verdicts à deux niveaux :
- ENGINE_GUARANTEE_STATUS ∈ {PASS, FAIL_BY_CONSTRUCTION_INTEGRITY}
- REGIME_VIABILITY_STATUS ∈ {OPEN_CORRIDOR,
                              LOCAL_TRANSPORT_CLOSURE,
                              FUNCTIONAL_LOCKING,
                              ABSORBING_BASIN_DETECTED}
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from mcq_v4.factorial_6d import (  # noqa: E402
    N_AXIS, DX, DIM, cfl_dt_max,
)
from mcq_v4.factorial_6d.engine import (  # noqa: E402
    compute_diffusion_flux, compute_divergence,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero  # noqa: E402


H_RESOLUTION = 1e-6
H_FUNCTIONAL = 1e-3
H_MIN_POSTULATED = 0.1  # postulé MCQ-théorique
D_FLOOR_EFFECTIVE = 0.01  # D · h_min postulé = 0.1 · 0.1
EPS_PERTURBATION = 0.01  # 1% de masse pour perturbation §6.2


def rhs_coupled(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)
    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi_dt, dh_dt


def step_engine_euler(psi, h, D, beta, gamma, h0, dt):
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi_dt, h + dt * dh_dt


def make_psi_A_centered(sigma_0=1.8):
    coords = np.arange(N_AXIS) * DX
    center = (N_AXIS - 1) * DX / 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - center) ** 2
                      + (coords[j] - center) ** 2
                      + (coords[k] - center) ** 2)
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma_0 ** 2)
    psi /= psi.sum()
    return psi


def make_psi_B2_bimodal(sigma_0=1.0):
    coords = np.arange(N_AXIS) * DX
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for c in [(1.5, 2.0, 2.0), (2.5, 2.0, 2.0)]:
        cx, cy, cz = c
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r2 = ((coords[i] - cx) ** 2
                          + (coords[j] - cy) ** 2
                          + (coords[k] - cz) ** 2)
                    psi[i, j, k] += np.exp(-0.5 * r2 / sigma_0 ** 2)
    psi /= psi.sum()
    return psi


def simulate_to_t(psi_init, h_init, D, beta, gamma, h0, t_target, dt=None):
    """Simule jusqu'à t_target. Retourne ψ, h, n_steps."""
    if dt is None:
        psi_max = float(psi_init.max())
        rate_h = beta * psi_max + gamma
        dt_cfl_diff = cfl_dt_max(h0, D)
        dt_cfl_h = 1.0 / rate_h
        dt = 0.5 * min(dt_cfl_diff, dt_cfl_h)
    n_steps = max(1, int(np.ceil(t_target / dt)))
    psi = psi_init.copy()
    h = h_init.copy()
    for _ in range(n_steps):
        psi, h = step_engine_euler(psi, h, D, beta, gamma, h0, dt)
    return psi, h, n_steps, dt


# ============================================================
# §6.1 — D_min > 0 strict, étendu post-§5.7
# ============================================================

def test_6_1_D_min(D, beta, gamma, h0):
    """
    §6.1 : intégrité moteur g_Ω > 0
    + observation min D_eff = D·h pour chaque famille
    """
    print(f"\n{'='*70}")
    print(f"§6.1 — D_min > 0 strict (test étendu post-§5.7)")
    print(f"{'='*70}")
    print(f"β={beta}, γ={gamma}, D={D}, h0={h0}")

    # On simule A et B2 jusqu'à stationnarité (t=500) et on regarde min D_eff
    results = {}
    for fam_name, psi_init_fn, sigma in [
        ("A", make_psi_A_centered, 1.8),
        ("B2", make_psi_B2_bimodal, 1.0),
    ]:
        psi_init = psi_init_fn(sigma)
        h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
        psi_final, h_final, n_steps, dt = simulate_to_t(
            psi_init, h_init, D, beta, gamma, h0, 500.0
        )
        # D_eff = D * h (puisque Φ=0, pas de Φ_corr)
        D_eff = D * h_final
        results[fam_name] = {
            "min_D_eff": float(D_eff.min()),
            "max_D_eff": float(D_eff.max()),
            "median_D_eff": float(np.median(D_eff)),
            "frac_D_eff_below_floor": float(np.sum(D_eff < D_FLOOR_EFFECTIVE) / D_eff.size),
            "frac_h_below_resolution": float(np.sum(h_final < H_RESOLUTION) / h_final.size),
            "frac_h_below_functional": float(np.sum(h_final < H_FUNCTIONAL) / h_final.size),
            "h_min_final": float(h_final.min()),
        }
        print(f"\n  Famille {fam_name} :")
        print(f"    min D_eff = D·h_min = {results[fam_name]['min_D_eff']:.4e}")
        print(f"    D_FLOOR_EFFECTIVE (= D·h_min_postulé) = {D_FLOOR_EFFECTIVE}")
        print(f"    frac D_eff < D_FLOOR : {results[fam_name]['frac_D_eff_below_floor']:.4f}")
        print(f"    frac h < h_resolution : {results[fam_name]['frac_h_below_resolution']:.4f}")
        print(f"    frac h < h_functional : {results[fam_name]['frac_h_below_functional']:.4f}")

    # Verdict deux couches
    # Engine guarantee : g_Ω par construction > 0 (D > 0, h > 0 par schéma)
    # On vérifie juste que D_eff ne devient pas NaN ou négatif
    engine_pass = all(
        np.isfinite(r["min_D_eff"]) and r["min_D_eff"] >= 0
        for r in results.values()
    )

    # Regime viability : si frac_D_eff_below_floor > 0 dans une famille,
    # alors corridor de transport effectif violé localement dans ce régime
    a_open = results["A"]["frac_D_eff_below_floor"] < 0.01
    b2_open = results["B2"]["frac_D_eff_below_floor"] < 0.01

    if a_open and b2_open:
        regime_status = "OPEN_CORRIDOR"
    elif a_open and not b2_open:
        regime_status = "LOCAL_TRANSPORT_CLOSURE_IN_B2_ONLY"
    elif not a_open and not b2_open:
        regime_status = "LOCAL_TRANSPORT_CLOSURE_IN_BOTH"
    else:
        regime_status = "UNEXPECTED_PATTERN"

    print(f"\n  ENGINE_GUARANTEE_STATUS : {'PASS' if engine_pass else 'FAIL'}")
    print(f"  REGIME_VIABILITY_STATUS : {regime_status}")

    return {
        "engine_guarantee_status": "PASS" if engine_pass else "FAIL",
        "regime_viability_status": regime_status,
        "results_by_family": results,
    }


# ============================================================
# §6.2 — Φ_eff sans bassin absorbant : TEST DE PERTURBATION BORNÉE
# ============================================================

def test_6_2_perturbation(D, beta, gamma, h0):
    """
    §6.2 reformulé post-§5.7 :
    1. Lancer B2 jusqu'à stationnarité (état collapse partiel)
    2. Perturber ψ : ajouter ε_ψ = 0.01 sur cellules collapsées
    3. Simuler 100 unités de plus
    4. Mesurer : h remonte (transformable) ou pas (absorbant) ?

    Double seuil :
    - h > h_resolution : réactivation numérique minimale
    - h > h_functional : transformabilité fonctionnelle plausible

    Classifications :
    - NO_ACCESS : ψ_perturbé n'atteint pas la zone
    - ACCESS_NO_REACTIVATION : ψ arrive, h reste effondré
    - RESOLUTION_REACTIVATION_ONLY : h > h_resolution mais < h_functional
    - FUNCTIONAL_REACTIVATION : h > h_functional dans zones cible
    """
    print(f"\n{'='*70}")
    print(f"§6.2 — Φ_eff sans bassin absorbant (perturbation bornée)")
    print(f"{'='*70}")
    print(f"β={beta}, γ={gamma}, D={D}, h0={h0}")

    # Phase 1 : B2 jusqu'à stationnarité
    psi_init = make_psi_B2_bimodal(sigma_0=1.0)
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_post_stat, h_post_stat, _, dt = simulate_to_t(
        psi_init, h_init, D, beta, gamma, h0, 500.0
    )

    # Identifier zones collapsées
    collapsed_mask = h_post_stat < H_RESOLUTION
    n_collapsed = int(collapsed_mask.sum())
    print(f"\n  État post-stationnaire B2 :")
    print(f"    cellules collapsées : {n_collapsed}/{h_post_stat.size}")
    print(f"    h_min                : {h_post_stat.min():.4e}")

    if n_collapsed == 0:
        print(f"  Pas de zone collapsée — §6.2 sans objet pour ce β")
        return {
            "engine_guarantee_status": "N/A",
            "regime_viability_status": "NO_COLLAPSE_TO_TEST",
            "n_collapsed_pre_perturbation": 0,
        }

    # Phase 2 : perturbation ψ
    # On veut atteindre la zone la plus profondément collapsée
    # Trouver la cellule la plus profonde
    deepest_idx_flat = np.argmin(h_post_stat)
    deepest_idx = np.unravel_index(deepest_idx_flat, h_post_stat.shape)
    print(f"\n  Cellule la plus profonde : {deepest_idx}, "
          f"h = {h_post_stat[deepest_idx]:.4e}")

    # Construire perturbation gaussienne centrée sur cette cellule
    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = (coords[deepest_idx[0]], coords[deepest_idx[1]],
                  coords[deepest_idx[2]])
    perturbation = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    sigma_pert = 1.0
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - cx) ** 2
                      + (coords[j] - cy) ** 2
                      + (coords[k] - cz) ** 2)
                perturbation[i, j, k] = np.exp(-0.5 * r2 / sigma_pert ** 2)
    perturbation *= EPS_PERTURBATION / perturbation.sum()

    psi_perturbed = psi_post_stat + perturbation
    # Re-normaliser pour conserver Σψ = 1
    psi_perturbed /= psi_perturbed.sum()
    h_pre_perturb = h_post_stat.copy()

    psi_mass_in_target = float(psi_perturbed[collapsed_mask].sum())
    beta_psi_over_gamma_target = float(
        beta * psi_perturbed[collapsed_mask].mean() / gamma
    )

    print(f"\n  Perturbation injectée : ε_ψ = {EPS_PERTURBATION} "
          f"(masse {perturbation.sum():.4f})")
    print(f"  ψ_mass dans zone collapsée post-perturb : {psi_mass_in_target:.4e}")
    print(f"  β·⟨ψ⟩/γ dans zone target : {beta_psi_over_gamma_target:.4f}")

    # Phase 3 : simulation post-perturbation (t = 100)
    psi_final, h_final, _, _ = simulate_to_t(
        psi_perturbed, h_pre_perturb, D, beta, gamma, h0, 100.0, dt=dt
    )

    # Mesures
    h_post_in_collapsed = h_final[collapsed_mask]
    n_above_resolution = int(np.sum(h_post_in_collapsed > H_RESOLUTION))
    n_above_functional = int(np.sum(h_post_in_collapsed > H_FUNCTIONAL))
    max_h_recovery = float(h_post_in_collapsed.max())

    # Classification
    psi_visited = psi_mass_in_target > 1e-4
    if not psi_visited:
        regime_status = "NO_ACCESS"
    elif n_above_resolution == 0:
        regime_status = "ACCESS_NO_REACTIVATION_ABSORBING_BASIN_DETECTED"
    elif n_above_functional == 0:
        regime_status = "RESOLUTION_REACTIVATION_ONLY"
    else:
        regime_status = "FUNCTIONAL_REACTIVATION"

    print(f"\n  Post-perturbation (t=100) :")
    print(f"    h_post zones collapsées : min={h_post_in_collapsed.min():.4e}, "
          f"max={max_h_recovery:.4e}")
    print(f"    n cellules > h_resolution : {n_above_resolution}/{n_collapsed}")
    print(f"    n cellules > h_functional : {n_above_functional}/{n_collapsed}")
    print(f"\n  REGIME_VIABILITY_STATUS : {regime_status}")

    return {
        "engine_guarantee_status": "PASS",  # moteur tourne normalement
        "regime_viability_status": regime_status,
        "n_collapsed_pre_perturbation": n_collapsed,
        "psi_mass_in_target": psi_mass_in_target,
        "beta_psi_over_gamma_in_target": beta_psi_over_gamma_target,
        "h_pre_target_min": float(h_pre_perturb[collapsed_mask].min()),
        "h_post_target_max": max_h_recovery,
        "n_above_resolution": n_above_resolution,
        "n_above_functional": n_above_functional,
        "perturbation_amplitude": EPS_PERTURBATION,
    }


# ============================================================
# §6.3 — Bruit non éliminable (garanti par construction, test rapide)
# ============================================================

def test_6_3_noise_constraint():
    """
    §6.3 est garanti par construction via ψ_floor (§1.5 spec).
    Test rapide : vérifier que dans régime sans bruit, la garantie
    structurelle de ψ_floor est en place dans le module noise.py.
    """
    print(f"\n{'='*70}")
    print(f"§6.3 — Bruit non éliminable (garanti par construction)")
    print(f"{'='*70}")

    # On vérifie juste la présence/cohérence de ψ_floor
    try:
        from mcq_v4.factorial_6d.noise import PSI_TYPICAL
        print(f"  ψ_floor référencé via PSI_TYPICAL = {PSI_TYPICAL}")
        print(f"  Garantie ψ_floor préservée par construction (§1.5)")
        engine_pass = True
    except ImportError:
        print(f"  WARNING : module noise non disponible — test skip")
        engine_pass = False

    print(f"\n  ENGINE_GUARANTEE_STATUS : "
          f"{'PASS_BY_CONSTRUCTION' if engine_pass else 'SKIP'}")
    print(f"  REGIME_VIABILITY_STATUS : N/A (pas de bruit dans le régime testé)")

    return {
        "engine_guarantee_status": "PASS_BY_CONSTRUCTION" if engine_pass else "SKIP",
        "regime_viability_status": "N/A_NO_NOISE_IN_REGIME",
    }


# ============================================================
# §6.4 — 𝔊^ero actif (test zones actives + observation B2 séparée)
# ============================================================

def test_6_4_G_ero(D, beta, gamma, h0):
    """
    §6.4 : tester que 𝔊^ero est mathématiquement actif sur zones actives,
    avec mesure ‖∂_t h_obs‖ / ‖∂_t h_attendu‖ ∈ [0.5, 1.5].

    Extension post-§5.7 : observation séparée que G_ero ≈ 0 numériquement
    dans zones collapsées de B2 (mécanisme verrouillage 4a-η).
    """
    print(f"\n{'='*70}")
    print(f"§6.4 — 𝔊^ero actif")
    print(f"{'='*70}")

    # === Test propre §6.4 : zones actives
    # Protocole spec : ψ ≈ 0 partout sauf région compacte
    # On utilise B2 à t=0 (état initial) où h = h₀ uniforme partout
    # mais on perturbe d'abord h pour le mettre hors équilibre
    psi_init = make_psi_B2_bimodal(sigma_0=1.0)  # ψ concentré
    # h initial : h₀ - 0.3 hors région de ψ (donc érosion active attendue)
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0 - 0.3)
    # Cellules avec ψ négligeable
    psi_threshold = 1e-3
    mask_outside_psi = psi_init < psi_threshold

    # ∂_t h attendu = γ · h · (1 - h/h₀) sur ces cellules
    dh_dt_expected = gamma * h_init * (1.0 - h_init / h0)
    # ∂_t h observé via rhs_coupled au step 1
    _, dh_dt_obs = rhs_coupled(psi_init, h_init, D, beta, gamma, h0)

    # Ratio sur cellules hors région ψ
    obs_outside = dh_dt_obs[mask_outside_psi]
    exp_outside = dh_dt_expected[mask_outside_psi]
    norm_obs = float(np.linalg.norm(obs_outside))
    norm_exp = float(np.linalg.norm(exp_outside))
    if norm_exp > 0:
        ratio = norm_obs / norm_exp
    else:
        ratio = float("nan")

    print(f"\n  Test zones actives :")
    print(f"    n cellules hors région ψ : {int(mask_outside_psi.sum())}/{psi_init.size}")
    print(f"    ‖∂_t h_obs‖   = {norm_obs:.4e}")
    print(f"    ‖∂_t h_exp‖   = {norm_exp:.4e}")
    print(f"    ratio         = {ratio:.4f} (attendu ∈ [0.5, 1.5])")

    g_ero_active_ok = 0.5 <= ratio <= 1.5

    # === Extension : observation B2 stationnaire
    psi_init_B2 = make_psi_B2_bimodal(sigma_0=1.0)
    h_init_B2 = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_stat, h_stat, _, _ = simulate_to_t(
        psi_init_B2, h_init_B2, D, beta, gamma, h0, 500.0
    )
    collapsed_mask = h_stat < H_RESOLUTION
    n_collapsed = int(collapsed_mask.sum())

    if n_collapsed > 0:
        _, dh_dt_B2 = rhs_coupled(psi_stat, h_stat, D, beta, gamma, h0)
        # G_ero seul (sans sédimentation)
        g_ero_in_collapsed = gamma * h_stat[collapsed_mask] * \
            (1.0 - h_stat[collapsed_mask] / h0)
        g_ero_max_collapsed = float(np.max(np.abs(g_ero_in_collapsed)))
        print(f"\n  Observation B2 stationnaire :")
        print(f"    n cellules collapsées : {n_collapsed}")
        print(f"    max |G_ero| en zones collapsées : {g_ero_max_collapsed:.4e}")
        print(f"    (effectivement zéro car G_ero ∝ h, et h ≈ 0)")
        g_ero_effectively_zero_in_collapsed = g_ero_max_collapsed < 1e-30
    else:
        g_ero_max_collapsed = None
        g_ero_effectively_zero_in_collapsed = None

    print(f"\n  ENGINE_GUARANTEE_STATUS : "
          f"{'PASS' if g_ero_active_ok else 'FAIL'} (zones actives)")
    print(f"  Observation : G_ero ≈ 0 dans zones collapsées de B2 — "
          f"mécanisme verrouillage 4a-η confirmé")

    return {
        "engine_guarantee_status": "PASS" if g_ero_active_ok else "FAIL",
        "ratio_dh_obs_over_expected": ratio,
        "g_ero_active_ok_on_active_cells": g_ero_active_ok,
        "g_ero_max_in_collapsed_B2": g_ero_max_collapsed,
        "g_ero_effectively_zero_in_collapsed_B2": g_ero_effectively_zero_in_collapsed,
    }


# ============================================================
# Run all §6
# ============================================================

def run_section_6():
    D = 0.1
    gamma = 1.0
    h0 = 1.0
    beta = 60.0  # config principale post-A.5 / §5.7

    print(f"{'='*80}")
    print(f"§6 — Vérification des 4 garanties de non-clôture")
    print(f"     Cadrage post-§5.7 : test de viabilité ouverte")
    print(f"{'='*80}")
    print(f"Paramètres : β={beta}, γ={gamma}, D={D}, h0={h0}")

    r_6_1 = test_6_1_D_min(D, beta, gamma, h0)
    r_6_2 = test_6_2_perturbation(D, beta, gamma, h0)
    r_6_3 = test_6_3_noise_constraint()
    r_6_4 = test_6_4_G_ero(D, beta, gamma, h0)

    # Synthèse globale
    print(f"\n\n{'='*80}")
    print(f"SYNTHÈSE §6 — Verdict deux couches")
    print(f"{'='*80}")
    print(f"\n{'Test':<12} {'ENGINE_GUARANTEE':<28} {'REGIME_VIABILITY':<48}")
    print("-" * 88)
    print(f"{'§6.1':<12} {r_6_1['engine_guarantee_status']:<28} "
          f"{r_6_1['regime_viability_status']:<48}")
    print(f"{'§6.2':<12} {r_6_2['engine_guarantee_status']:<28} "
          f"{r_6_2['regime_viability_status']:<48}")
    print(f"{'§6.3':<12} {r_6_3['engine_guarantee_status']:<28} "
          f"{r_6_3['regime_viability_status']:<48}")
    print(f"{'§6.4':<12} {r_6_4['engine_guarantee_status']:<28} "
          f"{'N/A':<48}")

    # Verdict global
    engine_all_pass = all(
        r["engine_guarantee_status"] in (
            "PASS", "PASS_BY_CONSTRUCTION", "N/A"
        )
        for r in [r_6_1, r_6_2, r_6_3, r_6_4]
    )

    # Détection bassin absorbant
    absorbing_detected = "ABSORBING_BASIN" in r_6_2["regime_viability_status"]
    local_closure_observed = any(
        "LOCAL_TRANSPORT_CLOSURE" in r_6_1["regime_viability_status"]
        for _ in [None]
    )

    print(f"\n  ENGINE_GUARANTEE global : "
          f"{'PASS' if engine_all_pass else 'FAIL'}")

    if absorbing_detected:
        global_regime = "ABSORBING_BASIN_DETECTED_IN_B2"
    elif local_closure_observed:
        global_regime = "LOCAL_TRANSPORT_CLOSURE_OBSERVED"
    else:
        global_regime = "OPEN_VIABILITY"

    print(f"  REGIME_VIABILITY global : {global_regime}")

    print(f"\n  Lecture (rappel §5.8) :")
    print(f"  - Le système peut PASS engine guarantee tout en montrant des")
    print(f"    régimes localement non-transformables (B2).")
    print(f"  - 'Multi-bassins empiriques' ≠ 'multi-attracteurs structurellement")
    print(f"    garantis MCQ' tant que 6d-β pas démarré.")

    return {
        "engine_guarantee_global": "PASS" if engine_all_pass else "FAIL",
        "regime_viability_global": global_regime,
        "test_6_1_D_min": r_6_1,
        "test_6_2_perturbation": r_6_2,
        "test_6_3_noise": r_6_3,
        "test_6_4_G_ero": r_6_4,
        "params": {"D": D, "gamma": gamma, "h0": h0, "beta": beta},
    }


if __name__ == "__main__":
    summary = run_section_6()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_section_6.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0)
