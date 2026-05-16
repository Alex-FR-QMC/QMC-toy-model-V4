"""
Test 6d-α micro-étape 4a-ζ — classification long terme du quasi-collapse.

Plan validé Alex :
- Régime C (β=125, γ=1, ψ décentré) en priorité aux 3 durées : t = 50, 500, 5000
- Régime A (β=1, γ=0.01) comme témoin stable à t = 500
- Pas de drift, pas de bruit (système minimal préservé)

Trois diagnostics :
1. Temps de retour τ_return par cellule + n_crossings du seuil
2. Volume / surface / périmètre normalisé du support actif au cours du temps
3. Mémoire longue via ||h - h₀||_L2, max|h - h₀|, 
   corr(-log(h/h₀), ∫(ψ - ψ_uniform)·dt)

Double seuil :
- h_resolution = 1e-6 (sous-résolution numérique)
- h_functional = 1e-3 (quasi-léthargie fonctionnelle)

Classification collapse_trend ∈ {EXPANDING, STABILIZED, SHRINKING}
avec tolérance relative 0.01 sur fraction_under, produite SÉPARÉMENT pour
les deux seuils :
- collapse_trend_resolution
- collapse_trend_functional

CAVEAT critique (rappel §4a-0) :
Le verdict est formulé en termes de PERSISTANCE OBSERVÉE dans la fenêtre
simulée, JAMAIS comme irréversibilité structurelle. Une absence de
réactivation à t=5000 borne empiriquement la durée d'observation, pas
le comportement asymptotique du système.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np
from scipy import ndimage

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


def rhs_coupled(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)
    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi_dt, dh_dt


def step_engine_euler(psi, h, D, beta, gamma, h0, dt):
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi_dt, h + dt * dh_dt


def make_psi_decentered(center, sigma_0=1.5):
    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = center
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i]-cx)**2 + (coords[j]-cy)**2 + (coords[k]-cz)**2
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma_0**2)
    psi /= psi.sum()
    return psi


def count_components(h: np.ndarray, threshold: float) -> int:
    mask = (h >= threshold).astype(int)
    struct = ndimage.generate_binary_structure(3, 1)
    _, n = ndimage.label(mask, structure=struct)
    return int(n)


def surface_active(h: np.ndarray, threshold: float) -> int:
    """Nombre de faces (i,j,k)/(i+1,j,k) où l'une est active et l'autre non."""
    mask = h >= threshold
    s = 0
    s += int(np.sum(mask[:-1, :, :] != mask[1:, :, :]))
    s += int(np.sum(mask[:, :-1, :] != mask[:, 1:, :]))
    s += int(np.sum(mask[:, :, :-1] != mask[:, :, 1:]))
    return s


def classify_trend(
    frac_t10: float,
    frac_t50: float,
    frac_t100: float,
    n_cells: int,
    max_crossings: int = 0,
    n_cells_multi_cross: int = 0,
) -> str:
    """
    Classification raffinée du régime collapse.

    Cinq labels possibles :
    - NO_COLLAPSE : frac reste ≈ 0 sur toute la fenêtre (régime stable)
    - OSCILLATING_BOUNDARY : nombreux franchissements du seuil
    - EXPANDING_COLLAPSE : frac croît
    - SHRINKING_COLLAPSE : frac décroît
    - STABILIZED_COLLAPSE : frac > 0 et stable (ni croît ni décroît)

    Règle d'ordre (premier match l'emporte) :
    1. NO_COLLAPSE si max(frac_samples) ≤ frac_tol
    2. OSCILLATING_BOUNDARY si max_crossings ≥ 3 OU
       n_cells_multi_cross > 5% des cellules
    3. EXPANDING si frac croît strictement
    4. SHRINKING si frac décroît strictement
    5. STABILIZED sinon (frac > 0 et plateau)
    """
    # Tolérance pour considérer "présence" significative
    frac_tol_present = 2.0 / n_cells  # au moins 2 cellules sous seuil pour parler de collapse
    # Tolérance pour considérer "variation" significative
    frac_tol_change = 0.01 * max(frac_t50, 1.0 / n_cells)

    max_frac = max(frac_t10, frac_t50, frac_t100)

    # 1. NO_COLLAPSE
    if max_frac <= frac_tol_present:
        return "NO_COLLAPSE"

    # 2. OSCILLATING_BOUNDARY (priorité haute pour ne pas masquer)
    if max_crossings >= 3 or n_cells_multi_cross > 0.05 * n_cells:
        return "OSCILLATING_BOUNDARY"

    # 3-5. Tendances monotones
    delta_early = frac_t50 - frac_t10
    delta_late = frac_t100 - frac_t50

    if delta_early > frac_tol_change and delta_late > frac_tol_change:
        return "EXPANDING_COLLAPSE"
    elif delta_early < -frac_tol_change and delta_late < -frac_tol_change:
        return "SHRINKING_COLLAPSE"
    else:
        return "STABILIZED_COLLAPSE"


def build_log_schedule(n_steps: int, n_dense: int = 30, n_sparse: int = 30) -> list[int]:
    """30 premiers steps + 30 points régulièrement espacés sur la suite."""
    if n_steps <= n_dense:
        return list(range(n_steps + 1))
    dense = list(range(n_dense + 1))
    if n_steps > n_dense:
        sparse_indices = np.linspace(n_dense, n_steps, n_sparse, dtype=int)
        sparse = [int(x) for x in sparse_indices if int(x) not in dense]
        return sorted(set(dense + sparse))
    return dense


def simulate_with_collapse_tracking(
    psi_init, h_init, D, beta, gamma, h0_target,
    n_steps, dt,
):
    psi = psi_init.copy()
    h = h_init.copy()
    integral_psi_centered = np.zeros_like(psi)
    psi_uniform = 1.0 / psi.size

    # Tracking step-by-step pour réactivation
    # below_now : état actuel (sous seuil h_resolution)
    # n_crossings : nombre de fois où cell passe au-dessus puis en-dessous (ou inverse)
    t_first_down = np.full(psi.shape, -1.0)
    t_first_reactivation = np.full(psi.shape, -1.0)
    n_crossings = np.zeros(psi.shape, dtype=int)
    below_prev = h < H_RESOLUTION

    log_schedule = set(build_log_schedule(n_steps))

    logs = {
        "t": [], "step": [],
        "psi_total": [], "psi_min": [], "psi_max": [],
        "h_min": [], "h_max": [], "h_mean": [],
        # Collapse instrumentation
        "frac_under_resolution": [], "frac_under_functional": [],
        "n_components_resolved": [], "n_components_functional": [],
        "surface_resolved": [], "surface_functional": [],
        "perimeter_normalized_resolved": [],
        "perimeter_normalized_functional": [],
        # Mémoire longue
        "h_minus_h0_L2": [],
        "h_minus_h0_max": [],
        "corr_logh_intpsi_centered": [],
    }

    def log_state(step, t, psi, h, integral_centered):
        logs["step"].append(step)
        logs["t"].append(t)
        logs["psi_total"].append(float(psi.sum()))
        logs["psi_min"].append(float(psi.min()))
        logs["psi_max"].append(float(psi.max()))
        logs["h_min"].append(float(h.min()))
        logs["h_max"].append(float(h.max()))
        logs["h_mean"].append(float(h.mean()))

        # Fractions sous seuil
        frac_res = float(np.sum(h < H_RESOLUTION) / h.size)
        frac_func = float(np.sum(h < H_FUNCTIONAL) / h.size)
        logs["frac_under_resolution"].append(frac_res)
        logs["frac_under_functional"].append(frac_func)

        # Composantes connexes
        n_res = count_components(h, H_RESOLUTION)
        n_func = count_components(h, H_FUNCTIONAL)
        logs["n_components_resolved"].append(n_res)
        logs["n_components_functional"].append(n_func)

        # Surface frontière
        surf_res = surface_active(h, H_RESOLUTION)
        surf_func = surface_active(h, H_FUNCTIONAL)
        logs["surface_resolved"].append(surf_res)
        logs["surface_functional"].append(surf_func)

        # Périmètre normalisé = surface / volume^(2/3) en 3D
        vol_res = (1 - frac_res) * h.size
        vol_func = (1 - frac_func) * h.size
        peri_res = surf_res / (vol_res ** (2/3)) if vol_res > 0 else float("nan")
        peri_func = surf_func / (vol_func ** (2/3)) if vol_func > 0 else float("nan")
        logs["perimeter_normalized_resolved"].append(peri_res)
        logs["perimeter_normalized_functional"].append(peri_func)

        # Mémoire longue
        diff_h_h0 = h - h0_target
        logs["h_minus_h0_L2"].append(float(np.sqrt(np.sum(diff_h_h0 ** 2))))
        logs["h_minus_h0_max"].append(float(np.max(np.abs(diff_h_h0))))

        # Corrélation -log(h/h0) vs ∫(ψ-ψ_uniform)·dt
        h_flat = h.flatten()
        log_ratio = -np.log(np.maximum(h_flat, 1e-30) / h0_target)
        I_flat = integral_centered.flatten()
        if log_ratio.std() > 0 and I_flat.std() > 0:
            corr_val = float(np.corrcoef(log_ratio, I_flat)[0, 1])
        else:
            corr_val = float("nan")
        logs["corr_logh_intpsi_centered"].append(corr_val)

    if 0 in log_schedule:
        log_state(0, 0.0, psi, h, integral_psi_centered)

    for step in range(1, n_steps + 1):
        psi_before = psi.copy()
        psi, h = step_engine_euler(psi, h, D, beta, gamma, h0_target, dt)
        # Intégrale ψ - ψ_uniform par trapèze
        integral_psi_centered += 0.5 * dt * ((psi_before - psi_uniform) + (psi - psi_uniform))

        # Tracking step-by-step
        below_now = h < H_RESOLUTION
        # first_down : cellule descend pour la première fois
        first_down_mask = below_now & (t_first_down < 0)
        t_first_down[first_down_mask] = step * dt
        # first_reactivation : était déjà descendue, remonte pour la première fois
        first_reactivation_mask = (
            (t_first_down >= 0) & (t_first_reactivation < 0) & ~below_now
        )
        t_first_reactivation[first_reactivation_mask] = step * dt
        # n_crossings : compter chaque transition
        crossings_mask = below_now != below_prev
        n_crossings += crossings_mask.astype(int)
        below_prev = below_now

        if step in log_schedule:
            log_state(step, step * dt, psi, h, integral_psi_centered)

    logs["psi_final"] = psi.copy()
    logs["h_final"] = h.copy()
    logs["integral_psi_centered"] = integral_psi_centered
    logs["t_first_down"] = t_first_down
    logs["t_first_reactivation"] = t_first_reactivation
    logs["n_crossings"] = n_crossings
    return logs


def test_one(beta, gamma, t_target, label, D=0.1, h0_target=1.0, sigma_psi=1.5):
    print(f"\n{'='*60}")
    print(f"Test 4a-ζ : {label}")
    print(f"β={beta}, γ={gamma}, t_target={t_target}")
    print(f"{'='*60}")

    psi_init = make_psi_decentered(center=(1.0, 2.0, 2.0), sigma_0=sigma_psi)
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0_target)
    psi_max_init = float(psi_init.max())

    rate_h_max = beta * psi_max_init + gamma
    dt_cfl_diff = cfl_dt_max(h0_target, D)
    dt_cfl_h = 1.0 / rate_h_max
    dt = 0.5 * min(dt_cfl_diff, dt_cfl_h)
    n_steps = max(1, int(np.ceil(t_target / dt)))

    print(f"  dt = {dt:.4f}, n_steps = {n_steps}")

    logs = simulate_with_collapse_tracking(
        psi_init, h_init, D, beta, gamma, h0_target, n_steps, dt
    )

    # === Métriques d'invariance
    mass_drift = max(abs(m - 1.0) for m in logs["psi_total"])
    psi_min_global = min(logs["psi_min"])
    h_min_global = min(logs["h_min"])

    # === Récup frac aux trois temps
    t_arr = np.array(logs["t"])
    t_final = t_arr[-1]
    # Trouver indices proches de t/10, t/2, t
    idx_10 = int(np.argmin(np.abs(t_arr - t_final * 0.1)))
    idx_50 = int(np.argmin(np.abs(t_arr - t_final * 0.5)))
    idx_100 = len(t_arr) - 1

    frac_res_at = [
        logs["frac_under_resolution"][idx_10],
        logs["frac_under_resolution"][idx_50],
        logs["frac_under_resolution"][idx_100],
    ]
    frac_func_at = [
        logs["frac_under_functional"][idx_10],
        logs["frac_under_functional"][idx_50],
        logs["frac_under_functional"][idx_100],
    ]

    n_cells = N_AXIS ** 3

    # Pour la classification raffinée, on a besoin des crossings AVANT d'appeler classify_trend
    # → on calcule d'abord la réactivation
    t_first_down = logs["t_first_down"]
    t_first_reactivation = logs["t_first_reactivation"]
    n_crossings = logs["n_crossings"]

    n_cells_ever_down = int(np.sum(t_first_down >= 0))
    n_cells_reactivated = int(np.sum(t_first_reactivation >= 0))
    max_crossings = int(np.max(n_crossings))
    n_cells_multi_cross = int(np.sum(n_crossings >= 2))

    trend_resolution = classify_trend(
        *frac_res_at, n_cells,
        max_crossings=max_crossings,
        n_cells_multi_cross=n_cells_multi_cross,
    )
    trend_functional = classify_trend(
        *frac_func_at, n_cells,
        max_crossings=max_crossings,
        n_cells_multi_cross=n_cells_multi_cross,
    )

    if n_cells_reactivated > 0:
        delays = t_first_reactivation[t_first_reactivation >= 0] \
            - t_first_down[t_first_reactivation >= 0]
        delay_mean = float(np.mean(delays))
        delay_max = float(np.max(delays))
        delay_min = float(np.min(delays))
    else:
        delay_mean = delay_max = delay_min = float("nan")

    # === Composantes
    n_comp_res_final = logs["n_components_resolved"][-1]
    n_comp_func_final = logs["n_components_functional"][-1]
    n_comp_res_max = max(logs["n_components_resolved"])
    n_comp_func_max = max(logs["n_components_functional"])

    # === Mémoire longue
    diff_h0_L2_final = logs["h_minus_h0_L2"][-1]
    diff_h0_L2_max = max(logs["h_minus_h0_L2"])
    diff_h0_max_final = logs["h_minus_h0_max"][-1]
    corr_logh_intpsi_final = logs["corr_logh_intpsi_centered"][-1]

    # === Affichage
    print(f"\n=== Invariants ===")
    print(f"  Σψ drift          : {mass_drift:.4e}")
    print(f"  positivité ψ min  : {psi_min_global:.4e}")
    print(f"  h_min global      : {h_min_global:.4e}")
    print(f"\n=== Fractions sous seuil ===")
    print(f"  Aux temps t/10, t/2, t :")
    print(f"    h_resolution : {frac_res_at[0]:.4f} → {frac_res_at[1]:.4f} → {frac_res_at[2]:.4f}")
    print(f"    h_functional : {frac_func_at[0]:.4f} → {frac_func_at[1]:.4f} → {frac_func_at[2]:.4f}")
    print(f"\n=== Classification tendance ===")
    print(f"  collapse_trend_resolution  : {trend_resolution}")
    print(f"  collapse_trend_functional  : {trend_functional}")
    print(f"\n=== Composantes connexes ===")
    print(f"  resolved   final : {n_comp_res_final},  max : {n_comp_res_max}")
    print(f"  functional final : {n_comp_func_final}, max : {n_comp_func_max}")
    print(f"\n=== Réactivation ===")
    print(f"  cellules passées sous seuil : {n_cells_ever_down}/{n_cells}")
    print(f"  cellules réactivées (≥1)    : {n_cells_reactivated}/{n_cells}")
    print(f"  max n_crossings sur cellule : {max_crossings}")
    print(f"  cellules avec ≥2 crossings  : {n_cells_multi_cross}/{n_cells}")
    if n_cells_reactivated > 0:
        print(f"  délai réactivation : min={delay_min:.4f}, mean={delay_mean:.4f}, max={delay_max:.4f}")
    print(f"\n=== Mémoire longue ===")
    print(f"  ||h-h0||_L2     final : {diff_h0_L2_final:.4e} (max: {diff_h0_L2_max:.4e})")
    print(f"  max|h-h0|       final : {diff_h0_max_final:.4e}")
    print(f"  corr(-log(h/h0), ∫(ψ-ψu)dt) final : {corr_logh_intpsi_final:+.4f}")

    # === VERDICT
    verdict = "PASS"
    reasons = []
    if mass_drift > 1e-10:
        verdict = "FAIL"
        reasons.append(f"Σψ drift = {mass_drift:.4e}")
    if psi_min_global < -1e-12:
        verdict = "FAIL"
        reasons.append(f"positivité ψ : min = {psi_min_global:.4e}")
    # Pas de critère "réactivation oui/non" — c'est l'observable, pas un critère

    print(f"\n=== VERDICT ===")
    if verdict == "PASS":
        print(f"  PASS ✓ pour {label}")
    else:
        print(f"  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "label": label,
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "beta": beta, "gamma": gamma, "D": D, "h0_target": h0_target,
            "psi_center_init": [1.0, 2.0, 2.0],
            "t_target": t_target, "dt": dt, "n_steps": n_steps,
            "h_resolution": H_RESOLUTION, "h_functional": H_FUNCTIONAL,
        },
        "invariants": {
            "mass_drift": float(mass_drift),
            "psi_min_global": float(psi_min_global),
            "h_min_global": float(h_min_global),
        },
        "collapse_classification": {
            "frac_resolution_at_t10_t50_t100": frac_res_at,
            "frac_functional_at_t10_t50_t100": frac_func_at,
            "collapse_trend_resolution": trend_resolution,
            "collapse_trend_functional": trend_functional,
        },
        "topology": {
            "n_components_resolved_final": n_comp_res_final,
            "n_components_functional_final": n_comp_func_final,
            "n_components_resolved_max": n_comp_res_max,
            "n_components_functional_max": n_comp_func_max,
        },
        "reactivation": {
            "n_cells_ever_down": n_cells_ever_down,
            "n_cells_reactivated": n_cells_reactivated,
            "max_n_crossings_per_cell": max_crossings,
            "n_cells_multi_crossings": n_cells_multi_cross,
            "delay_min": delay_min if not np.isnan(delay_min) else None,
            "delay_mean": delay_mean if not np.isnan(delay_mean) else None,
            "delay_max": delay_max if not np.isnan(delay_max) else None,
        },
        "long_memory": {
            "h_minus_h0_L2_final": float(diff_h0_L2_final),
            "h_minus_h0_L2_max": float(diff_h0_L2_max),
            "h_minus_h0_max_final": float(diff_h0_max_final),
            "corr_logh_intpsi_centered_final": float(corr_logh_intpsi_final),
        },
    }


def run_step_4a_zeta():
    configs = [
        # Témoin stable
        (1.0, 0.01, 500.0, "A_temoin_stable_t500"),
        # Étude collapse C aux trois durées
        (125.0, 1.0, 50.0, "C_collapse_t50"),
        (125.0, 1.0, 500.0, "C_collapse_t500"),
        (125.0, 1.0, 5000.0, "C_collapse_t5000"),
    ]
    results = {}
    for beta, gamma, t_target, label in configs:
        results[label] = test_one(beta, gamma, t_target, label)

    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*70}")
    print(f"VERDICT GLOBAL 4a-ζ : {global_verdict}")
    print(f"{'='*70}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    # Synthèse classification
    print(f"\n{'='*90}")
    print(f"SYNTHÈSE CLASSIFICATION")
    print(f"{'='*90}")
    print(f"{'config':<25} {'trend_res':<22} {'trend_func':<22} {'n_react':>8} {'multi_cross':>12}")
    for label, r in results.items():
        cc = r["collapse_classification"]
        rr = r["reactivation"]
        print(f"{label:<25} {cc['collapse_trend_resolution']:<22} "
              f"{cc['collapse_trend_functional']:<22} "
              f"{rr['n_cells_reactivated']:>8d} "
              f"{rr['n_cells_multi_crossings']:>12d}")

    # Étude de la progression sur C
    print(f"\n{'='*90}")
    print(f"PROGRESSION C : frac_under_functional aux trois durées")
    print(f"{'='*90}")
    print(f"{'durée':<25} {'frac_func(t/10)':<18} {'frac_func(t/2)':<18} {'frac_func(t)':<18}")
    for label in ["C_collapse_t50", "C_collapse_t500", "C_collapse_t5000"]:
        if label in results:
            cc = results[label]["collapse_classification"]
            vals = cc["frac_functional_at_t10_t50_t100"]
            print(f"{label:<25} {vals[0]:<18.4f} {vals[1]:<18.4f} {vals[2]:<18.4f}")

    return {"global_verdict": global_verdict, "cases": results}


if __name__ == "__main__":
    summary = run_step_4a_zeta()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_4a_zeta_long_term.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
