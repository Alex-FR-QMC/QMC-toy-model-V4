"""
Test 6d-α §5.7 — validation hors-distribution famille B.

acquired 4a + A.5: morphologic memory, feedback ψ↔h, stratified reactivation,
local algebraic tautology at coupled stationary point.
No strong MCQ conclusion yet — testing path-dependence here.

Plan validé Alex post-A.5 :
- §5 local au point fixe = identité algébrique (tautologie au stationnaire)
- §5.7 famille B = test de la GÉOMÉTRIE DE CONVERGENCE
- Question clé : trajectoires convergent vers même attracteur ?
  Si oui, par quelles géométries de relaxation ?

Famille A (référence) : gaussienne centrée σ_0=1.8
Famille B1 : gaussienne décentrée (1,2,2), σ=1.8
Famille B2 : bimodale, deux pics étroits σ=1.0
Famille B3 : uniforme ψ = 1/125 (contrôle chemin minimal)

β ∈ {45, 60, 80} (sélectionnés post-A.5)
β=45,60 candidats principaux ; β=80 stress-case

Métriques (3 niveaux) :
- Niveau 1 (trajectoriel A↔B) : D_ψ(t), D_h(t), AUC, t_50, t_10, max_D
- Niveau 2 (par famille) : t_collapse, frac_collapsed_max, C_T, mémoire intégrale
- Niveau 3 (comparaison observables) : différences A↔B sur observables

Lecture attendue (analyse a priori Alex) :
- D_ψ(t→∞) → 0 et D_h(t→∞) → 0 (attracteur unique)
- MAIS AUC_Dh élevé pour B1, B2 (sédimentation locale absente en B3)
- B3 PAS de collapse local (pas de concentration initiale)
- Si AUC_Dh élevé alors §5.7 révèle : géométrie de convergence non-triviale

Caveats :
- B2 stress-case potentiel si ψ_max trop haut → collapse massif
- B3 n'est pas un contrôle neutre mais un contrôle chemin minimal
- aucune conclusion MCQ forte tant que 6d-β pas démarré
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
EPS_SAT = 0.05


def rhs_coupled(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)
    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi_dt, dh_dt


def step_engine_euler(psi, h, D, beta, gamma, h0, dt):
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi_dt, h + dt * dh_dt


def make_psi_A_centered(sigma_0=1.8):
    """Famille A — gaussienne centrée selon spec §4.3."""
    coords = np.arange(N_AXIS) * DX
    center = (N_AXIS - 1) * DX / 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i] - center) ** 2 + \
                     (coords[j] - center) ** 2 + \
                     (coords[k] - center) ** 2
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma_0 ** 2)
    psi /= psi.sum()
    return psi


def make_psi_B1_decentered(sigma_0=1.8):
    """Famille B1 — gaussienne décentrée (1,2,2)."""
    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = 1.0, 2.0, 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i] - cx) ** 2 + \
                     (coords[j] - cy) ** 2 + \
                     (coords[k] - cz) ** 2
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma_0 ** 2)
    psi /= psi.sum()
    return psi


def make_psi_B2_bimodal(sigma_0=1.0):
    """Famille B2 — bimodale : deux pics étroits en (1.5,2,2) et (2.5,2,2)."""
    coords = np.arange(N_AXIS) * DX
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for c in [(1.5, 2.0, 2.0), (2.5, 2.0, 2.0)]:
        cx, cy, cz = c
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r2 = (coords[i] - cx) ** 2 + \
                         (coords[j] - cy) ** 2 + \
                         (coords[k] - cz) ** 2
                    psi[i, j, k] += np.exp(-0.5 * r2 / sigma_0 ** 2)
    psi /= psi.sum()
    return psi


def make_psi_B3_uniform():
    """Famille B3 — uniforme ψ = 1/N_total partout."""
    psi = np.full((N_AXIS, N_AXIS, N_AXIS), 1.0 / (N_AXIS ** 3))
    return psi


def simulate_with_snapshots(
    psi_init, h_init, D, beta, gamma, h0_target,
    n_steps, dt, snapshot_indices,
):
    """
    Simule et retourne snapshots aux indices de step demandés.
    Retourne aussi intégrale ψ-ψ_uniform.
    """
    psi = psi_init.copy()
    h = h_init.copy()
    psi_uniform = 1.0 / psi.size
    integral_psi_centered = np.zeros_like(psi)

    snapshots = {
        "t": [],
        "step": [],
        "psi": [],
        "h": [],
        "psi_total": [],
        "psi_min": [], "psi_max": [],
        "h_min": [], "h_max": [], "h_mean": [],
        "frac_collapsed": [],
        "frac_saturated_near_h0": [],
        "beta_psi_max_over_gamma": [],
    }

    def snap(step, t, psi, h):
        snapshots["step"].append(step)
        snapshots["t"].append(t)
        snapshots["psi"].append(psi.copy())
        snapshots["h"].append(h.copy())
        snapshots["psi_total"].append(float(psi.sum()))
        snapshots["psi_min"].append(float(psi.min()))
        snapshots["psi_max"].append(float(psi.max()))
        snapshots["h_min"].append(float(h.min()))
        snapshots["h_max"].append(float(h.max()))
        snapshots["h_mean"].append(float(h.mean()))
        snapshots["frac_collapsed"].append(
            float(np.sum(h < H_RESOLUTION) / h.size)
        )
        snapshots["frac_saturated_near_h0"].append(
            float(np.sum(h > h0_target - EPS_SAT * h0_target) / h.size)
        )
        snapshots["beta_psi_max_over_gamma"].append(
            beta * float(psi.max()) / gamma if gamma > 0 else float("inf")
        )

    snap(0, 0.0, psi, h)
    snapshot_set = set(snapshot_indices)

    for step in range(1, n_steps + 1):
        psi_before = psi.copy()
        psi, h = step_engine_euler(psi, h, D, beta, gamma, h0_target, dt)
        integral_psi_centered += 0.5 * dt * (
            (psi_before - psi_uniform) + (psi - psi_uniform)
        )
        if step in snapshot_set:
            snap(step, step * dt, psi, h)

    snapshots["integral_psi_centered_final"] = integral_psi_centered
    return snapshots


def compute_trajectory_distances(snapshots_A, snapshots_B):
    """
    Calcule D_ψ(t) et D_h(t) entre A et B aux mêmes snapshots.
    A et B ont été lancés avec MÊME dt et MÊMES snapshot_indices.
    """
    assert len(snapshots_A["t"]) == len(snapshots_B["t"]), \
        f"Mismatch snapshots A vs B : {len(snapshots_A['t'])} vs {len(snapshots_B['t'])}"

    D_psi = []
    D_h = []
    times = []
    for tA, tB, psiA, psiB, hA, hB in zip(
        snapshots_A["t"], snapshots_B["t"],
        snapshots_A["psi"], snapshots_B["psi"],
        snapshots_A["h"], snapshots_B["h"]
    ):
        if abs(tA - tB) > 1e-9:
            raise ValueError(f"Désalignement temporel A={tA} vs B={tB}")
        times.append(tA)
        norm_psi_A = max(np.linalg.norm(psiA), 1e-30)
        norm_h_A = max(np.linalg.norm(hA), 1e-30)
        D_psi.append(float(np.linalg.norm(psiB - psiA) / norm_psi_A))
        D_h.append(float(np.linalg.norm(hB - hA) / norm_h_A))

    times = np.array(times)
    D_psi_arr = np.array(D_psi)
    D_h_arr = np.array(D_h)

    # AUC par règle des trapèzes (np.trapezoid pour numpy >=2)
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    AUC_Dpsi = float(trapz(D_psi_arr, times)) if len(times) > 1 else 0.0
    AUC_Dh = float(trapz(D_h_arr, times)) if len(times) > 1 else 0.0

    # max et temps du max
    if len(D_psi_arr) > 0:
        idx_max_Dpsi = int(np.argmax(D_psi_arr))
        max_Dpsi = float(D_psi_arr[idx_max_Dpsi])
        t_max_Dpsi = float(times[idx_max_Dpsi])
    else:
        max_Dpsi = 0.0
        t_max_Dpsi = 0.0
    if len(D_h_arr) > 0:
        idx_max_Dh = int(np.argmax(D_h_arr))
        max_Dh = float(D_h_arr[idx_max_Dh])
        t_max_Dh = float(times[idx_max_Dh])
    else:
        max_Dh = 0.0
        t_max_Dh = 0.0

    # Temps de décroissance : à partir du max, quand passe-t-on à 50% / 10% ?
    def time_to_fraction(values, times, idx_start, fraction):
        if idx_start >= len(values) - 1:
            return None
        ref = values[idx_start]
        if ref <= 0:
            return None
        target = fraction * ref
        for i in range(idx_start + 1, len(values)):
            if values[i] <= target:
                return float(times[i] - times[idx_start])
        return None  # n'a jamais atteint la fraction

    t_50_Dpsi = time_to_fraction(D_psi_arr, times, idx_max_Dpsi, 0.5)
    t_10_Dpsi = time_to_fraction(D_psi_arr, times, idx_max_Dpsi, 0.1)
    t_50_Dh = time_to_fraction(D_h_arr, times, idx_max_Dh, 0.5)
    t_10_Dh = time_to_fraction(D_h_arr, times, idx_max_Dh, 0.1)

    # D finaux
    D_psi_final = float(D_psi_arr[-1])
    D_h_final = float(D_h_arr[-1])
    D_psi_initial = float(D_psi_arr[0])
    D_h_initial = float(D_h_arr[0])

    return {
        "times": times.tolist(),
        "D_psi_trajectory": D_psi_arr.tolist(),
        "D_h_trajectory": D_h_arr.tolist(),
        "D_psi_initial": D_psi_initial,
        "D_psi_final": D_psi_final,
        "D_h_initial": D_h_initial,
        "D_h_final": D_h_final,
        "max_Dpsi": max_Dpsi,
        "max_Dh": max_Dh,
        "t_max_Dpsi": t_max_Dpsi,
        "t_max_Dh": t_max_Dh,
        "AUC_Dpsi": AUC_Dpsi,
        "AUC_Dh": AUC_Dh,
        "t_50_Dpsi_after_max": t_50_Dpsi,
        "t_10_Dpsi_after_max": t_10_Dpsi,
        "t_50_Dh_after_max": t_50_Dh,
        "t_10_Dh_after_max": t_10_Dh,
    }


def compute_family_observables(snapshots, h0_target, beta, gamma):
    """Observables morphologiques par famille (Niveau 2)."""
    h_final = snapshots["h"][-1]
    psi_final = snapshots["psi"][-1]

    # t_collapse_first : premier temps où frac_collapsed > 0
    t_collapse_first = None
    for t, frac in zip(snapshots["t"], snapshots["frac_collapsed"]):
        if frac > 0.0 and t_collapse_first is None:
            t_collapse_first = float(t)
            break

    frac_collapsed_max_over_time = float(max(snapshots["frac_collapsed"]))

    # C_T par snapshot
    T_window_idx = max(1, len(snapshots["h"]) // 10)
    C_T_trajectory = []
    for i, h in enumerate(snapshots["h"]):
        if i >= T_window_idx:
            h_past = snapshots["h"][i - T_window_idx]
            C_T = float(np.linalg.norm(h - h_past))
            C_T_trajectory.append(C_T)
        else:
            C_T_trajectory.append(None)

    # Corrélation mémoire à la fin
    integral_centered = snapshots["integral_psi_centered_final"]
    h_flat = h_final.flatten()
    I_flat = integral_centered.flatten()
    log_ratio = -np.log(np.maximum(h_flat, 1e-30) / h0_target)
    if log_ratio.std() > 0 and I_flat.std() > 0:
        corr_logh_int = float(np.corrcoef(log_ratio, I_flat)[0, 1])
    else:
        corr_logh_int = float("nan")

    return {
        "t_collapse_first": t_collapse_first,
        "frac_collapsed_max_over_time": frac_collapsed_max_over_time,
        "frac_collapsed_final": snapshots["frac_collapsed"][-1],
        "h_min_global": float(min(snapshots["h_min"])),
        "h_max_global": float(max(snapshots["h_max"])),
        "h_mean_final": float(snapshots["h_mean"][-1]),
        "psi_max_initial": float(snapshots["psi_max"][0]),
        "psi_max_final": float(snapshots["psi_max"][-1]),
        "beta_psi_max_g_initial": float(snapshots["beta_psi_max_over_gamma"][0]),
        "beta_psi_max_g_final": float(snapshots["beta_psi_max_over_gamma"][-1]),
        "C_T_trajectory": C_T_trajectory,
        "C_T_final": (
            C_T_trajectory[-1] if C_T_trajectory and C_T_trajectory[-1] is not None
            else None
        ),
        "corr_logh_intpsi_centered_final": corr_logh_int,
    }


def run_one_beta(beta, gamma, D, h0_target, t_sim, n_snapshots=30):
    """Lance A + B1 + B2 + B3 pour un β donné, compare trajectoires."""
    print(f"\n{'='*70}")
    print(f"§5.7 — β = {beta}, γ = {gamma}")
    print(f"{'='*70}")

    # Initialisations
    psi_A = make_psi_A_centered(sigma_0=1.8)
    psi_B1 = make_psi_B1_decentered(sigma_0=1.8)
    psi_B2 = make_psi_B2_bimodal(sigma_0=1.0)
    psi_B3 = make_psi_B3_uniform()
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0_target)

    # CFL la plus contraignante parmi A, B1, B2, B3
    psi_max_global = max(
        float(psi_A.max()), float(psi_B1.max()),
        float(psi_B2.max()), float(psi_B3.max())
    )
    rate_h_max = beta * psi_max_global + gamma
    dt_cfl_diff = cfl_dt_max(h0_target, D)
    dt_cfl_h = 1.0 / rate_h_max
    dt = 0.5 * min(dt_cfl_diff, dt_cfl_h)
    n_steps = int(np.ceil(t_sim / dt))

    snapshot_indices = list(set(
        list(range(0, min(20, n_steps + 1))) +  # dense au début
        list(np.linspace(0, n_steps, n_snapshots, dtype=int))
    ))
    snapshot_indices = sorted(snapshot_indices)

    print(f"  dt = {dt:.4f}, n_steps = {n_steps}, t_sim = {t_sim}")
    print(f"  ψ_max init : A={psi_A.max():.4e}  B1={psi_B1.max():.4e}  "
          f"B2={psi_B2.max():.4e}  B3={psi_B3.max():.4e}")
    print(f"  β·ψ_max_B2/γ = {beta * psi_B2.max() / gamma:.4f}  "
          f"(stress-case si > 1)")

    # Lancer les 4 familles
    snapshots = {}
    for fam_name, psi_init in [
        ("A", psi_A), ("B1", psi_B1), ("B2", psi_B2), ("B3", psi_B3)
    ]:
        print(f"  Lancement famille {fam_name}...")
        snapshots[fam_name] = simulate_with_snapshots(
            psi_init, h_init, D, beta, gamma, h0_target,
            n_steps, dt, snapshot_indices,
        )

    # Observables par famille (Niveau 2)
    observables = {
        fam: compute_family_observables(snapshots[fam], h0_target, beta, gamma)
        for fam in ["A", "B1", "B2", "B3"]
    }

    # Comparaisons trajectorielles A↔B (Niveau 1)
    comparisons = {}
    for fam_B in ["B1", "B2", "B3"]:
        comparisons[f"A_vs_{fam_B}"] = compute_trajectory_distances(
            snapshots["A"], snapshots[fam_B]
        )

    # Affichage
    print(f"\n  --- Observables par famille (Niveau 2) ---")
    print(f"  {'famille':<5} {'h_mean_final':>12} {'frac_col_max':>14} "
          f"{'h_min':>12} {'t_collapse_first':>17} {'corr_logh_I':>12}")
    for fam in ["A", "B1", "B2", "B3"]:
        o = observables[fam]
        t_coll_str = f"{o['t_collapse_first']:.2f}" if o['t_collapse_first'] is not None else "N/A"
        print(f"  {fam:<5} {o['h_mean_final']:>12.4f} "
              f"{o['frac_collapsed_max_over_time']:>14.4f} "
              f"{o['h_min_global']:>12.4e} {t_coll_str:>17} "
              f"{o['corr_logh_intpsi_centered_final']:>+12.4f}")

    print(f"\n  --- Comparaisons trajectorielles A↔B (Niveau 1) ---")
    print(f"  {'compar':<10} {'D_ψ_init':>10} {'D_ψ_fin':>10} {'max_Dψ':>10} {'AUC_Dψ':>10} "
          f"{'D_h_init':>10} {'D_h_fin':>10} {'max_Dh':>10} {'AUC_Dh':>10}")
    for compar_name, c in comparisons.items():
        print(f"  {compar_name:<10} {c['D_psi_initial']:>10.4f} {c['D_psi_final']:>10.4e} "
              f"{c['max_Dpsi']:>10.4f} {c['AUC_Dpsi']:>10.2e} "
              f"{c['D_h_initial']:>10.4f} {c['D_h_final']:>10.4e} "
              f"{c['max_Dh']:>10.4f} {c['AUC_Dh']:>10.2e}")

    # Vérifier convergence finale (D_psi_final et D_h_final petits)
    print(f"\n  --- Lecture verdict ---")
    converged_to_same = all(
        c["D_psi_final"] < 1e-3 and c["D_h_final"] < 1e-3
        for c in comparisons.values()
    )
    print(f"  Toutes familles convergent vers même attracteur (D_final < 1e-3) : "
          f"{'OUI' if converged_to_same else 'NON'}")

    # AUC max sur les comparaisons
    auc_h_max = max(c["AUC_Dh"] for c in comparisons.values())
    auc_psi_max = max(c["AUC_Dpsi"] for c in comparisons.values())
    print(f"  AUC_Dh max sur comparaisons : {auc_h_max:.4f}")
    print(f"  AUC_Dpsi max sur comparaisons : {auc_psi_max:.4f}")
    print(f"  → Mesure de la 'longueur du chemin de convergence'")

    return {
        "beta": beta, "gamma": gamma,
        "params": {
            "D": D, "h0_target": h0_target, "t_sim": t_sim,
            "dt": dt, "n_steps": n_steps,
            "sigma_0_A": 1.8, "sigma_0_B1": 1.8, "sigma_0_B2": 1.0,
        },
        "observables_by_family": observables,
        "trajectory_comparisons": comparisons,
        "verdict_aggregated": {
            "converged_to_same_attractor": bool(converged_to_same),
            "AUC_Dh_max": float(auc_h_max),
            "AUC_Dpsi_max": float(auc_psi_max),
        },
    }


def run_test_5_7():
    """Exécute §5.7 famille B sur β=45,60,80."""
    gamma = 1.0
    D = 0.1
    h0_target = 1.0
    t_sim = 500.0

    beta_values = [45.0, 60.0, 80.0]
    results = {}
    for beta in beta_values:
        results[f"beta_{int(beta)}"] = run_one_beta(
            beta, gamma, D, h0_target, t_sim
        )

    # Synthèse globale
    print(f"\n\n{'='*80}")
    print(f"SYNTHÈSE §5.7 — DÉPENDANCE AU CHEMIN")
    print(f"{'='*80}")
    print(f"{'β':>5} {'converge_attractor':>20} {'AUC_Dh_max':>12} {'AUC_Dpsi_max':>14}")
    for label, r in results.items():
        v = r["verdict_aggregated"]
        print(f"{r['beta']:>5.0f} {str(v['converged_to_same_attractor']):>20} "
              f"{v['AUC_Dh_max']:>12.4f} {v['AUC_Dpsi_max']:>14.4f}")

    print(f"\nLecture :")
    print(f"  - converge_attractor=True ET AUC_Dh proche de 0 → tautologie complète")
    print(f"  - converge_attractor=True ET AUC_Dh élevé → géométrie de convergence non-triviale")
    print(f"  - converge_attractor=False → multi-bassins (résultat MCQ fort)")

    return {"results": results}


if __name__ == "__main__":
    summary = run_test_5_7()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_5_7_family_B.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return None  # snapshots ψ et h pas sauvegardés
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0)
