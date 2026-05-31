"""
Test 6d-β session 3 — Sensibilité à l'échelle temporelle d'observation.

Rituel d'ouverture (validé) :
- Niveau visé : 2 (toujours)
- Mouvement : pas de nouvelle observable, cartographie t_sim
- Risque actif : "recherche de la vraie limite" (purification déguisée)

Question discriminante :
«Comment les signatures morphodynamiques se transforment-elles
quand on change l'échelle temporelle d'observation ?»

PAS : "quelle est la limite asymptotique ?"
PAS : "qu'est-ce qui est réel ?"

La migration des signatures est le résultat, pas un échec.

Discipline minimaliste :
- t_sim ∈ {500, 1000, 2000} (extension homothétique ×1/×2/×4)
- 3 cas (FAIBLE/MOYEN/ÉLEVÉ) — pas seulement MOYEN
- dt FIXÉ à dt_0 (pas de croisement dt × t_sim)
- 3 × 3 = 9 simulations

Lectures concurrentes documentées (préambule rituel étape 4) :
L1, L2, L3, L4 — toutes acceptables a priori, aucune hiérarchie.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mcq_v4.factorial_6d import N_AXIS, DX, cfl_dt_max  # noqa: E402
from mcq_v4.factorial_6d.engine import (  # noqa: E402
    compute_diffusion_flux, compute_divergence,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero  # noqa: E402

from instrumentation_6d_beta.temporal_structure_Dh import (  # noqa: E402
    compute_temporal_structure,
)
from instrumentation_6d_beta.redivergence_map import (  # noqa: E402
    compute_redivergence_map,
)


def rhs_coupled(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)
    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi_dt, dh_dt


def step_engine_euler(psi, h, D, beta, gamma, h0, dt):
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi_dt, h + dt * dh_dt


def make_psi_A(sigma_0=1.8):
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
    return psi / psi.sum()


def make_psi_B2(sigma_0=1.0):
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
    return psi / psi.sum()


def make_psi_B3():
    return np.full((N_AXIS, N_AXIS, N_AXIS), 1.0 / (N_AXIS ** 3))


def simulate_pair(psi_A_init, psi_B_init, h_init, D, beta, gamma, h0,
                  t_sim, dt, n_snapshots=80):
    """Simule pair (A, B) en parallèle. Retourne trajectoires h alignées."""
    n_steps = int(np.ceil(t_sim / dt))
    snapshot_indices = sorted(set(
        list(range(0, min(20, n_steps + 1)))
        + list(np.linspace(0, n_steps, n_snapshots, dtype=int))
    ))
    snapshot_set = set(snapshot_indices)

    psi_A = psi_A_init.copy()
    h_A = h_init.copy()
    psi_B = psi_B_init.copy()
    h_B = h_init.copy()

    t_list = [0.0]
    h_A_list = [h_A.copy()]
    h_B_list = [h_B.copy()]

    for step in range(1, n_steps + 1):
        psi_A, h_A = step_engine_euler(psi_A, h_A, D, beta, gamma, h0, dt)
        psi_B, h_B = step_engine_euler(psi_B, h_B, D, beta, gamma, h0, dt)
        if step in snapshot_set:
            t_list.append(step * dt)
            h_A_list.append(h_A.copy())
            h_B_list.append(h_B.copy())

    times = np.array(t_list)
    Dh = []
    for hA, hB in zip(h_A_list, h_B_list):
        norm_A = max(np.linalg.norm(hA), 1e-30)
        Dh.append(float(np.linalg.norm(hB - hA) / norm_A))
    return times, np.array(Dh)


def run_session_3():
    gamma = 1.0
    D = 0.1
    h0 = 1.0

    cases = [
        ("FAIBLE", 80.0, "B3", make_psi_B3),
        ("MOYEN", 45.0, "B2", lambda: make_psi_B2(sigma_0=1.0)),
        ("ELEVE", 60.0, "B2", lambda: make_psi_B2(sigma_0=1.0)),
    ]
    t_sim_values = [500.0, 1000.0, 2000.0]

    print(f"{'='*78}")
    print(f"6d-β SESSION 3 — Sensibilité à l'échelle temporelle")
    print(f"  Question : comment les signatures se transforment-elles ?")
    print(f"  PAS de recherche de 'vraie limite'")
    print(f"{'='*78}")

    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    results = {}

    for case_label, beta, fam_B_name, psi_B_factory in cases:
        print(f"\n{'─'*78}")
        print(f"Cas {case_label} : β={beta}, A↔{fam_B_name}")
        print(f"{'─'*78}")

        psi_A = make_psi_A(sigma_0=1.8)
        psi_B = psi_B_factory()
        psi_max = max(float(psi_A.max()), float(psi_B.max()))
        rate_h = beta * psi_max + gamma
        dt_0 = 0.5 * min(cfl_dt_max(h0, D), 1.0 / rate_h)

        print(f"  dt fixé : {dt_0:.6f}")
        results[case_label] = {
            "params": {"beta": beta, "fam_B": fam_B_name, "dt_0": dt_0},
            "by_t_sim": {},
        }

        for t_sim in t_sim_values:
            print(f"\n  ── t_sim = {t_sim} ──")
            times, Dh = simulate_pair(
                psi_A, psi_B, h_init, D, beta, gamma, h0,
                t_sim, dt_0, n_snapshots=80,
            )

            # Structure temporelle globale
            struct = compute_temporal_structure(times, Dh)
            # Cartographie des re-divergences
            rmap = compute_redivergence_map(times, Dh)

            print(f"    Structure temporelle :")
            print(f"      t_max_Dh           = {struct.t_max_Dh:.4f}")
            print(f"      max_Dh             = {struct.max_Dh:.4e}")
            print(f"      tau_div            = {struct.tau_div:.4f}")
            print(f"      tau_rec_10pct      = "
                  f"{struct.tau_rec_10pct:.4f}" if struct.tau_rec_10pct
                  else f"      tau_rec_10pct      = non atteint")
            print(f"      tau_rec_1pct       = "
                  f"{struct.tau_rec_1pct:.4f}" if struct.tau_rec_1pct
                  else f"      tau_rec_1pct       = non atteint")
            print(f"      plateau_detected   = {struct.plateau_detected}")
            if struct.plateau_detected:
                print(f"        plateau_duration = {struct.plateau_duration:.4f}")
                print(f"        plateau_value    = {struct.plateau_value:.4e}")
            print(f"      max_well_defined   = {struct.max_well_defined}")

            print(f"    Re-divergences :")
            print(f"      n_events           = {rmap.n_events}")
            print(f"      amp_max            = {rmap.amplitude_max:.4e}")
            print(f"      amp_median         = {rmap.amplitude_median:.4e}")
            print(f"      range_log10_decades= "
                  f"{np.log10(rmap.amplitude_max / max(rmap.amplitude_min, 1e-300)):.2f}"
                  if rmap.amplitude_max > 0 else "      range_log10_decades= N/A")
            print(f"      temporal_distrib   = {rmap.temporal_distribution}")
            print(f"      n_transitions      = {len(rmap.transitions)}")

            results[case_label]["by_t_sim"][f"t_sim_{int(t_sim)}"] = {
                "t_sim": t_sim,
                "structure": {
                    "t_max_Dh": struct.t_max_Dh,
                    "max_Dh": struct.max_Dh,
                    "tau_div": struct.tau_div,
                    "tau_rec_10pct": struct.tau_rec_10pct,
                    "tau_rec_1pct": struct.tau_rec_1pct,
                    "plateau_detected": struct.plateau_detected,
                    "plateau_duration": struct.plateau_duration,
                    "plateau_value": struct.plateau_value,
                    "max_well_defined": struct.max_well_defined,
                    "monotonic_after_peak": struct.monotonic_after_peak,
                    "n_redivergence_episodes": struct.n_redivergence_episodes,
                },
                "redivergence_map": {
                    "n_events": rmap.n_events,
                    "amplitude_min": rmap.amplitude_min,
                    "amplitude_median": rmap.amplitude_median,
                    "amplitude_max": rmap.amplitude_max,
                    "temporal_distribution": rmap.temporal_distribution,
                    "n_transitions": len(rmap.transitions),
                    "distance_median_to_transition":
                        rmap.distance_to_nearest_transition_median,
                    "transitions_by_type": {
                        ttype: sum(1 for t in rmap.transitions
                                   if t.transition_type == ttype)
                        for ttype in ['peak', 'plateau_entry', 'plateau_exit',
                                      'slope_change']
                    },
                },
            }

    # ─── Analyse cross-t_sim : MIGRATION DES SIGNATURES ───
    print(f"\n\n{'='*78}")
    print(f"ANALYSE CROSS-T_SIM — Migration des signatures (pas asymptote)")
    print(f"{'='*78}")

    for case_label, case_data in results.items():
        print(f"\n  ── {case_label} ──")
        print(f"  {'t_sim':<10} {'max_Dh':>12} {'tau_div':>10} "
              f"{'tau_rec_10':>12} {'plateau':>10} {'n_redivg':>10} "
              f"{'amp_max':>12}")
        for ts_key, ts_data in case_data["by_t_sim"].items():
            s = ts_data["structure"]
            r = ts_data["redivergence_map"]
            t10 = f"{s['tau_rec_10pct']:.2f}" if s['tau_rec_10pct'] else "N/A"
            plat = "OUI" if s['plateau_detected'] else "non"
            print(f"  {ts_key:<10} {s['max_Dh']:>12.3e} "
                  f"{s['tau_div']:>10.2f} {t10:>12} {plat:>10} "
                  f"{r['n_events']:>10} {r['amplitude_max']:>12.3e}")

    print(f"\n\n{'='*78}")
    print(f"GRILLE A/B/C — Session 3 (à remplir après lecture)")
    print(f"{'='*78}")

    return results


if __name__ == "__main__":
    summary = run_session_3()
    output_dir = REPO_ROOT / "results" / "phase6d_beta"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_3_temporal_extension.json"

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
