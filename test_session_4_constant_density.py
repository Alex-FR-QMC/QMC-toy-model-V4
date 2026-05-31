"""
Test 6d-β session 4 — Cartographie sous densité d'échantillonnage constante.

Rituel d'ouverture (validé) :
- Niveau visé : 2 (toujours)
- Mouvement : cartographie, pas nouvelle observable
- Risque actif : "recherche de la vraie densité" (dérive purificatrice)

Question discriminante (reformulée par Alex) :
«Comment certaines signatures changent-elles lorsqu'on fixe la
granularité d'observation plutôt que le nombre total d'échantillons ?»

PAS : "quelle est la vraie densité ?"
PAS : "quelle est la mesure non-biaisée ?"

L'objectif est de cartographier ce qui reste invariant quand on
change la géométrie d'échantillonnage.

Discipline minimaliste :
- t_sim ∈ {500, 1000, 2000} (identique à session 3)
- 3 cas (FAIBLE/MOYEN/ÉLEVÉ)
- dt FIXÉ à dt_0 (identique session 3)
- Densité d'échantillonnage CONSTANTE : n_snapshots ∝ t_sim
  → n_snapshots = 80 × (t_sim/500) ∈ {80, 160, 320}

Lectures concurrentes documentées (préambule rituel étape 4) :
L1 (artefact pur), L2 (fenêtre bornée), L3 (couplage), L4 (hétérogène).
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


def simulate_pair_constant_density(
    psi_A_init, psi_B_init, h_init, D, beta, gamma, h0,
    t_sim, dt, n_snapshots,
):
    """Simule pair (A, B) avec n_snapshots variable proportionnel à t_sim."""
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


def run_session_4():
    gamma = 1.0
    D = 0.1
    h0 = 1.0

    cases = [
        ("FAIBLE", 80.0, "B3", make_psi_B3),
        ("MOYEN", 45.0, "B2", lambda: make_psi_B2(sigma_0=1.0)),
        ("ELEVE", 60.0, "B2", lambda: make_psi_B2(sigma_0=1.0)),
    ]

    # Densité constante = 80 snapshots / 500 unités = 0.16 / unité
    # → n_snapshots ∝ t_sim
    configurations = [
        (500.0, 80),
        (1000.0, 160),
        (2000.0, 320),
    ]

    print(f"{'='*78}")
    print(f"6d-β SESSION 4 — Densité d'échantillonnage constante")
    print(f"  Question : ce qui reste invariant en fixant la granularité")
    print(f"  Densité = 0.16 snapshots/unité de temps (fixe)")
    print(f"  NOT 'recherche de la vraie densité'")
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

        for t_sim, n_snap in configurations:
            density = n_snap / t_sim
            print(f"\n  ── t_sim={t_sim}, n_snapshots={n_snap} "
                  f"(densité {density:.4f}/unité) ──")
            times, Dh = simulate_pair_constant_density(
                psi_A, psi_B, h_init, D, beta, gamma, h0,
                t_sim, dt_0, n_snap,
            )

            struct = compute_temporal_structure(times, Dh)
            rmap = compute_redivergence_map(times, Dh)

            print(f"    Structure :")
            print(f"      max_Dh             = {struct.max_Dh:.4e}")
            print(f"      t_max_Dh           = {struct.t_max_Dh:.4f}")
            print(f"      tau_div            = {struct.tau_div:.4f}")
            print(f"      plateau_detected   = {struct.plateau_detected}")
            if struct.plateau_detected:
                print(f"        plateau_value    = {struct.plateau_value:.4e}")
            print(f"    Re-divergences :")
            print(f"      n_events           = {rmap.n_events}")
            print(f"      amp_max            = {rmap.amplitude_max:.4e}")
            print(f"      amp_median         = {rmap.amplitude_median:.4e}")
            print(f"      temporal_distrib   = {rmap.temporal_distribution}")
            print(f"      n_transitions      = {len(rmap.transitions)}")

            results[case_label]["by_t_sim"][f"t_sim_{int(t_sim)}"] = {
                "t_sim": t_sim,
                "n_snapshots": n_snap,
                "density_per_unit": density,
                "structure": {
                    "max_Dh": struct.max_Dh,
                    "t_max_Dh": struct.t_max_Dh,
                    "tau_div": struct.tau_div,
                    "tau_rec_10pct": struct.tau_rec_10pct,
                    "tau_rec_1pct": struct.tau_rec_1pct,
                    "plateau_detected": struct.plateau_detected,
                    "plateau_duration": struct.plateau_duration,
                    "plateau_value": struct.plateau_value,
                    "max_well_defined": struct.max_well_defined,
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
                },
            }

    # ─── Cartographie d'invariance ───
    print(f"\n\n{'='*78}")
    print(f"CARTOGRAPHIE D'INVARIANCE — densité constante")
    print(f"{'='*78}")

    for case_label, case_data in results.items():
        print(f"\n  ── {case_label} ──")
        print(f"  {'t_sim':<10} {'n_snap':>8} {'max_Dh':>12} {'tau_div':>10} "
              f"{'plat_val':>12} {'n_events':>10} {'amp_max':>12}")
        for ts_key, ts_data in case_data["by_t_sim"].items():
            s = ts_data["structure"]
            r = ts_data["redivergence_map"]
            plat_v = (f"{s['plateau_value']:.3e}"
                      if s['plateau_value'] is not None else "N/A")
            print(f"  {ts_key:<10} {ts_data['n_snapshots']:>8} "
                  f"{s['max_Dh']:>12.3e} {s['tau_div']:>10.2f} "
                  f"{plat_v:>12} {r['n_events']:>10} "
                  f"{r['amplitude_max']:>12.3e}")

    return results


if __name__ == "__main__":
    summary = run_session_4()
    output_dir = REPO_ROOT / "results" / "phase6d_beta"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_4_constant_density.json"

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
