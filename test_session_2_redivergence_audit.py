"""
Test 6d-β session 2 — Audit de stabilité interprétative des re-divergences.

Rituel d'ouverture appliqué (voir préambule discussion) :
- §11.4 et §6.4 relus
- Niveau visé : 2 (toujours)
- Mouvement : PAS de circulation §11.4 cette session
- Risque actif : purification déguisée

Question discriminante :
«Dans quelles conditions les re-divergences changent-elles de nature ?»

Pas : "sont-elles réelles ou artefact ?"

Discipline minimaliste :
- 3 cas représentatifs × 3 valeurs de dt = 9 simulations
- Mesures (a)+(b)+(d) du module redivergence_map
- Pas d'inflation : pas de multi-seed, pas de n_snapshots varié, pas d'auto-corrélation

Lectures concurrentes à tester :
- L1 : toutes re-divergences de même nature numérique → invalidation
- L2 : aucune localisation préférentielle → invalidation enrichissement (b)
- L3 : régime intermédiaire (1e-9) reste indécidable → suspension §4.3

Cas choisis (3 ordres de grandeur des amplitudes max session 1) :
- Faible    : β=80 A↔B3 → 4e-17 (quasi flottant)
- Moyen     : β=45 A↔B2 → 8e-9 (zone dangereuse, vraie discrimination)
- Élevé     : β=60 A↔B2 → 5e-3 (clairement non-trivial)
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

from instrumentation_6d_beta.redivergence_map import (  # noqa: E402
    compute_redivergence_map, summarize_redivergence_map,
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


def simulate_with_dt(
    psi_init_A, psi_init_B, h_init, D, beta, gamma, h0, t_sim, dt,
    n_snapshots=80,
):
    """Simule A et B avec dt explicite (identique pour les deux)."""
    n_steps = int(np.ceil(t_sim / dt))
    snapshot_indices = sorted(set(
        list(range(0, min(20, n_steps + 1)))
        + list(np.linspace(0, n_steps, n_snapshots, dtype=int))
    ))
    snapshot_set = set(snapshot_indices)

    snapshots_A = {"t": [0.0], "h": [h_init.copy()]}
    snapshots_B = {"t": [0.0], "h": [h_init.copy()]}

    psi_A = psi_init_A.copy()
    h_A = h_init.copy()
    psi_B = psi_init_B.copy()
    h_B = h_init.copy()

    for step in range(1, n_steps + 1):
        psi_A, h_A = step_engine_euler(psi_A, h_A, D, beta, gamma, h0, dt)
        psi_B, h_B = step_engine_euler(psi_B, h_B, D, beta, gamma, h0, dt)
        if step in snapshot_set:
            t = step * dt
            snapshots_A["t"].append(t)
            snapshots_A["h"].append(h_A.copy())
            snapshots_B["t"].append(t)
            snapshots_B["h"].append(h_B.copy())

    return snapshots_A, snapshots_B


def compute_Dh_traj(snap_A, snap_B):
    times = np.array(snap_A["t"])
    Dh = []
    for hA, hB in zip(snap_A["h"], snap_B["h"]):
        norm_A = max(np.linalg.norm(hA), 1e-30)
        Dh.append(float(np.linalg.norm(hB - hA) / norm_A))
    return times, np.array(Dh)


def run_session_2():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_sim = 500.0

    # Trois cas représentatifs
    cases = [
        # (label, beta, fam_B_name, psi_B_factory)
        ("FAIBLE_4e-17", 80.0, "B3", make_psi_B3),
        ("MOYEN_8e-9", 45.0, "B2", lambda: make_psi_B2(sigma_0=1.0)),
        ("ELEVE_5e-3", 60.0, "B2", lambda: make_psi_B2(sigma_0=1.0)),
    ]

    # Trois résolutions temporelles
    # dt_0 = celui calculé naturellement par CFL en session 1
    # On le calcule pour chaque cas puis on prend dt_0, dt_0/2, dt_0/4
    dt_ratios = [1.0, 0.5, 0.25]

    print(f"{'='*78}")
    print(f"6d-β SESSION 2 — Audit stabilité interprétative re-divergences")
    print(f"  Question : dans quelles conditions changent-elles de nature ?")
    print(f"  PAS de purification réel/artefact")
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

        results[case_label] = {
            "params": {"beta": beta, "fam_B": fam_B_name, "dt_0": dt_0},
            "by_dt": {},
        }

        for ratio in dt_ratios:
            dt = dt_0 * ratio
            print(f"\n  ── dt = {dt:.6f} (ratio {ratio}) ──")

            snap_A, snap_B = simulate_with_dt(
                psi_A, psi_B, h_init, D, beta, gamma, h0, t_sim, dt,
                n_snapshots=80,
            )
            times, Dh_traj = compute_Dh_traj(snap_A, snap_B)
            rmap = compute_redivergence_map(times, Dh_traj)

            print(summarize_redivergence_map(rmap))

            # Sauvegarder pour analyse cross-dt
            results[case_label]["by_dt"][f"ratio_{ratio}"] = {
                "dt": dt,
                "n_events": rmap.n_events,
                "amplitude_min": rmap.amplitude_min,
                "amplitude_median": rmap.amplitude_median,
                "amplitude_max": rmap.amplitude_max,
                "temporal_distribution": rmap.temporal_distribution,
                "n_transitions": len(rmap.transitions),
                "transitions_by_type": {
                    ttype: sum(1 for t in rmap.transitions if t.transition_type == ttype)
                    for ttype in ['peak', 'plateau_entry', 'plateau_exit', 'slope_change']
                },
                "distance_median_to_transition": rmap.distance_to_nearest_transition_median,
                "histogram_log_bins": [
                    {"bin_min": b[0], "bin_max": b[1], "count": b[2]}
                    for b in rmap.histogram_log_bins
                ],
                "Dh_max_observed": float(np.max(Dh_traj)),
            }

    # ─── Analyse cross-dt ───
    print(f"\n\n{'='*78}")
    print(f"ANALYSE CROSS-DT — Stabilité des signatures")
    print(f"{'='*78}")

    for case_label, case_data in results.items():
        print(f"\n  ── {case_label} ──")
        print(f"  {'ratio':<10} {'n_events':>10} {'amp_max':>12} "
              f"{'amp_median':>12} {'distribution':>16}")
        for ratio_key, dt_data in case_data["by_dt"].items():
            print(f"  {ratio_key:<10} {dt_data['n_events']:>10} "
                  f"{dt_data['amplitude_max']:>12.3e} "
                  f"{dt_data['amplitude_median']:>12.3e} "
                  f"{dt_data['temporal_distribution']:>16}")

    # ─── Lecture finale : grille A/B/C ───
    print(f"\n\n{'='*78}")
    print(f"GRILLE A/B/C — Session 2")
    print(f"{'='*78}")

    print(f"\n(A) Certitudes :")
    print(f"  À remplir après lecture des résultats")
    print(f"\n(B) Hypothèses :")
    print(f"  À remplir après lecture des résultats")
    print(f"\n(C) Inconnus :")
    print(f"  À remplir après lecture des résultats")

    return results


if __name__ == "__main__":
    summary = run_session_2()
    output_dir = REPO_ROOT / "results" / "phase6d_beta"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_2_redivergence_audit.json"

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
