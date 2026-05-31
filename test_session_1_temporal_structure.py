"""
Test 6d-β session 1 — Structure temporelle de D_h(t) sur §5.7.

Rituel d'ouverture appliqué :
- §11.4 et §6.4 relus
- Niveau visé : 2 (morphodynamique)
- Mouvement : construction nouvelle observable niveau 2
- Lectures concurrentes plausibles documentées dans module
  instrumentation_6d_beta/temporal_structure_Dh.py

Objectif : appliquer l'instrumentation de structure temporelle aux
trajectoires D_h(t) déjà mesurées en §5.7, pour révéler des
géométries de convergence qui étaient compressées dans AUC_Dh.

Ce qu'on cherche à voir :
- Y a-t-il des trajectoires non-monotones (re-divergence après pic) ?
- Y a-t-il des plateaux trajectoriels intermédiaires ?
- τ_div et τ_rec varient-ils significativement entre A↔B1, A↔B2, A↔B3 ?

Ce qu'on NE cherche PAS :
- Pas de tentative de lecture MCQ
- Pas d'identification avec un opérateur Ch3
- Pas de quantification "score de robustesse"
- Pas d'inflation logging
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
    compute_temporal_structure, summarize_structure,
)


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


def make_psi_B1_decentered(sigma_0=1.8):
    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = 1.0, 2.0, 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - cx) ** 2
                      + (coords[j] - cy) ** 2
                      + (coords[k] - cz) ** 2)
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


def make_psi_B3_uniform():
    return np.full((N_AXIS, N_AXIS, N_AXIS), 1.0 / (N_AXIS ** 3))


def run_trajectory_with_dense_snapshots(
    psi_init, h_init, D, beta, gamma, h0, t_sim, n_snapshots=80,
):
    """
    Snapshots denses paramétriques — hook d'observabilité §6.6.
    Pas de modification du moteur, juste logging structuré.

    n_snapshots = 80 (vs 30 en 6d-α) pour résolution temporelle
    permettant de détecter plateaux et re-divergences.
    """
    psi_max = float(psi_init.max())
    rate_h = beta * psi_max + gamma
    dt_cfl_diff = cfl_dt_max(h0, D)
    dt_cfl_h = 1.0 / rate_h
    dt = 0.5 * min(dt_cfl_diff, dt_cfl_h)
    n_steps = int(np.ceil(t_sim / dt))

    snapshot_indices = sorted(set(
        list(range(0, min(20, n_steps + 1)))
        + list(np.linspace(0, n_steps, n_snapshots, dtype=int))
    ))
    snapshot_set = set(snapshot_indices)

    snapshots_t = []
    snapshots_psi = []
    snapshots_h = []

    psi = psi_init.copy()
    h = h_init.copy()
    snapshots_t.append(0.0)
    snapshots_psi.append(psi.copy())
    snapshots_h.append(h.copy())

    for step in range(1, n_steps + 1):
        psi, h = step_engine_euler(psi, h, D, beta, gamma, h0, dt)
        if step in snapshot_set:
            snapshots_t.append(step * dt)
            snapshots_psi.append(psi.copy())
            snapshots_h.append(h.copy())

    return {
        "t": np.array(snapshots_t),
        "psi": snapshots_psi,
        "h": snapshots_h,
        "dt": dt,
        "n_steps": n_steps,
    }


def compute_D_h_trajectory(snapshots_A, snapshots_B):
    """D_h(t) trajectoriel entre A et B aux mêmes snapshots."""
    assert len(snapshots_A["t"]) == len(snapshots_B["t"])
    D_h = []
    for hA, hB in zip(snapshots_A["h"], snapshots_B["h"]):
        norm_A = max(np.linalg.norm(hA), 1e-30)
        D_h.append(float(np.linalg.norm(hB - hA) / norm_A))
    return snapshots_A["t"], np.array(D_h)


def run_test_session_1():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_sim = 500.0

    beta_values = [45.0, 60.0, 80.0]
    results = {}

    print(f"{'='*78}")
    print(f"6d-β SESSION 1 — Structure temporelle D_h(t)")
    print(f"  Niveau visé : 2 (morphodynamique global)")
    print(f"  Question : qu'est-ce qui disparaît dans AUC_Dh ?")
    print(f"{'='*78}")

    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)

    for beta in beta_values:
        print(f"\n{'─'*78}")
        print(f"β = {beta}, γ = {gamma}")
        print(f"{'─'*78}")

        # Initialisations
        psi_A = make_psi_A_centered(sigma_0=1.8)
        psi_B1 = make_psi_B1_decentered(sigma_0=1.8)
        psi_B2 = make_psi_B2_bimodal(sigma_0=1.0)
        psi_B3 = make_psi_B3_uniform()

        # CFL la plus contraignante
        psi_max_global = max(
            float(psi_A.max()), float(psi_B1.max()),
            float(psi_B2.max()), float(psi_B3.max())
        )
        rate_h = beta * psi_max_global + gamma
        dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / rate_h)
        n_steps = int(np.ceil(t_sim / dt))
        snapshot_indices = sorted(set(
            list(range(0, min(20, n_steps + 1)))
            + list(np.linspace(0, n_steps, 80, dtype=int))
        ))

        snapshots = {}
        for fam_name, psi_init in [
            ("A", psi_A), ("B1", psi_B1), ("B2", psi_B2), ("B3", psi_B3)
        ]:
            psi = psi_init.copy()
            h = h_init.copy()
            t_list = [0.0]
            psi_list = [psi.copy()]
            h_list = [h.copy()]
            snapshot_set = set(snapshot_indices)
            for step in range(1, n_steps + 1):
                psi, h = step_engine_euler(psi, h, D, beta, gamma, h0, dt)
                if step in snapshot_set:
                    t_list.append(step * dt)
                    psi_list.append(psi.copy())
                    h_list.append(h.copy())
            snapshots[fam_name] = {
                "t": np.array(t_list),
                "psi": psi_list,
                "h": h_list,
            }

        comparisons_structures = {}
        for fam_B in ["B1", "B2", "B3"]:
            t, D_h_traj = compute_D_h_trajectory(
                snapshots["A"], snapshots[fam_B]
            )
            struct = compute_temporal_structure(t, D_h_traj)
            comparisons_structures[f"A_vs_{fam_B}"] = struct

            print(f"\n  ─── A vs {fam_B} ───")
            print(summarize_structure(struct))

        # Synthèse comparée
        print(f"\n  ─── Synthèse comparée (β={beta}) ───")
        print(f"  {'compar':<10} {'τ_div':>8} {'τ_rec_10':>10} "
              f"{'τ_rec_1':>10} {'n_redivg':>10} {'plateau':>10}")
        for compar_name, s in comparisons_structures.items():
            t10_str = f"{s.tau_rec_10pct:.2f}" if s.tau_rec_10pct else "N/A"
            t1_str = f"{s.tau_rec_1pct:.2f}" if s.tau_rec_1pct else "N/A"
            plat_str = "OUI" if s.plateau_detected else "non"
            print(f"  {compar_name:<10} {s.tau_div:>8.2f} {t10_str:>10} "
                  f"{t1_str:>10} {s.n_redivergence_episodes:>10} "
                  f"{plat_str:>10}")

        results[f"beta_{int(beta)}"] = {
            f"A_vs_{fam_B}": {
                "tau_div": s.tau_div,
                "tau_rec_10pct": s.tau_rec_10pct,
                "tau_rec_1pct": s.tau_rec_1pct,
                "max_Dh": s.max_Dh,
                "t_max_Dh": s.t_max_Dh,
                "n_redivergence_episodes": s.n_redivergence_episodes,
                "redivergence_amplitude_max": s.redivergence_amplitude_max,
                "monotonic_after_peak": s.monotonic_after_peak,
                "plateau_detected": s.plateau_detected,
                "plateau_duration": s.plateau_duration,
                "plateau_value": s.plateau_value,
                "max_well_defined": s.max_well_defined,
                "diagnostic_notes": s.diagnostic_notes,
            }
            for fam_B, s in [
                ("B1", comparisons_structures["A_vs_B1"]),
                ("B2", comparisons_structures["A_vs_B2"]),
                ("B3", comparisons_structures["A_vs_B3"]),
            ]
        }

    # Synthèse cross-β
    print(f"\n\n{'='*78}")
    print(f"SYNTHÈSE CROSS-β — Géométries de transformation révélées")
    print(f"{'='*78}")

    n_non_monotonic = 0
    n_with_plateau = 0
    n_well_defined = 0
    n_total = 0
    for label, beta_res in results.items():
        for compar, struct_dict in beta_res.items():
            n_total += 1
            if not struct_dict["monotonic_after_peak"]:
                n_non_monotonic += 1
            if struct_dict["plateau_detected"]:
                n_with_plateau += 1
            if struct_dict["max_well_defined"]:
                n_well_defined += 1

    print(f"\n  Total trajectoires analysées       : {n_total}")
    print(f"  Pics bien définis                  : {n_well_defined}/{n_total}")
    print(f"  Trajectoires non-monotones post-pic : {n_non_monotonic}/{n_total}")
    print(f"  Plateaux détectés                  : {n_with_plateau}/{n_total}")

    print(f"\n  Lecture morphodynamique (niveau 2) :")
    if n_non_monotonic > 0:
        print(f"    - {n_non_monotonic} trajectoires présentent des épisodes")
        print(f"      de re-divergence après le pic principal. Cela révèle")
        print(f"      une structure non-monotone que AUC_Dh masquait.")
    else:
        print(f"    - Toutes trajectoires sont monotones post-pic. La")
        print(f"      structure temporelle interne n'ajoute rien à AUC.")
        print(f"      Lecture concurrente 1 (redondance) confirmée pour ce régime.")

    if n_with_plateau > 0:
        print(f"    - {n_with_plateau} trajectoires présentent un plateau")
        print(f"      trajectoriel. Phase de stabilisation intermédiaire avant")
        print(f"      relaxation finale, invisible dans D_h_final.")

    print(f"\n  Statut session 1 :")
    print(f"    Niveau habité : 2 (morphodynamique)")
    print(f"    Observable construite : structure temporelle D_h(t)")
    print(f"    Test §5.3 (survie autonome) : PASSÉ avant code")
    print(f"    Lectures concurrentes documentées : 1, 2, 3 (module)")
    print(f"    À inscrire en §11.4 : oui, après audit Alex")

    return results


if __name__ == "__main__":
    summary = run_test_session_1()
    output_dir = REPO_ROOT / "results" / "phase6d_beta"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_1_temporal_structure_Dh.json"

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
