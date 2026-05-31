"""
Test 6d-β session 6 — Modification minimale opérateur de détection.

Question (formulée par Alex) :
«Que devient la régularité observée en session 5 si l'on modifie
légèrement l'opérateur de détection, sans chercher à produire un
détecteur meilleur ou plus vrai ?»

PAS : "trouver le bon détecteur"
PAS : "purifier la régularité observée"
PAS : "construire une nouvelle taxonomie de détecteurs"

Modification minimale : ajout d'un seuil absolu relatif à max(Dh).
- seuil 0       → reproduit S5 exactement
- seuil 1e-6×max → modification minimale

Cas : ÉLEVÉ uniquement (β=60 A↔B2)
Protocole : densité constante, t_sim ∈ {500, 1000, 2000}, dt fixé
Sondes : les mêmes que S5 (statut symétrique préservé)

Reprise minimale. Pas de nouveau cadre méthodologique.
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
    RedivergenceEvent, detect_morphological_transitions,
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


def detect_redivergence_events_with_threshold(
    times, Dh, threshold_relative,
):
    """Modification minimale du détecteur S2 :
    on ne compte un événement que si la remontée dépasse
    threshold_relative × max(Dh).

    threshold_relative = 0 reproduit S2 exactement.
    """
    idx_max = int(np.argmax(Dh))
    Dh_max = float(Dh[idx_max])
    threshold_abs = threshold_relative * Dh_max
    events = []
    if idx_max >= len(Dh) - 1:
        return events

    for i in range(idx_max, len(Dh) - 1):
        delta = Dh[i + 1] - Dh[i]
        if delta > threshold_abs:
            events.append(RedivergenceEvent(
                t=float(times[i + 1]),
                t_index=i + 1,
                amplitude=float(delta),
                Dh_before=float(Dh[i]),
                Dh_after=float(Dh[i + 1]),
                fraction_of_max=float(delta) / max(Dh_max, 1e-30),
            ))
    return events


def simulate_pair_constant_density(
    psi_A_init, psi_B_init, h_init, D, beta, gamma, h0,
    t_sim, dt, n_snapshots,
):
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


def probe_interval_distribution(events):
    if len(events) < 2:
        return {"n_intervals": 0}
    ts = sorted(e.t for e in events)
    deltas = np.diff(ts)
    return {
        "n_intervals": len(deltas),
        "delta_median": float(np.median(deltas)),
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
        "CV": float(deltas.std() / max(deltas.mean(), 1e-30)),
    }


def probe_rank_amplitude_correlation(events):
    if len(events) < 3:
        return {"n": len(events)}
    sorted_events = sorted(events, key=lambda e: e.t)
    amps = [e.amplitude for e in sorted_events]
    n = len(amps)
    ranks_t = list(range(n))
    ranks_a = list(np.argsort(np.argsort(amps)))
    d = np.array(ranks_t) - np.array(ranks_a)
    rho = float(1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1)))
    return {"n": n, "spearman_rho": rho}


def probe_log_amplitude(events, n_bins=10):
    if len(events) < 2:
        return {"n": len(events)}
    amps = np.array([e.amplitude for e in events])
    amps = amps[amps > 0]
    if len(amps) < 2:
        return {"n": 0}
    log_amps = np.log10(amps)
    log_min, log_max = float(log_amps.min()), float(log_amps.max())
    if log_max - log_min < 1e-9:
        return {"n": len(amps), "decades_spanned": 0.0}
    bins = np.linspace(log_min, log_max, n_bins + 1)
    hist, _ = np.histogram(log_amps, bins=bins)
    nonzero = hist[hist > 0]
    return {
        "n": len(amps),
        "decades_spanned": log_max - log_min,
        "n_bins_empty": int(np.sum(hist == 0)),
        "max_min_ratio": (float(nonzero.max() / nonzero.min())
                          if len(nonzero) >= 2 else None),
    }


def run_session_6():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0

    configurations = [(500.0, 80), (1000.0, 160), (2000.0, 320)]
    thresholds = [0.0, 1e-6]  # S5 reproduit + modification minimale

    print(f"{'='*78}")
    print(f"6d-β SESSION 6 — Modification minimale opérateur de détection")
    print(f"  cas ÉLEVÉ, densité constante")
    print(f"  seuils relatifs : {thresholds}")
    print(f"{'='*78}")

    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_A = make_psi_A(sigma_0=1.8)
    psi_B = make_psi_B2(sigma_0=1.0)
    psi_max = max(float(psi_A.max()), float(psi_B.max()))
    rate_h = beta * psi_max + gamma
    dt_0 = 0.5 * min(cfl_dt_max(h0, D), 1.0 / rate_h)
    print(f"  dt fixé : {dt_0:.6f}\n")

    results = {"params": {"beta": beta, "dt_0": dt_0}, "by_threshold": {}}

    # Pré-calcul des trajectoires (mêmes pour les deux seuils)
    trajectories = {}
    for t_sim, n_snap in configurations:
        times, Dh = simulate_pair_constant_density(
            psi_A, psi_B, h_init, D, beta, gamma, h0,
            t_sim, dt_0, n_snap,
        )
        trajectories[t_sim] = (times, Dh)

    for threshold in thresholds:
        key = f"threshold_{threshold:.0e}"
        print(f"\n{'─'*78}")
        print(f"Seuil relatif = {threshold:.0e} × max(Dh)")
        print(f"{'─'*78}")

        results["by_threshold"][key] = {"threshold_relative": threshold,
                                         "by_t_sim": {}}

        for t_sim, n_snap in configurations:
            times, Dh = trajectories[t_sim]
            events = detect_redivergence_events_with_threshold(
                times, Dh, threshold,
            )
            transitions = detect_morphological_transitions(times, Dh)

            p_int = probe_interval_distribution(events)
            p_rk = probe_rank_amplitude_correlation(events)
            p_la = probe_log_amplitude(events)

            print(f"\n  t_sim = {t_sim} (n_snap={n_snap})")
            print(f"    n_events            = {len(events)}")
            if "CV" in p_int:
                print(f"    CV des Δt           = {p_int['CV']:.4f}")
                print(f"    delta_median        = {p_int['delta_median']:.4f}")
            if "spearman_rho" in p_rk:
                print(f"    Spearman ρ          = {p_rk['spearman_rho']:+.4f}")
            if "decades_spanned" in p_la:
                print(f"    décades couvertes   = {p_la['decades_spanned']:.2f}")
                print(f"    bins vides          = "
                      f"{p_la.get('n_bins_empty', 'N/A')}/10")

            results["by_threshold"][key]["by_t_sim"][f"t_sim_{int(t_sim)}"] = {
                "t_sim": t_sim,
                "n_events": len(events),
                "probe_intervals": p_int,
                "probe_rank_amplitude": p_rk,
                "probe_log_amplitude": p_la,
            }

    # Synthèse comparative
    print(f"\n\n{'='*78}")
    print(f"COMPARAISON CROSS-SEUIL (n_events)")
    print(f"{'='*78}\n")
    print(f"  {'t_sim':<10} {'seuil=0':>12} {'seuil=1e-6':>14}")
    for t_sim, n_snap in configurations:
        n0 = results["by_threshold"]["threshold_0e+00"]["by_t_sim"][
            f"t_sim_{int(t_sim)}"]["n_events"]
        n1 = results["by_threshold"]["threshold_1e-06"]["by_t_sim"][
            f"t_sim_{int(t_sim)}"]["n_events"]
        print(f"  {t_sim:<10} {n0:>12} {n1:>14}")

    print(f"\n{'='*78}")
    print(f"COMPARAISON CROSS-SEUIL (Spearman ρ et CV)")
    print(f"{'='*78}\n")
    for t_sim, _ in configurations:
        print(f"  t_sim = {t_sim}")
        for key in ["threshold_0e+00", "threshold_1e-06"]:
            d = results["by_threshold"][key]["by_t_sim"][f"t_sim_{int(t_sim)}"]
            rho = d["probe_rank_amplitude"].get("spearman_rho", "N/A")
            cv = d["probe_intervals"].get("CV", "N/A")
            rho_str = f"{rho:+.4f}" if isinstance(rho, float) else rho
            cv_str = f"{cv:.4f}" if isinstance(cv, float) else cv
            print(f"    {key:<20} ρ = {rho_str}, CV = {cv_str}")

    return results


if __name__ == "__main__":
    summary = run_session_6()
    output_dir = REPO_ROOT / "results" / "phase6d_beta"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_6_minimal_threshold.json"

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
    print(f"\nRésultats : {output_path}")
