"""
Test 6d-β session 5 — Sondes de structure interne des 65 événements ÉLEVÉS.

Rituel d'ouverture appliqué (préambule discussion) :
- Niveau visé : 2 (toujours)
- Mouvement : interrogation ciblée, pas de circulation prévue
- Risque actif : micro-purification structurelle

Garde-fous explicites (validés Alex) :
- "Absence d'organisation détectable" = résultat positif recevable
- "Stabilité de structure ≠ persistance d'événements individuels"
- Les 65 événements existent relativement à l'algorithme de détection
- "Compatible avec" ≠ "décrit par"
- Aucune mesure n'est centrale a priori — symétrie statutaire

Question :
«La structure interne éventuelle d'un cas déjà stabilisé
existe-t-elle réellement ?»

PAS : "Y a-t-il une structure cachée à révéler ?"
PAS : "Quelle loi décrit ces événements ?"

5 sondes hétérogènes (statut symétrique) :
(1) Distribution Δt inter-événements
(2) Corrélation rang-amplitude (Spearman)
(3) Distribution log-amplitude (cartographie, pas fit)
(4) Distances aux transitions
(5) Identité événementielle approximative (cross-t_sim matching)

Cas : ÉLEVÉ uniquement (β=60 A↔B2)
Protocole : densité constante (n_snapshots = 0.16 × t_sim)
t_sim ∈ {500, 1000, 2000}
dt fixé
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
    detect_redivergence_events, detect_morphological_transitions,
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


# ─── Sondes (5 + 1 méta pour identité événementielle) ───

def probe_1_interval_distribution(events):
    """Distribution des Δt inter-événements. Comparaison à exponentielle
    (Poisson) uniquement comme référence, pas comme privilège."""
    if len(events) < 2:
        return {"n_intervals": 0, "note": "trop peu d'événements"}
    times = sorted(e.t for e in events)
    deltas = np.diff(times)
    log_deltas = np.log10(deltas)
    return {
        "n_intervals": len(deltas),
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
        "delta_median": float(np.median(deltas)),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "log_delta_min": float(log_deltas.min()),
        "log_delta_max": float(log_deltas.max()),
        "coefficient_of_variation": float(deltas.std() / max(deltas.mean(), 1e-30)),
        # CV ≈ 1 pour Poisson, < 1 pour régulier, > 1 pour clusterisé
        "interpretation_marker": "CV~1 compat. Poisson / <1 compat. régulier / >1 compat. cluster",
    }


def probe_2_rank_amplitude_correlation(events):
    """Corrélation Spearman entre rang temporel et amplitude."""
    if len(events) < 3:
        return {"n": len(events), "note": "trop peu d'événements"}
    # Trier par temps, puis prendre les amplitudes dans cet ordre
    sorted_events = sorted(events, key=lambda e: e.t)
    amplitudes = [e.amplitude for e in sorted_events]
    ranks_temporal = list(range(len(amplitudes)))
    ranks_amplitude = list(np.argsort(np.argsort(amplitudes)))
    # Spearman par formule directe
    n = len(amplitudes)
    d = np.array(ranks_temporal) - np.array(ranks_amplitude)
    spearman_rho = float(1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1)))
    return {
        "n": n,
        "spearman_rho_temporal_vs_amplitude": spearman_rho,
        "interpretation_marker": "|ρ| proche 0 = ordre indépendant ; |ρ| proche 1 = ordre lié",
    }


def probe_3_log_amplitude_distribution(events, n_bins=10):
    """Distribution log-amplitude (cartographie pas fit)."""
    if len(events) < 2:
        return {"n": len(events), "note": "trop peu d'événements"}
    amps = np.array([e.amplitude for e in events])
    amps = amps[amps > 0]
    if len(amps) < 2:
        return {"n": 0, "note": "amplitudes positives insuffisantes"}
    log_amps = np.log10(amps)
    log_min, log_max = log_amps.min(), log_amps.max()
    if log_max - log_min < 1e-9:
        return {"n": len(amps), "note": "amplitudes quasi identiques"}
    bins = np.linspace(log_min, log_max, n_bins + 1)
    hist, _ = np.histogram(log_amps, bins=bins)
    # Calcul d'indicateurs minimaux (PAS de fit)
    # - rapport max/min count (uniforme si ~1, clusterisé sinon)
    nonzero_hist = hist[hist > 0]
    max_min_ratio = (float(nonzero_hist.max() / nonzero_hist.min())
                     if len(nonzero_hist) >= 2 else None)
    return {
        "n": int(len(amps)),
        "log_amplitude_range": [float(log_min), float(log_max)],
        "decades_spanned": float(log_max - log_min),
        "n_bins": n_bins,
        "histogram_counts": [int(h) for h in hist],
        "bin_edges_log10": [float(b) for b in bins],
        "n_empty_bins": int(np.sum(hist == 0)),
        "max_min_count_ratio": max_min_ratio,
        "interpretation_marker": "ratio~1 + 0 bin vide = compat. uniforme ; sinon non uniforme",
    }


def probe_4_distance_to_transitions(events, transitions):
    """Distribution des distances de chaque événement à transition la plus proche."""
    if not events or not transitions:
        return {"n_events": len(events), "n_transitions": len(transitions),
                "note": "données insuffisantes"}
    distances = []
    nearest_types = []
    for e in events:
        d_min = float('inf')
        nearest_type = None
        for tr in transitions:
            d = abs(e.t - tr.t)
            if d < d_min:
                d_min = d
                nearest_type = tr.transition_type
        distances.append(d_min)
        nearest_types.append(nearest_type)
    distances = np.array(distances)
    return {
        "n_events": len(events),
        "n_transitions": len(transitions),
        "distance_min": float(distances.min()),
        "distance_max": float(distances.max()),
        "distance_median": float(np.median(distances)),
        "distance_mean": float(distances.mean()),
        "fraction_near_transition_below_10pct_duration": None,  # rempli ailleurs
        "nearest_transition_type_counts": {
            t: int(nearest_types.count(t))
            for t in set(nearest_types) if t is not None
        },
        "interpretation_marker": "distances petites + concentrées sur 1 type = liés ; sinon indépendants",
    }


def probe_5_event_identity_across_t_sim(events_by_t_sim, t_tolerance=0.5,
                                          amp_relative_tolerance=0.1):
    """
    Comparaison d'identité événementielle entre t_sim.

    Un événement à (t1, a1) dans simulation A est dit "identifié à"
    un événement à (t2, a2) dans simulation B si :
    - |t1 - t2| < t_tolerance
    - |a1 - a2| / max(a1, a2) < amp_relative_tolerance

    GARDE-FOU MAJEUR (rappel Alex) :
    «stabilité d'une structure d'événements ≠ persistance des
    événements individuels»

    Cette sonde tente de tester cette distinction empiriquement.
    """
    t_sims = sorted(events_by_t_sim.keys())
    if len(t_sims) < 2:
        return {"note": "besoin d'au moins 2 t_sim"}

    matches = {}
    for i, ts_a in enumerate(t_sims):
        for ts_b in t_sims[i + 1:]:
            events_a = events_by_t_sim[ts_a]
            events_b = events_by_t_sim[ts_b]
            if not events_a or not events_b:
                continue
            # Matching greedy
            matched_count = 0
            matched_pairs = []
            for ea in events_a:
                best_match = None
                best_score = float('inf')
                for eb in events_b:
                    if abs(ea.t - eb.t) < t_tolerance:
                        amp_max = max(ea.amplitude, eb.amplitude, 1e-300)
                        rel_diff = abs(ea.amplitude - eb.amplitude) / amp_max
                        if rel_diff < amp_relative_tolerance:
                            score = abs(ea.t - eb.t) + rel_diff
                            if score < best_score:
                                best_score = score
                                best_match = eb
                if best_match is not None:
                    matched_count += 1
                    matched_pairs.append((ea.t, ea.amplitude,
                                          best_match.t, best_match.amplitude))
            matches[f"{ts_a}_vs_{ts_b}"] = {
                "n_events_a": len(events_a),
                "n_events_b": len(events_b),
                "n_matched": matched_count,
                "fraction_matched_a": matched_count / max(len(events_a), 1),
                "fraction_matched_b": matched_count / max(len(events_b), 1),
                "first_5_matches_sample": matched_pairs[:5],
            }

    return {
        "matching_criteria": {
            "t_tolerance": t_tolerance,
            "amp_relative_tolerance": amp_relative_tolerance,
        },
        "pairwise_matches": matches,
        "interpretation_marker": (
            "matching fraction proche 1 = persistance événementielle ; "
            "proche 0 = structure agrégée stable sans persistance individuelle"
        ),
    }


def run_session_5():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0

    configurations = [
        (500.0, 80),
        (1000.0, 160),
        (2000.0, 320),
    ]

    print(f"{'='*78}")
    print(f"6d-β SESSION 5 — Structure interne des 65 événements ÉLEVÉS")
    print(f"  Question : organisation interne réelle, ou stabilité agrégée seule ?")
    print(f"  Garde-fous :")
    print(f"  - absence d'organisation = résultat positif")
    print(f"  - stabilité structure ≠ persistance événements individuels")
    print(f"  - 5 sondes à statut symétrique")
    print(f"{'='*78}")

    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_A = make_psi_A(sigma_0=1.8)
    psi_B = make_psi_B2(sigma_0=1.0)
    psi_max = max(float(psi_A.max()), float(psi_B.max()))
    rate_h = beta * psi_max + gamma
    dt_0 = 0.5 * min(cfl_dt_max(h0, D), 1.0 / rate_h)
    print(f"\n  dt fixé : {dt_0:.6f}")

    events_by_t_sim = {}
    transitions_by_t_sim = {}
    results = {"params": {"beta": beta, "dt_0": dt_0}, "by_t_sim": {}}

    for t_sim, n_snap in configurations:
        print(f"\n{'─'*78}")
        print(f"t_sim = {t_sim}, n_snapshots = {n_snap}")
        print(f"{'─'*78}")

        times, Dh = simulate_pair_constant_density(
            psi_A, psi_B, h_init, D, beta, gamma, h0,
            t_sim, dt_0, n_snap,
        )
        events = detect_redivergence_events(times, Dh)
        transitions = detect_morphological_transitions(times, Dh)
        events_by_t_sim[t_sim] = events
        transitions_by_t_sim[t_sim] = transitions

        print(f"  n_events détectés = {len(events)}")
        print(f"  n_transitions     = {len(transitions)}")

        p1 = probe_1_interval_distribution(events)
        p2 = probe_2_rank_amplitude_correlation(events)
        p3 = probe_3_log_amplitude_distribution(events)
        p4 = probe_4_distance_to_transitions(events, transitions)

        # Affichage minimal pour chaque sonde
        print(f"\n  Sonde 1 — Δt inter-événements :")
        if "n_intervals" in p1 and p1.get("n_intervals", 0) > 0:
            print(f"    CV = {p1['coefficient_of_variation']:.4f}")
            print(f"    delta_median = {p1['delta_median']:.4f}")
            print(f"    delta_range  = [{p1['delta_min']:.4f}, {p1['delta_max']:.4f}]")
            print(f"    {p1['interpretation_marker']}")

        print(f"\n  Sonde 2 — Corrélation rang-amplitude :")
        if "spearman_rho_temporal_vs_amplitude" in p2:
            print(f"    Spearman ρ = {p2['spearman_rho_temporal_vs_amplitude']:+.4f}")
            print(f"    {p2['interpretation_marker']}")

        print(f"\n  Sonde 3 — Distribution log-amplitude :")
        if "decades_spanned" in p3:
            print(f"    n_amplitudes positives = {p3['n']}")
            print(f"    décades couvertes      = {p3['decades_spanned']:.2f}")
            print(f"    n_bins vides           = "
                  f"{p3['n_empty_bins']}/{p3['n_bins']}")
            if p3.get('max_min_count_ratio') is not None:
                print(f"    ratio max/min count    = "
                      f"{p3['max_min_count_ratio']:.2f}")
            print(f"    {p3['interpretation_marker']}")

        print(f"\n  Sonde 4 — Distance aux transitions :")
        if "distance_median" in p4:
            print(f"    distance_median = {p4['distance_median']:.4f}")
            print(f"    distance_range  = [{p4['distance_min']:.4f}, "
                  f"{p4['distance_max']:.4f}]")
            print(f"    types les plus proches : "
                  f"{p4['nearest_transition_type_counts']}")
            print(f"    {p4['interpretation_marker']}")

        results["by_t_sim"][f"t_sim_{int(t_sim)}"] = {
            "t_sim": t_sim,
            "n_snapshots": n_snap,
            "n_events": len(events),
            "n_transitions": len(transitions),
            "probe_1_intervals": p1,
            "probe_2_rank_amplitude": p2,
            "probe_3_log_amplitude": p3,
            "probe_4_distance_to_transitions": p4,
        }

    # Sonde 5 : identité événementielle cross-t_sim
    print(f"\n\n{'='*78}")
    print(f"Sonde 5 — Identité événementielle cross-t_sim")
    print(f"  (test garde-fou Alex : structure stable ≠ persistance d'événements)")
    print(f"{'='*78}")
    p5 = probe_5_event_identity_across_t_sim(events_by_t_sim)
    print(f"\n  Critères de matching :")
    print(f"    t_tolerance         = {p5['matching_criteria']['t_tolerance']}")
    print(f"    amp_relative_tol    = "
          f"{p5['matching_criteria']['amp_relative_tolerance']}")
    print(f"\n  Résultats pairwise :")
    for pair_name, match_data in p5["pairwise_matches"].items():
        print(f"\n    {pair_name} :")
        print(f"      n_events_a    = {match_data['n_events_a']}")
        print(f"      n_events_b    = {match_data['n_events_b']}")
        print(f"      n_matched     = {match_data['n_matched']}")
        print(f"      frac_matched_a = {match_data['fraction_matched_a']:.4f}")
        print(f"      frac_matched_b = {match_data['fraction_matched_b']:.4f}")
    print(f"\n  {p5['interpretation_marker']}")

    results["probe_5_event_identity"] = p5

    # ─── Synthèse cartographique (PAS de verdict) ───
    print(f"\n\n{'='*78}")
    print(f"SYNTHÈSE — Cartographie des 5 sondes")
    print(f"{'='*78}")
    print(f"\n  Aucune sonde n'est privilégiée. Les résultats sont à lire en")
    print(f"  parallèle pour identifier si une organisation interne se dégage")
    print(f"  ou si la stabilité ne se manifeste qu'au niveau agrégé.")

    return results


if __name__ == "__main__":
    summary = run_session_5()
    output_dir = REPO_ROOT / "results" / "phase6d_beta"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_5_internal_structure.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, tuple):
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
