"""
Instrumentation session 2 6d-β — cartographie de granularité des
instabilités trajectorielles.

Discipline (validée Alex) :
- Ne cherche PAS à décider "réel vs artefact"
- Cherche à comprendre dans quelles conditions les re-divergences
  changent de nature
- Le but n'est pas de nettoyer les données jusqu'à obtenir une
  observable propre, mais de comprendre quelles irrégularités
  survivent à l'audit

Conformité §6.6 : couche externe au moteur. Aucune modification de
dynamique. Hooks de logging structuré uniquement.

Mesures session 2 :
(a) Distribution des amplitudes de re-divergences (log-spaced)
(b) Localisation temporelle relative aux transitions morphologiques
    majeures de D_h(t) (entrée/sortie de plateau, cassures de pente)
(d) Comparaison qualitative inter-régimes pour 3 cas × 3 dt

(c) auto-corrélation reportée — risque inflation analytique.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RedivergenceEvent:
    """Un événement de re-divergence détecté dans D_h(t)."""
    t: float           # temps de l'événement
    t_index: int       # index dans la trajectoire snapshot
    amplitude: float   # delta D_h positif (D_h(t+dt) - D_h(t))
    Dh_before: float   # valeur de D_h juste avant
    Dh_after: float    # valeur de D_h juste après
    fraction_of_max: float  # amplitude / max(D_h)


@dataclass
class MorphologicalTransition:
    """Transition morphologique majeure détectée sur D_h(t)."""
    t: float
    t_index: int
    transition_type: str  # 'peak', 'plateau_entry', 'plateau_exit', 'slope_change'
    severity: float       # mesure quantitative de la transition


@dataclass
class RedivergenceMap:
    """
    Cartographie complète des re-divergences pour une trajectoire
    D_h(t) donnée. Pas de jugement réel/artefact ici — juste
    cartographie structurée.
    """
    n_events: int
    events: list[RedivergenceEvent]

    # Distribution
    amplitudes: list[float]
    amplitude_min: float
    amplitude_max: float
    amplitude_median: float

    # Histogramme log-spaced
    histogram_log_bins: list[tuple[float, float, int]]  # (bin_min, bin_max, count)

    # Localisation temporelle
    transitions: list[MorphologicalTransition]
    events_near_transitions: list[tuple[int, int, float]]
    # (event_index, transition_index, time_distance)

    # Synthèse qualitative
    temporal_distribution: str  # 'concentrated', 'distributed', 'clustered', 'sparse'
    distance_to_nearest_transition_median: Optional[float]


def detect_redivergence_events(
    times: np.ndarray,
    Dh: np.ndarray,
    min_amplitude_filter: float = 0.0,  # 0 = pas de filtre, on garde tout
) -> list[RedivergenceEvent]:
    """
    Détecte tous les épisodes de re-divergence (Dh[i+1] > Dh[i]) après
    le pic principal.

    Pas de filtrage en amplitude par défaut : on garde tout pour
    cartographier la distribution complète.
    """
    idx_max = int(np.argmax(Dh))
    Dh_max = float(Dh[idx_max])
    events = []
    if idx_max >= len(Dh) - 1:
        return events

    for i in range(idx_max, len(Dh) - 1):
        if Dh[i + 1] > Dh[i]:
            amp = float(Dh[i + 1] - Dh[i])
            if amp >= min_amplitude_filter:
                events.append(RedivergenceEvent(
                    t=float(times[i + 1]),
                    t_index=i + 1,
                    amplitude=amp,
                    Dh_before=float(Dh[i]),
                    Dh_after=float(Dh[i + 1]),
                    fraction_of_max=amp / max(Dh_max, 1e-30),
                ))
    return events


def detect_morphological_transitions(
    times: np.ndarray,
    Dh: np.ndarray,
    plateau_relative_variation: float = 0.05,
    plateau_min_duration_frac: float = 0.05,
) -> list[MorphologicalTransition]:
    """
    Détecte transitions morphologiques majeures :
    - peak (max global)
    - plateau_entry / plateau_exit
    - slope_change (cassure de pente significative)

    Approche minimaliste — pas de filtre statistique sophistiqué.
    """
    transitions = []
    n = len(Dh)
    if n < 5:
        return transitions

    Dh_max = float(np.max(Dh))
    idx_max = int(np.argmax(Dh))

    # Peak
    transitions.append(MorphologicalTransition(
        t=float(times[idx_max]),
        t_index=idx_max,
        transition_type='peak',
        severity=Dh_max,
    ))

    # Plateaux : détection par fenêtres glissantes
    total_duration = float(times[-1] - times[0])
    min_duration = plateau_min_duration_frac * total_duration
    in_plateau = False
    plateau_start = -1
    for i in range(n - 2):
        # Évaluation locale : variation relative sur fenêtre [i, i+window]
        for win in [3, 5, 10]:
            j = min(i + win, n - 1)
            window = Dh[i:j + 1]
            window_mean = float(window.mean())
            if window_mean < 1e-30:
                continue
            rel_var = float((window.max() - window.min()) / window_mean)
            window_duration = float(times[j] - times[i])
            if rel_var < plateau_relative_variation and window_duration >= min_duration:
                if not in_plateau:
                    in_plateau = True
                    plateau_start = i
                break
        else:
            if in_plateau:
                # Sortie de plateau
                in_plateau = False
                if plateau_start >= 0:
                    plateau_end = i
                    transitions.append(MorphologicalTransition(
                        t=float(times[plateau_start]),
                        t_index=plateau_start,
                        transition_type='plateau_entry',
                        severity=float(Dh[plateau_start]),
                    ))
                    transitions.append(MorphologicalTransition(
                        t=float(times[plateau_end]),
                        t_index=plateau_end,
                        transition_type='plateau_exit',
                        severity=float(Dh[plateau_end]),
                    ))

    # Cassures de pente : seconde dérivée discrète
    if n >= 5:
        slopes = np.diff(Dh) / np.maximum(np.diff(times), 1e-30)
        slope_changes = np.diff(slopes)
        # Top 3 cassures les plus marquées
        if len(slope_changes) > 3:
            top_indices = np.argsort(np.abs(slope_changes))[-3:]
            slope_change_median_abs = float(np.median(np.abs(slope_changes)))
            for idx in top_indices:
                # Seuil : cassure > 10× médiane absolue
                if abs(slope_changes[idx]) > 10 * slope_change_median_abs:
                    transitions.append(MorphologicalTransition(
                        t=float(times[idx + 1]),
                        t_index=int(idx + 1),
                        transition_type='slope_change',
                        severity=float(abs(slope_changes[idx])),
                    ))

    # Trier par temps
    transitions.sort(key=lambda x: x.t)
    return transitions


def compute_redivergence_map(
    times: np.ndarray,
    Dh: np.ndarray,
    n_log_bins: int = 8,
) -> RedivergenceMap:
    """Cartographie complète des re-divergences pour une trajectoire."""
    events = detect_redivergence_events(times, Dh, min_amplitude_filter=0.0)
    transitions = detect_morphological_transitions(times, Dh)

    if not events:
        return RedivergenceMap(
            n_events=0,
            events=[],
            amplitudes=[],
            amplitude_min=0.0,
            amplitude_max=0.0,
            amplitude_median=0.0,
            histogram_log_bins=[],
            transitions=transitions,
            events_near_transitions=[],
            temporal_distribution='no_events',
            distance_to_nearest_transition_median=None,
        )

    amplitudes = [e.amplitude for e in events]
    amp_min = float(min(amplitudes))
    amp_max = float(max(amplitudes))
    amp_median = float(np.median(amplitudes))

    # Histogramme log-spaced
    if amp_min > 0 and amp_max > amp_min:
        log_min = np.log10(amp_min)
        log_max = np.log10(amp_max)
        bin_edges = np.logspace(log_min, log_max, n_log_bins + 1)
        hist, _ = np.histogram(amplitudes, bins=bin_edges)
        histogram_log_bins = [
            (float(bin_edges[i]), float(bin_edges[i + 1]), int(hist[i]))
            for i in range(n_log_bins)
        ]
    else:
        histogram_log_bins = [(amp_min, amp_max, len(amplitudes))]

    # Localisation : pour chaque événement, trouver la transition la plus proche
    events_near_transitions = []
    distances = []
    for i_event, event in enumerate(events):
        if transitions:
            nearest_dist = float('inf')
            nearest_idx = -1
            for i_trans, trans in enumerate(transitions):
                d = abs(event.t - trans.t)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = i_trans
            events_near_transitions.append((i_event, nearest_idx, nearest_dist))
            distances.append(nearest_dist)

    # Synthèse qualitative
    if len(events) <= 2:
        temporal_distribution = 'sparse'
    else:
        # Mesurer la concentration : ratio inter-quartile vs durée totale
        event_times = np.array([e.t for e in events])
        q25 = float(np.percentile(event_times, 25))
        q75 = float(np.percentile(event_times, 75))
        total_duration = float(times[-1] - times[0])
        concentration = (q75 - q25) / max(total_duration, 1e-30)
        if concentration < 0.2:
            temporal_distribution = 'concentrated'
        elif concentration < 0.6:
            temporal_distribution = 'clustered'
        else:
            temporal_distribution = 'distributed'

    distance_median = float(np.median(distances)) if distances else None

    return RedivergenceMap(
        n_events=len(events),
        events=events,
        amplitudes=amplitudes,
        amplitude_min=amp_min,
        amplitude_max=amp_max,
        amplitude_median=amp_median,
        histogram_log_bins=histogram_log_bins,
        transitions=transitions,
        events_near_transitions=events_near_transitions,
        temporal_distribution=temporal_distribution,
        distance_to_nearest_transition_median=distance_median,
    )


def summarize_redivergence_map(rmap: RedivergenceMap) -> str:
    """Synthèse textuelle minimale pour rapport."""
    if rmap.n_events == 0:
        return "  Aucun événement de re-divergence détecté."

    lines = [
        f"  n_events                : {rmap.n_events}",
        f"  amplitude_min           : {rmap.amplitude_min:.4e}",
        f"  amplitude_median        : {rmap.amplitude_median:.4e}",
        f"  amplitude_max           : {rmap.amplitude_max:.4e}",
        f"  range (log10 décades)   : "
        f"{np.log10(rmap.amplitude_max / max(rmap.amplitude_min, 1e-300)):.2f}",
        f"  temporal_distribution   : {rmap.temporal_distribution}",
    ]
    if rmap.distance_to_nearest_transition_median is not None:
        lines.append(
            f"  dist_median_to_transition: "
            f"{rmap.distance_to_nearest_transition_median:.4f}"
        )
    lines.append(f"  n_transitions_detected  : {len(rmap.transitions)}")
    # Compte par type de transition
    types_count = {}
    for t in rmap.transitions:
        types_count[t.transition_type] = types_count.get(t.transition_type, 0) + 1
    for ttype, count in types_count.items():
        lines.append(f"    {ttype} : {count}")

    # Histogramme
    if rmap.histogram_log_bins:
        lines.append("  histogramme amplitudes (log-spaced) :")
        for bin_min, bin_max, count in rmap.histogram_log_bins:
            if count > 0:
                lines.append(
                    f"    [{bin_min:.2e}, {bin_max:.2e}] : {count}"
                )
    return "\n".join(lines)
