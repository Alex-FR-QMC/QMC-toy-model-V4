"""
Instrumentation trajectorielle 6d-β — couche externe au moteur.

Statut : niveau 2 (morphodynamique global) selon 6d-β-numerics §2.
Mouvement : construction d'une nouvelle observable niveau 2.

Question guide (§6.5) :
"Qu'est-ce qui disparaît encore quand on réduit la trajectoire à
quelques snapshots ou à un état terminal ?"

Réponse instrumentée : la structure temporelle interne de la
divergence trajectorielle D_h(t) entre deux dynamiques différant
par leur condition initiale.

Métriques :
- τ_div : temps caractéristique de divergence (atteinte du max)
- τ_rec : temps caractéristique de recohérence (descente sous ε·max)
- monotonicité : nombre d'épisodes de re-divergence après pic
- plateau : durée de phase quasi-stationnaire

Tensions à conserver (test étape 4 §6.2) :
- Lecture concurrente 1 (redondance) : si D_h(t) monotone partout,
  ces métriques sont redondantes avec AUC + max. À tester.
- Lecture concurrente 2 (fausse résolution) : si max(D_h) mal défini
  (plateau), τ_div/τ_rec deviennent artificiels. Garde-fou intégré :
  détection explicite de plateau, dégradation gracieuse.
- Lecture concurrente 3 (projection MCQ) : "temps caractéristiques"
  ressemble à grille MCQ. Test §5.3 passé : l'observable survit
  au retrait du vocabulaire MCQ comme objet morphodynamique pur.

Conformité §6.6 : pas de modification engine. Hooks d'observabilité
strictement externes. Cette fonction prend en entrée des trajectoires
déjà calculées par le moteur, ne modifie rien dans la dynamique.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TemporalStructureDh:
    """
    Structure temporelle d'une trajectoire de divergence D_h(t).

    Tous les temps sont dans l'unité de temps de la simulation
    (généralement adimensionnée par γ).

    Champs d'auto-diagnostic (max_well_defined, plateau_detected,
    etc.) permettent la dégradation gracieuse face à des courbes
    bruitées ou ambiguës.
    """
    # Temps caractéristiques
    t_max_Dh: float  # temps du maximum de D_h(t)
    max_Dh: float    # valeur du maximum
    tau_div: float   # temps de divergence (= t_max - t_0)
    tau_rec_10pct: Optional[float]  # temps pour descendre à 10% de max
    tau_rec_1pct: Optional[float]   # temps pour descendre à 1% de max

    # Structure de la décroissance après pic
    n_redivergence_episodes: int   # épisodes où D_h(t+dt) > D_h(t) post-pic
    redivergence_amplitude_max: float  # amplitude max des rebonds

    # Détection de plateau
    plateau_detected: bool          # courbe quasi-stationnaire ?
    plateau_duration: Optional[float]  # durée de la phase plateau
    plateau_value: Optional[float]   # valeur stationnaire du plateau

    # Auto-diagnostic — qualité de la définition
    max_well_defined: bool          # un pic clair existe ?
    monotonic_after_peak: bool      # décroissance strictement monotone ?
    diagnostic_notes: list[str]     # notes textuelles d'auto-diagnostic


def compute_temporal_structure(
    times: np.ndarray,
    D_h_trajectory: np.ndarray,
    plateau_relative_variation_threshold: float = 0.05,
    plateau_min_duration_frac: float = 0.1,
) -> TemporalStructureDh:
    """
    Extrait la structure temporelle d'une courbe D_h(t).

    Paramètres
    ----------
    times : array (N,) — temps des snapshots, monotonement croissant
    D_h_trajectory : array (N,) — valeurs D_h(t) aux mêmes temps
    plateau_relative_variation_threshold : float
        Une phase est considérée plateau si la variation relative
        sur la fenêtre est inférieure à ce seuil
    plateau_min_duration_frac : float
        Durée minimale du plateau, en fraction de la durée totale,
        pour être détecté

    Retour
    ------
    TemporalStructureDh avec champs renseignés et auto-diagnostic.

    Notes méthodologiques
    --------------------
    - Si max_Dh est très bas (<10⁻⁹), la courbe est essentiellement
      plate dès le départ → cas dégénéré, métriques peu informatives
    - Si plateau détecté avant le pic, la définition de τ_div est
      ambiguë → flag dans diagnostic_notes
    - L'auto-diagnostic permet la dégradation gracieuse plutôt que
      des valeurs forcées potentiellement trompeuses
    """
    assert len(times) == len(D_h_trajectory), \
        "times et D_h_trajectory doivent avoir la même longueur"
    assert len(times) >= 3, \
        "Au moins 3 points requis pour structure temporelle"

    times = np.asarray(times, dtype=float)
    Dh = np.asarray(D_h_trajectory, dtype=float)
    notes = []

    # === Pic ===
    idx_max = int(np.argmax(Dh))
    t_max = float(times[idx_max])
    Dh_max = float(Dh[idx_max])
    tau_div = t_max - float(times[0])

    # Auto-diagnostic : le pic est-il bien défini ?
    # Critère : max est-il distinct de la médiane ou de la valeur finale ?
    Dh_median = float(np.median(Dh))
    Dh_final = float(Dh[-1])
    Dh_initial = float(Dh[0])

    if Dh_max < 1e-12:
        max_well_defined = False
        notes.append(
            f"Cas dégénéré : max(D_h) = {Dh_max:.2e} essentiellement nul. "
            f"Toutes métriques temporelles sont peu informatives."
        )
    elif Dh_max - Dh_median < 0.1 * Dh_max:
        max_well_defined = False
        notes.append(
            f"Pic mal défini : max/median = {Dh_max / max(Dh_median, 1e-30):.2f} "
            f"trop proche de 1. Courbe quasi-plate, τ_div peu significatif."
        )
    else:
        max_well_defined = True

    # === Recohérence : temps pour descendre à 10% et 1% du max ===
    # Cherche après le pic
    tau_rec_10pct: Optional[float] = None
    tau_rec_1pct: Optional[float] = None

    if max_well_defined and idx_max < len(Dh) - 1:
        target_10pct = 0.1 * Dh_max
        target_1pct = 0.01 * Dh_max

        for i in range(idx_max + 1, len(Dh)):
            if tau_rec_10pct is None and Dh[i] <= target_10pct:
                tau_rec_10pct = float(times[i] - t_max)
            if tau_rec_1pct is None and Dh[i] <= target_1pct:
                tau_rec_1pct = float(times[i] - t_max)
                break

        if tau_rec_10pct is None:
            notes.append(
                "Recohérence à 10% non atteinte dans la fenêtre simulée. "
                "Allonger t_sim ou réinterpréter."
            )
        if tau_rec_1pct is None and tau_rec_10pct is not None:
            notes.append(
                "Recohérence à 1% non atteinte (10% oui). "
                "Décroissance ralentie en queue de trajectoire."
            )

    # === Monotonicité de la décroissance après pic ===
    n_redivergence = 0
    redivergence_amplitudes = []

    if idx_max < len(Dh) - 1:
        post_peak = Dh[idx_max:]
        diffs = np.diff(post_peak)
        # Episodes où la courbe remonte
        for i, d in enumerate(diffs):
            if d > 0:
                n_redivergence += 1
                redivergence_amplitudes.append(float(d))

    monotonic_after_peak = (n_redivergence == 0)
    redivergence_amplitude_max = (
        float(max(redivergence_amplitudes))
        if redivergence_amplitudes else 0.0
    )

    if not monotonic_after_peak:
        notes.append(
            f"Décroissance non monotone : {n_redivergence} épisodes de "
            f"re-divergence après pic, amplitude max = "
            f"{redivergence_amplitude_max:.3e}. Signe possible d'hystérésis "
            f"trajectorielle."
        )

    # === Détection de plateau ===
    # Stratégie : chercher la plus longue fenêtre [i, j] dans la
    # trajectoire post-pic où la variation relative reste sous seuil
    plateau_detected = False
    plateau_duration: Optional[float] = None
    plateau_value: Optional[float] = None

    if len(Dh) >= 5:
        min_window_duration = plateau_min_duration_frac * (times[-1] - times[0])
        # Recherche par fenêtre glissante sur la trajectoire entière
        best_plateau_duration = 0.0
        best_plateau_value: Optional[float] = None
        for i in range(len(Dh) - 2):
            for j in range(i + 2, len(Dh)):
                window = Dh[i:j+1]
                window_mean = window.mean()
                if window_mean < 1e-30:
                    rel_var = 0.0
                else:
                    rel_var = (window.max() - window.min()) / window_mean
                duration = times[j] - times[i]
                if (rel_var < plateau_relative_variation_threshold
                        and duration > best_plateau_duration
                        and duration >= min_window_duration):
                    best_plateau_duration = duration
                    best_plateau_value = float(window_mean)

        if best_plateau_value is not None:
            plateau_detected = True
            plateau_duration = best_plateau_duration
            plateau_value = best_plateau_value
            notes.append(
                f"Plateau détecté : durée {plateau_duration:.2f}, "
                f"valeur ≈ {plateau_value:.3e}. Phase de stabilisation "
                f"intermédiaire ou attracteur quasi-stationnaire."
            )

    return TemporalStructureDh(
        t_max_Dh=t_max,
        max_Dh=Dh_max,
        tau_div=tau_div,
        tau_rec_10pct=tau_rec_10pct,
        tau_rec_1pct=tau_rec_1pct,
        n_redivergence_episodes=n_redivergence,
        redivergence_amplitude_max=redivergence_amplitude_max,
        plateau_detected=plateau_detected,
        plateau_duration=plateau_duration,
        plateau_value=plateau_value,
        max_well_defined=max_well_defined,
        monotonic_after_peak=monotonic_after_peak,
        diagnostic_notes=notes,
    )


def summarize_structure(struct: TemporalStructureDh) -> str:
    """Synthèse textuelle pour rapport."""
    lines = [
        f"  t_max_Dh         : {struct.t_max_Dh:.4f}",
        f"  max_Dh           : {struct.max_Dh:.4e}",
        f"  tau_div          : {struct.tau_div:.4f}",
        f"  tau_rec_10pct    : "
        f"{struct.tau_rec_10pct:.4f}" if struct.tau_rec_10pct is not None else
        "  tau_rec_10pct    : non atteint",
        f"  tau_rec_1pct     : "
        f"{struct.tau_rec_1pct:.4f}" if struct.tau_rec_1pct is not None else
        "  tau_rec_1pct     : non atteint",
        f"  n_redivergence   : {struct.n_redivergence_episodes}",
        f"  monotonic_post   : {struct.monotonic_after_peak}",
        f"  plateau_detected : {struct.plateau_detected}",
    ]
    if struct.plateau_detected:
        lines.append(f"    plateau_duration : {struct.plateau_duration:.4f}")
        lines.append(f"    plateau_value    : {struct.plateau_value:.4e}")
    lines.append(f"  max_well_defined : {struct.max_well_defined}")
    if struct.diagnostic_notes:
        lines.append("  diagnostic_notes :")
        for note in struct.diagnostic_notes:
            lines.append(f"    - {note}")
    return "\n".join(lines)
