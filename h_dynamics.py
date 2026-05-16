"""
6d-α étape 4a : générateurs h-dynamiques.

Conformément à numerics-doc §4 :
    ∂_t h = 𝔊^sed + 𝔊^ero

avec :
    𝔊^sed(ψ, h) = -β · ψ · h    (sédimentation : présence ψ "use" h)
    𝔊^ero(h)    = +γ · h · (1 - h/h₀)   (érosion : restauration logistique vers h₀)

ATTENTION — rupture conceptuelle (numerics-doc §4a-0) :
- Le support devient historique : h accumule une trace de l'histoire de ψ.
- La référence matricielle fixe disparaît.
- Les invariants stationnaires classiques disparaissent probablement.
- Trois couches d'invariants (héritage 3a/3b) restent à distinguer :
  (A) structurels exacts, (B) statistiques, (C) faibles/cassables.

Ce module fournit les opérateurs h-dynamiques ISOLÉS pour les
micro-étapes 4a-α (𝔊^sed seul), 4a-β (𝔊^ero seul), 4a-γ (combinés).
Le couplage ψ-h complet (4a-δ) viendra après validation des micros.

Décision Option 1 conservée (signe drift) — Cohérent ici :
- 𝔊^sed négatif (h décroît sous présence ψ)
- 𝔊^ero positif (h se régénère vers h₀)
"""

from __future__ import annotations
import numpy as np
from .state import State6dMinimal, N_AXIS, DX


def G_sed(psi: np.ndarray, h: np.ndarray, beta: float) -> np.ndarray:
    """
    Générateur de sédimentation : 𝔊^sed = -β · ψ · h.

    À ψ fixe uniforme constante = ψ₀, la solution analytique est :
        h(t) = h_init · exp(-β · ψ₀ · t)

    Pour ψ inhomogène, le taux de décroissance varie spatialement.
    Pas de couplage spatial dans 𝔊^sed lui-même (terme local cellulaire).
    """
    return -beta * psi * h


def G_ero(h: np.ndarray, gamma: float, h0: float) -> np.ndarray:
    """
    Générateur d'érosion (restauration logistique) :
    𝔊^ero = +γ · h · (1 - h/h₀).

    Sans présence ψ et sans 𝔊^sed, solution analytique logistique :
        h(t) = h₀ · h_init / (h_init + (h₀ - h_init)·exp(-γ·t))

    h_init < h₀ → croissance vers h₀ (h₀ = équilibre stable).
    h_init > h₀ → décroissance vers h₀.
    h_init = 0 → reste à 0 (équilibre instable).
    """
    return gamma * h * (1.0 - h / h0)


def step_h_only_explicit(
    state: State6dMinimal,
    psi_fixed: np.ndarray,
    beta: float,
    gamma: float,
    h0_target: float,
    dt: float,
    include_sed: bool = True,
    include_ero: bool = True,
) -> State6dMinimal:
    """
    Step h-only avec ψ FIXE en paramètre (pas variable d'état).

    Pour micro-étapes 4a-α, 4a-β, 4a-γ.

    L'utilisateur peut activer/désactiver 𝔊^sed et 𝔊^ero individuellement
    pour isoler chaque générateur.

    Note CFL : pour 𝔊^sed, le taux est β·ψ, donc dt < 1/(β·ψ_max) pour
    stabilité explicite. Pour 𝔊^ero, le taux max est γ (à h = h₀/2),
    donc dt < 1/γ. CFL combinée : dt < 1/(β·ψ_max + γ).
    """
    dh_dt = np.zeros_like(state.h)

    if include_sed:
        dh_dt += G_sed(psi_fixed, state.h, beta)
    if include_ero:
        dh_dt += G_ero(state.h, gamma, h0_target)

    h_new = state.h + dt * dh_dt

    # Pas de clipping h = max(h, h_min) explicite pour cette micro-étape
    # — on veut MESURER si la positivité h tient ou non, pas la forcer.
    return State6dMinimal(
        psi=state.psi.copy(),  # ψ ne change pas dans cette étape
        h=h_new,
        h0=state.h0,
        h_min=state.h_min,
    )


def simulate_h_only(
    state_init: State6dMinimal,
    psi_fixed: np.ndarray,
    beta: float,
    gamma: float,
    h0_target: float,
    n_steps: int,
    dt: float,
    include_sed: bool = True,
    include_ero: bool = True,
    log_every: int = 1,
) -> tuple[State6dMinimal, dict]:
    """
    Simule l'évolution h-only avec ψ fixe en paramètre.

    Logs trackés :
    - t, n_steps
    - h_min, h_max, h_mean (statistiques globales)
    - h_at_some_cells : h aux cellules de référence pour comparaison analytique
    """
    logs = {
        "step": [], "t": [],
        "h_min": [], "h_max": [], "h_mean": [],
        "h_center": [],  # h(2,2,2)
        "h_corner": [],  # h(0,0,0)
        "h_face": [],    # h(2,2,0) une face
        "psi_min": [], "psi_max": [], "psi_mean": [],  # devraient être constants
        "dt_used": dt,
        "n_steps": n_steps,
        "beta": beta, "gamma": gamma, "h0_target": h0_target,
        "include_sed": include_sed, "include_ero": include_ero,
    }

    def log_state(step: int, t: float, s: State6dMinimal) -> None:
        logs["step"].append(step)
        logs["t"].append(t)
        logs["h_min"].append(float(s.h.min()))
        logs["h_max"].append(float(s.h.max()))
        logs["h_mean"].append(float(s.h.mean()))
        logs["h_center"].append(float(s.h[2, 2, 2]))
        logs["h_corner"].append(float(s.h[0, 0, 0]))
        logs["h_face"].append(float(s.h[2, 2, 0]))
        logs["psi_min"].append(float(s.psi.min()))
        logs["psi_max"].append(float(s.psi.max()))
        logs["psi_mean"].append(float(s.psi.mean()))

    state = state_init
    log_state(0, 0.0, state)

    for step in range(1, n_steps + 1):
        state = step_h_only_explicit(
            state, psi_fixed, beta, gamma, h0_target, dt,
            include_sed=include_sed, include_ero=include_ero,
        )
        if step % log_every == 0:
            log_state(step, step * dt, state)

    return state, logs


# ============================================================
# Solutions analytiques pour références
# ============================================================

def solution_sed_uniform(
    h_init: float, beta: float, psi_uniform: float, t: float
) -> float:
    """
    Pour 𝔊^sed seul avec ψ uniforme constante = ψ₀ :
        h(t) = h_init · exp(-β·ψ₀·t)
    """
    return h_init * np.exp(-beta * psi_uniform * t)


def solution_ero_logistic(
    h_init: float, gamma: float, h0: float, t: float
) -> float:
    """
    Pour 𝔊^ero seul (logistique vers h₀) :
        h(t) = h₀·h_init / (h_init + (h₀ - h_init)·exp(-γ·t))

    Note : si h_init = 0, retourne 0 (équilibre instable).
    """
    if h_init <= 1e-300:
        return 0.0
    return h0 * h_init / (h_init + (h0 - h_init) * np.exp(-gamma * t))


def solution_combined_pointfix(
    h_init: float, beta: float, psi_uniform: float,
    gamma: float, h0: float
) -> float:
    """
    Point fixe pour ψ uniforme constante :
    ∂_t h = -β·ψ·h + γ·h·(1 - h/h₀) = 0
    → h = 0 (trivial) ou h = h₀·(1 - β·ψ/γ) (valide si β·ψ < γ)
    """
    coef = 1.0 - beta * psi_uniform / gamma
    if coef > 0:
        return h0 * coef
    else:
        return 0.0  # collapse vers 0


def solution_combined_trajectory(
    h_init: float, beta: float, psi_uniform: float,
    gamma: float, h0: float, t: float
) -> float:
    """
    Solution analytique exacte de la trajectoire combinée 𝔊^sed + 𝔊^ero
    avec ψ uniforme constante :

        dh/dt = r·h - (γ/h₀)·h²   avec r = γ - β·ψ

    Trois régimes :

    Cas r ≠ 0 (sous-critique ou sur-critique) :
        h(t) = r·h_init·exp(r·t) /
               [r + (γ·h_init/h₀)·(exp(r·t) - 1)]

        Pour r > 0 : h(t) → K = h₀·(1 - β·ψ/γ) > 0
        Pour r < 0 : h(t) → 0 exponentiellement

    Cas r = 0 (critique exact) :
        h(t) = h_init / (1 + (γ·h_init/h₀)·t)
        décroissance polynomiale O(1/t) vers 0

    Cas h_init = 0 : reste 0 (point fixe).
    """
    if h_init <= 0:
        return 0.0

    r = gamma - beta * psi_uniform

    if abs(r) < 1e-15:
        # Cas critique r = 0
        return h_init / (1.0 + (gamma * h_init / h0) * t)

    # Cas général r ≠ 0
    exp_rt = np.exp(r * t)
    numerator = r * h_init * exp_rt
    denominator = r + (gamma * h_init / h0) * (exp_rt - 1.0)
    return numerator / denominator
