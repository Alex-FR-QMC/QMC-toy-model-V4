"""
Phase 6c-B — Structural signatures.

Implements as readers (NOT as operators) the multi-scale interpretive
layer required by the plan: STR/RSR/MI/MV/RR2/T*/opacity. These are
diagnostics applied to existing trajectories — the dynamics is unchanged
relative to 6b/6c-A.

KNV-aligned thresholds for 𝒢 (transformable gradient) and Δ (corridor):

  G_min = 0.001     (below: gradient transformable collapsed)
  delta_min = 0.005 (below: morphodynamic corridor floor — KNV)
  delta_crit = 0.30 (above: dispersion — KNV)

Reading conventions:

  STR (stationary transformable):
      ‖dτ'/dt‖ < threshold AND Var(τ') > threshold AND 𝒢 > G_min
      Distinguishes dynamic stability from dead inertia.

  RSR (reorganisation–stabilisation–resumption):
      damped oscillations of τ' AND Γ″ ≈ 0 (acceleration vanishing)
      Distinguishes restoration from chaotic dispersion.

  MI (maintained integrability):
      Δ stays within corridor over a window (no boundary excursion)

  MV (viable variation):
      Δ exits corridor briefly then returns — reconfiguration event

  RR² (inter-modular regulation):
      Latency of R_psi propagation after isolated perturbation of one
      module to the other modules.

  𝕋* (effective cadence):
      ‖∂_t h‖ per module on a window — scalar reading.

  RR³ (anti-petrification):
      NOT_MEASURABLE_WITHOUT_G_OMEGA — explicit caveat. Without g_Ω
      modulating D_eff, no readable signal.

  Opacity:
      F3a/F3b gap per module (already computed in 6b — re-exposed).

Window default: w = max(20, dominant_period_estimate). The dominant
period is estimated from the autocorrelation of τ' (lag of first
zero crossing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mcq_v4.factorial.state import (
    FactorialState, FactorialEngineConfig, THETA_T, THETA_M, THETA_I,
)
from mcq_v4.factorial.three_module_system import ThreeModuleSystem
from mcq_v4.factorial.observables import compute_observables
from mcq_v4.factorial.tau_prime import compute_modular_contributions


# ============================================================================
# KNV-aligned thresholds
# ============================================================================

KNV_THRESHOLDS_6C_B = {
    'G_min': 0.001,                # transformable gradient floor
    'delta_min': 0.005,             # corridor floor (KNV 5)
    'delta_crit': 0.30,             # corridor ceiling (KNV 6)
    'STR_dtau_max': 0.005,          # stationary if ||dτ'/dt|| below this
    'STR_var_min': 0.0001,          # AND var(τ') above this (alive, not dead)
    'RSR_damping_min': 0.5,         # 50% amplitude decay over window for "damped"
    'RSR_gamma_max': 0.01,          # ||Γ″|| below this for "stabilising"
    'MV_excursion_steps_min': 3,    # min consecutive steps outside corridor for MV event
    'window_default': 20,
    'window_max': 100,
}


# ============================================================================
# Window estimation (period-aware)
# ============================================================================

def estimate_dominant_period(traj: np.ndarray, max_lag: int = 50) -> int:
    """
    Estimate the dominant period of a 1D trajectory via autocorrelation.

    Returns the lag of the first zero-crossing of the autocorrelation
    function (a proxy for half the dominant period), capped between 5
    and `max_lag`. If no clear period detected, returns the default
    window.
    """
    n = len(traj)
    if n < 10:
        return KNV_THRESHOLDS_6C_B['window_default']

    x = traj - np.mean(traj)
    if np.std(x) < 1e-12:
        return KNV_THRESHOLDS_6C_B['window_default']

    # Compute autocorrelation
    acf = np.correlate(x, x, mode='full')[n - 1:]
    acf = acf / acf[0]

    # First zero-crossing
    sign_changes = np.where(np.diff(np.sign(acf)) < 0)[0]
    if len(sign_changes) == 0:
        return KNV_THRESHOLDS_6C_B['window_default']

    half_period = int(sign_changes[0])
    full_period = max(2 * half_period, 5)
    return min(full_period, max_lag)


def adaptive_window(traj: np.ndarray) -> int:
    """w = max(20, dominant period estimate)."""
    est = estimate_dominant_period(traj)
    return max(KNV_THRESHOLDS_6C_B['window_default'], est)


# ============================================================================
# 𝒢 (transformable gradient) and Δ (corridor)
# ============================================================================

def compute_G_modular(state: FactorialState) -> dict:
    """
    Modular 𝒢 PROXY: mean inter-cell ψ-roughness on Θ_m.

    NOT the full transformable gradient 𝒢 = ‖∂τ'/∂Γ_meta‖. This is a
    proxy based on finite differences of ψ across the 3D grid:

        G_proxy = mean over axes of |Δ_axis ψ|

    The proxy correlates with 𝒢 in the regime where ψ-shape changes
    follow τ'-gradient changes (i.e. when the metric is not driving
    independent τ' variations). It is sufficient to detect the
    qualitative collapse condition (G_proxy → 0 implies ψ flattened)
    but does not measure the full structural transformability.

    The full 𝒢 requires Γ_meta as an explicit functional of metric
    history — deferred to Phase 6d/7 when Γ_meta becomes operative.

    All thresholds and verdicts using this function should be labelled
    with `_G_proxy` suffix to avoid claiming MCQ-full compliance.
    """
    psi = state.psi
    G_T_neighbors = np.abs(np.diff(psi, axis=0)).mean()
    G_M_neighbors = np.abs(np.diff(psi, axis=1)).mean()
    G_I_neighbors = np.abs(np.diff(psi, axis=2)).mean()
    G_total = float((G_T_neighbors + G_M_neighbors + G_I_neighbors) / 3.0)
    return {
        'G_proxy_T': float(G_T_neighbors),
        'G_proxy_M': float(G_M_neighbors),
        'G_proxy_I': float(G_I_neighbors),
        'G_proxy_total': G_total,
        'G_total': G_total,  # kept for backward compatibility but renamed in spirit
        'above_floor': bool(G_total > KNV_THRESHOLDS_6C_B['G_min']),
        'note': 'G_proxy: ψ-roughness, not full 𝒢 = ‖∂τ′/∂Γ_meta‖.',
    }


def compute_delta_modular(state: FactorialState) -> dict:
    """
    Modular Δ: std of CENTRED 𝔼[ψ_marg] over the three axes.

    Each axis is centred around its zero-point: T and I are already at
    [-2, +2], M is at [0, 4] so we subtract M_CENTRE_IDX=2.

    For comparison with KNV corridor:
      delta_min < Δ < delta_crit means the system is in the morphodynamic
      corridor (KNV 5 and 6 both respected).
    """
    psi_T_marg = state.psi.sum(axis=(1, 2))
    psi_M_marg = state.psi.sum(axis=(0, 2))
    psi_I_marg = state.psi.sum(axis=(0, 1))

    E_T = float((THETA_T * psi_T_marg).sum())  # centred at 0
    E_M_raw = float((THETA_M * psi_M_marg).sum())
    E_M = E_M_raw - 2.0  # M_CENTRE_IDX maps to 2.0
    E_I = float((THETA_I * psi_I_marg).sum())  # centred at 0

    delta = float(np.std([E_T, E_M, E_I]))
    in_corridor = bool(
        KNV_THRESHOLDS_6C_B['delta_min'] < delta < KNV_THRESHOLDS_6C_B['delta_crit']
    )
    return {
        'delta': delta,
        'E_T': E_T, 'E_M_centred': E_M, 'E_M_raw': E_M_raw, 'E_I': E_I,
        'in_corridor': in_corridor,
        'corridor': [KNV_THRESHOLDS_6C_B['delta_min'],
                     KNV_THRESHOLDS_6C_B['delta_crit']],
    }


def compute_G_systemic(states: dict) -> dict:
    """
    Systemic 𝒢: mean inter-modular contrast on shared T-axis contributions.

    Uses the contrastive readings of τ'_shared. A vanishing systemic 𝒢
    means the three modules' contributions to shared factors have
    collapsed to identical values.
    """
    contribs = {name: compute_modular_contributions(states[name])
                for name in ['A', 'B', 'C']}
    # Inter-modular contrast on T (shared axis)
    T_vals = [contribs[n]['T'] for n in ['A', 'B', 'C']]
    G_systemic = float(np.std(T_vals))
    return {
        'G_systemic': G_systemic,
        'contrib_T_A': contribs['A']['T'],
        'contrib_T_B': contribs['B']['T'],
        'contrib_T_C': contribs['C']['T'],
        'above_floor': bool(G_systemic > KNV_THRESHOLDS_6C_B['G_min']),
    }


# ============================================================================
# Γ″ (acceleration of τ')
# ============================================================================

def compute_gamma_double_prime(tau_prime_history: np.ndarray, dt: float) -> np.ndarray:
    """
    Γ″ = ∂²_t τ' via second finite difference.

    Returns array of shape (n_steps - 2, dim_tau). For RSR detection,
    we want ‖Γ″‖ near zero at the END of an oscillation (system has
    settled).
    """
    if tau_prime_history.shape[0] < 3:
        return np.zeros((0, tau_prime_history.shape[1] if tau_prime_history.ndim > 1 else 1))
    # Second finite difference along time
    gamma = (tau_prime_history[2:] - 2 * tau_prime_history[1:-1] + tau_prime_history[:-2]) / (dt * dt)
    return gamma


# ============================================================================
# STR / RSR readers
# ============================================================================

def read_STR_RSR(
    tau_prime_history: np.ndarray,
    G_modular_history: list,
    dt: float,
    window: Optional[int] = None,
) -> dict:
    """
    Classify dynamic regime per window, MCQ-aligned.

    STR (stationary transformable):
      ‖dτ'/dt‖ < STR_dtau_max
      AND Var(τ'_window) > STR_var_min       (alive, not dead)
      AND 𝒢_modular > G_min                  (gradient still transformable)

    RSR (reorganisation-stabilisation):
      damped oscillation: amplitude decreases by RSR_damping_min over window
      AND ‖Γ″‖ < RSR_gamma_max at end of window

    Otherwise: TRANSITIONING.

    Returns per-window classification.
    """
    if window is None:
        # Estimate dominant period from full trajectory
        if tau_prime_history.ndim == 1:
            window = adaptive_window(tau_prime_history)
        else:
            # Use first component to estimate
            window = adaptive_window(tau_prime_history[:, 0])

    n_steps = tau_prime_history.shape[0]
    if n_steps < window + 3:
        return {'classifications': [],
                'window': window,
                'note': 'trajectory too short'}

    # Compute Γ″ once
    gamma = compute_gamma_double_prime(tau_prime_history, dt)

    classifications = []
    for start in range(0, n_steps - window, window // 2):  # 50% overlap
        end = start + window
        if end > n_steps:
            break

        tau_window = tau_prime_history[start:end]

        # ‖dτ'/dt‖ in window: mean absolute first difference
        if tau_window.ndim == 1:
            dtau = np.abs(np.diff(tau_window)) / max(dt, 1e-12)
        else:
            dtau = np.linalg.norm(np.diff(tau_window, axis=0), axis=1) / max(dt, 1e-12)
        dtau_mean = float(dtau.mean()) if len(dtau) > 0 else 0.0

        # Variance over window
        if tau_window.ndim == 1:
            var_w = float(np.var(tau_window))
        else:
            var_w = float(np.mean(np.var(tau_window, axis=0)))

        # 𝒢 modular at window end (worst-case across modules)
        if start < len(G_modular_history) and end - 1 < len(G_modular_history):
            G_at_end_min = min(
                G_modular_history[end - 1][m]['G_total']
                for m in ['A', 'B', 'C']
            )
        else:
            G_at_end_min = 0.0

        # Γ″ at window end
        if end - 2 - 1 < gamma.shape[0] and end > 2:
            gamma_end_norm = float(np.linalg.norm(gamma[end - 3]))
        else:
            gamma_end_norm = 0.0

        # Damping check: amplitude(2nd half) < (1 - damping_min) × amplitude(1st half)
        half = window // 2
        if tau_window.ndim == 1:
            amp_first = float(np.ptp(tau_window[:half]))
            amp_second = float(np.ptp(tau_window[half:]))
        else:
            amp_first = float(np.mean([np.ptp(tau_window[:half, k])
                                       for k in range(tau_window.shape[1])]))
            amp_second = float(np.mean([np.ptp(tau_window[half:, k])
                                        for k in range(tau_window.shape[1])]))
        damping_ok = (amp_first > 1e-9) and (
            amp_second < (1 - KNV_THRESHOLDS_6C_B['RSR_damping_min']) * amp_first
        )

        # Classification
        is_STR = (
            dtau_mean < KNV_THRESHOLDS_6C_B['STR_dtau_max']
            and var_w > KNV_THRESHOLDS_6C_B['STR_var_min']
            and G_at_end_min > KNV_THRESHOLDS_6C_B['G_min']
        )
        is_RSR = (
            damping_ok
            and gamma_end_norm < KNV_THRESHOLDS_6C_B['RSR_gamma_max']
        )

        if is_STR:
            cls = 'STR'
        elif is_RSR:
            cls = 'RSR'
        else:
            # Distinguish dead inertia (low dτ but G or var collapsed) from active transition
            if dtau_mean < KNV_THRESHOLDS_6C_B['STR_dtau_max']:
                if G_at_end_min <= KNV_THRESHOLDS_6C_B['G_min']:
                    cls = 'DEAD_INERTIA_G_COLLAPSED'
                elif var_w <= KNV_THRESHOLDS_6C_B['STR_var_min']:
                    cls = 'DEAD_INERTIA_VAR_COLLAPSED'
                else:
                    cls = 'TRANSITIONING_QUIET'
            else:
                cls = 'TRANSITIONING_ACTIVE'

        classifications.append({
            'window_start': start,
            'window_end': end,
            'classification': cls,
            'dtau_mean': dtau_mean,
            'var_window': var_w,
            'G_min_modular': G_at_end_min,
            'gamma_norm_end': gamma_end_norm,
            'amp_first': amp_first,
            'amp_second': amp_second,
            'damping_ratio': (amp_second / amp_first) if amp_first > 1e-9 else 1.0,
        })

    # Aggregate
    cls_counts = {}
    for w in classifications:
        cls_counts[w['classification']] = cls_counts.get(w['classification'], 0) + 1

    return {
        'classifications': classifications,
        'window': window,
        'aggregate_counts': cls_counts,
        'dominant_regime': (
            max(cls_counts.keys(), key=lambda k: cls_counts[k])
            if cls_counts else 'NO_DATA'
        ),
    }


# ============================================================================
# MI / MV readers
# ============================================================================

def read_MI_MV(
    delta_history: list,
    min_excursion_steps: Optional[int] = None,
) -> dict:
    """
    MI: integrability maintained = Δ does not violate corridor in a
        pathological direction (delta > delta_crit OR delta dropped
        sharply from previous corridor membership).
    MV: viable variation = Δ briefly exits corridor then returns.

    Refinement (Alex's audit, session 2):
        Low Δ at REST is NOT a KNV violation — a centred quiescent
        system has Δ ≈ 0 and is in STR. Only consider Δ < delta_min
        as a KNV indicator if the system PREVIOUSLY had Δ in corridor
        and dropped (a collapse from corridor).

    delta_history: list of {'delta', 'in_corridor'} per step.
    """
    if min_excursion_steps is None:
        min_excursion_steps = KNV_THRESHOLDS_6C_B['MV_excursion_steps_min']

    if len(delta_history) == 0:
        return {'status': 'NO_DATA'}

    n = len(delta_history)
    deltas = [d['delta'] for d in delta_history]

    # Distinguish rest-low (always low, never had high Δ) from collapse-low
    # (had high Δ early, fell to low Δ).
    max_delta_so_far = 0.0
    state_history = []  # 'low_rest', 'in_corridor', 'high_excursion', 'collapse_low'
    for i, d in enumerate(deltas):
        max_delta_so_far = max(max_delta_so_far, d)
        if d > KNV_THRESHOLDS_6C_B['delta_crit']:
            state_history.append('high_excursion')
        elif d < KNV_THRESHOLDS_6C_B['delta_min']:
            # Distinguish rest from collapse based on history
            if max_delta_so_far < KNV_THRESHOLDS_6C_B['delta_min'] * 1.5:
                state_history.append('low_rest')  # never been higher → at rest
            else:
                state_history.append('collapse_low')  # was higher, now low
        else:
            state_history.append('in_corridor')

    # MI status: fraction of steps in 'in_corridor' OR 'low_rest'
    n_in_corridor = state_history.count('in_corridor')
    n_low_rest = state_history.count('low_rest')
    n_high_excursion = state_history.count('high_excursion')
    n_collapse_low = state_history.count('collapse_low')

    fraction_viable = (n_in_corridor + n_low_rest) / n
    fraction_pathological = (n_high_excursion + n_collapse_low) / n

    # Excursions: contiguous runs of 'high_excursion' or 'collapse_low'
    excursions = []
    i = 0
    while i < n:
        if state_history[i] in ('high_excursion', 'collapse_low'):
            j = i
            kind = state_history[i]
            while j < n and state_history[j] == kind:
                j += 1
            length = j - i
            returned = (j < n and state_history[j] in ('in_corridor', 'low_rest'))
            excursions.append({
                'start': i, 'end': j, 'length': length, 'kind': kind,
                'returned_to_viable': returned,
            })
            i = j
        else:
            i += 1

    mv_events = [e for e in excursions
                 if e['length'] >= min_excursion_steps and e['returned_to_viable']]
    knv_events = [e for e in excursions if not e['returned_to_viable']]

    return {
        'fraction_in_corridor': float(n_in_corridor / n),
        'fraction_low_rest': float(n_low_rest / n),
        'fraction_viable_total': float(fraction_viable),
        'fraction_pathological': float(fraction_pathological),
        'n_high_excursions': n_high_excursion,
        'n_collapse_low_steps': n_collapse_low,
        'n_excursions': len(excursions),
        'n_MV_events': len(mv_events),
        'n_KNV_events': len(knv_events),
        'mv_events': mv_events,
        'knv_events': knv_events,
        'MI_status': (
            'MAINTAINED' if fraction_viable > 0.95
            else 'PARTIAL' if fraction_viable > 0.7
            else 'COMPROMISED'
        ),
        'note': (
            "Δ < delta_min while system has never been higher = LOW_REST "
            "(viable, STR-compatible). Δ < delta_min after Δ was in corridor "
            "= COLLAPSE_LOW (pathological)."
        ),
    }


# ============================================================================
# RR² latency reader
# ============================================================================

def read_RR2_latency(
    R_psi_history_pre: list,
    R_psi_history_post: list,
    dt: float,
    drop_threshold: float = 0.05,
) -> dict:
    """
    RR² as inter-modular regulation latency.

    After perturbation of one module, measure how fast R_psi to other
    modules drops by `drop_threshold`. Latency = time of first drop
    > threshold from pre-perturbation baseline.

    Inputs are histories of {pair: R_value}. Pre is the steady-state
    baseline (last few steps before pert), post is the trajectory after.
    """
    if not R_psi_history_pre or not R_psi_history_post:
        return {'status': 'NO_DATA'}

    # Baseline: mean of pre history per pair
    pairs = list(R_psi_history_pre[0].keys())
    baselines = {}
    for pair in pairs:
        vals = [r.get(pair, 0.0) for r in R_psi_history_pre]
        baselines[pair] = float(np.mean(vals))

    # Find latency per pair
    latencies = {}
    for pair in pairs:
        latency = None
        for t, r in enumerate(R_psi_history_post):
            if r.get(pair, 0.0) < baselines[pair] - drop_threshold:
                latency = t * dt
                break
        latencies[pair] = latency

    return {
        'baselines': baselines,
        'latencies_per_pair': latencies,
        'drop_threshold': drop_threshold,
        'note': "RR² latency: time after perturbation for R_psi to drop. None = no drop observed.",
    }


# ============================================================================
# 𝕋* reader (effective cadence per module via ‖∂_t h‖)
# ============================================================================

def read_T_star(h_history: list, dt: float, window: int = 20) -> dict:
    """
    Effective cadence per module: ‖∂_t h‖_window_mean.

    h_history: list of {name: {'h_T': arr, 'h_M': arr, 'h_I': arr}} per step.
    """
    if len(h_history) < 2:
        return {'status': 'NO_DATA'}

    per_module = {}
    for name in ['A', 'B', 'C']:
        T_norms = []
        M_norms = []
        I_norms = []
        for t in range(1, len(h_history)):
            prev = h_history[t - 1].get(name, {})
            curr = h_history[t].get(name, {})
            if not prev or not curr:
                continue
            dT = np.linalg.norm(curr['h_T'] - prev['h_T']) / dt
            dM = np.linalg.norm(curr['h_M'] - prev['h_M']) / dt
            dI = np.linalg.norm(curr['h_I'] - prev['h_I']) / dt
            T_norms.append(float(dT))
            M_norms.append(float(dM))
            I_norms.append(float(dI))

        if not T_norms:
            per_module[name] = {'status': 'NO_DATA'}
            continue

        # Recent window mean
        recent = slice(-window, None)
        T_recent = T_norms[recent] if len(T_norms) >= window else T_norms
        M_recent = M_norms[recent] if len(M_norms) >= window else M_norms
        I_recent = I_norms[recent] if len(I_norms) >= window else I_norms

        all_recent = T_recent + M_recent + I_recent
        cadence = float(np.mean(all_recent))

        # Classification
        if cadence < 1e-4:
            status = 'LOCKED'
        elif cadence < 1e-2:
            status = 'SLOW'
        else:
            status = 'MOVING'

        per_module[name] = {
            'cadence_recent_mean': cadence,
            'cadence_T_recent': float(np.mean(T_recent)),
            'cadence_M_recent': float(np.mean(M_recent)),
            'cadence_I_recent': float(np.mean(I_recent)),
            'status': status,
        }

    return {
        'per_module': per_module,
        'window': window,
        'note': "T* = ‖∂_t h‖ per module. LOCKED suggests petrification (RR³ would respond if g_Ω present).",
    }


# ============================================================================
# RR³ — explicitly NOT measurable
# ============================================================================

def read_RR3() -> dict:
    """
    RR³ requires g_Ω to modulate D_eff in response to ‖∂_t h‖ → 0
    (anti-petrification). Phase 6c-B keeps the engine intact (no g_Ω),
    so RR³ has no readable trace.
    """
    return {
        'status': 'NOT_MEASURABLE_WITHOUT_G_OMEGA',
        'rationale': (
            "Phase 6c-B keeps the 6a engine unchanged. D_eff is constant "
            "(equal to D_0). g_Ω modulation of D_eff in response to "
            "‖∂_t h‖ → 0 is the architectural mechanism for RR³ "
            "(anti-petrification), but g_Ω is deferred to Phase 6d. "
            "Without it, RR³ produces no observable signal."
        ),
    }


# ============================================================================
# Opacity from F3a/F3b gap
# ============================================================================

def read_opacity_from_F3(f3a_per_module: dict, f3b_per_module: dict) -> dict:
    """
    Opacity per module = gap F3a vs F3b.

    F3a measures whether perturbations are visible to τ' (typically
    INVISIBLE in marginal-h setup).
    F3b measures internal morphodynamic transformability of the same
    perturbations.

    Opacity gap = signal_F3b - signal_F3a. Larger gap = stronger
    structural opacity (transformable internally but invisible externally).
    """
    per_module = {}
    for name in ['A', 'B', 'C']:
        a = f3a_per_module.get(name, {})
        b = f3b_per_module.get(name, {})
        a_signal = a.get('max_cardinality', 0)
        b_signal = b.get('div_var_M', 0.0)
        per_module[name] = {
            'F3a_outcome': a.get('outcome'),
            'F3a_max_cardinality': a_signal,
            'F3b_outcome': b.get('outcome'),
            'F3b_div_var_M': b_signal,
            'opacity_gap': float(b_signal),  # F3a is structurally 0-signal canonical
            'gap_qualitative': (
                'STRONG' if b_signal > 0.02
                else 'WEAK' if b_signal > 0.005
                else 'ABSENT'
            ),
        }
    return {
        'per_module': per_module,
        'note': "Opacity = transformable internally (F3b) but invisible externally (F3a).",
    }
