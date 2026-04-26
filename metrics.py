"""
Phase 6a metrics — F2 (tripartition), F2' (h_M frozen), F3 (polymorphism 𝒟),
MCF intra, and invariants with mass drift classification.

All metrics adopt the philosophy established in the Phase 6a specification:

  - Numerical thresholds are INITIAL_CALIBRATION values, to be revalidated
    against first runs.
  - Each metric reports a `signal_quality` field (CLEAN / NOISY / ABSENT)
    in addition to the outcome verdict. A clean signal below a threshold
    is classified as BORDERLINE / CAPACITY_RESULT, not FAIL.
  - F2' has three possible outcomes: PASS / WEAK_MEDIATION /
    MARGINAL_APPROXIMATION_LIMIT, with explicit caveat that the third
    outcome cannot architecturally distinguish (a) tripartition inert
    from (b) marginal h approximation insufficient.
  - The mass drift classification (ACCEPTABLE / NUMERICAL_WARNING /
    NUMERICAL_INVALID) cascades: NUMERICAL_INVALID runs do not get
    interpreted at the F2 / F2' / F3 level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mcq_v4.factorial.state import (
    FactorialState, FactorialEngineConfig, StepDiagnostic,
    THETA_T, THETA_M, THETA_I,
)
from mcq_v4.factorial.observables import (
    compute_observables, compute_tau_prime_modular,
)


# ============================================================================
# Threshold definitions (INITIAL_CALIBRATION — to revalidate)
# ============================================================================

THRESHOLDS = {
    # Mass drift
    'mass_drift_acceptable': 1e-4,
    'mass_drift_warning': 1e-3,

    # F2 — tripartition
    'F2_var_growth_min': 0.3,        # σ_M and σ_I must grow at least this much
    'F2_var_T_preservation': 0.5,    # σ_T should retain at least this fraction

    # F2' — h_M mediation
    'F2p_pass': 0.05,                # ≥ 5% relative diff → PASS
    'F2p_weak': 0.01,                # 1-5% → WEAK_MEDIATION; <1% → MARGINAL_APPROXIMATION_LIMIT

    # F3 — polymorphism
    'F3_G_min': 0.001,               # response threshold for direction in 𝒟
    'F3_perturb_eps': 0.01,          # perturbation amplitude
    'F3_min_identity_changes': 3,    # number of changes in identity of 𝒟

    # MCF intra
    'MCF_var_relative_diff_max': 0.05,  # 5% relative diff between runs
    'MCF_H_relative_diff_max': 0.05,
}


# ============================================================================
# Mass drift / invariants
# ============================================================================

def metric_invariants(
    history: list[FactorialState],
    diagnostics: list[StepDiagnostic],
    cfg: FactorialEngineConfig,
) -> dict:
    """
    Standard invariants + numerical stability classification.

    Mass drift classification:
      < 1e-4              : ACCEPTABLE
      [1e-4, 1e-3)        : NUMERICAL_WARNING (interpretable but borderline)
      ≥ 1e-3              : NUMERICAL_INVALID (NOT interpretable)
    """
    mass_over_time = np.array([s.psi.sum() for s in history])
    mass_drift_over_time = np.abs(1.0 - mass_over_time)
    mass_drift_final = float(mass_drift_over_time[-1])
    mass_drift_max = float(mass_drift_over_time.max())

    if mass_drift_final < THRESHOLDS['mass_drift_acceptable']:
        classification = 'ACCEPTABLE'
        run_interpretable = True
    elif mass_drift_final < THRESHOLDS['mass_drift_warning']:
        classification = 'NUMERICAL_WARNING'
        run_interpretable = True
    else:
        classification = 'NUMERICAL_INVALID'
        run_interpretable = False

    # Aggregate clip statistics
    total_clip_events = int(sum(d.n_clip_events for d in diagnostics))
    total_clipped_mass = float(sum(d.clipped_mass for d in diagnostics))
    max_clip_overall = float(max((d.max_clip_magnitude for d in diagnostics), default=0.0))

    # h-bound checks
    h_T_in_bounds = all(
        cfg.h_min - 1e-9 <= s.h_T.min() and s.h_T.max() <= cfg.h_0 + 1e-9
        for s in history
    )
    h_M_in_bounds = all(
        cfg.h_min - 1e-9 <= s.h_M.min() and s.h_M.max() <= cfg.h_0 + 1e-9
        for s in history
    )
    h_I_in_bounds = all(
        cfg.h_min - 1e-9 <= s.h_I.min() and s.h_I.max() <= cfg.h_0 + 1e-9
        for s in history
    )

    # Positivity (after clip — should always hold)
    positivity = all(s.psi.min() >= -1e-12 for s in history)

    return {
        'mass_drift_final': mass_drift_final,
        'mass_drift_max': mass_drift_max,
        'mass_drift_classification': classification,
        'run_interpretable': run_interpretable,
        'total_clip_events': total_clip_events,
        'total_clipped_mass': total_clipped_mass,
        'max_clip_magnitude_overall': max_clip_overall,
        'h_T_in_bounds': h_T_in_bounds,
        'h_M_in_bounds': h_M_in_bounds,
        'h_I_in_bounds': h_I_in_bounds,
        'positivity': positivity,
        'thresholds': {
            'acceptable': THRESHOLDS['mass_drift_acceptable'],
            'warning': THRESHOLDS['mass_drift_warning'],
        },
        'thresholds_status': 'INITIAL_CALIBRATION',
    }


# ============================================================================
# F2 — Tripartition opérationnelle
# ============================================================================

def metric_F2(history: list[FactorialState]) -> dict:
    """
    Initial: ψ concentrated on T axis (uniform on T, point on M=2, I=2).

    Prediction (MCQ): the regulation Φ_corr together with 3D diffusion must
    redistribute mass to axes M and I — σ_M and σ_I must grow.

    Outcome:
      PASS               : σ_M(T_final) and σ_I(T_final) grew by ≥ 0.3 each
                           AND σ_T retained ≥ 50% of initial value
      BORDERLINE         : signal clean but below threshold
      FAIL               : signal absent — σ_M, σ_I stay near 0
    """
    var_T = np.array([compute_observables(s.psi)['var_T'] for s in history])
    var_M = np.array([compute_observables(s.psi)['var_M'] for s in history])
    var_I = np.array([compute_observables(s.psi)['var_I'] for s in history])

    var_T_init, var_T_final = float(var_T[0]), float(var_T[-1])
    var_M_init, var_M_final = float(var_M[0]), float(var_M[-1])
    var_I_init, var_I_final = float(var_I[0]), float(var_I[-1])

    growth_M = var_M_final - var_M_init
    growth_I = var_I_final - var_I_init
    growth_T = var_T_final - var_T_init

    # Signal quality assessment: do M and I show *any* fluctuation,
    # or are they identically zero?
    var_M_range = float(var_M.max() - var_M.min())
    var_I_range = float(var_I.max() - var_I.min())
    signal_amplitude = max(var_M_range, var_I_range)

    if signal_amplitude < 1e-6:
        signal_quality = 'ABSENT'
    elif signal_amplitude < 0.05:
        signal_quality = 'NOISY'
    else:
        signal_quality = 'CLEAN'

    # Outcome assignment
    threshold = THRESHOLDS['F2_var_growth_min']
    preserve_T = THRESHOLDS['F2_var_T_preservation']
    T_preserved = (var_T_init == 0) or (var_T_final >= preserve_T * var_T_init)

    if signal_quality == 'ABSENT':
        outcome = 'FAIL'
    elif growth_M >= threshold and growth_I >= threshold and T_preserved:
        outcome = 'PASS'
    elif growth_M > 0 and growth_I > 0:
        outcome = 'BORDERLINE'
    else:
        outcome = 'BORDERLINE'

    return {
        'var_T_init': var_T_init, 'var_T_final': var_T_final,
        'var_M_init': var_M_init, 'var_M_final': var_M_final,
        'var_I_init': var_I_init, 'var_I_final': var_I_final,
        'growth_T': growth_T,
        'growth_M': growth_M,
        'growth_I': growth_I,
        'var_M_range': var_M_range,
        'var_I_range': var_I_range,
        'signal_amplitude': signal_amplitude,
        'signal_quality': signal_quality,
        'T_preserved': bool(T_preserved),
        'outcome': outcome,
        'thresholds': {
            'F2_var_growth_min': threshold,
            'F2_var_T_preservation': preserve_T,
        },
        'thresholds_status': 'INITIAL_CALIBRATION',
        'var_T_trajectory': var_T.tolist(),
        'var_M_trajectory': var_M.tolist(),
        'var_I_trajectory': var_I.tolist(),
    }


# ============================================================================
# F2' — Garde-fou non-cosméticité de M
# ============================================================================

def metric_F2_prime(
    history_free: list[FactorialState],
    history_frozen: list[FactorialState],
    cfg: FactorialEngineConfig,
) -> dict:
    """
    Compare τ' trajectories between h_M free and h_M frozen runs.

    Three outcomes:
      PASS                            : rel_diff ≥ 0.05 (h_M effectively mediates τ')
      WEAK_MEDIATION                  : 0.01 ≤ rel_diff < 0.05 (non-zero, borderline)
      MARGINAL_APPROXIMATION_LIMIT    : rel_diff < 0.01 (indistinguishable)

    Note: MARGINAL_APPROXIMATION_LIMIT is AMBIGUOUS — could indicate
      (a) tripartition inert, OR (b) marginal h approximation insufficient.
    Disambiguation requires Phase 6a-bis with full h_a(θ_T, θ_M, θ_I).
    """
    tau_T_free = np.array([
        compute_tau_prime_modular(s.psi, s.h_T, s.h_I)['tau_T'] for s in history_free
    ])
    tau_T_frozen = np.array([
        compute_tau_prime_modular(s.psi, s.h_T, s.h_I)['tau_T'] for s in history_frozen
    ])
    tau_I_free = np.array([
        compute_tau_prime_modular(s.psi, s.h_T, s.h_I)['tau_I'] for s in history_free
    ])
    tau_I_frozen = np.array([
        compute_tau_prime_modular(s.psi, s.h_T, s.h_I)['tau_I'] for s in history_frozen
    ])

    # Time-integrated absolute differences
    L1_T = float(np.abs(tau_T_free - tau_T_frozen).sum() * cfg.dt)
    L1_I = float(np.abs(tau_I_free - tau_I_frozen).sum() * cfg.dt)

    # Reference amplitude: max range of free run, with safety floor
    amp_T = max(float(tau_T_free.max() - tau_T_free.min()), 1e-12)
    amp_I = max(float(tau_I_free.max() - tau_I_free.min()), 1e-12)

    # Total simulation time
    T_total = cfg.dt * cfg.T_steps

    # Relative differences (L1 / (amplitude · time))
    rel_diff_T = L1_T / max(amp_T * T_total, 1e-12)
    rel_diff_I = L1_I / max(amp_I * T_total, 1e-12)
    rel_diff_max = max(rel_diff_T, rel_diff_I)

    # Signal quality on the difference itself
    diff_T = np.abs(tau_T_free - tau_T_frozen)
    diff_I = np.abs(tau_I_free - tau_I_frozen)
    signal_amp_diff = float(max(diff_T.max(), diff_I.max()))

    if signal_amp_diff < 1e-9:
        signal_quality = 'ABSENT'
    elif signal_amp_diff < 1e-4:
        signal_quality = 'NOISY'
    else:
        signal_quality = 'CLEAN'

    # Outcome
    if rel_diff_max >= THRESHOLDS['F2p_pass']:
        outcome = 'PASS'
    elif rel_diff_max >= THRESHOLDS['F2p_weak']:
        outcome = 'WEAK_MEDIATION'
    else:
        outcome = 'MARGINAL_APPROXIMATION_LIMIT'

    return {
        'tau_T_free': tau_T_free.tolist(),
        'tau_T_frozen': tau_T_frozen.tolist(),
        'tau_I_free': tau_I_free.tolist(),
        'tau_I_frozen': tau_I_frozen.tolist(),
        'L1_diff_tau_T': L1_T,
        'L1_diff_tau_I': L1_I,
        'amplitude_T': amp_T,
        'amplitude_I': amp_I,
        'rel_diff_T': rel_diff_T,
        'rel_diff_I': rel_diff_I,
        'rel_diff_max': rel_diff_max,
        'signal_amp_diff': signal_amp_diff,
        'signal_quality': signal_quality,
        'outcome': outcome,
        'thresholds': {
            'PASS': THRESHOLDS['F2p_pass'],
            'WEAK_MEDIATION': THRESHOLDS['F2p_weak'],
        },
        'thresholds_status': 'INITIAL_CALIBRATION',
        'caveat': (
            "MARGINAL_APPROXIMATION_LIMIT outcome is AMBIGUOUS: could indicate "
            "(a) tripartition is dynamically inert in this engine, OR "
            "(b) the marginal h approximation is too weak to expose the mediation. "
            "Disambiguation requires Phase 6a-bis with full h_a(θ_T, θ_M, θ_I)."
        ),
    }


# ============================================================================
# F3 — Polymorphie 𝒟(t) directionnelle
# ============================================================================

def _perturb_psi(
    psi: np.ndarray, axis: str, sign: int, eps: float,
) -> np.ndarray:
    """
    Shift ε of mass along (axis, sign): from current centre-of-mass cell
    to its neighbour in direction sign on the given axis.
    Returns perturbed psi (renormalised to original mass).
    """
    if axis == 'T':
        marg = psi.sum(axis=(1, 2))
        com_idx = int(np.argmax(marg))
        dst_idx = com_idx + sign
        if dst_idx < 0 or dst_idx > 4:
            return psi.copy()
        psi_p = psi.copy()
        slab_src = psi_p[com_idx, :, :].copy()
        delta = eps * slab_src
        psi_p[com_idx, :, :] -= delta
        psi_p[dst_idx, :, :] += delta
    elif axis == 'M':
        marg = psi.sum(axis=(0, 2))
        com_idx = int(np.argmax(marg))
        dst_idx = com_idx + sign
        if dst_idx < 0 or dst_idx > 4:
            return psi.copy()
        psi_p = psi.copy()
        slab_src = psi_p[:, com_idx, :].copy()
        delta = eps * slab_src
        psi_p[:, com_idx, :] -= delta
        psi_p[:, dst_idx, :] += delta
    else:  # I
        marg = psi.sum(axis=(0, 1))
        com_idx = int(np.argmax(marg))
        dst_idx = com_idx + sign
        if dst_idx < 0 or dst_idx > 4:
            return psi.copy()
        psi_p = psi.copy()
        slab_src = psi_p[:, :, com_idx].copy()
        delta = eps * slab_src
        psi_p[:, :, com_idx] -= delta
        psi_p[:, :, dst_idx] += delta

    return psi_p


def metric_F3_at_instant(
    state: FactorialState,
    eps: float = None,
    G_min: float = None,
) -> dict:
    """
    Compute |𝒟(t)| at one instant by discrete perturbation on 6 directions.

    For each (axis, sign) ∈ {T, M, I} × {+, -}:
      - shift ε of mass in that direction
      - measure ‖τ'(perturbed) - τ'(unperturbed)‖ / ε
      - direction is in 𝒟(t) iff response > G_min
    """
    eps = eps if eps is not None else THRESHOLDS['F3_perturb_eps']
    G_min = G_min if G_min is not None else THRESHOLDS['F3_G_min']

    tau_current = compute_tau_prime_modular(state.psi, state.h_T, state.h_I)
    directions_active = []
    responses = {}

    for axis in ['T', 'M', 'I']:
        for sign in [+1, -1]:
            psi_p = _perturb_psi(state.psi, axis, sign, eps)
            tau_p = compute_tau_prime_modular(psi_p, state.h_T, state.h_I)
            d_tau_T = (tau_p['tau_T'] - tau_current['tau_T']) / eps
            d_tau_I = (tau_p['tau_I'] - tau_current['tau_I']) / eps
            response = float(np.sqrt(d_tau_T ** 2 + d_tau_I ** 2))
            responses[f"{axis}{'+' if sign > 0 else '-'}"] = response
            if response > G_min:
                directions_active.append(f"{axis}{'+' if sign > 0 else '-'}")

    return {
        'cardinality': len(directions_active),
        'directions_active': directions_active,
        'responses': responses,
    }


def metric_F3_temporal(
    history: list[FactorialState],
    sample_every: int = 10,
) -> dict:
    """
    Apply F3 across time, sampled.

    PASS criteria:
      - 0 < min_cardinality (no collapse to 𝒟 = ∅)
      - max_cardinality < 6 (some directions sediment)
      - identity_changes ≥ 3 (𝒟 evolves — productive forgetting active)

    FAIL modes:
      - cardinality == 6 throughout         → no sedimentation directionality
      - cardinality → 0                     → 𝒢 collapse (KNV 1)
      - identities fixed (no changes)       → no productive forgetting
    """
    sampled_indices = list(range(0, len(history), sample_every))
    if sampled_indices[-1] != len(history) - 1:
        sampled_indices.append(len(history) - 1)

    cardinality_over_time = []
    identities_over_time = []
    responses_over_time = []
    for idx in sampled_indices:
        result = metric_F3_at_instant(history[idx])
        cardinality_over_time.append(result['cardinality'])
        identities_over_time.append(tuple(sorted(result['directions_active'])))
        responses_over_time.append(result['responses'])

    cardinality_over_time = np.array(cardinality_over_time)
    min_card = int(cardinality_over_time.min())
    max_card = int(cardinality_over_time.max())

    # Count identity changes (consecutive differences)
    identity_changes = sum(
        1 for a, b in zip(identities_over_time[:-1], identities_over_time[1:]) if a != b
    )

    # Signal quality
    if max_card == 0:
        signal_quality = 'ABSENT'
    elif min_card == max_card and identity_changes == 0:
        signal_quality = 'NOISY'  # constant cardinality and no identity change is suspicious
    else:
        signal_quality = 'CLEAN'

    # Outcome
    has_polymorphism = (min_card > 0)
    has_sedimentation = (max_card < 6)
    has_identity_evolution = (identity_changes >= THRESHOLDS['F3_min_identity_changes'])

    if signal_quality == 'ABSENT':
        outcome = 'FAIL_NO_POLYMORPHISM'
    elif min_card == 0:
        outcome = 'FAIL_COLLAPSE'  # 𝒢 collapse occurred
    elif max_card == 6:
        if identity_changes == 0:
            outcome = 'FAIL_NO_SEDIMENTATION'
        else:
            outcome = 'BORDERLINE'  # all directions active but identities flicker
    elif not has_identity_evolution:
        outcome = 'FAIL_NO_PRODUCTIVE_FORGETTING'
    elif has_polymorphism and has_sedimentation and has_identity_evolution:
        outcome = 'PASS'
    else:
        outcome = 'BORDERLINE'

    return {
        'min_cardinality': min_card,
        'max_cardinality': max_card,
        'cardinality_over_time': cardinality_over_time.tolist(),
        'identity_changes': identity_changes,
        'identities_over_time': [list(idn) for idn in identities_over_time],
        'sample_indices': sampled_indices,
        'has_polymorphism': has_polymorphism,
        'has_sedimentation': has_sedimentation,
        'has_identity_evolution': has_identity_evolution,
        'signal_quality': signal_quality,
        'outcome': outcome,
        'thresholds': {
            'G_min': THRESHOLDS['F3_G_min'],
            'eps_perturb': THRESHOLDS['F3_perturb_eps'],
            'min_identity_changes': THRESHOLDS['F3_min_identity_changes'],
        },
        'thresholds_status': 'INITIAL_CALIBRATION',
    }


# ============================================================================
# MCF intra-modulaire
# ============================================================================

def metric_MCF_intra(
    history_init1: list[FactorialState],
    history_init2: list[FactorialState],
) -> dict:
    """
    Cold-start convergence test: two runs from very different initial ψ
    should converge to similar stationary observables.

    Compares the final values of (var_T, var_M, var_I, H_global) across
    runs, normalised by their average magnitude.

    PASS: all relative differences < 5%.
    """
    obs1 = compute_observables(history_init1[-1].psi)
    obs2 = compute_observables(history_init2[-1].psi)

    def rel_diff(x, y):
        avg = 0.5 * (abs(x) + abs(y))
        if avg < 1e-12:
            return 0.0
        return abs(x - y) / avg

    diffs = {
        'var_T': rel_diff(obs1['var_T'], obs2['var_T']),
        'var_M': rel_diff(obs1['var_M'], obs2['var_M']),
        'var_I': rel_diff(obs1['var_I'], obs2['var_I']),
        'H_global': rel_diff(obs1['H_global'], obs2['H_global']),
    }
    max_rel_diff = max(diffs.values())

    threshold = THRESHOLDS['MCF_var_relative_diff_max']
    if max_rel_diff < threshold:
        outcome = 'PASS'
    elif max_rel_diff < 2 * threshold:
        outcome = 'BORDERLINE'
    else:
        outcome = 'FAIL'

    return {
        'final_obs_run1': {
            'var_T': obs1['var_T'], 'var_M': obs1['var_M'], 'var_I': obs1['var_I'],
            'H_global': obs1['H_global'],
        },
        'final_obs_run2': {
            'var_T': obs2['var_T'], 'var_M': obs2['var_M'], 'var_I': obs2['var_I'],
            'H_global': obs2['H_global'],
        },
        'relative_diffs': diffs,
        'max_relative_diff': max_rel_diff,
        'outcome': outcome,
        'threshold': threshold,
        'thresholds_status': 'INITIAL_CALIBRATION',
    }


# ============================================================================
# Cross-mode mediation synthesis (F2' across FULL / ALPHA_ONLY / BETA_ONLY)
# ============================================================================

def synthesize_mediation(results_per_mode: dict) -> dict:
    """
    Cross-mode analysis of F2' outcomes to localise the source of M-mediation.

    Possible verdicts:
      M_MEDIATES_VIA_BOTH         : both ALPHA_ONLY and BETA_ONLY pass
      M_MEDIATES_VIA_ALPHA        : ALPHA_ONLY pass, BETA_ONLY does not
      M_MEDIATES_VIA_BETA         : BETA_ONLY pass, ALPHA_ONLY does not
      M_MEDIATES_VIA_JOINT        : FULL pass, but neither alpha nor beta alone
      M_INERT_OR_APPROX_LIMIT     : FULL also fails (ambiguous: architecture or approximation)
      INCONSISTENT_INVESTIGATE    : alpha or beta passes but FULL doesn't (anomalous)
      PARTIAL_INVALID             : at least one mode is NUMERICAL_INVALID
    """
    f2p_outcomes = {}
    for mode, r in results_per_mode.items():
        if r.get('status') == 'INTERPRETABLE':
            f2p_outcomes[mode] = r['F2_prime']['outcome']
        else:
            f2p_outcomes[mode] = 'NUMERICAL_INVALID'

    if any(o == 'NUMERICAL_INVALID' for o in f2p_outcomes.values()):
        return {
            'verdict': 'PARTIAL_INVALID',
            'outcomes_per_mode': f2p_outcomes,
            'caveat': "At least one mode produced a numerically invalid run.",
        }

    full = f2p_outcomes.get('FULL', '')
    alpha = f2p_outcomes.get('ALPHA_ONLY', '')
    beta = f2p_outcomes.get('BETA_ONLY', '')

    full_ok = (full == 'PASS')
    alpha_ok = (alpha == 'PASS')
    beta_ok = (beta == 'PASS')

    if full_ok:
        if alpha_ok and beta_ok:
            verdict = 'M_MEDIATES_VIA_BOTH'
        elif alpha_ok and not beta_ok:
            verdict = 'M_MEDIATES_VIA_ALPHA'
        elif beta_ok and not alpha_ok:
            verdict = 'M_MEDIATES_VIA_BETA'
        else:
            verdict = 'M_MEDIATES_VIA_JOINT'
    else:
        # FULL not PASS
        if alpha_ok or beta_ok:
            verdict = 'INCONSISTENT_INVESTIGATE'
        else:
            verdict = 'M_INERT_OR_APPROX_LIMIT'

    return {
        'verdict': verdict,
        'outcomes_per_mode': f2p_outcomes,
        'interpretation': {
            'M_MEDIATES_VIA_BOTH': "M-mediation is robust: both 3D diffusion and Φ_corr regulation independently mediate h_M's effect on τ'.",
            'M_MEDIATES_VIA_ALPHA': "M-mediation is via 3D diffusion only. Φ_corr does not contribute to the h_M effect.",
            'M_MEDIATES_VIA_BETA': "M-mediation is via Φ_corr regulation only. The diffusion structure does not transmit h_M to τ'.",
            'M_MEDIATES_VIA_JOINT': "M-mediation requires both α and β jointly. Neither alone is sufficient to expose the effect.",
            'M_INERT_OR_APPROX_LIMIT': (
                "Either the tripartition is dynamically inert in this engine "
                "(architectural failure of the factorial structure), OR the "
                "marginal h approximation is insufficient to expose the mediation "
                "(numerical limitation, calls for Phase 6a-bis with full h_a)."
            ),
            'INCONSISTENT_INVESTIGATE': "Anomalous: a mode passes but FULL does not. Investigate before interpreting.",
        }.get(verdict, "Unknown verdict."),
    }
