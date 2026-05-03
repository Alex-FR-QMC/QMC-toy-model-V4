"""
Phase 6c-A diagnostics.

Provides perspective-specific diagnostics that go beyond Phase 6b's
F1-F6 metrics. The Phase 6c question is: does the perspective-dependent
coupling produce a STRUCTURALLY distinct signal from the label-based 6b?

Diagnostics defined here:

  - h_divergence(state_i, state_j)
        L1 elementwise divergence of the marginal metrics, per axis + total

  - R_metric(state_i, state_j, coupling_form)
        cosine-similarity-based overlap on the perspectival novelty fields,
        in [0, 1]. Loggued ALONGSIDE norm_ratio to detect intensity-only
        perspective vs shape perspective.

  - delta_R = |R_psi - R_metric|
        the structural-perspective signal: independent of F1, detects
        whether the system is operationally label-based (ΔR ≈ 0) or
        operationally perspectival (ΔR > 0).

  - phi_extra_diagnostics(extras_baseline, extras_form)
        amplification_ratio + pattern_dissimilarity + temporal_stability
        triple-constraint to distinguish:
          * numerical amplification of the same pattern
          * real perspectival pattern
          * dynamical artefact (high dissimilarity, low stability)

  - hypotheses H1, H2, H3, H4 with explicit discrimination criteria.

The h_div_min protocol:

  - During warmup: log h_divergence(t)
  - At end of warmup: ε_struct = 0.2 * max(h_div_warmup)
  - If max(h_div_warmup) < some absolute floor (1e-3): warmup INSUFFICIENT
  - During measurement: log h_div_ratio(t) = h_div(t) / max(h_div_warmup)
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from mcq_v4.factorial.state import FactorialState, FactorialEngineConfig
from mcq_v4.factorial.three_module_system import ThreeModuleSystem, CouplingConfig
from mcq_v4.factorial.coupling import (
    _h_field_product, _h_open_field, _compute_novelty_and_form,
)
from mcq_v4.factorial.overlaps import compute_pairwise_R_psi


THRESHOLDS_6C = {
    # H1: F1 differential — 6c must produce ≥ 2× the label-based F1
    'H1_F1_amplification_factor': 2.0,
    # H2: structural ΔR — perspective is detectable if ΔR exceeds this
    'H2_delta_R_min': 0.05,
    # H3: F4 metric divergence evolution — 6c must produce a
    # statistically distinct trajectory (relative deviation)
    'H3_F4_relative_change': 0.10,  # 10% deviation from 6b baseline
    # H4: H4 is supported when H1=H2=H3 are all NOT_SUPPORTED AND
    # h_div was sufficient (not just because warmup was inadequate)
    'H4_compression_kills_perspective': 'derived',  # see evaluator
    # ε_struct: relative threshold on h_divergence
    'epsilon_struct_fraction': 0.2,
    # Absolute floor below which warmup is INSUFFICIENT (cannot calibrate)
    'h_div_absolute_floor': 1e-3,
    # Temporal stability — corr threshold above which pattern is stable
    'temporal_stability_min': 0.5,
    # Pattern dissimilarity threshold for "real perspective" classification
    'pattern_dissimilarity_min': 0.1,
    # norm_ratio thresholds for "intensity perspective"
    'norm_ratio_significant': 0.2,  # |1 - norm_ratio| > this = significant
}


# ============================================================================
# h_divergence
# ============================================================================

def compute_h_divergence_pair(state_i: FactorialState,
                              state_j: FactorialState) -> dict:
    """
    Per-axis L1 divergence of marginal metrics + total.

    L1 (not L2) because L2 is invariant under permutation, which would
    miss the case where h_i and h_j have the same shape sedimented at
    different positions. L1 is sensitive to position.
    """
    div_T = float(np.sum(np.abs(state_i.h_T - state_j.h_T)))
    div_M = float(np.sum(np.abs(state_i.h_M - state_j.h_M)))
    div_I = float(np.sum(np.abs(state_i.h_I - state_j.h_I)))
    return {
        'T': div_T, 'M': div_M, 'I': div_I,
        'total': div_T + div_M + div_I,
    }


def compute_pairwise_h_divergence(states: dict[str, FactorialState]) -> dict:
    """All three pairs (A,B), (B,C), (A,C)."""
    return {
        ('A', 'B'): compute_h_divergence_pair(states['A'], states['B']),
        ('B', 'C'): compute_h_divergence_pair(states['B'], states['C']),
        ('A', 'C'): compute_h_divergence_pair(states['A'], states['C']),
    }


# ============================================================================
# R_metric — overlap on perspectival novelty fields
# ============================================================================

def _compute_field_for_metric(
    state: FactorialState,
    coupling_form: str,
    cfg_engine: FactorialEngineConfig,
    prev_h_field_window: Optional[list] = None,
) -> np.ndarray:
    """
    Compute the perspectival field that would be used by `coupling_form`.
    For label-based forms, returns ψ. For perspectival forms, returns
    the novelty/form field as defined in coupling.py.
    """
    if coupling_form in ('positive', 'contrastive'):
        return state.psi.copy()

    if coupling_form.startswith('perspectival_'):
        from mcq_v4.factorial.three_module_system import CouplingConfig
        cfg = CouplingConfig(coupling_form=coupling_form)
        return _compute_novelty_and_form(
            state, cfg,
            prev_h_field=prev_h_field_window,
            h_min=cfg_engine.h_min,
            h_0=cfg_engine.h_0,
            dt=cfg_engine.dt,
        )

    raise ValueError(f"Unknown coupling_form: {coupling_form!r}")


def compute_R_metric_pair(
    state_i: FactorialState,
    state_j: FactorialState,
    coupling_form: str,
    cfg_engine: FactorialEngineConfig,
    prev_h_field_i: Optional[list] = None,
    prev_h_field_j: Optional[list] = None,
) -> dict:
    """
    Compute R_metric and norm_ratio between two states for a given
    coupling form.

      R_metric    = 0.5 * (1 + cos_sim(field_i, field_j))   ∈ [0, 1]
      norm_ratio  = ‖field_i‖ / ‖field_j‖                  (raw, not bounded)

    Reading:
      - cos_sim ≈ 1 AND norm_ratio ≈ 1: no perspective (fields aligned and
        same magnitude)
      - cos_sim ≈ 1 AND norm_ratio far from 1: perspective in INTENSITY
      - cos_sim < 1: perspective in SHAPE

    The combination distinguishes "fields differ in magnitude only"
    (intensity perspective, invisible to cos_sim alone) from "fields
    differ in shape" (true pattern perspective).
    """
    f_i = _compute_field_for_metric(state_i, coupling_form, cfg_engine,
                                    prev_h_field_i)
    f_j = _compute_field_for_metric(state_j, coupling_form, cfg_engine,
                                    prev_h_field_j)

    norm_i = float(np.linalg.norm(f_i))
    norm_j = float(np.linalg.norm(f_j))

    if norm_i < 1e-12 or norm_j < 1e-12:
        cos_sim = 0.0
    else:
        cos_sim = float(np.sum(f_i * f_j) / (norm_i * norm_j))
        cos_sim = max(-1.0, min(1.0, cos_sim))

    R_metric = 0.5 * (1.0 + cos_sim)
    norm_ratio = norm_i / max(norm_j, 1e-12)

    return {
        'R_metric': R_metric,
        'cos_sim': cos_sim,
        'norm_ratio': norm_ratio,
        'norm_i': norm_i,
        'norm_j': norm_j,
        'norm_ratio_deviation_from_unity': abs(1.0 - norm_ratio),
    }


def compute_pairwise_R_metric(
    states: dict[str, FactorialState],
    coupling_form: str,
    cfg_engine: FactorialEngineConfig,
    prev_h_fields: Optional[dict] = None,
) -> dict:
    """All three pairs."""
    if prev_h_fields is None:
        prev_h_fields = {}

    return {
        ('A', 'B'): compute_R_metric_pair(
            states['A'], states['B'], coupling_form, cfg_engine,
            prev_h_field_i=prev_h_fields.get('A'),
            prev_h_field_j=prev_h_fields.get('B'),
        ),
        ('B', 'C'): compute_R_metric_pair(
            states['B'], states['C'], coupling_form, cfg_engine,
            prev_h_field_i=prev_h_fields.get('B'),
            prev_h_field_j=prev_h_fields.get('C'),
        ),
        ('A', 'C'): compute_R_metric_pair(
            states['A'], states['C'], coupling_form, cfg_engine,
            prev_h_field_i=prev_h_fields.get('A'),
            prev_h_field_j=prev_h_fields.get('C'),
        ),
    }


# ============================================================================
# ΔR = |R_psi - R_metric|
# ============================================================================

def compute_delta_R(R_psi_pairs: dict, R_metric_pairs: dict) -> dict:
    """
    ΔR per pair: |R_psi - R_metric|.

    Reading:
      - ΔR ≈ 0 → system operationally label-based (no perspective signal)
      - ΔR > threshold → perspective is structurally active

    Independent of F1 (does not depend on dynamic effect).
    """
    delta = {}
    for pair in [('A', 'B'), ('B', 'C'), ('A', 'C')]:
        r_psi = R_psi_pairs.get(pair, 0.0)
        r_met = R_metric_pairs.get(pair, {}).get('R_metric', r_psi)
        delta[pair] = abs(r_psi - r_met)
    return {
        'pairs': delta,
        'mean': float(np.mean(list(delta.values()))),
        'max': float(np.max(list(delta.values()))),
    }


# ============================================================================
# Phi_extra diagnostics — triple constraint
# ============================================================================

def compute_phi_extra_diagnostics(
    extras_baseline: list[dict[str, np.ndarray]],
    extras_form: list[dict[str, np.ndarray]],
) -> dict:
    """
    Compare the extras trajectories of two coupling forms (baseline vs form).

    Inputs are lists of dicts {name: phi_extra_array}, one per step.

    Returns three-dimensional diagnostic per module:

      amplification_ratio: max|phi_form| / max|phi_baseline|
        > 1 = form has stronger amplitude

      pattern_dissimilarity: 1 - mean over time of cos_sim(phi_form, phi_baseline)
        > threshold = form has different spatial pattern

      temporal_stability: mean autocorrelation of phi_form across consecutive steps
        > threshold = pattern is stable (not noise)

    Classification per module:
      - amplification ↑, dissimilarity ≈ 0  : numerical amplification only
      - amplification ≈ 1, dissimilarity ↑, stability ↑ : real perspective
      - dissimilarity ↑, stability ↓ : dynamical artefact
    """
    diagnostics = {}

    for name in ['A', 'B', 'C']:
        baseline_traj = [e[name] for e in extras_baseline if e[name] is not None]
        form_traj = [e[name] for e in extras_form if e[name] is not None]

        # If empty, skip
        if len(baseline_traj) == 0 or len(form_traj) == 0:
            diagnostics[name] = {
                'amplification_ratio': 0.0,
                'pattern_dissimilarity': 0.0,
                'temporal_stability': 0.0,
                'classification': 'NO_DATA',
            }
            continue

        # Amplification
        max_baseline = max((float(np.abs(b).max()) for b in baseline_traj), default=1e-12)
        max_form = max((float(np.abs(f).max()) for f in form_traj), default=0.0)
        amplification = max_form / max(max_baseline, 1e-12)

        # Pattern dissimilarity (mean over time)
        T = min(len(baseline_traj), len(form_traj))
        cos_sims = []
        for t in range(T):
            b = baseline_traj[t].flatten()
            f = form_traj[t].flatten()
            nb = float(np.linalg.norm(b))
            nf = float(np.linalg.norm(f))
            if nb < 1e-12 or nf < 1e-12:
                cos_sims.append(1.0)
            else:
                c = float(np.sum(b * f) / (nb * nf))
                c = max(-1.0, min(1.0, c))
                cos_sims.append(c)
        mean_cos = float(np.mean(cos_sims)) if cos_sims else 1.0
        pattern_dissimilarity = 1.0 - mean_cos

        # Temporal stability — autocorrelation of form's pattern at lag 1
        if len(form_traj) >= 2:
            stabs = []
            for t in range(len(form_traj) - 1):
                a = form_traj[t].flatten()
                b = form_traj[t + 1].flatten()
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na < 1e-12 or nb < 1e-12:
                    continue
                s = float(np.sum(a * b) / (na * nb))
                s = max(-1.0, min(1.0, s))
                stabs.append(s)
            temporal_stability = float(np.mean(stabs)) if stabs else 0.0
        else:
            temporal_stability = 0.0

        # Classification
        amp_high = amplification > 1.5  # significantly amplified
        dis_high = pattern_dissimilarity > THRESHOLDS_6C['pattern_dissimilarity_min']
        stab_high = temporal_stability > THRESHOLDS_6C['temporal_stability_min']

        if amp_high and not dis_high:
            classification = 'NUMERICAL_AMPLIFICATION_ONLY'
        elif dis_high and stab_high:
            classification = 'REAL_PERSPECTIVE'
        elif dis_high and not stab_high:
            classification = 'DYNAMICAL_ARTEFACT'
        else:
            classification = 'INDISTINGUISHABLE_FROM_BASELINE'

        diagnostics[name] = {
            'amplification_ratio': amplification,
            'pattern_dissimilarity': pattern_dissimilarity,
            'temporal_stability': temporal_stability,
            'classification': classification,
        }

    # Aggregate classification: REAL_PERSPECTIVE on at least 2 of 3 modules
    real_count = sum(1 for d in diagnostics.values()
                     if d['classification'] == 'REAL_PERSPECTIVE')
    aggregate = 'REAL_PERSPECTIVE' if real_count >= 2 else 'NOT_PERSPECTIVAL_ENOUGH'

    return {
        'per_module': diagnostics,
        'aggregate_classification': aggregate,
        'real_perspective_count': real_count,
    }


# ============================================================================
# Hypotheses H1-H4 evaluation
# ============================================================================

def evaluate_hypothesis_H1(
    f1_baseline: dict,
    f1_form: dict,
) -> dict:
    """
    H1: F1 trajectorial differential.
    Form is supported if rel_diff_pure(form) >= H1_factor * rel_diff_pure(baseline).
    """
    rdp_b = f1_baseline.get('rel_diff_pure', 0.0)
    rdp_f = f1_form.get('rel_diff_pure', 0.0)
    factor = THRESHOLDS_6C['H1_F1_amplification_factor']

    if rdp_b < 1e-9:
        # Baseline F1 is zero — any non-zero F1 in form supports H1 if measurable
        if rdp_f > 1e-5:
            verdict = 'SUPPORTED_FORM_HAS_INTERFERENCE_BASELINE_DOES_NOT'
        else:
            verdict = 'INCONCLUSIVE_BOTH_ZERO'
    else:
        ratio = rdp_f / rdp_b
        if ratio >= factor:
            verdict = 'SUPPORTED'
        elif ratio >= 1.0:
            verdict = 'WEAKLY_SUPPORTED'
        else:
            verdict = 'NOT_SUPPORTED'

    return {
        'hypothesis': 'H1_F1_differential',
        'criterion': f"rel_diff_pure(form) >= {factor} * rel_diff_pure(baseline)",
        'rel_diff_pure_baseline': rdp_b,
        'rel_diff_pure_form': rdp_f,
        'ratio': rdp_f / max(rdp_b, 1e-12),
        'verdict': verdict,
    }


def evaluate_hypothesis_H2(
    delta_R_history: list[dict],
) -> dict:
    """
    H2: Structural perspective via ΔR.
    Supported if mean ΔR over the trajectory exceeds threshold.
    """
    if len(delta_R_history) == 0:
        return {'hypothesis': 'H2_structural_perspective', 'verdict': 'NO_DATA'}

    means = [d['mean'] for d in delta_R_history]
    mean_delta_R = float(np.mean(means))
    max_delta_R = float(np.max(means))
    threshold = THRESHOLDS_6C['H2_delta_R_min']

    if mean_delta_R >= threshold:
        verdict = 'SUPPORTED'
    elif max_delta_R >= threshold:
        verdict = 'WEAKLY_SUPPORTED'
    else:
        verdict = 'NOT_SUPPORTED'

    return {
        'hypothesis': 'H2_structural_perspective',
        'criterion': f"mean ΔR(t) >= {threshold}",
        'mean_delta_R': mean_delta_R,
        'max_delta_R': max_delta_R,
        'verdict': verdict,
    }


def evaluate_hypothesis_H3(
    f4_baseline: dict,
    f4_form: dict,
) -> dict:
    """
    H3: F4 metric divergence evolution differs between baseline and form.
    """
    div_b = f4_baseline.get('metric_divergence_mean', 0.0)
    div_f = f4_form.get('metric_divergence_mean', 0.0)

    if div_b < 1e-12:
        return {'hypothesis': 'H3_F4_evolution',
                'verdict': 'INCONCLUSIVE_NO_BASELINE'}

    rel_change = abs(div_f - div_b) / div_b
    threshold = THRESHOLDS_6C['H3_F4_relative_change']

    verdict = 'SUPPORTED' if rel_change >= threshold else 'NOT_SUPPORTED'

    return {
        'hypothesis': 'H3_F4_evolution',
        'criterion': f"|F4_form - F4_baseline| / F4_baseline >= {threshold}",
        'F4_baseline': div_b,
        'F4_form': div_f,
        'relative_change': rel_change,
        'verdict': verdict,
    }


def evaluate_hypothesis_H4(
    h1_result: dict, h2_result: dict, h3_result: dict,
    h_div_max_warmup: float,
) -> dict:
    """
    H4: Compression T/M/I kills perspective.
    Supported if H1, H2, H3 are all NOT_SUPPORTED AND warmup produced
    sufficient h_divergence (so the absence of perspective is not just
    due to inadequate warmup).
    """
    floor = THRESHOLDS_6C['h_div_absolute_floor']
    warmup_ok = h_div_max_warmup >= floor

    h1_neg = h1_result.get('verdict') in ('NOT_SUPPORTED', 'INCONCLUSIVE_BOTH_ZERO')
    h2_neg = h2_result.get('verdict') == 'NOT_SUPPORTED'
    h3_neg = h3_result.get('verdict') == 'NOT_SUPPORTED'

    all_negative = h1_neg and h2_neg and h3_neg

    if not warmup_ok:
        verdict = 'INCONCLUSIVE_INSUFFICIENT_WARMUP'
    elif all_negative:
        verdict = 'SUPPORTED_COMPRESSION_KILLS_PERSPECTIVE'
    else:
        verdict = 'NOT_SUPPORTED_PERSPECTIVE_HAS_SIGNAL'

    return {
        'hypothesis': 'H4_compression_kills_perspective',
        'criterion': "All of H1/H2/H3 NOT_SUPPORTED AND warmup OK",
        'warmup_ok': warmup_ok,
        'h_div_max_warmup': h_div_max_warmup,
        'h_div_floor': floor,
        'H1_negative': h1_neg, 'H2_negative': h2_neg, 'H3_negative': h3_neg,
        'verdict': verdict,
    }


def synthesize_3_levels(
    h1: dict, h2: dict, h3: dict, h4: dict,
    extras_diag: dict,
) -> dict:
    """
    Three-level interpretation:
      Level 1: NO_PERSPECTIVE  — H2 NOT_SUPPORTED (ΔR ≈ 0)
      Level 2: STRUCTURAL_PERSPECTIVE — H2 SUPPORTED (ΔR > 0) AND
                  extras_diag aggregate is REAL_PERSPECTIVE
      Level 3: DYNAMICAL_EFFECT — Level 2 AND (H1 SUPPORTED OR H3 SUPPORTED)
    """
    h2_pos = h2.get('verdict') in ('SUPPORTED', 'WEAKLY_SUPPORTED')
    extras_real = extras_diag.get('aggregate_classification') == 'REAL_PERSPECTIVE'
    h1_pos = h1.get('verdict') in ('SUPPORTED', 'WEAKLY_SUPPORTED',
                                    'SUPPORTED_FORM_HAS_INTERFERENCE_BASELINE_DOES_NOT')
    h3_pos = h3.get('verdict') == 'SUPPORTED'

    if not h2_pos:
        level = 'LEVEL_1_NO_PERSPECTIVE'
    elif h2_pos and extras_real and (h1_pos or h3_pos):
        level = 'LEVEL_3_DYNAMICAL_EFFECT'
    elif h2_pos and extras_real:
        level = 'LEVEL_2_STRUCTURAL_PERSPECTIVE_NO_DYNAMIC_EFFECT'
    elif h2_pos and not extras_real:
        level = 'LEVEL_2_PARTIAL_STRUCTURAL_SIGNAL_ONLY'
    else:
        level = 'INCONCLUSIVE'

    return {
        'level': level,
        'H1_supports_dynamic_effect': h1_pos,
        'H2_supports_structural_perspective': h2_pos,
        'H3_supports_F4_evolution': h3_pos,
        'extras_aggregate_real_perspective': extras_real,
        'H4_compression_kills_perspective_verdict': h4.get('verdict'),
    }
