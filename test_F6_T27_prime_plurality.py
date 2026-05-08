"""
Phase 6c-B — F6.T27' Distributed plurality (LABEL-SPACE TRANSPOSITION).

⚠ CRITICAL CAVEAT (Alex's guardrail #1 + #7):
    This is NOT a native MCQᴺ test. It is a compressed intra-instance
    transposition: 3 modules with HOMOLOGOUS-but-NOT-SHARED axes Θ_A,
    Θ_B, Θ_C aggregating T/M/I. The "global union" of plurality masks
    is computed in homologous INDEX space, not in a shared geometry.

    A native MCQᴺ test would require either:
      - explicit shared Θ across modules
      - inter-modular metric projection
      - 15 factors / 7 modules architecture

Original Phase 5 T27:
    Each instance starts with a different plurality pattern. Run the
    coupled system. Measure: do local pluralities decay (universal
    fusion) or persist while distributing (distributed plurality)?

Transposition for 6c-B (Alex's guardrail #3):
    - Each module gets a different multi-modal initial ψ
    - Measure n_modes_local_i(t) per module (6-connected components,
      Alex's guardrail #4, on a thresholded ψ support)
    - Compute global union over the homologous index space:
      union_mask = mask_A ∪ mask_B ∪ mask_C (per-cell in 5×5×5)
    - Count n_modes_union via 6-connectivity on the union

Classification (Alex's guardrail #3 + #6 + #9):

    DISTRIBUTED_PLURALITY_LABEL_SPACE
        Local pluralities decreased: mean(n_modes_local_final) <
            mean(n_modes_local_initial)
        AND global union plurality persists or grows:
            n_modes_union_final >= 1.5 × max_i(n_modes_local_i_final)
        AND local plurality genuinely decayed (not just preserved):
            mean(n_modes_local_final) < 0.80 × mean(n_modes_local_initial)
        Reading: plurality has been REDISTRIBUTED to the union space —
        no module is "locally plural" but the whole carries plurality.
        NB: this is a label-space distributed plurality, not geometric.

    ALREADY_PLURAL_LOCALLY
        Local pluralities preserved: mean(n_modes_local_final) >=
            0.80 × mean(n_modes_local_initial)
        Reading: T27 does not measure escape from saturation here —
        the plurality was not destroyed locally, so the test does not
        probe the redistribution mechanism.

    UNIVERSAL_FUSION
        Local pluralities collapsed to ~1: max_i(n_modes_local_final) <= 1
        AND n_modes_union_final <= 1
        Reading: all modules converged to a single mode each, and the
        union does not recover plurality. NOTE (Alex's guardrail #5):
        with 5×5×5 resolution, this could be a sub-resolution artefact.
        Initial/final masks are logged.

    OUT_OF_PERTURBATIVE_REGIME
        Engine numerical instability OR mass invariants violated.

    INVARIANT_FAILURE
        Mass not conserved beyond tolerance, or h bounds violated, or
        psi negativity. Distinct from OUT_OF_PERTURBATIVE — the run
        completed but produced inconsistent state.

Connectivity (guardrail #4): scipy.ndimage.label with the 6-connected
structure element [[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],
[[0,0,0],[0,1,0],[0,0,0]]] in 3D. NOT 26-connectivity (which would
fuse diagonally adjacent modes).

Resolution caveat (guardrail #5): 5×5×5 is coarse. A spurious
UNIVERSAL_FUSION may reflect grid coarseness, not dynamics. Initial
and final masks are logged so this can be inspected.

Invariants (guardrail #8): per-step mass, positivity, h bounds are
recorded. T27 conclusions are gated on these holding.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label as ndi_label

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    FactorialEngineConfig, EngineMode, FactorialState, THETA_T,
)
from mcq_v4.factorial.three_module_system import (
    DIFFERENTIATED_WEIGHTS, CouplingConfig, build_three_module_system,
)
from mcq_v4.factorial.coupling import step_three_modules


# ============================================================================
# Initial multi-modal ψ patterns — different per module
# ============================================================================

def make_plural_psi_A() -> np.ndarray:
    """
    Module A initial: TWO main modes on T-axis (θ_T = -2 and θ_T = +2)
    plus a central saddle, centred on M=2, I=0. The "tensional bipolar"
    pattern.

    NOTE on cardinality (Alex's session 7 caveat):
    With the default threshold_factor=0.20 applied to max(ψ)=0.40,
    the central saddle (ψ=0.20) is at exactly the threshold and gets
    counted as a third mode. Result: A has cardinality 3, even though
    the design intent was 2 main poles. Modules A, B, C all end up
    cardinally identical (3/3/3) while being GEOMETRICALLY differentiated
    (different axes hosting the modes). The label-space union DOES
    benefit from the geometric differentiation (initial union = 7, not 3).
    """
    psi = np.zeros((5, 5, 5))
    psi[0, 2, 2] = 0.4   # θ_T = -2
    psi[4, 2, 2] = 0.4   # θ_T = +2
    psi[2, 2, 2] = 0.2   # saddle — counted as mode at threshold 0.20
    psi /= psi.sum()
    return psi


def make_plural_psi_B() -> np.ndarray:
    """
    Module B initial: TWO main modes on M-axis (θ_M = 0, θ_M = 4) plus
    central saddle. Cf. caveat in make_plural_psi_A about cardinality.
    """
    psi = np.zeros((5, 5, 5))
    psi[2, 0, 2] = 0.4
    psi[2, 4, 2] = 0.4
    psi[2, 2, 2] = 0.2
    psi /= psi.sum()
    return psi


def make_plural_psi_C() -> np.ndarray:
    """
    Module C initial: THREE genuine modes (corners on T×I diagonal +
    central). Cardinality 3 by design.
    """
    psi = np.zeros((5, 5, 5))
    psi[0, 2, 0] = 0.30
    psi[4, 2, 4] = 0.30
    psi[2, 2, 2] = 0.40
    psi /= psi.sum()
    return psi


# ============================================================================
# 6-connected mode counting (guardrail #4)
# ============================================================================

# Explicit 6-connectivity structuring element for scipy.ndimage.label
# in 3D: only face-adjacent voxels are connected (not edges/corners).
STRUCT_6_CONN = np.array([
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
], dtype=np.int8)


def support_mask(psi: np.ndarray, threshold_factor: float = 0.20) -> np.ndarray:
    """
    Threshold-based support mask: cells with ψ >= threshold_factor × max(ψ).

    threshold_factor=0.20 means cells holding at least 20% of the peak
    density are considered part of the support. This is a standard
    "mode" definition for plurality counting.
    """
    if psi.max() < 1e-12:
        return np.zeros_like(psi, dtype=bool)
    threshold = threshold_factor * psi.max()
    return psi >= threshold


def count_modes_6conn(psi: np.ndarray, threshold_factor: float = 0.20) -> tuple:
    """Return (n_modes, mask) using 6-connectivity on threshold support."""
    mask = support_mask(psi, threshold_factor)
    if not mask.any():
        return 0, mask
    _, n_modes = ndi_label(mask.astype(np.int8), structure=STRUCT_6_CONN)
    return int(n_modes), mask


def union_modes_6conn(masks: list, threshold_factor: float = 0.20) -> int:
    """Count 6-connected modes in the union of multiple boolean masks."""
    if not masks:
        return 0
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union = union | m
    if not union.any():
        return 0
    _, n = ndi_label(union.astype(np.int8), structure=STRUCT_6_CONN)
    return int(n)


# ============================================================================
# Engine invariants check (guardrail #8)
# ============================================================================

def check_invariants(state: FactorialState,
                     engine_cfg: FactorialEngineConfig,
                     mass_tol: float = 1e-3) -> dict:
    """Mass conservation, ψ positivity, h bounds.

    h_min and h_0 are on the engine config (FactorialEngineConfig),
    not the per-module ModuleConfig that lives on state.cfg.
    """
    psi = state.psi
    mass = float(psi.sum())
    mass_ok = abs(mass - 1.0) < mass_tol
    positivity_ok = bool((psi >= -1e-9).all())
    h_T_ok = bool((state.h_T >= engine_cfg.h_min - 1e-9).all()
                  and (state.h_T <= engine_cfg.h_0 + 1e-9).all())
    h_M_ok = bool((state.h_M >= engine_cfg.h_min - 1e-9).all()
                  and (state.h_M <= engine_cfg.h_0 + 1e-9).all())
    h_I_ok = bool((state.h_I >= engine_cfg.h_min - 1e-9).all()
                  and (state.h_I <= engine_cfg.h_0 + 1e-9).all())
    return {
        'mass': mass,
        'mass_ok': mass_ok,
        'positivity_ok': positivity_ok,
        'h_T_ok': h_T_ok,
        'h_M_ok': h_M_ok,
        'h_I_ok': h_I_ok,
        'all_ok': bool(mass_ok and positivity_ok and h_T_ok and h_M_ok and h_I_ok),
    }


# ============================================================================
# Main run
# ============================================================================

def measure_T27_run(
    cfg_engine: FactorialEngineConfig,
    coupling_form: str,
    eps: float,
    base_seed: int,
    warmup_steps: int = 30,
    main_steps: int = 200,
    threshold_factor: float = 0.20,
    run_no_coupling_baseline: bool = True,
) -> dict:
    """
    Single T27' run for one coupling form, with parallel no-coupling
    baseline (Alex-style control branch).

    The baseline isolates engine-driven fusion (diffusion + Φ_corr in
    each module independently) from coupling-driven fusion. Without
    this, T27 would conflate intrinsic module dynamics with the
    inter-modular signal we actually want to measure.

    NOTE: warmup is SHORT (30 steps) and uses the plural initial ψ
    (no smoothing) so that the multi-modality is preserved. Coupling
    is OFF during warmup, then switched ON for main_steps with the
    targeted form. The baseline keeps coupling OFF throughout main_steps.
    """
    initial_psi = {
        'A': make_plural_psi_A(),
        'B': make_plural_psi_B(),
        'C': make_plural_psi_C(),
    }

    # Initial mode counts
    initial_modes = {}
    initial_masks = {}
    for m, psi in initial_psi.items():
        n, mask = count_modes_6conn(psi, threshold_factor)
        initial_modes[m] = n
        initial_masks[m] = mask
    initial_union_modes = union_modes_6conn(list(initial_masks.values()),
                                             threshold_factor)
    max_initial_local = max(initial_modes.values())

    # Build TWO systems: measurement (with coupling) and baseline (without)
    # Both share base_seed → same warmup trajectory
    coupling_cfg_warmup = CouplingConfig(epsilon=0.0, coupling_form='contrastive')
    sys_obj = build_three_module_system(
        cfg_engine, coupling_cfg_warmup, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )
    if run_no_coupling_baseline:
        sys_baseline = build_three_module_system(
            cfg_engine, coupling_cfg_warmup, DIFFERENTIATED_WEIGHTS,
            initial_psi, base_seed=base_seed,
        )
    else:
        sys_baseline = None

    # Short warmup with coupling OFF for both branches
    for _ in range(warmup_steps):
        sys_obj, _ = step_three_modules(sys_obj, coupling_active=False)
        if sys_baseline is not None:
            sys_baseline, _ = step_three_modules(sys_baseline, coupling_active=False)

    # Inject coupling on the measurement branch only
    sys_obj.coupling_cfg = CouplingConfig(epsilon=eps, coupling_form=coupling_form)

    # Trajectories
    n_modes_local_traj = {m: [] for m in ['A', 'B', 'C']}
    n_modes_union_traj = []
    n_modes_local_baseline_traj = {m: [] for m in ['A', 'B', 'C']}
    n_modes_union_baseline_traj = []
    invariants_per_step = []
    numerical_failure_step = None

    masks_post_warmup = {}
    for m in ['A', 'B', 'C']:
        n, mask = count_modes_6conn(getattr(sys_obj, f'state_{m}').psi,
                                     threshold_factor)
        n_modes_local_traj[m].append(n)
        masks_post_warmup[m] = mask
    n_modes_union_traj.append(
        union_modes_6conn(list(masks_post_warmup.values()), threshold_factor)
    )
    if sys_baseline is not None:
        bm = {}
        for m in ['A', 'B', 'C']:
            n, mask = count_modes_6conn(
                getattr(sys_baseline, f'state_{m}').psi, threshold_factor
            )
            n_modes_local_baseline_traj[m].append(n)
            bm[m] = mask
        n_modes_union_baseline_traj.append(
            union_modes_6conn(list(bm.values()), threshold_factor)
        )

    for step in range(main_steps):
        try:
            sys_obj, _ = step_three_modules(sys_obj, coupling_active=True)
            if sys_baseline is not None:
                sys_baseline, _ = step_three_modules(sys_baseline, coupling_active=False)
        except Exception:
            numerical_failure_step = step
            break

        # Invariants check (measurement branch)
        inv = {m: check_invariants(getattr(sys_obj, f'state_{m}'), cfg_engine)
               for m in ['A', 'B', 'C']}
        invariants_per_step.append(inv)
        if not all(inv[m]['all_ok'] for m in ['A', 'B', 'C']):
            numerical_failure_step = step
            break

        # Mode counts (measurement)
        masks_step = {}
        for m in ['A', 'B', 'C']:
            n, mask = count_modes_6conn(getattr(sys_obj, f'state_{m}').psi,
                                         threshold_factor)
            n_modes_local_traj[m].append(n)
            masks_step[m] = mask
        n_modes_union_traj.append(
            union_modes_6conn(list(masks_step.values()), threshold_factor)
        )

        # Mode counts (baseline, no coupling)
        if sys_baseline is not None:
            bmasks = {}
            for m in ['A', 'B', 'C']:
                n, mask = count_modes_6conn(
                    getattr(sys_baseline, f'state_{m}').psi, threshold_factor
                )
                n_modes_local_baseline_traj[m].append(n)
                bmasks[m] = mask
            n_modes_union_baseline_traj.append(
                union_modes_6conn(list(bmasks.values()), threshold_factor)
            )

    # Final state
    final_modes = {m: n_modes_local_traj[m][-1] if n_modes_local_traj[m] else 0
                   for m in ['A', 'B', 'C']}
    final_masks = {}
    for m in ['A', 'B', 'C']:
        _, fm = count_modes_6conn(getattr(sys_obj, f'state_{m}').psi,
                                   threshold_factor)
        final_masks[m] = fm
    final_union_modes = union_modes_6conn(list(final_masks.values()),
                                           threshold_factor)
    max_final_local = max(final_modes.values())
    mean_initial_local = float(np.mean(list(initial_modes.values())))
    mean_final_local = float(np.mean(list(final_modes.values())))

    # Baseline final state
    if sys_baseline is not None and n_modes_local_baseline_traj['A']:
        baseline_final_modes = {m: n_modes_local_baseline_traj[m][-1]
                                for m in ['A', 'B', 'C']}
        baseline_final_union = (n_modes_union_baseline_traj[-1]
                                if n_modes_union_baseline_traj else 0)
        baseline_max_final_local = max(baseline_final_modes.values())
        baseline_mean_final_local = float(np.mean(list(baseline_final_modes.values())))
    else:
        baseline_final_modes = None
        baseline_final_union = None
        baseline_max_final_local = None
        baseline_mean_final_local = None

    # ─────────────────────────────────────────────────────────────────
    # EARLY-WINDOW DIAGNOSTIC (Alex's spec, session 7)
    # The transition phase is structurally informative:
    # before engine-driven fusion completes, each form may show a
    # different timing of collapse. Even if all paths end at 1/1/1,
    # the trajectory through the transition can differ.
    #
    # Metrics computed on a fixed analysis window AND on event-based
    # observables:
    #   - union_decay_rate over a ramp window
    #   - time_to_local_fusion_X for each module (X ∈ {A, B, C})
    #   - time_to_union_collapse_to_K for K ∈ {3, 1}
    #   - measurement_minus_baseline at fixed event times
    # ─────────────────────────────────────────────────────────────────
    def first_step_at_or_below(traj: list, threshold: int):
        """Return the first index i where traj[i] <= threshold, or None."""
        for i, v in enumerate(traj):
            if v <= threshold:
                return i
        return None

    # time_to_local_fusion_X = first step where module X has 1 mode
    time_to_local_fusion = {}
    for m in ['A', 'B', 'C']:
        time_to_local_fusion[m] = first_step_at_or_below(n_modes_local_traj[m], 1)

    time_to_local_fusion_baseline = {}
    if sys_baseline is not None:
        for m in ['A', 'B', 'C']:
            time_to_local_fusion_baseline[m] = first_step_at_or_below(
                n_modes_local_baseline_traj[m], 1)
    else:
        time_to_local_fusion_baseline = {m: None for m in ['A', 'B', 'C']}

    # time_to_union_collapse_to_K
    time_to_union_collapse = {
        K: first_step_at_or_below(n_modes_union_traj, K) for K in [5, 3, 1]
    }
    time_to_union_collapse_baseline = {
        K: (first_step_at_or_below(n_modes_union_baseline_traj, K)
            if sys_baseline is not None else None)
        for K in [5, 3, 1]
    }

    # union_decay_rate: mean drop in n_modes_union over the early window
    # Window: 5 steps before union first dropped, to 5 steps after collapse
    early_window_start = 0
    early_window_end = min(60, len(n_modes_union_traj))
    if early_window_end > early_window_start + 1:
        union_segment = n_modes_union_traj[early_window_start:early_window_end]
        # Net decay
        union_decay = union_segment[0] - union_segment[-1]
        union_decay_rate = union_decay / max(early_window_end - early_window_start, 1)
    else:
        union_decay = 0.0
        union_decay_rate = 0.0

    if sys_baseline is not None and len(n_modes_union_baseline_traj) > 1:
        b_segment = n_modes_union_baseline_traj[early_window_start:early_window_end]
        union_decay_baseline = b_segment[0] - b_segment[-1] if b_segment else 0.0
        union_decay_rate_baseline = union_decay_baseline / max(
            early_window_end - early_window_start, 1)
    else:
        union_decay = union_decay_rate
        union_decay_baseline = None
        union_decay_rate_baseline = None

    # measurement minus baseline at fixed event times
    # For each step in the early window, compute (measurement_count - baseline_count)
    early_diff_traj = {}
    if sys_baseline is not None:
        for m in ['A', 'B', 'C']:
            n_match = min(len(n_modes_local_traj[m]),
                          len(n_modes_local_baseline_traj[m]))
            early_diff_traj[m] = [
                n_modes_local_traj[m][i] - n_modes_local_baseline_traj[m][i]
                for i in range(min(n_match, early_window_end))
            ]
        n_match_u = min(len(n_modes_union_traj),
                        len(n_modes_union_baseline_traj))
        early_diff_traj['union'] = [
            n_modes_union_traj[i] - n_modes_union_baseline_traj[i]
            for i in range(min(n_match_u, early_window_end))
        ]
    else:
        early_diff_traj = {m: [] for m in ['A', 'B', 'C', 'union']}

    # Has the measurement EVER differed from baseline by >= 1 mode in the window?
    early_window_signal_present = {}
    for k in ['A', 'B', 'C', 'union']:
        diffs = early_diff_traj.get(k, [])
        early_window_signal_present[k] = bool(
            len(diffs) > 0 and any(abs(d) >= 1 for d in diffs)
        )

    # Maximum absolute deviation from baseline within the early window
    early_window_max_abs_dev = {
        k: (max(abs(d) for d in early_diff_traj.get(k, [])) if early_diff_traj.get(k, []) else 0)
        for k in ['A', 'B', 'C', 'union']
    }

    early_window = {
        'window_steps': [early_window_start, early_window_end],
        'time_to_local_fusion': time_to_local_fusion,
        'time_to_local_fusion_baseline': time_to_local_fusion_baseline,
        'time_to_union_collapse': time_to_union_collapse,
        'time_to_union_collapse_baseline': time_to_union_collapse_baseline,
        'union_decay_in_window': union_decay,
        'union_decay_rate_per_step': union_decay_rate,
        'union_decay_baseline': union_decay_baseline,
        'union_decay_rate_baseline': union_decay_rate_baseline,
        'early_diff_traj': early_diff_traj,
        'early_window_signal_present': early_window_signal_present,
        'early_window_max_abs_dev': early_window_max_abs_dev,
    }

    # ─────────────────────────────────────────────────────────────────
    # Diagnostics — atomic flags
    # ─────────────────────────────────────────────────────────────────
    invariants_held = (
        numerical_failure_step is None
        and len(invariants_per_step) > 0
        and all(invariants_per_step[-1][m]['all_ok'] for m in ['A', 'B', 'C'])
    )

    diagnostics = {
        # Local plurality evolution (measurement vs initial)
        'mean_local_decreased': bool(mean_final_local < mean_initial_local),
        'mean_local_decreased_substantially': bool(
            mean_final_local < 0.80 * mean_initial_local
        ),
        'mean_local_preserved_80pct': bool(
            mean_final_local >= 0.80 * mean_initial_local
        ),

        # Universal fusion signal
        'all_local_collapsed_to_1': bool(max_final_local <= 1),
        'union_collapsed_to_1': bool(final_union_modes <= 1),

        # Distributed plurality signal
        'union_dominates_max_local_15x': bool(
            max_final_local >= 1
            and final_union_modes >= 1.5 * max_final_local
        ),

        # Baseline-comparative (engine-driven vs coupling-driven)
        'baseline_also_fused': (
            baseline_max_final_local is not None
            and baseline_max_final_local <= 1
        ),
        'measurement_more_fused_than_baseline': (
            baseline_mean_final_local is not None
            and mean_final_local < baseline_mean_final_local - 0.5
        ),
        'measurement_less_fused_than_baseline': (
            baseline_mean_final_local is not None
            and mean_final_local > baseline_mean_final_local + 0.5
        ),
        'union_differentiates_baseline': (
            baseline_final_union is not None
            and abs(final_union_modes - baseline_final_union) >= 1
        ),

        # Invariant state
        'invariants_held': invariants_held,
        'numerical_failure': bool(numerical_failure_step is not None),
    }

    # ─────────────────────────────────────────────────────────────────
    # Regime classification (non-exclusive)
    # ─────────────────────────────────────────────────────────────────
    regimes = []

    if not diagnostics['invariants_held']:
        regimes.append('INVARIANT_FAILURE')
    if diagnostics['numerical_failure']:
        regimes.append('OUT_OF_PERTURBATIVE_REGIME')

    if (diagnostics['all_local_collapsed_to_1']
            and diagnostics['union_collapsed_to_1']):
        # Distinguish: is fusion attributable to coupling, or already
        # present in the no-coupling baseline?
        if diagnostics['baseline_also_fused']:
            regimes.append('ENGINE_DRIVEN_FUSION_NOT_COUPLING_ATTRIBUTABLE')
        else:
            regimes.append('UNIVERSAL_FUSION')

    if diagnostics['mean_local_preserved_80pct']:
        regimes.append('ALREADY_PLURAL_LOCALLY')

    if (diagnostics['mean_local_decreased_substantially']
            and diagnostics['union_dominates_max_local_15x']):
        regimes.append('DISTRIBUTED_PLURALITY_LABEL_SPACE')

    if (diagnostics['mean_local_decreased']
            and not diagnostics['mean_local_decreased_substantially']
            and not diagnostics['all_local_collapsed_to_1']):
        regimes.append('PARTIAL_LOCAL_REDUCTION_NO_DISTRIBUTED_SIGNAL')

    if diagnostics['measurement_more_fused_than_baseline']:
        regimes.append('COUPLING_ACCELERATES_FUSION')
    if diagnostics['measurement_less_fused_than_baseline']:
        regimes.append('COUPLING_PRESERVES_PLURALITY_VS_BASELINE')

    # Early-window transient signal (Alex's session 7 spec)
    any_early_signal = any(early_window_signal_present.values())
    if any_early_signal:
        regimes.append('EARLY_WINDOW_TRANSIENT_DIFFERENTIATION')

    if not regimes:
        regimes.append('UNCLASSIFIED')

    # ─────────────────────────────────────────────────────────────────
    # Tensions (preserved)
    # ─────────────────────────────────────────────────────────────────
    tensions = []

    # NEW: engine-driven fusion (the most important tension to flag)
    if diagnostics['baseline_also_fused'] and diagnostics['all_local_collapsed_to_1']:
        tensions.append({
            'kind': 'fusion_present_in_baseline_too',
            'description': (
                "The no-coupling baseline ALSO ends in fusion. "
                "T27 conclusions cannot attribute fusion to inter-modular "
                "coupling. The intra-modular dynamics (diffusion + Φ_corr) "
                "destroys plurality on its own at 5×5×5 resolution. The "
                "test does not probe the redistribution mechanism here — "
                "it would require either coarser plurality preservation "
                "(deeper grid) or weaker intra-modular fusion."
            ),
            'baseline_final_modes': baseline_final_modes,
            'measurement_final_modes': final_modes,
        })

    if (diagnostics['mean_local_decreased_substantially']
            and not diagnostics['union_dominates_max_local_15x']
            and not diagnostics['baseline_also_fused']):
        tensions.append({
            'kind': 'local_decay_without_union_pickup',
            'description': (
                "Local plurality decayed substantially but the union "
                "does not recover plurality. Plurality is being LOST "
                "rather than redistributed."
            ),
            'mean_initial': mean_initial_local,
            'mean_final': mean_final_local,
            'union_final': final_union_modes,
            'max_final_local': max_final_local,
        })

    if (diagnostics['all_local_collapsed_to_1']
            and final_union_modes >= 2
            and not diagnostics['baseline_also_fused']):
        tensions.append({
            'kind': 'local_singletons_with_global_plurality',
            'description': (
                "Each module collapsed to a single local mode, but the "
                "union still shows plurality (>=2 modes). Plurality is "
                "purely INTER-modular. Strong label-space distribution "
                "signal, but vulnerable to resolution caveat."
            ),
            'final_local_modes': final_modes,
            'final_union_modes': final_union_modes,
        })

    if (numerical_failure_step is None and len(invariants_per_step) > 0
            and any(not invariants_per_step[-1][m]['mass_ok']
                    for m in ['A', 'B', 'C'])):
        tensions.append({
            'kind': 'invariant_drift_without_engine_failure',
            'description': (
                "Run completed without exception but mass conservation "
                "drifted. T27 conclusions on this run should not be "
                "treated as MCQ-meaningful."
            ),
            'masses_final': {m: invariants_per_step[-1][m]['mass']
                             for m in ['A', 'B', 'C']},
        })

    # Early-window timing differential — preserves info even when
    # final state is identical to baseline
    timing_diffs = {}
    for m in ['A', 'B', 'C']:
        t_meas = time_to_local_fusion.get(m)
        t_base = time_to_local_fusion_baseline.get(m)
        if t_meas is not None and t_base is not None:
            timing_diffs[m] = t_meas - t_base
    union_timing_diff = None
    if (time_to_union_collapse[1] is not None
            and time_to_union_collapse_baseline[1] is not None):
        union_timing_diff = (time_to_union_collapse[1]
                             - time_to_union_collapse_baseline[1])

    any_timing_differs = (
        any(abs(d) >= 1 for d in timing_diffs.values())
        or (union_timing_diff is not None and abs(union_timing_diff) >= 1)
    )

    if any_timing_differs:
        tensions.append({
            'kind': 'fusion_timing_differs_in_early_window',
            'description': (
                "Final state identical to baseline but TIMING of fusion "
                "differs. The coupling does affect the dynamics during "
                "the transition — it just doesn't change where the system "
                "ends up. T27' detects a transient signal even when the "
                "long-run verdict says the test is non-discriminating."
            ),
            'time_to_local_fusion_diff_vs_baseline': timing_diffs,
            'time_to_union_collapse_diff_vs_baseline': union_timing_diff,
        })

    return {
        'coupling_form': coupling_form,
        'epsilon': eps,
        'threshold_factor': threshold_factor,
        'warmup_steps': warmup_steps,
        'main_steps': main_steps,

        # Initial vs final
        'initial_modes': initial_modes,
        'initial_union_modes': initial_union_modes,
        'final_modes': final_modes,
        'final_union_modes': final_union_modes,
        'max_initial_local': max_initial_local,
        'max_final_local': max_final_local,
        'mean_initial_local': mean_initial_local,
        'mean_final_local': mean_final_local,

        # Baseline (no coupling)
        'baseline_final_modes': baseline_final_modes,
        'baseline_final_union_modes': baseline_final_union,
        'baseline_max_final_local': baseline_max_final_local,
        'baseline_mean_final_local': baseline_mean_final_local,

        # Trajectories
        'n_modes_local_traj': n_modes_local_traj,
        'n_modes_union_traj': n_modes_union_traj,
        'n_modes_local_baseline_traj': n_modes_local_baseline_traj,
        'n_modes_union_baseline_traj': n_modes_union_baseline_traj,

        # Mask snapshots
        'masks_initial': {m: initial_masks[m].tolist()
                          for m in ['A', 'B', 'C']},
        'masks_final': {m: final_masks[m].tolist()
                        for m in ['A', 'B', 'C']},

        # Diagnostics + regimes + tensions
        'diagnostics_local': diagnostics,
        'regimes_detected': regimes,
        'tensions_observed': tensions,

        # Early-window diagnostic (transient signal during fusion)
        'early_window': early_window,

        # Invariants summary
        'numerical_failure_step': numerical_failure_step,
        'final_invariants': (invariants_per_step[-1] if invariants_per_step
                             else None),
    }


def run_T27_test(base_seed: int = 42) -> dict:
    print("=" * 70)
    print("F6.T27' — Distributed plurality (LABEL-SPACE TRANSPOSITION)")
    print("⚠ Not native MCQᴺ. Compressed intra-instance, homologous indices.")
    print("=" * 70)

    cfg = FactorialEngineConfig(
        dt=0.05, T_steps=300, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    eps = 0.005
    forms = ['contrastive',
             'perspectival_INV_H',
             'perspectival_H_OPEN',
             'perspectival_MORPHO_ACTIVE']

    per_form = {}
    for form in forms:
        print(f"\n  Form: {form}")
        r = measure_T27_run(cfg, form, eps, base_seed)
        per_form[form] = r
        print(f"    initial_modes:    A={r['initial_modes']['A']} "
              f"B={r['initial_modes']['B']} C={r['initial_modes']['C']}, "
              f"union={r['initial_union_modes']}")
        print(f"    final_modes:      A={r['final_modes']['A']} "
              f"B={r['final_modes']['B']} C={r['final_modes']['C']}, "
              f"union={r['final_union_modes']}")
        if r['baseline_final_modes'] is not None:
            print(f"    baseline_final:   A={r['baseline_final_modes']['A']} "
                  f"B={r['baseline_final_modes']['B']} C={r['baseline_final_modes']['C']}, "
                  f"union={r['baseline_final_union_modes']}")
        print(f"    mean_local: init={r['mean_initial_local']:.2f} → "
              f"final(coup)={r['mean_final_local']:.2f}, "
              f"final(baseline)={r['baseline_mean_final_local']:.2f}")
        print(f"    invariants_held: {r['diagnostics_local']['invariants_held']}")

        # Early-window timings
        ew = r['early_window']
        tlf = ew['time_to_local_fusion']
        tlf_b = ew['time_to_local_fusion_baseline']
        print(f"    EARLY-WINDOW timing:")
        print(f"      time_to_local_fusion(A,B,C) measurement: "
              f"({tlf['A']}, {tlf['B']}, {tlf['C']})")
        print(f"      time_to_local_fusion(A,B,C) baseline:    "
              f"({tlf_b['A']}, {tlf_b['B']}, {tlf_b['C']})")
        print(f"      time_to_union_collapse to (5,3,1) meas: "
              f"({ew['time_to_union_collapse'][5]}, "
              f"{ew['time_to_union_collapse'][3]}, "
              f"{ew['time_to_union_collapse'][1]})")
        print(f"      time_to_union_collapse to (5,3,1) base: "
              f"({ew['time_to_union_collapse_baseline'][5]}, "
              f"{ew['time_to_union_collapse_baseline'][3]}, "
              f"{ew['time_to_union_collapse_baseline'][1]})")
        print(f"      union_decay_rate measurement={ew['union_decay_rate_per_step']:.4f} "
              f"baseline={ew['union_decay_rate_baseline']:.4f}")
        print(f"      max_abs_deviation_vs_baseline: "
              f"A={ew['early_window_max_abs_dev']['A']} "
              f"B={ew['early_window_max_abs_dev']['B']} "
              f"C={ew['early_window_max_abs_dev']['C']} "
              f"union={ew['early_window_max_abs_dev']['union']}")

        print(f"    regimes: {r['regimes_detected']}")
        if r['tensions_observed']:
            print(f"    tensions:")
            for t in r['tensions_observed']:
                print(f"      • {t['kind']}")

    # Cross-form early-window comparison
    print(f"\n{'═' * 70}")
    print("  CROSS-FORM EARLY-WINDOW COMPARISON")
    print(f"{'═' * 70}")
    print(f"  {'form':<32s} | t_fusion(A,B,C) | t_union(5,3,1) | union_rate")
    for form in forms:
        ew = per_form[form]['early_window']
        tlf = ew['time_to_local_fusion']
        tuc = ew['time_to_union_collapse']
        print(f"  {form:<32s} | "
              f"({tlf['A']},{tlf['B']},{tlf['C']})        | "
              f"({tuc[5]},{tuc[3]},{tuc[1]})       | "
              f"{ew['union_decay_rate_per_step']:.4f}")

    return {
        'test': 'F6_T27_prime_distributed_plurality_label_space',
        'per_form': per_form,
        'base_seed': base_seed,
        'epsilon': eps,
        'caveats': [
            "LABEL-SPACE TRANSPOSITION, NOT NATIVE MCQᴺ. Compressed "
            "intra-instance with homologous-but-not-shared Θ_A/B/C; "
            "global union computed in homologous index space.",
            "5×5×5 grid resolution may produce sub-resolution UNIVERSAL_FUSION "
            "artefacts. Initial/final masks are logged for inspection.",
            "Connectivity = 6-conn (face-adjacent only), not 26-conn.",
            "Mass / positivity / h-bound invariants checked per step. "
            "Regime conclusions are gated on these holding.",
            "Native geometric distributed plurality requires either shared "
            "Θ across modules, inter-modular metric projection, or the "
            "15-factor / 7-module architecture.",
        ],
    }


if __name__ == "__main__":
    result = run_T27_test(base_seed=42)
    out = Path("/home/claude/mcq_v4/results/phase6c_b/F6_T27_prime_plurality.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
