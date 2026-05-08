"""
Phase 6c-B — Bloc 2: Reader application on existing trajectories.

Applies the diagnostic readers from signatures_6c_b.py to trajectories
already captured by Bloc 1 tests (T25' and T27'). This produces a
multi-scale interpretive layer (STR/RSR/MI/MV/T*/opacity/RR3) on top
of the regime classifications, without re-running the engine.

What this provides:
  - STR/RSR per form per sweep config (regime: stationary transformable
    vs reorganisation-stabilisation vs transitioning vs dead inertia)
  - MI/MV from Δ_centred AND Δ_shape histories (corridor maintenance
    vs viable variation vs collapse)
  - 𝕋* effective cadence — but on tau_T trajectory (h is not in the
    captured T25' state since we only logged h_M_mean per step;
    T*-from-h would require a re-run with full h trajectories)
  - RR3 — explicit NOT_MEASURABLE caveat
  - Opacity — derived from 6b F3a/F3b artefacts (re-loaded from 6b)

This Bloc 2 layer is INTERPRETIVE: it does not change the underlying
regime verdicts. It adds reading lenses that put the verdicts in MCQ
multi-scale temporality (t / κ(t) / 𝕋 / 𝕋*).

Caveat: the readers were designed assuming continuous trajectories.
For T27', mode counts are integer-valued per step → STR/RSR's
amplitude-based damping criterion is inappropriate. So T27' uses
only MI/MV (on union mode trajectory) and 𝕋* (on mode-decay timing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.signatures_6c_b import (
    KNV_THRESHOLDS_6C_B,
    estimate_dominant_period, adaptive_window,
    read_STR_RSR, read_MI_MV, read_T_star, read_RR3,
    compute_gamma_double_prime,
)


# ============================================================================
# T25' application — STR/RSR + MI/MV (Δ_centred + Δ_shape) + simplified T*
# ============================================================================

def apply_readers_to_T25_run(t25_run: dict, dt: float = 0.05) -> dict:
    """
    Apply STR/RSR, MI/MV, T* readers to a single T25' sweep config result.

    T25' captures per-step:
      var_M_A_traj, delta_centred_A_traj, delta_shape_A_traj,
      G_proxy_A_traj, tau_T_A_traj, h_M_mean_A_traj
      (and B, C variants)

    All in `signals_raw`.
    """
    sigs = t25_run['signals_raw']

    # τ' as 2D vector (T, I) on module A
    tau_T = np.asarray(sigs.get('tau_T_A_traj', []), dtype=float)
    tau_I = np.asarray(sigs.get('tau_I_A_traj', []), dtype=float)
    if tau_T.size == 0:
        return {'error': 'NO_TRAJECTORY_DATA'}
    tau_prime_A = np.column_stack([tau_T, tau_I])

    # ─── STR / RSR on tau_prime_A ───
    # Construct G_modular_history — list per step of dict {A,B,C}
    G_A = sigs.get('G_proxy_A_traj', [])
    n_steps = len(tau_T)
    G_modular_history = [
        {'A': {'G_total': G_A[i] if i < len(G_A) else 0.0},
         'B': {'G_total': 1.0}, 'C': {'G_total': 1.0}}  # B,C absent here, dummy
        for i in range(n_steps)
    ]
    str_rsr = read_STR_RSR(tau_prime_A, G_modular_history, dt=dt)

    # ─── MI/MV on Δ_centred and Δ_shape ───
    delta_centred = sigs.get('delta_centred_A_traj', [])
    delta_shape = sigs.get('delta_shape_A_traj', [])
    delta_centred_history = [
        {'delta': float(d),
         'in_corridor': bool(KNV_THRESHOLDS_6C_B['delta_min'] < d
                             < KNV_THRESHOLDS_6C_B['delta_crit'])}
        for d in delta_centred
    ]
    delta_shape_history = [
        {'delta': float(d),
         'in_corridor': bool(KNV_THRESHOLDS_6C_B['delta_min'] < d
                             < KNV_THRESHOLDS_6C_B['delta_crit'])}
        for d in delta_shape
    ]
    mi_mv_centred = read_MI_MV(delta_centred_history)
    mi_mv_shape = read_MI_MV(delta_shape_history)

    # ─── Simplified 𝕋* on h_M_mean ───
    # Real T* would need full h_T/h_M/h_I trajectories. We have only
    # h_M_mean per step. We compute a scalar cadence proxy.
    h_M_mean = np.asarray(sigs.get('h_M_mean_A_traj', []), dtype=float)
    if h_M_mean.size > 1:
        d_h_M = np.abs(np.diff(h_M_mean)) / max(dt, 1e-12)
        window = min(20, len(d_h_M))
        recent = d_h_M[-window:] if len(d_h_M) >= window else d_h_M
        cadence_h_M_recent = float(np.mean(recent))
        if cadence_h_M_recent < 1e-4:
            T_star_status = 'LOCKED_h_M'
        elif cadence_h_M_recent < 1e-2:
            T_star_status = 'SLOW_h_M'
        else:
            T_star_status = 'MOVING_h_M'
    else:
        cadence_h_M_recent = 0.0
        T_star_status = 'NO_DATA'

    T_star_simplified = {
        'cadence_h_M_recent_mean': cadence_h_M_recent,
        'status': T_star_status,
        'note': "Simplified T* — only h_M_mean available; full T* requires h_T/h_M/h_I per step.",
    }

    # ─── Cross-reader synthesis ───
    synthesis = {
        'dominant_dynamic_regime': str_rsr.get('dominant_regime', 'NO_DATA'),
        'MI_status_on_centred': mi_mv_centred.get('MI_status'),
        'MI_status_on_shape': mi_mv_shape.get('MI_status'),
        'T_star_status': T_star_status,
        'reader_layer_consistent_with_regime': None,  # filled below
    }

    # Consistency check: does the reader layer agree with the run's regime?
    regimes = t25_run.get('regimes_detected', [])
    knv_present = any(r.startswith('KNV_') for r in regimes)
    expansion_present = any('EXPANSION' in r for r in regimes)
    contraction_present = any(
        ('CONTRACTION' in r) and not r.startswith('NUMERICAL')
        for r in regimes
    )

    # If KNV is in regimes, MI on shape should be COMPROMISED or PARTIAL
    if knv_present and synthesis['MI_status_on_shape'] in ('COMPROMISED', 'PARTIAL'):
        synthesis['reader_layer_consistent_with_regime'] = True
    elif not knv_present and synthesis['MI_status_on_shape'] == 'MAINTAINED':
        synthesis['reader_layer_consistent_with_regime'] = True
    elif knv_present and synthesis['MI_status_on_shape'] == 'MAINTAINED':
        synthesis['reader_layer_consistent_with_regime'] = False
    elif not knv_present and synthesis['MI_status_on_shape'] != 'MAINTAINED':
        synthesis['reader_layer_consistent_with_regime'] = False
    else:
        synthesis['reader_layer_consistent_with_regime'] = None

    return {
        'STR_RSR': {
            'dominant_regime': str_rsr.get('dominant_regime'),
            'aggregate_counts': str_rsr.get('aggregate_counts'),
            'window_used': str_rsr.get('window'),
        },
        'MI_MV_on_delta_centred': {
            'fraction_in_corridor': mi_mv_centred.get('fraction_in_corridor'),
            'fraction_low_rest': mi_mv_centred.get('fraction_low_rest'),
            'fraction_viable_total': mi_mv_centred.get('fraction_viable_total'),
            'n_excursions': mi_mv_centred.get('n_excursions'),
            'n_MV_events': mi_mv_centred.get('n_MV_events'),
            'n_KNV_events': mi_mv_centred.get('n_KNV_events'),
            'MI_status': mi_mv_centred.get('MI_status'),
        },
        'MI_MV_on_delta_shape': {
            'fraction_in_corridor': mi_mv_shape.get('fraction_in_corridor'),
            'fraction_low_rest': mi_mv_shape.get('fraction_low_rest'),
            'fraction_viable_total': mi_mv_shape.get('fraction_viable_total'),
            'n_excursions': mi_mv_shape.get('n_excursions'),
            'n_MV_events': mi_mv_shape.get('n_MV_events'),
            'n_KNV_events': mi_mv_shape.get('n_KNV_events'),
            'MI_status': mi_mv_shape.get('MI_status'),
        },
        'T_star_simplified': T_star_simplified,
        'RR3': read_RR3(),
        'synthesis': synthesis,
    }


# ============================================================================
# T27' application — MI/MV on n_modes_union trajectory
# ============================================================================

def apply_readers_to_T27_run(t27_run: dict, dt: float = 0.05) -> dict:
    """
    For T27', the meaningful continuous-time signal is the n_modes_union
    trajectory. STR/RSR don't apply (integer-valued mode counts), but
    MI/MV adapted to "is the mode count in a viable range" is informative.

    We define a viable plurality corridor: 3 <= n_modes_union <= 7.
      - n_modes_union > 7: above the initial union → expansion of plurality
                          (impossible with 5×5×5 grid in practice; capped)
      - 1 <= n_modes_union < 3: collapsed plurality (post-fusion regime)
      - n_modes_union = 1: total fusion

    𝕋* on union mode decay timing: how rapid is the transition from
    plural to fused state.
    """
    union_traj = t27_run.get('n_modes_union_traj', [])
    union_baseline = t27_run.get('n_modes_union_baseline_traj', [])
    if not union_traj:
        return {'error': 'NO_TRAJECTORY_DATA'}

    # MI/MV on union trajectory — viable range = [3, 7]
    PLURAL_VIABLE_LOW = 3
    PLURAL_VIABLE_HIGH = 7

    union_history = [
        {'delta': float(u),  # using 'delta' field name for read_MI_MV API
         'in_corridor': bool(PLURAL_VIABLE_LOW <= u <= PLURAL_VIABLE_HIGH)}
        for u in union_traj
    ]
    # We can't reuse read_MI_MV directly (its rest-low logic is corridor-
    # specific). Compute manually:
    n = len(union_traj)
    n_in_corridor = sum(1 for h in union_history if h['in_corridor'])
    n_below_corridor = sum(1 for h in union_history
                           if not h['in_corridor'] and h['delta'] < PLURAL_VIABLE_LOW)
    fraction_plural_viable = n_in_corridor / n if n > 0 else 0.0
    fraction_collapsed = n_below_corridor / n if n > 0 else 0.0

    # Time to first leave the plural corridor (fusion onset)
    first_leave_corridor = None
    for i, h in enumerate(union_history):
        if not h['in_corridor'] and h['delta'] < PLURAL_VIABLE_LOW:
            first_leave_corridor = i
            break

    # Same for baseline
    if union_baseline:
        first_leave_corridor_baseline = None
        for i, u in enumerate(union_baseline):
            if u < PLURAL_VIABLE_LOW:
                first_leave_corridor_baseline = i
                break
    else:
        first_leave_corridor_baseline = None

    # 𝕋* on mode decay rate
    if len(union_traj) > 1:
        union_arr = np.array(union_traj, dtype=float)
        diffs = np.abs(np.diff(union_arr)) / max(dt, 1e-12)
        # transition window: where most of the decay happens
        if first_leave_corridor is not None:
            window_start = max(0, first_leave_corridor - 5)
            window_end = min(len(diffs), first_leave_corridor + 15)
            transition_diffs = diffs[window_start:window_end]
            mean_decay_rate_in_transition = float(np.mean(transition_diffs)) if len(transition_diffs) > 0 else 0.0
        else:
            mean_decay_rate_in_transition = 0.0
    else:
        mean_decay_rate_in_transition = 0.0

    return {
        'plurality_corridor': [PLURAL_VIABLE_LOW, PLURAL_VIABLE_HIGH],
        'fraction_plural_viable': fraction_plural_viable,
        'fraction_collapsed': fraction_collapsed,
        'first_leave_corridor_step': first_leave_corridor,
        'first_leave_corridor_baseline_step': first_leave_corridor_baseline,
        'leave_corridor_diff_vs_baseline': (
            (first_leave_corridor - first_leave_corridor_baseline)
            if (first_leave_corridor is not None
                and first_leave_corridor_baseline is not None)
            else None
        ),
        'T_star_proxy_mode_decay_rate': mean_decay_rate_in_transition,
        'RR3': read_RR3(),
        'synthesis': {
            'plurality_maintenance': (
                'NEVER_VIABLE' if fraction_plural_viable < 0.1
                else 'TRANSIENT' if fraction_plural_viable < 0.5
                else 'PARTIAL' if fraction_plural_viable < 0.9
                else 'MAINTAINED'
            ),
        },
    }


# ============================================================================
# Orchestrator — load JSON results from previous tests and apply readers
# ============================================================================

def run_bloc2(results_dir: Path = None, dt: float = 0.05) -> dict:
    if results_dir is None:
        results_dir = Path("/home/claude/mcq_v4/results/phase6c_b")
    print(f"\n{'#' * 70}")
    print(f"# Bloc 2 — Reader application on existing trajectories")
    print(f"# Source: {results_dir}")
    print(f"{'#' * 70}\n")

    out = {'T25_readers': {}, 'T27_readers': {}}

    # ─── T25' ───
    t25_path = results_dir / "F6_T25_prime_contraction.json"
    if t25_path.exists():
        with open(t25_path) as f:
            t25 = json.load(f)
        print("[T25'] Applying STR/RSR + MI/MV (Δ_centred + Δ_shape) + 𝕋* + RR3 readers")
        for form in t25.get('per_form', {}):
            out['T25_readers'][form] = {}
            sweep = t25['per_form'][form].get('sweep_results', {})
            for label, run in sweep.items():
                out['T25_readers'][form][label] = apply_readers_to_T25_run(run, dt=dt)
            # Print summary line per form for the gentle-most config
            first_label = next(iter(sweep)) if sweep else None
            if first_label:
                rl = out['T25_readers'][form][first_label]
                print(f"  {form:<32s} [{first_label}]")
                str_rsr = rl.get('STR_RSR', {})
                mi_c = rl.get('MI_MV_on_delta_centred', {})
                mi_s = rl.get('MI_MV_on_delta_shape', {})
                ts = rl.get('T_star_simplified', {})
                print(f"    STR/RSR dominant_regime={str_rsr.get('dominant_regime')}")
                print(f"    MI(Δ_centred)={mi_c.get('MI_status')}  "
                      f"frac_corridor={mi_c.get('fraction_in_corridor', 0):.2f}  "
                      f"frac_low_rest={mi_c.get('fraction_low_rest', 0):.2f}  "
                      f"MV_events={mi_c.get('n_MV_events')}  "
                      f"KNV_events={mi_c.get('n_KNV_events')}")
                print(f"    MI(Δ_shape)  ={mi_s.get('MI_status')}  "
                      f"frac_corridor={mi_s.get('fraction_in_corridor', 0):.2f}  "
                      f"MV_events={mi_s.get('n_MV_events')}  "
                      f"KNV_events={mi_s.get('n_KNV_events')}")
                print(f"    T_star_simplified={ts.get('status')} "
                      f"cadence={ts.get('cadence_h_M_recent_mean', 0):.5f}")
                print(f"    consistent_with_regime={rl.get('synthesis', {}).get('reader_layer_consistent_with_regime')}")
    else:
        print(f"[T25'] File not found: {t25_path}")

    # ─── T27' ───
    t27_path = results_dir / "F6_T27_prime_plurality.json"
    if t27_path.exists():
        with open(t27_path) as f:
            t27 = json.load(f)
        print("\n[T27'] Applying plurality MI + 𝕋*-mode-decay readers")
        for form in t27.get('per_form', {}):
            run = t27['per_form'][form]
            r = apply_readers_to_T27_run(run, dt=dt)
            out['T27_readers'][form] = r
            print(f"  {form:<32s}")
            print(f"    plurality_maintenance={r.get('synthesis', {}).get('plurality_maintenance')}  "
                  f"frac_plural_viable={r.get('fraction_plural_viable', 0):.2f}")
            print(f"    first_leave_corridor: meas={r.get('first_leave_corridor_step')} "
                  f"base={r.get('first_leave_corridor_baseline_step')} "
                  f"diff={r.get('leave_corridor_diff_vs_baseline')}")
            print(f"    T*_proxy_mode_decay_rate={r.get('T_star_proxy_mode_decay_rate', 0):.3f}")
    else:
        print(f"[T27'] File not found: {t27_path}")

    # RR3 — uniform caveat across all
    print(f"\n[RR3] {read_RR3()['rationale']}")

    return out


if __name__ == "__main__":
    result = run_bloc2()
    out = Path("/home/claude/mcq_v4/results/phase6c_b/bloc2_readers.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
