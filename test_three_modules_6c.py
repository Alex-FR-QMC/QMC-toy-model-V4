"""
Phase 6c-A — Main runner.

Differential measurement of perspectival coupling (6c) vs label-based
contrastive baseline (6b), all on the same post-warmup state.

Protocol:
  1. Run 6b preflight suite (the existing 7 preflights)
  2. Run preflight 8 (perspectival sign micro-test)
  3. Warmup with coupling OFF (option A): produces a common post-warmup
     state with h_A ≠ h_B ≠ h_C from differentiated weights only
  4. Validate warmup: max(h_div_warmup) >= h_div_floor
     ε_struct = 0.2 × max(h_div_warmup)
  5. From post-warmup state, run 200 steps with each coupling form:
       'contrastive' (6b baseline), 'perspectival_INV_H',
       'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE'
     ALL with same RNG state (cloned at end of warmup)
  6. Compute F1, F4, R_metric, ΔR, h_div_ratio, extras diagnostics
  7. Evaluate H1-H4 with explicit criteria
  8. Synthesize 3-level interpretation
  9. Save verdict with mandatory fields

Phase 6c-A is a baseline gradient coupling — 𝒞^{mod} as Φ_extra.
The non-gradient extension is deferred to Phase 6d.

Run:
    PYTHONPATH=src python tests/phase6c/test_three_modules_6c.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

# Add phase6b for preflight orchestrator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase6b"))

from mcq_v4.factorial.state import (
    FactorialEngineConfig, EngineMode, ModuleConfig, THETA_T, FactorialState,
)
from mcq_v4.factorial.engine import make_initial_state
from mcq_v4.factorial.three_module_system import (
    DIFFERENTIATED_WEIGHTS, CouplingConfig, build_three_module_system,
)
from mcq_v4.factorial.coupling import (
    run_three_modules, step_three_modules, _h_field_product,
)
from mcq_v4.factorial.tau_prime import compute_modular_contributions
from mcq_v4.factorial import metrics_6b as m6b
from mcq_v4.factorial import diagnostics_6c as d6c

from preflight_suite import run_preflight_suite


WARMUP_PROTOCOL_6C = {
    'n_warmup_steps': 100,
    'coupling_during_warmup': 'OFF',
    'base_seed': 42,
    'h_div_floor': 1e-3,
    'epsilon_struct_fraction': 0.2,
}


# ============================================================================
# Initial conditions
# ============================================================================

def make_initial_psi():
    def init_T():
        p = np.exp(-(THETA_T**2)/2.0); p /= p.sum()
        psi = np.zeros((5,5,5)); psi[:,2,2] = p; return psi
    def init_M():
        p = np.exp(-((np.arange(5)-2)**2)/2.0); p /= p.sum()
        psi = np.zeros((5,5,5)); psi[2,:,2] = p; return psi
    def init_I():
        p = np.exp(-(THETA_T**2)/2.0); p /= p.sum()
        psi = np.zeros((5,5,5)); psi[2,2,:] = p; return psi
    return {'A': init_T(), 'B': init_M(), 'C': init_I()}


# ============================================================================
# Warmup (option A — coupling OFF)
# ============================================================================

def run_warmup(cfg_engine, initial_psi, base_seed, n_warmup):
    """
    Pure intra-modular dynamics warmup. No coupling.
    Returns post-warmup system + h_div trajectory + max h_div.
    """
    coupling_cfg = CouplingConfig(epsilon=0.0, coupling_form='contrastive')
    sys = build_three_module_system(cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
                                    initial_psi, base_seed=base_seed)

    h_div_traj = []
    for t in range(n_warmup):
        sys, _ = step_three_modules(sys, coupling_active=False)
        # h_divergence at this step
        pairs_div = d6c.compute_pairwise_h_divergence(sys.states)
        h_div_traj.append({
            pair: pairs_div[pair]['total']
            for pair in [('A', 'B'), ('B', 'C'), ('A', 'C')]
        })

    max_h_div = max(
        max(d.values()) for d in h_div_traj
    ) if h_div_traj else 0.0

    return {
        'system': sys,
        'h_div_trajectory': h_div_traj,
        'max_h_div_warmup': float(max_h_div),
        'epsilon_struct': 0.2 * float(max_h_div),
        'n_warmup_steps': n_warmup,
    }


def clone_system_state(sys):
    """Deep-copy a system's state and engine RNG bit_generator state."""
    state_dict = {
        name: FactorialState(
            psi=getattr(sys, f'state_{name}').psi.copy(),
            h_T=getattr(sys, f'state_{name}').h_T.copy(),
            h_M=getattr(sys, f'state_{name}').h_M.copy(),
            h_I=getattr(sys, f'state_{name}').h_I.copy(),
            cfg=getattr(sys, f'state_{name}').cfg,
        )
        for name in ['A', 'B', 'C']
    }
    rng_states = {
        name: copy.deepcopy(getattr(sys, f'engine_{name}').rng.bit_generator.state)
        for name in ['A', 'B', 'C']
    }
    prev_h_fields = (
        copy.deepcopy(sys.prev_h_fields)
        if sys.prev_h_fields is not None else None
    )
    return state_dict, rng_states, prev_h_fields


def make_system_from_clone(cfg_engine, coupling_cfg, state_dict,
                           rng_states, prev_h_fields, base_seed):
    """Build a fresh ThreeModuleSystem with cloned states + RNGs."""
    initial_psi = {n: state_dict[n].psi.copy() for n in ['A', 'B', 'C']}
    sys = build_three_module_system(cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
                                    initial_psi, base_seed=base_seed)
    # Overwrite states with cloned ones (psi + h)
    for name in ['A', 'B', 'C']:
        st = state_dict[name]
        new_st = FactorialState(
            psi=st.psi.copy(), h_T=st.h_T.copy(),
            h_M=st.h_M.copy(), h_I=st.h_I.copy(),
            cfg=st.cfg,
        )
        if name == 'A': sys.state_A = new_st
        elif name == 'B': sys.state_B = new_st
        else: sys.state_C = new_st
    # Restore RNG states
    for name in ['A', 'B', 'C']:
        getattr(sys, f'engine_{name}').rng.bit_generator.state = copy.deepcopy(rng_states[name])
    # Restore prev_h_fields
    sys.prev_h_fields = copy.deepcopy(prev_h_fields) if prev_h_fields else None
    return sys


# ============================================================================
# Per-form measurement
# ============================================================================

def measure_form(cfg_engine, coupling_form, eps,
                 state_dict, rng_states, prev_h_fields,
                 max_h_div_warmup, base_seed, n_measurement=200):
    """
    Run n_measurement steps from the same post-warmup state with the
    given coupling form. Logs trajectories + diagnostics.
    """
    coupling_cfg = CouplingConfig(epsilon=eps, coupling_form=coupling_form)
    sys = make_system_from_clone(cfg_engine, coupling_cfg, state_dict,
                                 rng_states, prev_h_fields, base_seed)

    # Trajectories to log
    h_div_traj = []
    R_psi_traj = []
    R_metric_traj = []
    delta_R_traj = []
    extras_traj = []  # list of {'A': arr, 'B': arr, 'C': arr}

    for t in range(n_measurement):
        # h_div BEFORE step
        h_div = d6c.compute_pairwise_h_divergence(sys.states)
        h_div_total_pairs = {pair: h_div[pair]['total']
                             for pair in [('A','B'), ('B','C'), ('A','C')]}
        max_pair = max(h_div_total_pairs.values())
        h_div_traj.append({
            'pairs': h_div_total_pairs,
            'max_pair': max_pair,
            'h_div_ratio': max_pair / max(max_h_div_warmup, 1e-12),
        })

        # R_psi and R_metric BEFORE step
        from mcq_v4.factorial.overlaps import compute_pairwise_R_psi
        R_psi = compute_pairwise_R_psi(sys.states)
        R_metric = d6c.compute_pairwise_R_metric(
            sys.states, coupling_form, cfg_engine,
            prev_h_fields=sys.prev_h_fields,
        )
        delta_R = d6c.compute_delta_R(R_psi, {p: R_metric[p] for p in R_metric})
        R_psi_traj.append(R_psi)
        R_metric_traj.append({p: R_metric[p] for p in R_metric})
        delta_R_traj.append(delta_R)

        # Step (computes extras internally — we recompute them here for logging)
        # Actually capture from step_log
        sys, log = step_three_modules(sys, coupling_active=True)

        # Extras: re-compute for this step (since step_log doesn't return them directly)
        # For brevity we approximate by storing per-module extra phi diagnostics
        # which include max_phi_extra. Pattern dissimilarity needs the array,
        # so we recompute using the pre-step state we just left.
        # Simpler: capture extras manually — but we already passed through step.
        # Use what step_three_modules already logged (extra_phi_diagnostics has scalars only).
        extras_traj.append({
            name: log['extra_phi_diagnostics'][name].get('max_phi_extra', 0.0)
            for name in ['A', 'B', 'C']
        })

    # Final F4 (use existing 6b function on the trajectory we just produced)
    # We need the system histories for F4 — let's rebuild minimally
    # For F4 we just need final h profiles
    final_state = {'A': sys.state_A, 'B': sys.state_B, 'C': sys.state_C}
    final_metric_div = d6c.compute_pairwise_h_divergence(final_state)
    metric_div_mean = float(np.mean([
        final_metric_div[p]['total'] / 3.0  # divide by number of axes for per-axis avg
        for p in [('A','B'), ('B','C'), ('A','C')]
    ]))

    return {
        'coupling_form': coupling_form,
        'epsilon': eps,
        'h_div_trajectory': h_div_traj,
        'R_psi_trajectory_means': {
            'A-B': float(np.mean([r[('A','B')] for r in R_psi_traj])),
            'B-C': float(np.mean([r[('B','C')] for r in R_psi_traj])),
            'A-C': float(np.mean([r[('A','C')] for r in R_psi_traj])),
        },
        'R_metric_trajectory_means': {
            'A-B': float(np.mean([r[('A','B')]['R_metric'] for r in R_metric_traj])),
            'B-C': float(np.mean([r[('B','C')]['R_metric'] for r in R_metric_traj])),
            'A-C': float(np.mean([r[('A','C')]['R_metric'] for r in R_metric_traj])),
        },
        'norm_ratio_means': {
            'A-B': float(np.mean([r[('A','B')]['norm_ratio'] for r in R_metric_traj])),
            'B-C': float(np.mean([r[('B','C')]['norm_ratio'] for r in R_metric_traj])),
            'A-C': float(np.mean([r[('A','C')]['norm_ratio'] for r in R_metric_traj])),
        },
        'delta_R_history': delta_R_traj,
        'extras_max_per_step': extras_traj,
        'F4_metric_divergence_final': metric_div_mean,
        'final_system': sys,
    }


def measure_F1_per_form(cfg_engine, coupling_form, eps,
                        state_dict, rng_states, prev_h_fields,
                        base_seed, n_steps=100):
    """
    Compute F1 trajectorial differential post-warmup for one form.
    Same approach as 6b's metric_F1_trajectorial but starting from a
    cloned post-warmup state.
    """
    # Branch 1: coupled
    sys_c = make_system_from_clone(
        cfg_engine, CouplingConfig(epsilon=eps, coupling_form=coupling_form),
        state_dict, rng_states, prev_h_fields, base_seed,
    )
    contribs_coupled_T = []
    history = [sys_c]
    for t in range(n_steps):
        sys_c, _ = step_three_modules(sys_c, coupling_active=True)
        history.append(sys_c)
    contribs_coupled_T = np.array([
        compute_modular_contributions(s.state_A)['T']
        + compute_modular_contributions(s.state_B)['T']
        for s in history
    ])

    # Branch 2: solo parallel (eps=0, same starting state)
    sys_p = make_system_from_clone(
        cfg_engine, CouplingConfig(epsilon=0.0, coupling_form=coupling_form),
        state_dict, rng_states, prev_h_fields, base_seed,
    )
    history_p = [sys_p]
    for t in range(n_steps):
        sys_p, _ = step_three_modules(sys_p, coupling_active=False)
        history_p.append(sys_p)
    tau_AB_par = np.array([
        compute_modular_contributions(s.state_A)['T']
        + compute_modular_contributions(s.state_B)['T']
        for s in history_p
    ])

    dt = cfg_engine.dt
    L1_pure = float(np.abs(contribs_coupled_T - tau_AB_par).sum() * dt)
    amp = max(float(tau_AB_par.max() - tau_AB_par.min()), 1e-12)
    T_total = dt * n_steps
    rel_pure = L1_pure / max(amp * T_total, 1e-12)

    diff_pure = np.abs(contribs_coupled_T - tau_AB_par)
    if diff_pure.max() < 1e-9:
        signal = 'ABSENT'
    elif diff_pure.max() < 1e-5:
        signal = 'NOISY'
    else:
        signal = 'CLEAN'

    return {
        'rel_diff_pure': rel_pure,
        'L1_diff_pure': L1_pure,
        'amp_reference': amp,
        'signal_quality': signal,
    }


# ============================================================================
# Pattern dissimilarity & temporal stability via re-running with arrays kept
# ============================================================================

def measure_extras_arrays(cfg_engine, coupling_form, eps,
                          state_dict, rng_states, prev_h_fields,
                          base_seed, n_steps=50):
    """
    Re-run the form, capturing the actual phi_extra arrays at each step.
    Used for pattern dissimilarity and temporal stability metrics.
    """
    from mcq_v4.factorial.coupling import compute_extra_phi_for_module
    from mcq_v4.factorial.overlaps import compute_pairwise_R_psi

    coupling_cfg = CouplingConfig(epsilon=eps, coupling_form=coupling_form)
    sys = make_system_from_clone(cfg_engine, coupling_cfg, state_dict,
                                 rng_states, prev_h_fields, base_seed)

    extras_arrays = []
    for t in range(n_steps):
        R_pairs = compute_pairwise_R_psi(sys.states)
        extras_step = {}
        for name in ['A', 'B', 'C']:
            extras_step[name] = compute_extra_phi_for_module(
                name, sys.states, R_pairs, coupling_cfg,
                prev_h_fields=sys.prev_h_fields,
                h_min=cfg_engine.h_min, h_0=cfg_engine.h_0, dt=cfg_engine.dt,
            )
        extras_arrays.append(extras_step)
        sys, _ = step_three_modules(sys, coupling_active=True)
    return extras_arrays


# ============================================================================
# JSON safety + verdict assembly
# ============================================================================

def _make_json_safe(obj):
    from enum import Enum
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
        return f"<{type(obj).__name__}>"
    return obj


MANDATORY_FIELDS_6C = [
    'phase', 'phase_status', 'preflight_status',
    'warmup', 'forms_tested', 'hypotheses', 'synthesis',
    'seeds', 'global_status',
]


def assert_mandatory_fields(verdict):
    missing = [f for f in MANDATORY_FIELDS_6C if f not in verdict]
    assert not missing, f"6c verdict missing mandatory fields: {missing}"
    assert verdict['seeds'].get('rngs_independent') is True, \
        "rngs_independent must be True"


def save_verdict(verdict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _make_json_safe(verdict)
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)


# ============================================================================
# Main runner
# ============================================================================

def run_phase_6c_a():
    print("#" * 70)
    print("# PHASE 6c-A — perspectival coupling (gradient approximation)")
    print("# 6c stays in Φ_extra architecture; non-gradient is deferred to 6d.")
    print("#" * 70)
    print()

    base_seed = WARMUP_PROTOCOL_6C['base_seed']
    cfg_engine = FactorialEngineConfig(
        dt=0.05, T_steps=300, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )

    # ---- 1. Preflight 6b suite ----
    print("=" * 70)
    print("STEP 1 — Preflight 6b (7 preflights)")
    print("=" * 70)
    preflight_6b = run_preflight_suite(verbose=False)
    print(f"  Status: {preflight_6b['status']}")
    if preflight_6b['status'] != 'OK':
        print("  ❌ Preflight 6b failed. 6c-A cannot proceed.")
        return {'phase': '6c-A', 'global_status': preflight_6b['status']}

    # ---- 2. Preflight 8 (perspectival sign micro-test) ----
    print()
    print("=" * 70)
    print("STEP 2 — Preflight 8 (perspectival sign micro-test)")
    print("=" * 70)
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'test_perspectival_sign.py')],
        capture_output=True, text=True, timeout=120,
        env={**__import__('os').environ, 'PYTHONPATH': str(_SRC_PATH)},
    )
    preflight_8_pass = (proc.returncode == 0)
    print(f"  Status: {'PASS' if preflight_8_pass else 'FAIL'}")
    if not preflight_8_pass:
        print("  ❌ Preflight 8 failed.")
        return {'phase': '6c-A', 'global_status': 'PREFLIGHT_8_FAILED'}

    # ---- 3. Warmup ----
    print()
    print("=" * 70)
    print("STEP 3 — Warmup (coupling OFF, n=100 steps)")
    print("=" * 70)
    initial_psi = make_initial_psi()
    warmup_result = run_warmup(cfg_engine, initial_psi, base_seed,
                               n_warmup=WARMUP_PROTOCOL_6C['n_warmup_steps'])
    max_hd = warmup_result['max_h_div_warmup']
    floor = WARMUP_PROTOCOL_6C['h_div_floor']
    eps_struct = warmup_result['epsilon_struct']
    print(f"  max h_div during warmup: {max_hd:.4f}")
    print(f"  h_div floor required:    {floor}")
    print(f"  ε_struct (0.2 × max):    {eps_struct:.4f}")
    if max_hd < floor:
        print("  ❌ Warmup INSUFFICIENT — h did not diverge enough.")
        return {'phase': '6c-A', 'global_status': 'WARMUP_INSUFFICIENT',
                'warmup': warmup_result}
    print(f"  ✓ Warmup adequate.")

    # Clone post-warmup state for measurements
    state_dict, rng_states, prev_h_fields = clone_system_state(warmup_result['system'])
    print(f"  ✓ Post-warmup state cloned (RNG snapshot taken).")

    # ---- 4. Per-form measurements ----
    print()
    print("=" * 70)
    print("STEP 4 — Per-form measurements (200 steps each, same starting state)")
    print("=" * 70)

    eps = 0.005
    forms = ['contrastive',
             'perspectival_INV_H',
             'perspectival_H_OPEN',
             'perspectival_MORPHO_ACTIVE']

    measurements = {}
    f1_results = {}
    extras_arrays_per_form = {}

    for form in forms:
        print(f"  {form}...", end=" ", flush=True)
        meas = measure_form(cfg_engine, form, eps,
                            state_dict, rng_states, prev_h_fields,
                            max_hd, base_seed, n_measurement=200)
        # Strip non-serializable system from measurements
        meas_save = {k: v for k, v in meas.items() if k != 'final_system'}
        measurements[form] = meas_save

        f1 = measure_F1_per_form(cfg_engine, form, eps,
                                 state_dict, rng_states, prev_h_fields,
                                 base_seed, n_steps=100)
        f1_results[form] = f1

        ext_arr = measure_extras_arrays(cfg_engine, form, eps,
                                        state_dict, rng_states, prev_h_fields,
                                        base_seed, n_steps=50)
        extras_arrays_per_form[form] = ext_arr

        print(f"F1={f1['rel_diff_pure']:.5f}, "
              f"R_metric_AB={meas['R_metric_trajectory_means']['A-B']:.4f}, "
              f"F4_div={meas['F4_metric_divergence_final']:.4f}")

    # ---- 5. Phi_extra diagnostics (triple constraint) ----
    print()
    print("=" * 70)
    print("STEP 5 — Phi_extra triple-constraint diagnostics")
    print("=" * 70)
    baseline_extras = extras_arrays_per_form['contrastive']
    extras_diagnostics = {}
    for form in forms:
        if form == 'contrastive':
            continue
        diag = d6c.compute_phi_extra_diagnostics(
            baseline_extras, extras_arrays_per_form[form])
        extras_diagnostics[form] = diag
        print(f"  {form}:")
        for name in ['A', 'B', 'C']:
            d = diag['per_module'][name]
            print(f"    {name}: amp={d['amplification_ratio']:.3f}, "
                  f"dis={d['pattern_dissimilarity']:.3f}, "
                  f"stab={d['temporal_stability']:.3f} → {d['classification']}")
        print(f"    aggregate: {diag['aggregate_classification']}")

    # ---- 6. Hypothesis evaluation H1-H4 ----
    print()
    print("=" * 70)
    print("STEP 6 — Hypothesis evaluation H1/H2/H3/H4")
    print("=" * 70)
    f1_baseline = f1_results['contrastive']
    f4_baseline = {'metric_divergence_mean':
                   measurements['contrastive']['F4_metric_divergence_final']}

    hypotheses_per_form = {}
    for form in forms:
        if form == 'contrastive':
            continue
        h1 = d6c.evaluate_hypothesis_H1(f1_baseline, f1_results[form])
        h2 = d6c.evaluate_hypothesis_H2(measurements[form]['delta_R_history'])
        h3 = d6c.evaluate_hypothesis_H3(
            f4_baseline,
            {'metric_divergence_mean': measurements[form]['F4_metric_divergence_final']},
        )
        h4 = d6c.evaluate_hypothesis_H4(h1, h2, h3, max_hd)
        synthesis = d6c.synthesize_3_levels(h1, h2, h3, h4, extras_diagnostics[form])
        hypotheses_per_form[form] = {
            'H1': h1, 'H2': h2, 'H3': h3, 'H4': h4,
            'synthesis_3_levels': synthesis,
        }
        print(f"  {form}:")
        print(f"    H1 (F1 diff):           {h1['verdict']}")
        print(f"    H2 (ΔR structural):     {h2['verdict']} (mean ΔR={h2.get('mean_delta_R', 0):.4f})")
        print(f"    H3 (F4 evolution):      {h3['verdict']}")
        print(f"    H4 (compression kills): {h4['verdict']}")
        print(f"    → Level: {synthesis['level']}")

    # ---- 7. Verdict assembly ----
    print()
    print("=" * 70)
    print("STEP 7 — Verdict assembly")
    print("=" * 70)

    seeds_log = warmup_result['system'].seed_log()

    verdict = {
        'phase': '6c-A',
        'phase_status': (
            "Phase 6c-A — perspectival coupling in gradient approximation "
            "(Φ_extra architecture inherited from 6b). Tests whether the "
            "perspective-dependent formula 𝒞_ij ∝ R · (novelty_j^h_j − R · form_i^h_i) "
            "produces a structurally distinct signal from the label-based 6b "
            "baseline. NON-GRADIENT native coupling (𝒞^{mod} as a separate "
            "term in ∂_t ψ, not in Φ_eff) is deferred to Phase 6d."
        ),
        'preflight_status': preflight_6b['status'],
        'preflight_8_perspectival_sign': preflight_8_pass,
        'engine_config': {
            'dt': cfg_engine.dt, 'mode': str(cfg_engine.mode.value),
            'D_0': cfg_engine.D_0, 'beta_0': cfg_engine.beta_0,
            'gamma_0': cfg_engine.gamma_0, 'h_0': cfg_engine.h_0,
            'h_min': cfg_engine.h_min, 'lambda_KNV': cfg_engine.lambda_KNV,
        },
        'warmup': {
            'protocol': WARMUP_PROTOCOL_6C,
            'max_h_div': max_hd,
            'epsilon_struct': eps_struct,
            'h_div_floor': floor,
            'sufficient': max_hd >= floor,
        },
        'forms_tested': {
            form: {
                'F1': f1_results[form],
                'measurement_summary': measurements[form],
                'phi_extra_diagnostics': extras_diagnostics.get(form),
            }
            for form in forms
        },
        'hypotheses': hypotheses_per_form,
        'synthesis': {
            form: hypotheses_per_form[form]['synthesis_3_levels']
            for form in forms if form != 'contrastive'
        },
        'seeds': seeds_log,
        'epsilon': eps,
        'global_status': 'INTERPRETABLE',
    }

    assert_mandatory_fields(verdict)

    out = Path("/home/claude/mcq_v4/results/phase6c/verdict_phase6c_a.json")
    save_verdict(verdict, out)
    print(f"  ✓ Verdict saved to {out}")

    return verdict


if __name__ == "__main__":
    verdict = run_phase_6c_a()
    print()
    print("#" * 70)
    print(f"# Phase 6c-A — global status: {verdict.get('global_status')}")
    print("#" * 70)
    if 'synthesis' in verdict:
        for form, syn in verdict['synthesis'].items():
            print(f"# {form:<32s} → {syn['level']}")
        print("#" * 70)
