"""
Phase 6b — Main runner.

Orchestrates the full Phase 6b verdict generation:

  1. Preflight suite (7 tests) — fail-fast on any blocking failure
  2. F1 trajectorial interference (3-branch)
  3. F2-bis and F2'-bis per module under active coupling
  4. F3a / F3b per module under active coupling
  5. F4 modular differentiation (priority on h_a)
  6. F5 weight ablation
  7. F6 ε sweep (regime map — primary result)
  8. Global verdict with mandatory fields, asserted before JSON save

Status policy:
  - If preflight status != OK → 6b verdict status = preflight status,
    no F1-F6 interpretation.
  - Otherwise F1-F6 are run and the global verdict is interpretable.

Phase 6b stays explicitly NON-PERSPECTIVAL — see phase_status field.

Run from project root:
    PYTHONPATH=src python tests/phase6b/test_three_modules.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    FactorialEngineConfig, EngineMode, ModuleConfig, THETA_T,
)
from mcq_v4.factorial.engine import FactorialEngine, make_initial_state
from mcq_v4.factorial.three_module_system import (
    ThreeModuleSystem, CouplingConfig,
    DIFFERENTIATED_WEIGHTS, build_three_module_system,
)
from mcq_v4.factorial.coupling import (
    run_three_modules, step_three_modules, compute_extra_phi_for_module,
)
from mcq_v4.factorial.tau_prime import compute_modular_contributions
from mcq_v4.factorial.overlaps import compute_pairwise_R_psi
from mcq_v4.factorial import metrics as m6a       # F2, F2_prime, F3_temporal, F3b_morphodynamic, invariants
from mcq_v4.factorial import metrics_6b as m6b    # F1, F4, F5, F6, synthesis

# Preflight orchestrator
from preflight_suite import run_preflight_suite


# ============================================================================
# Initial conditions (canonical Phase 6b ICs — break symmetry across modules)
# ============================================================================

def init_T_centred() -> np.ndarray:
    """Gaussian on T axis at θ_T=0, point at M=2, I=2."""
    p_T = np.exp(-(THETA_T ** 2) / 2.0); p_T /= p_T.sum()
    psi = np.zeros((5, 5, 5)); psi[:, 2, 2] = p_T
    psi /= psi.sum(); return psi


def init_M_centred() -> np.ndarray:
    p_M = np.exp(-((np.arange(5) - 2) ** 2) / 2.0); p_M /= p_M.sum()
    psi = np.zeros((5, 5, 5)); psi[2, :, 2] = p_M
    psi /= psi.sum(); return psi


def init_I_centred() -> np.ndarray:
    p_I = np.exp(-(THETA_T ** 2) / 2.0); p_I /= p_I.sum()
    psi = np.zeros((5, 5, 5)); psi[2, 2, :] = p_I
    psi /= psi.sum(); return psi


CANONICAL_INITIAL_PSI = {
    'A': init_T_centred,
    'B': init_M_centred,
    'C': init_I_centred,
}


def make_initial_psi() -> dict:
    return {name: fn() for name, fn in CANONICAL_INITIAL_PSI.items()}


# ============================================================================
# F2-bis and F2'-bis per module under active coupling
# ============================================================================

def metric_F2_bis_per_module(
    cfg_engine: FactorialEngineConfig,
    coupling_cfg: CouplingConfig,
    initial_psi: dict,
    base_seed: int = 42,
    n_steps: int = 200,
) -> dict:
    """
    Run the 3-module system under coupling and apply F2 (tripartition)
    on each module's history.

    Each module starts from its assigned initial ψ. F2 measures growth
    of σ_M and σ_I from initial values.
    """
    sys = build_three_module_system(cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
                                    initial_psi, base_seed=base_seed)
    history, logs = run_three_modules(sys, n_steps=n_steps, coupling_active=True)

    # Per-module state histories
    histories = {
        'A': [s.state_A for s in history],
        'B': [s.state_B for s in history],
        'C': [s.state_C for s in history],
    }

    f2_per_module = {}
    for name in ['A', 'B', 'C']:
        f2_per_module[name] = m6a.metric_F2(histories[name])

    return {
        'per_module': f2_per_module,
        'all_pass': all(r['outcome'] == 'PASS' for r in f2_per_module.values()),
        'note': "F2 applied per module under active coupling. Tripartition must persist under coupling.",
    }


def metric_F2_prime_bis_per_module(
    cfg_engine: FactorialEngineConfig,
    coupling_cfg: CouplingConfig,
    initial_psi: dict,
    base_seed: int = 42,
    n_steps: int = 200,
) -> dict:
    """
    For each module M ∈ {A, B, C}, run two coupled simulations:
      - free_M : all h evolve normally
      - frozen_M : h_M frozen for module M, others normal

    Compare τ' trajectories per module. Tests whether the inter-modular
    coupling preserves or alters the h_M mediation pathway found in 6a.
    """
    results = {}
    for target in ['A', 'B', 'C']:
        # Free run
        sys_free = build_three_module_system(
            cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS, initial_psi,
            base_seed=base_seed,
        )
        history_free = [sys_free]
        for t in range(n_steps):
            sys_free, _ = step_three_modules(sys_free,
                                             freeze_h_M={'A': False, 'B': False, 'C': False},
                                             coupling_active=True)
            history_free.append(sys_free)

        # Frozen run (h_M frozen for `target` only)
        sys_frozen = build_three_module_system(
            cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS, initial_psi,
            base_seed=base_seed,
        )
        freeze_dict = {n: (n == target) for n in ['A', 'B', 'C']}
        history_frozen = [sys_frozen]
        for t in range(n_steps):
            sys_frozen, _ = step_three_modules(sys_frozen,
                                               freeze_h_M=freeze_dict,
                                               coupling_active=True)
            history_frozen.append(sys_frozen)

        # Per-target module: extract free vs frozen histories
        h_free = [getattr(s, f'state_{target}') for s in history_free]
        h_frozen = [getattr(s, f'state_{target}') for s in history_frozen]

        # Apply F2_prime metric on these histories
        results[target] = m6a.metric_F2_prime(h_free, h_frozen, cfg_engine)

    return {
        'per_module': results,
        'all_pass': all(r.get('outcome') == 'PASS' for r in results.values()),
        'note': (
            "F2' applied per module under active coupling. Compares τ' "
            "trajectory of each module with its h_M evolving freely vs "
            "frozen, while the coupled system runs around it. Tests "
            "whether the α-mediated h_M effect persists in the coupled regime."
        ),
    }


# ============================================================================
# F3a and F3b per module under active coupling
# ============================================================================

def metric_F3a_per_module(
    cfg_engine: FactorialEngineConfig,
    coupling_cfg: CouplingConfig,
    initial_psi: dict,
    base_seed: int = 42,
    n_steps: int = 200,
) -> dict:
    """F3a (tensional-visible polymorphism) per module, under coupling."""
    sys = build_three_module_system(cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
                                    initial_psi, base_seed=base_seed)
    history, _ = run_three_modules(sys, n_steps=n_steps, coupling_active=True)

    histories = {
        'A': [s.state_A for s in history],
        'B': [s.state_B for s in history],
        'C': [s.state_C for s in history],
    }

    return {
        'per_module': {name: m6a.metric_F3_temporal(histories[name], sample_every=10)
                       for name in ['A', 'B', 'C']},
        'note': (
            "F3a per module under coupling. The marginal-h structural "
            "artefact (M directions invisible to τ') should persist."
        ),
    }


def metric_F3b_per_module(
    cfg_engine: FactorialEngineConfig,
    coupling_cfg: CouplingConfig,
    initial_psi: dict,
    base_seed: int = 42,
    n_steps_warmup: int = 100,
    horizon: int = 50,
) -> dict:
    """
    F3b per module under coupling.

    For each module M, perturb its ψ on the M axis after warmup, then
    measure trajectory divergence vs unperturbed reference, while the
    coupled system continues to run.

    This is more complex than F3b in 6a because the perturbation must
    propagate through coupling without breaking RNG synchrony. We
    implement it as a parallel run of two ThreeModuleSystem instances
    sharing initial RNG state but with the perturbation applied to one.
    """
    from mcq_v4.factorial.metrics import _perturb_psi
    from mcq_v4.factorial.observables import compute_observables
    from mcq_v4.factorial.tau_prime import compute_modular_contributions

    results = {}
    for target in ['A', 'B', 'C']:
        # Reference run with warmup + horizon
        sys_ref = build_three_module_system(
            cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS, initial_psi,
            base_seed=base_seed,
        )
        # Warmup
        for t in range(n_steps_warmup):
            sys_ref, _ = step_three_modules(sys_ref, coupling_active=True)

        # Snapshot RNG states after warmup
        rng_states_after_warmup = {
            n: getattr(sys_ref, f'engine_{n}').rng.bit_generator.state
            for n in ['A', 'B', 'C']
        }

        # Continue reference for horizon
        ref_obs_traj = []
        sys_continued = sys_ref
        for t in range(horizon + 1):
            obs = compute_observables(getattr(sys_continued, f'state_{target}').psi)
            h_M_mean = float(getattr(sys_continued, f'state_{target}').h_M.mean())
            ref_obs_traj.append({
                'var_T': obs['var_T'], 'var_M': obs['var_M'], 'var_I': obs['var_I'],
                'H': obs['H_global'], 'h_M_mean': h_M_mean,
            })
            if t < horizon:
                sys_continued, _ = step_three_modules(sys_continued,
                                                     coupling_active=True)

        # Perturbed branch: rebuild from same warmup point
        sys_pert = build_three_module_system(
            cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS, initial_psi,
            base_seed=base_seed,
        )
        for t in range(n_steps_warmup):
            sys_pert, _ = step_three_modules(sys_pert, coupling_active=True)

        # Apply perturbation to target module's psi (M axis)
        target_state = getattr(sys_pert, f'state_{target}')
        psi_perturbed = _perturb_psi(target_state.psi, axis='M', sign=+1, eps=0.05)
        target_state_new = type(target_state)(
            psi=psi_perturbed,
            h_T=target_state.h_T.copy(),
            h_M=target_state.h_M.copy(),
            h_I=target_state.h_I.copy(),
            cfg=target_state.cfg,
        )
        # Replace in sys_pert
        states = sys_pert.states.copy()
        states[target] = target_state_new
        sys_pert = sys_pert.replace_states(states)

        # Restore RNG states post-warmup so noise is identical
        for n in ['A', 'B', 'C']:
            getattr(sys_pert, f'engine_{n}').rng.bit_generator.state = rng_states_after_warmup[n]

        # Continue perturbed for horizon
        pert_obs_traj = []
        for t in range(horizon + 1):
            obs = compute_observables(getattr(sys_pert, f'state_{target}').psi)
            h_M_mean = float(getattr(sys_pert, f'state_{target}').h_M.mean())
            pert_obs_traj.append({
                'var_T': obs['var_T'], 'var_M': obs['var_M'], 'var_I': obs['var_I'],
                'H': obs['H_global'], 'h_M_mean': h_M_mean,
            })
            if t < horizon:
                sys_pert, _ = step_three_modules(sys_pert, coupling_active=True)

        # Divergences
        div_var_M = max(abs(p['var_M'] - r['var_M'])
                        for p, r in zip(pert_obs_traj, ref_obs_traj))
        div_var_T = max(abs(p['var_T'] - r['var_T'])
                        for p, r in zip(pert_obs_traj, ref_obs_traj))
        div_var_I = max(abs(p['var_I'] - r['var_I'])
                        for p, r in zip(pert_obs_traj, ref_obs_traj))
        div_H = max(abs(p['H'] - r['H']) for p, r in zip(pert_obs_traj, ref_obs_traj))
        div_h_M = max(abs(p['h_M_mean'] - r['h_M_mean'])
                      for p, r in zip(pert_obs_traj, ref_obs_traj))

        # Outcome
        if div_var_M < 1e-9:
            outcome = 'INVISIBLE'; signal = 'ABSENT'
        elif div_var_M >= 0.05:
            outcome = 'PASS'; signal = 'CLEAN'
        elif div_var_M >= 0.005:
            outcome = 'WEAK_DIVERGENCE'; signal = 'CLEAN'
        else:
            outcome = 'INVISIBLE'; signal = 'NOISY'

        results[target] = {
            'div_var_M': float(div_var_M),
            'div_var_T': float(div_var_T),
            'div_var_I': float(div_var_I),
            'div_H': float(div_H),
            'div_h_M': float(div_h_M),
            'outcome': outcome,
            'signal_quality': signal,
            'horizon': horizon,
            'warmup': n_steps_warmup,
        }

    return {
        'per_module': results,
        'all_have_signal': all(r['outcome'] in ['PASS', 'WEAK_DIVERGENCE']
                               for r in results.values()),
        'note': (
            "F3b per module under coupling. δψ_M perturbation propagates "
            "through coupling. RNG states are synchronised between reference "
            "and perturbed branches so divergence is purely due to perturbation."
        ),
    }


# ============================================================================
# Final mass-drift aggregator
# ============================================================================

def aggregate_mass_drifts(history: list[ThreeModuleSystem]) -> dict:
    """Compute final mass per module + total, classify."""
    final = history[-1]
    per_module = {
        'A': float(final.state_A.psi.sum()),
        'B': float(final.state_B.psi.sum()),
        'C': float(final.state_C.psi.sum()),
    }
    total = sum(per_module.values())
    expected_total = 3.0  # three independent modules each with mass 1
    drift_per_module = {n: abs(1.0 - m) for n, m in per_module.items()}
    drift_total = abs(expected_total - total)

    if max(drift_per_module.values()) < 1e-4:
        classification = 'ACCEPTABLE'
        interpretable = True
    elif max(drift_per_module.values()) < 1e-3:
        classification = 'NUMERICAL_WARNING'
        interpretable = True
    else:
        classification = 'NUMERICAL_INVALID'
        interpretable = False

    return {
        'mass_per_module': per_module,
        'mass_total': total,
        'mass_drift_per_module': drift_per_module,
        'mass_drift_total': drift_total,
        'classification': classification,
        'interpretable': interpretable,
    }


# ============================================================================
# Verdict assembly + assertion
# ============================================================================

def _make_json_safe(obj):
    from enum import Enum
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
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
    return obj


MANDATORY_VERDICT_FIELDS = [
    'phase_status',
    'preflight_status',
    'seeds',
    'mass_drift_per_module',
    'mass_drift_total',
    'phase',
    'global_status',
]


def assert_mandatory_fields(verdict: dict):
    """
    Verify every mandatory field is present in the verdict before saving.
    Raises AssertionError if any is missing.
    """
    missing = [f for f in MANDATORY_VERDICT_FIELDS if f not in verdict]
    assert not missing, (
        f"Verdict 6b is missing mandatory fields: {missing}. "
        f"Refusing to save incomplete verdict."
    )

    # Additional structural checks
    seeds = verdict['seeds']
    assert seeds.get('rngs_independent') is True, (
        "Verdict claims rngs_independent=False — this should never reach save. "
        "Investigate."
    )
    for n in ['seed_A', 'seed_B', 'seed_C', 'base_seed']:
        assert n in seeds, f"seeds field missing: {n}"

    drifts = verdict['mass_drift_per_module']
    for m in ['A', 'B', 'C']:
        assert m in drifts, f"mass_drift_per_module missing module {m}"


def save_verdict(verdict: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _make_json_safe(verdict)
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)


# ============================================================================
# Main runner
# ============================================================================

def run_phase_6b():
    print("#" * 70)
    print("# PHASE 6b — three-module factorial perturbative coupling")
    print("# Status: NON-PERSPECTIVAL baseline. Not MCQ^N native validation.")
    print("#" * 70)
    print()

    # ----- 1. Preflight suite -----
    print("=" * 70)
    print("STEP 1 — Preflight suite")
    print("=" * 70)
    preflight = run_preflight_suite(verbose=True)
    print()
    print(f"Preflight status: {preflight['status']}")

    if preflight['status'] != 'OK':
        print()
        print("❌ Preflight suite did NOT pass. F1-F6 are NOT interpreted.")
        print(f"   Blocking failures: {preflight['blocking_failures']}")
        if preflight['warnings']:
            print(f"   Warnings: {preflight['warnings']}")

        # Save a partial verdict reflecting the preflight failure
        verdict = {
            'phase': '6b',
            'phase_status': "Phase 6b preflight FAILED — verdict NOT interpretable.",
            'preflight_status': preflight['status'],
            'preflight_blocking_failures': preflight['blocking_failures'],
            'preflight_warnings': preflight['warnings'],
            'global_status': preflight['status'],
            'seeds': {
                'base_seed': 42, 'seed_A': 143, 'seed_B': 244, 'seed_C': 345,
                'rngs_independent': False,
            },
            'mass_drift_per_module': {'A': None, 'B': None, 'C': None},
            'mass_drift_total': None,
            'F1': None, 'F2_bis': None, 'F2_prime_bis': None,
            'F3a_per_module': None, 'F3b_per_module': None,
            'F4': None, 'F5': None, 'F6': None,
        }
        out = Path("/home/claude/mcq_v4/results/phase6b/verdict_phase6b.json")
        save_verdict(verdict, out)
        print(f"   Partial verdict saved to {out}")
        return verdict

    print("✓ Preflight passed. Proceeding to F1-F6.")
    print()

    # ----- 2. Setup -----
    base_seed = 42
    cfg_engine_kwargs = dict(
        dt=0.05, T_steps=300, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    cfg_engine = FactorialEngineConfig(**cfg_engine_kwargs)
    coupling_cfg_default = CouplingConfig(epsilon=0.005, coupling_form='contrastive')
    initial_psi = make_initial_psi()

    # ----- 3. F1 trajectorial interference -----
    print("=" * 70)
    print("STEP 2 — F1 trajectorial interference (3-branch)")
    print("=" * 70)
    f1 = m6b.metric_F1_trajectorial(
        cfg_engine, coupling_cfg_default,
        DIFFERENTIATED_WEIGHTS, initial_psi,
        base_seed=base_seed, n_steps=200,
    )
    print(f"  outcome: {f1['outcome']} ({f1['signal_quality']})")
    print(f"  rel_diff_pure (effect of coupling): {f1['rel_diff_pure']:.6f}")
    print(f"  rel_diff_setup (system isolation):  {f1['rel_diff_setup']:.6e}")
    print(f"  ratio diagnostics: {f1['logs_coupled_extra_diagnostics']}")
    print()

    # ----- 4. F2-bis per module -----
    print("=" * 70)
    print("STEP 3 — F2-bis per module (tripartition under coupling)")
    print("=" * 70)
    f2_bis = metric_F2_bis_per_module(
        cfg_engine, coupling_cfg_default, initial_psi,
        base_seed=base_seed, n_steps=200,
    )
    for name, r in f2_bis['per_module'].items():
        print(f"  {name}: {r['outcome']} (growth_M={r['growth_M']:.3f}, "
              f"growth_I={r['growth_I']:.3f})")
    print()

    # ----- 5. F2'-bis per module -----
    print("=" * 70)
    print("STEP 4 — F2'-bis per module (h_M frozen under coupling)")
    print("=" * 70)
    f2_prime_bis = metric_F2_prime_bis_per_module(
        cfg_engine, coupling_cfg_default, initial_psi,
        base_seed=base_seed, n_steps=200,
    )
    for name, r in f2_prime_bis['per_module'].items():
        print(f"  {name}: {r['outcome']} (rel_diff={r['rel_diff_max']:.4f})")
    print()

    # ----- 6. F3a / F3b per module -----
    print("=" * 70)
    print("STEP 5 — F3a / F3b per module under coupling")
    print("=" * 70)
    f3a = metric_F3a_per_module(cfg_engine, coupling_cfg_default, initial_psi,
                                base_seed=base_seed, n_steps=200)
    for name, r in f3a['per_module'].items():
        print(f"  F3a {name}: {r['outcome']} (max_card={r['max_cardinality']})")

    f3b = metric_F3b_per_module(cfg_engine, coupling_cfg_default, initial_psi,
                                base_seed=base_seed, n_steps_warmup=100, horizon=50)
    for name, r in f3b['per_module'].items():
        print(f"  F3b {name}: {r['outcome']} (div_var_M={r['div_var_M']:.4e})")
    print()

    # ----- 7. F4 modular differentiation -----
    print("=" * 70)
    print("STEP 6 — F4 modular differentiation (priority: h_a)")
    print("=" * 70)
    sys_for_f4 = build_three_module_system(
        cfg_engine, coupling_cfg_default, DIFFERENTIATED_WEIGHTS, initial_psi,
        base_seed=base_seed,
    )
    history_f4, logs_f4 = run_three_modules(sys_for_f4, n_steps=300,
                                            coupling_active=True)
    f4 = m6b.metric_F4_modular_differentiation(history_f4, logs_f4)
    print(f"  outcome: {f4['outcome']}")
    print(f"  metric_divergence_mean: {f4['metric_divergence_mean']:.4f}")
    print(f"  R_psi means: {f4['R_psi_means']}")
    print(f"  R_tau means: {f4['R_tau_means']}")
    print()

    # Aggregate mass drift from F4 run
    mass_agg = aggregate_mass_drifts(history_f4)
    print(f"  Mass drift per module: {mass_agg['mass_drift_per_module']}")
    print(f"  Mass drift total: {mass_agg['mass_drift_total']:.2e} ({mass_agg['classification']})")
    print()

    # ----- 8. F5 weight ablation -----
    print("=" * 70)
    print("STEP 7 — F5 weight ablation")
    print("=" * 70)
    f5 = m6b.metric_F5_weight_ablation(
        cfg_engine, coupling_cfg_default, initial_psi,
        base_seed=base_seed, n_steps=300,
    )
    print(f"  outcome: {f5['outcome']}")
    print(f"  differentiated metric_div: {f5['differentiated']['metric_divergence_mean']:.4f}")
    print(f"  identical metric_div:      {f5['identical']['metric_divergence_mean']:.4f}")
    print(f"  ratio: {f5['ratio_id_to_diff']:.4f}")
    print()

    # ----- 9. F6 ε sweep (PRIMARY RESULT) -----
    print("=" * 70)
    print("STEP 8 — F6 ε sweep (regime map — PRIMARY RESULT)")
    print("=" * 70)
    f6 = m6b.metric_F6_epsilon_sweep(
        cfg_engine, initial_psi, weights_dict=DIFFERENTIATED_WEIGHTS,
        base_seed=base_seed, n_steps=200,
        epsilons=[0.001, 0.005, 0.01, 0.05, 0.1],
        coupling_forms=['contrastive', 'positive'],
    )
    for form, by_eps in f6['sweep'].items():
        print(f"\n  Form: {form}")
        print(f"  {'eps':>8} {'max_ratio':>12} {'regime':>14} "
              f"{'F4_outcome':>32} {'metric_div':>12} {'mass_drift_max':>14}")
        for k, v in by_eps.items():
            print(f"  {v['epsilon']:>8.4f} "
                  f"{v['ratio_diagnostics']['max_ratio']:>12.4f} "
                  f"{v['ratio_diagnostics']['regime']:>14} "
                  f"{v['F4_outcome']:>32} "
                  f"{v['F4_metric_divergence']:>12.4f} "
                  f"{v['mass_drift_max']:>14.2e}")
    print()

    # ----- 10. Synthesis -----
    print("=" * 70)
    print("STEP 9 — Synthesis")
    print("=" * 70)
    synthesis = m6b.synthesize_phase_6b(f1, f4, f5, f6)
    print(f"  Overall verdict: {synthesis['overall_verdict']}")
    print(f"  Differentiating coupling confirmed: {synthesis['differentiating_coupling_confirmed']}")
    print(f"  F6 regime summary: {synthesis['F6_regime_summary']}")
    print()

    # ----- 11. Verdict assembly -----
    seeds_log = {
        'base_seed': base_seed,
        'seed_A': base_seed + 101,
        'seed_B': base_seed + 202,
        'seed_C': base_seed + 303,
        'derivation': 'base_seed + (101, 202, 303)',
        'rngs_independent': True,
    }

    verdict = {
        'phase': '6b',
        'phase_status': (
            "Phase 6b — factorial perturbative coupling baseline established. "
            "NON-PERSPECTIVAL by construction. Not MCQ^N native validation. "
            "Phase 6c will introduce perspective-dependent native coupling."
        ),
        'preflight_status': preflight['status'],
        'preflight_summary': {
            k: r.get('pass', False)
            for k, r in preflight['results'].items()
        },
        'preflight_warnings': preflight['warnings'],
        'global_status': 'INTERPRETABLE' if mass_agg['interpretable'] else 'NUMERICAL_INVALID',
        'engine_config': cfg_engine_kwargs,
        'coupling_config': {
            'epsilon_default': coupling_cfg_default.epsilon,
            'coupling_form': coupling_cfg_default.coupling_form,
        },
        'weights': DIFFERENTIATED_WEIGHTS,
        'seeds': seeds_log,
        'mass_drift_per_module': mass_agg['mass_drift_per_module'],
        'mass_drift_total': mass_agg['mass_drift_total'],
        'mass_classification': mass_agg['classification'],
        'F1': f1,
        'F2_bis': f2_bis,
        'F2_prime_bis': f2_prime_bis,
        'F3a_per_module': f3a,
        'F3b_per_module': f3b,
        'F4': f4,
        'F5': f5,
        'F6': f6,
        'synthesis': synthesis,
    }

    # ----- 12. Assert mandatory fields, then save -----
    assert_mandatory_fields(verdict)
    out = Path("/home/claude/mcq_v4/results/phase6b/verdict_phase6b.json")
    save_verdict(verdict, out)
    print(f"Verdict saved to: {out}")

    return verdict


if __name__ == "__main__":
    verdict = run_phase_6b()
    print()
    print("#" * 70)
    print(f"# Phase 6b — global status: {verdict['global_status']}")
    print(f"# Synthesis verdict: {verdict.get('synthesis', {}).get('overall_verdict', 'N/A')}")
    print("#" * 70)
