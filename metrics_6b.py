"""
Phase 6b metrics — F1 (trajectorial interference), F4 (modular
differentiation via metrics), F5 (ablation weights), F6 (ε sweep map).

Conventions
-----------
- All thresholds marked INITIAL_CALIBRATION
- Each metric reports `signal_quality` (CLEAN/NOISY/ABSENT)
- Differentiation is measured PRIORITARILY on h_a divergence (F4),
  not on R_psi alone. R_psi and R_tau are secondary diagnostics.
- The verdict DIFFERENTIATING_INTERFERENCE_OBSERVED reflects the
  anti-fusion construction of the MCQ coupling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mcq_v4.factorial.state import FactorialState, FactorialEngineConfig, ModuleConfig
from mcq_v4.factorial.engine import FactorialEngine, make_initial_state
from mcq_v4.factorial.three_module_system import (
    ThreeModuleSystem, CouplingConfig, build_three_module_system,
    DIFFERENTIATED_WEIGHTS, IDENTICAL_WEIGHTS,
)
from mcq_v4.factorial.coupling import run_three_modules
from mcq_v4.factorial.tau_prime import (
    compute_tau_prime_3modules, compute_modular_contributions,
)
from mcq_v4.factorial.overlaps import compute_pairwise_R_psi


THRESHOLDS_6B = {
    'F1_PASS': 0.05,
    'F1_WEAK': 0.01,
    'F4_metric_diff_PASS': 0.30,
    'F4_metric_diff_WEAK': 0.10,
    'F5_ratio_threshold': 0.5,
    'F6_extra_to_intra_perturbative': 0.1,
    'F6_extra_to_intra_overcoupled': 0.5,
}


# ============================================================================
# Ratio diagnostics across a run
# ============================================================================

def compute_ratio_diagnostics(logs: list[dict]) -> dict:
    """
    Aggregate the extra/lambda_KNV ratio across all steps and modules.

    We use ratio_to_lambda_KNV as the primary stability indicator because
    Phi_intra can be near zero in the viable corridor (var ∈ [var_min, var_max]),
    which would inflate ratio_to_intra spuriously. lambda_KNV is the
    regulation amplitude scale and is constant throughout the run.

    Regime classification:
      BUFFERED      : max ratio < 0.1 — extra is small relative to regulation scale
      PERTURBATIVE  : 0.1 ≤ max ratio < 0.5
      OVERCOUPLED   : max ratio ≥ 0.5
    """
    all_ratios = []
    all_intra_ratios = []
    for log in logs:
        for name in ['A', 'B', 'C']:
            d = log['extra_phi_diagnostics'][name]
            all_ratios.append(d['ratio_extra_to_lambda_KNV'])
            all_intra_ratios.append(d['ratio_extra_to_intra'])

    arr = np.array(all_ratios)
    arr_intra = np.array(all_intra_ratios)
    if arr.size == 0:
        return {'max_ratio': 0.0, 'mean_ratio': 0.0,
                'fraction_above_0.1': 0.0, 'fraction_above_0.5': 0.0,
                'regime': 'BUFFERED',
                'max_ratio_to_intra': 0.0,
                'note': 'no logs',}

    max_r = float(arr.max())
    return {
        'max_ratio': max_r,
        'mean_ratio': float(arr.mean()),
        'fraction_above_0.1': float((arr > 0.1).mean()),
        'fraction_above_0.5': float((arr > 0.5).mean()),
        'regime': (
            'BUFFERED' if max_r < THRESHOLDS_6B['F6_extra_to_intra_perturbative']
            else 'PERTURBATIVE' if max_r < THRESHOLDS_6B['F6_extra_to_intra_overcoupled']
            else 'OVERCOUPLED'
        ),
        # Secondary diagnostic: ratio to intra (less stable but informative)
        'max_ratio_to_intra': float(arr_intra.max()),
        'mean_ratio_to_intra': float(arr_intra.mean()),
        'note': (
            "Primary regime classification uses ratio extra/lambda_KNV "
            "(stable). Secondary ratio_to_intra is logged but not used for "
            "regime classification because Phi_intra can be near zero in the "
            "viable corridor."
        ),
    }


# ============================================================================
# F1 — Trajectorial interference (3 branches)
# ============================================================================

def metric_F1_trajectorial(
    cfg_engine: FactorialEngineConfig,
    coupling_cfg: CouplingConfig,
    weights_dict: dict,
    initial_psi: dict,
    base_seed: int = 42,
    n_steps: int = 200,
) -> dict:
    """
    Three-branch trajectorial interference test on shared factor k_AB.
    """
    seed_offsets = {'A': 101, 'B': 202}

    # Branch 1: solo separated
    histories_solo_sep = {}
    for name in ['A', 'B']:
        mcfg = ModuleConfig(name=name, weights=weights_dict[name],
                            seed=base_seed + seed_offsets[name])
        engine = FactorialEngine(cfg_engine, mcfg)
        state = make_initial_state(initial_psi[name], mcfg)
        history = [state]
        for t in range(n_steps):
            state, _ = engine.step(state)
            history.append(state)
        histories_solo_sep[name] = history

    contribs_solo_sep_T = {
        name: np.array([compute_modular_contributions(s)['T']
                        for s in histories_solo_sep[name]])
        for name in ['A', 'B']
    }
    tau_AB_solo_sep = contribs_solo_sep_T['A'] + contribs_solo_sep_T['B']

    # Branch 2: solo parallel (3-module system, ε=0)
    coupling_cfg_zero = CouplingConfig(epsilon=0.0,
                                       coupling_form=coupling_cfg.coupling_form)
    sys_par = build_three_module_system(
        cfg_engine, coupling_cfg_zero, weights_dict, initial_psi,
        base_seed=base_seed,
    )
    history_par, _ = run_three_modules(sys_par, n_steps=n_steps,
                                       coupling_active=False)
    contribs_par_T = {
        'A': np.array([compute_modular_contributions(s.state_A)['T']
                       for s in history_par]),
        'B': np.array([compute_modular_contributions(s.state_B)['T']
                       for s in history_par]),
    }
    tau_AB_par = contribs_par_T['A'] + contribs_par_T['B']

    # Branch 3: coupled
    sys_coupled = build_three_module_system(
        cfg_engine, coupling_cfg, weights_dict, initial_psi,
        base_seed=base_seed,
    )
    history_coupled, logs_coupled = run_three_modules(
        sys_coupled, n_steps=n_steps, coupling_active=True,
    )
    contribs_coupled_T = {
        'A': np.array([compute_modular_contributions(s.state_A)['T']
                       for s in history_coupled]),
        'B': np.array([compute_modular_contributions(s.state_B)['T']
                       for s in history_coupled]),
    }
    tau_AB_coupled = contribs_coupled_T['A'] + contribs_coupled_T['B']

    dt = cfg_engine.dt
    L1_pure = float(np.abs(tau_AB_coupled - tau_AB_par).sum() * dt)
    L1_setup = float(np.abs(tau_AB_par - tau_AB_solo_sep).sum() * dt)
    amp = max(float(tau_AB_par.max() - tau_AB_par.min()), 1e-12)
    T_total = dt * n_steps
    rel_pure = L1_pure / max(amp * T_total, 1e-12)
    rel_setup = L1_setup / max(amp * T_total, 1e-12)

    diff_pure = np.abs(tau_AB_coupled - tau_AB_par)
    if diff_pure.max() < 1e-9:
        signal_quality = 'ABSENT'
    elif diff_pure.max() < 1e-5:
        signal_quality = 'NOISY'
    else:
        signal_quality = 'CLEAN'

    if signal_quality == 'ABSENT':
        outcome = 'ADDITIVE'
    elif rel_pure >= THRESHOLDS_6B['F1_PASS']:
        outcome = 'DIFFERENTIATING_INTERFERENCE_OBSERVED'
    elif rel_pure >= THRESHOLDS_6B['F1_WEAK']:
        outcome = 'WEAK_INTERFERENCE'
    else:
        outcome = 'ADDITIVE'

    return {
        'tau_AB_solo_separated_final': float(tau_AB_solo_sep[-1]),
        'tau_AB_solo_parallel_final': float(tau_AB_par[-1]),
        'tau_AB_coupled_final': float(tau_AB_coupled[-1]),
        'L1_diff_pure': L1_pure,
        'L1_diff_setup': L1_setup,
        'rel_diff_pure': rel_pure,
        'rel_diff_setup': rel_setup,
        'amp_reference': amp,
        'signal_quality': signal_quality,
        'outcome': outcome,
        'thresholds': {'PASS': THRESHOLDS_6B['F1_PASS'],
                       'WEAK': THRESHOLDS_6B['F1_WEAK']},
        'thresholds_status': 'INITIAL_CALIBRATION',
        'logs_coupled_extra_diagnostics': compute_ratio_diagnostics(logs_coupled),
        'note': (
            "L1_diff_pure measures the effect of coupling proper. "
            "L1_diff_setup should be very small if isolation is clean. "
            "DIFFERENTIATING_INTERFERENCE_OBSERVED is the anti-fusion verdict: "
            "the coupling converts inter-modular novelty into a gradient of "
            "differentiation (cf. coupling sign micro-test = COUPLING_REPULSIVE)."
        ),
    }


# ============================================================================
# F4 — Modular differentiation
# ============================================================================

def compute_metric_divergence(states: dict[str, FactorialState]) -> dict:
    pairs = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    axes = ['T', 'M', 'I']
    divs = {}
    for (i, j) in pairs:
        per_axis = {}
        for a in axes:
            h_i = getattr(states[i], f'h_{a}')
            h_j = getattr(states[j], f'h_{a}')
            num = float(np.abs(h_i - h_j).mean())
            den = float(np.maximum(np.abs(h_i).mean(), np.abs(h_j).mean()) + 1e-12)
            per_axis[a] = num / den
        divs[f'{i}-{j}'] = {**per_axis,
                            'mean': float(np.mean(list(per_axis.values())))}
    overall = float(np.mean([v['mean'] for v in divs.values()]))
    return {'pairs': divs, 'overall_mean': overall}


def metric_F4_modular_differentiation(
    history: list[ThreeModuleSystem],
    logs: list[dict],
    second_half_fraction: float = 0.5,
) -> dict:
    """
    Modular differentiation, evaluated PRIORITARILY on metrics h_a, with
    R_psi and R_tau as secondary diagnostics.

    Reading: ψ similar + h different = real morphodynamic differentiation.
    """
    n = len(history)
    start = int(n * (1 - second_half_fraction))
    second_half = history[start:]
    second_half_logs = logs[start - 1:] if start > 0 else logs

    metric_divs = [compute_metric_divergence(sys.states) for sys in second_half]
    metric_div_means = [d['overall_mean'] for d in metric_divs]
    metric_div_mean = float(np.mean(metric_div_means))
    metric_div_std = float(np.std(metric_div_means))

    R_psi_AB = [log['R_psi'][('A', 'B')] for log in second_half_logs]
    R_psi_BC = [log['R_psi'][('B', 'C')] for log in second_half_logs]
    R_psi_AC = [log['R_psi'][('A', 'C')] for log in second_half_logs]
    R_tau_AB = [log['R_tau'][('A', 'B')] for log in second_half_logs]
    R_tau_BC = [log['R_tau'][('B', 'C')] for log in second_half_logs]
    R_tau_AC = [log['R_tau'][('A', 'C')] for log in second_half_logs]

    if metric_div_mean >= THRESHOLDS_6B['F4_metric_diff_PASS']:
        outcome = 'PASS_METRIC_DIFFERENTIATION'
    elif metric_div_mean >= THRESHOLDS_6B['F4_metric_diff_WEAK']:
        outcome = 'WEAK_METRIC_DIFFERENTIATION'
    else:
        outcome = 'FUSION_OR_NEAR_FUSION'

    return {
        'metric_divergence_mean': metric_div_mean,
        'metric_divergence_std': metric_div_std,
        'metric_divergence_per_pair_final': metric_divs[-1]['pairs'],
        'R_psi_means': {
            'A-B': float(np.mean(R_psi_AB)),
            'B-C': float(np.mean(R_psi_BC)),
            'A-C': float(np.mean(R_psi_AC)),
        },
        'R_tau_means': {
            'A-B': float(np.mean(R_tau_AB)),
            'B-C': float(np.mean(R_tau_BC)),
            'A-C': float(np.mean(R_tau_AC)),
        },
        'outcome': outcome,
        'thresholds': {
            'PASS': THRESHOLDS_6B['F4_metric_diff_PASS'],
            'WEAK': THRESHOLDS_6B['F4_metric_diff_WEAK'],
        },
        'thresholds_status': 'INITIAL_CALIBRATION',
        'note': (
            "Differentiation measured PRIORITARILY on h_a divergence "
            "(metric profiles), not R_psi alone. ψ similar + h different "
            "= real morphodynamic differentiation. R_psi/R_tau are secondary."
        ),
    }


# ============================================================================
# F5 — Weight ablation
# ============================================================================

def metric_F5_weight_ablation(
    cfg_engine: FactorialEngineConfig,
    coupling_cfg: CouplingConfig,
    initial_psi: dict,
    base_seed: int = 42,
    n_steps: int = 500,
) -> dict:
    """
    Differentiated weights vs identical weights ablation.

    Identical weights should SIGNIFICANTLY REDUCE differentiation relative
    to differentiated weights (not necessarily produce total fusion).
    """
    sys_diff = build_three_module_system(
        cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS, initial_psi,
        base_seed=base_seed,
    )
    history_diff, logs_diff = run_three_modules(sys_diff, n_steps=n_steps,
                                                coupling_active=True)
    f4_diff = metric_F4_modular_differentiation(history_diff, logs_diff)

    sys_id = build_three_module_system(
        cfg_engine, coupling_cfg, IDENTICAL_WEIGHTS, initial_psi,
        base_seed=base_seed,
    )
    history_id, logs_id = run_three_modules(sys_id, n_steps=n_steps,
                                            coupling_active=True)
    f4_id = metric_F4_modular_differentiation(history_id, logs_id)

    diff_d = f4_diff['metric_divergence_mean']
    diff_i = f4_id['metric_divergence_mean']
    ratio = diff_i / max(diff_d, 1e-12)

    if ratio < THRESHOLDS_6B['F5_ratio_threshold']:
        outcome = 'WEIGHTS_DRIVE_DIFFERENTIATION'
    else:
        outcome = 'WEIGHTS_INCONCLUSIVE'

    return {
        'differentiated': {
            'metric_divergence_mean': diff_d,
            'outcome_F4': f4_diff['outcome'],
            'R_psi_means': f4_diff['R_psi_means'],
        },
        'identical': {
            'metric_divergence_mean': diff_i,
            'outcome_F4': f4_id['outcome'],
            'R_psi_means': f4_id['R_psi_means'],
        },
        'ratio_id_to_diff': ratio,
        'outcome': outcome,
        'threshold': THRESHOLDS_6B['F5_ratio_threshold'],
        'thresholds_status': 'INITIAL_CALIBRATION',
        'note': (
            "Identical weights should reduce differentiation significantly "
            "vs differentiated weights. Ratio < 0.5 → weights are the "
            "primary driver. Total fusion (ratio=0) NOT required."
        ),
    }


# ============================================================================
# F6 — ε sweep
# ============================================================================

def metric_F6_epsilon_sweep(
    cfg_engine: FactorialEngineConfig,
    initial_psi: dict,
    weights_dict: Optional[dict] = None,
    base_seed: int = 42,
    n_steps: int = 300,
    epsilons: Optional[list] = None,
    coupling_forms: Optional[list] = None,
) -> dict:
    """
    Sweep coupling strength ε across regimes. Output: regime map.

    The result of Phase 6b is the MAP, not a chosen ε.
    """
    if weights_dict is None:
        weights_dict = DIFFERENTIATED_WEIGHTS
    if epsilons is None:
        epsilons = [0.001, 0.005, 0.01, 0.05]
    if coupling_forms is None:
        coupling_forms = ['contrastive', 'positive']

    sweep = {}
    for form in coupling_forms:
        sweep[form] = {}
        for eps in epsilons:
            coupling_cfg = CouplingConfig(epsilon=eps, coupling_form=form)
            sys = build_three_module_system(
                cfg_engine, coupling_cfg, weights_dict, initial_psi,
                base_seed=base_seed,
            )
            history, logs = run_three_modules(sys, n_steps=n_steps,
                                              coupling_active=(eps > 0))
            ratio_diag = compute_ratio_diagnostics(logs)
            f4 = metric_F4_modular_differentiation(history, logs)

            drift_per_module = {
                'A': float(abs(1.0 - history[-1].state_A.psi.sum())),
                'B': float(abs(1.0 - history[-1].state_B.psi.sum())),
                'C': float(abs(1.0 - history[-1].state_C.psi.sum())),
            }

            sweep[form][f'eps_{eps:.4f}'] = {
                'epsilon': eps,
                'coupling_form': form,
                'ratio_diagnostics': ratio_diag,
                'F4_outcome': f4['outcome'],
                'F4_metric_divergence': f4['metric_divergence_mean'],
                'F4_R_psi_means': f4['R_psi_means'],
                'F4_R_tau_means': f4['R_tau_means'],
                'mass_drift': drift_per_module,
                'mass_drift_max': max(drift_per_module.values()),
            }

    return {
        'sweep': sweep,
        'epsilons_tested': epsilons,
        'coupling_forms_tested': coupling_forms,
        'note': (
            "The ε sweep is the PRIMARY result of Phase 6b. No single ε "
            "is chosen as 'the right value' — the map of regimes IS the "
            "verdict. An ε_working can be reported for figures only, "
            "in the perturbative regime."
        ),
    }


# ============================================================================
# Phase 6b synthesis
# ============================================================================

def synthesize_phase_6b(
    f1_result: dict,
    f4_result: dict,
    f5_result: dict,
    f6_result: dict,
) -> dict:
    f1 = f1_result.get('outcome', 'UNKNOWN')
    f4 = f4_result.get('outcome', 'UNKNOWN')
    f5 = f5_result.get('outcome', 'UNKNOWN')

    sweep_summary = {}
    for form, by_eps in f6_result['sweep'].items():
        regimes = [v['ratio_diagnostics']['regime'] for v in by_eps.values()]
        sweep_summary[form] = {
            'BUFFERED': regimes.count('BUFFERED'),
            'PERTURBATIVE': regimes.count('PERTURBATIVE'),
            'OVERCOUPLED': regimes.count('OVERCOUPLED'),
        }

    if (f1 == 'DIFFERENTIATING_INTERFERENCE_OBSERVED'
            and f4 == 'PASS_METRIC_DIFFERENTIATION'
            and f5 == 'WEIGHTS_DRIVE_DIFFERENTIATION'):
        verdict = 'DIFFERENTIATING_INTERFERENCE_OBSERVED'
    elif f1 == 'ADDITIVE':
        verdict = 'ADDITIVE_NO_INTERFERENCE'
    elif f4 == 'FUSION_OR_NEAR_FUSION':
        verdict = 'FUSION_DOMINATES'
    else:
        verdict = 'MIXED_SEE_DETAILS'

    return {
        'phase_status': (
            "Phase 6b — factorial perturbative coupling baseline established. "
            "NON-PERSPECTIVAL by construction. Not MCQ^N native validation."
        ),
        'F1': f1, 'F4': f4, 'F5': f5,
        'F6_regime_summary': sweep_summary,
        'overall_verdict': verdict,
        'differentiating_coupling_confirmed': (
            f4 in ['PASS_METRIC_DIFFERENTIATION', 'WEAK_METRIC_DIFFERENTIATION']
            and f5 == 'WEIGHTS_DRIVE_DIFFERENTIATION'
        ),
    }
