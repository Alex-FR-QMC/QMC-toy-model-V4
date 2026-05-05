"""
Phase 6c-B — F6.T25' Morphodynamic constrained contraction (MCQ-aligned).

Replaces F6.T25 (brute swap version) which conflated KNV violations
with numerical instability and never reached the regime
`var_max < var_init AND stable`.

T25' protocol (Alex's spec):

Phase 1 — Warmup
    System 3 modules stabilised, coupling OFF, n=100 steps.

Phase 2 — Progressive contraction ramp on M_A
    var_max(t) = max(var_init - α·(t-t₀), var_floor)
    α small (0.001–0.01 per step), var_floor ≥ 0.2
    No brutal swap → Γ″ numerical divergence avoided.

Phase 3 — MCQ monitoring (REQUIRED)
    For module A (and B, C for propagation):
      - 𝒢_proxy(A): from compute_G_modular (note: ψ-roughness proxy,
                     not full 𝒢 = ‖∂τ′/∂Γ_meta‖)
      - Δ(A): from compute_delta_modular
      - Γ″(A) proxy: second time derivative of contrib_T (not full
                     ‖Γ″‖ on Γ_meta — proxy)
      - var_M(A): the contraction observable

Classification (5 categories — Alex's spec):

    LOCAL_CONTRACTION_VIABLE
        var_M(A) ↓ significantly
        AND 𝒢_proxy(A) > G_min (no diff. KNV)
        AND Δ(A) < Δ_crit (no top. KNV)
        AND Γ″ proxy bounded and damped (no morph. KNV)
        AND propagation to B,C below threshold

    PROPAGATED_CONTRACTION
        var_M(A,B,C) ↓ together AND MCQ conditions met

    STRUCTURAL_RESISTANCE      ← KEY new category
        var_M(A) does not decrease despite progressive constraint
        AND no KNV violation
        Reading: system actively resists contraction without breaking 𝒱
        This is NOT failure — it is a viable steady response.

    KNV_COLLAPSE
        Genuine KNV violation: 𝒢_proxy → 0 OR Δ exceeds Δ_crit
        OR Γ″ proxy unbounded (non-dampable)

    NUMERICAL_INSTABILITY      ← separate from KNV
        NaN / overflow / numerical divergence
        Distinct from morphogenic non-damping. Engine breakdown,
        not an MCQ-meaningful regime.

References (cf. /mnt/project/QMC_Chap1.pdf, Executive Summary):
    §1.5 — three KNV foundations: differential 𝒢→0, topological Δ
           outside corridor, morphogenic Γ″ non-dampable.
    §1.7 — 𝓔_QMC = 𝒱 ∩ 𝒟_QMC: viability is necessary but not sufficient;
           dynamic existence requires also Var(Δφ_{ij}) > 0,
           0 < Ξ_obs < Ξ_crit, ℒ < ℒ_crit.
    §1.4 — RTS regime: dτ'/dt ≈ 0 AND Var(τ') > 0 AND Γ' ≠ 0 —
           the signature of viable stability under constraint
           (relevant for STRUCTURAL_RESISTANCE).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    FactorialEngineConfig, EngineMode, FactorialState, THETA_T,
)
from mcq_v4.factorial.three_module_system import (
    DIFFERENTIATED_WEIGHTS, CouplingConfig, build_three_module_system,
)
from mcq_v4.factorial.coupling import step_three_modules
from mcq_v4.factorial.observables import compute_observables
from mcq_v4.factorial.tau_prime import compute_modular_contributions
from mcq_v4.factorial.signatures_6c_b import (
    compute_G_modular, compute_delta_modular, compute_gamma_double_prime,
    KNV_THRESHOLDS_6C_B,
)


def make_initial_psi():
    def init_T():
        p = np.exp(-(THETA_T**2) / 2.0); p /= p.sum()
        psi = np.zeros((5, 5, 5)); psi[:, 2, 2] = p; return psi
    def init_M():
        p = np.exp(-((np.arange(5) - 2) ** 2) / 2.0); p /= p.sum()
        psi = np.zeros((5, 5, 5)); psi[2, :, 2] = p; return psi
    def init_I():
        p = np.exp(-(THETA_T ** 2) / 2.0); p /= p.sum()
        psi = np.zeros((5, 5, 5)); psi[2, 2, :] = p; return psi
    return {'A': init_T(), 'B': init_M(), 'C': init_I()}


def clone_system(sys_obj):
    state_clones = {
        name: FactorialState(
            psi=getattr(sys_obj, f'state_{name}').psi.copy(),
            h_T=getattr(sys_obj, f'state_{name}').h_T.copy(),
            h_M=getattr(sys_obj, f'state_{name}').h_M.copy(),
            h_I=getattr(sys_obj, f'state_{name}').h_I.copy(),
            cfg=getattr(sys_obj, f'state_{name}').cfg,
        )
        for name in ['A', 'B', 'C']
    }
    rng_states = {
        name: copy.deepcopy(getattr(sys_obj, f'engine_{name}').rng.bit_generator.state)
        for name in ['A', 'B', 'C']
    }
    prev_h = copy.deepcopy(sys_obj.prev_h_fields) if sys_obj.prev_h_fields else None
    return state_clones, rng_states, prev_h


def make_system_from_clone(cfg_engine, coupling_cfg, state_clones,
                           rng_states, prev_h, base_seed):
    initial_psi = {n: state_clones[n].psi.copy() for n in ['A', 'B', 'C']}
    new_sys = build_three_module_system(
        cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )
    for name in ['A', 'B', 'C']:
        st = state_clones[name]
        new_st = FactorialState(
            psi=st.psi.copy(), h_T=st.h_T.copy(),
            h_M=st.h_M.copy(), h_I=st.h_I.copy(),
            cfg=st.cfg,
        )
        if name == 'A':
            new_sys.state_A = new_st
        elif name == 'B':
            new_sys.state_B = new_st
        else:
            new_sys.state_C = new_st
    for name in ['A', 'B', 'C']:
        getattr(new_sys, f'engine_{name}').rng.bit_generator.state = \
            copy.deepcopy(rng_states[name])
    new_sys.prev_h_fields = copy.deepcopy(prev_h) if prev_h else None
    return new_sys


def make_swapped_cfg(base_cfg: FactorialEngineConfig, var_max: float,
                     lambda_KNV: float) -> FactorialEngineConfig:
    return FactorialEngineConfig(
        dt=base_cfg.dt, T_steps=base_cfg.T_steps, mode=base_cfg.mode,
        D_0=base_cfg.D_0, D_min=base_cfg.D_min,
        beta_0=base_cfg.beta_0, gamma_0=base_cfg.gamma_0,
        h_0=base_cfg.h_0, h_min=base_cfg.h_min,
        sigma_eta=base_cfg.sigma_eta,
        var_min=base_cfg.var_min, var_max=var_max,
        H_min=base_cfg.H_min, lambda_KNV=lambda_KNV,
    )


def is_finite_state(state: FactorialState) -> bool:
    return (
        np.all(np.isfinite(state.psi))
        and np.all(np.isfinite(state.h_T))
        and np.all(np.isfinite(state.h_M))
        and np.all(np.isfinite(state.h_I))
    )


def measure_T25_prime_run(
    cfg_engine: FactorialEngineConfig,
    coupling_form: str,
    eps: float,
    base_seed: int,
    alpha: float,
    var_floor: float,
    lambda_KNV_target: float,
    warmup_steps: int = 100,
    contraction_steps: int = 200,
) -> dict:
    """
    One T25' run with progressive ramp.

    var_max(step) = max(var_init - alpha · step, var_floor)
    The ramp starts from base var_max at step 0 (post-warmup) and decreases
    linearly until reaching var_floor.
    """
    initial_psi = make_initial_psi()

    coupling_cfg_warmup = CouplingConfig(epsilon=0.0, coupling_form='contrastive')
    sys_warmup = build_three_module_system(
        cfg_engine, coupling_cfg_warmup, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )
    for _ in range(warmup_steps):
        sys_warmup, _ = step_three_modules(sys_warmup, coupling_active=False)

    state_clones, rng_states, prev_h = clone_system(sys_warmup)
    coupling_cfg_meas = CouplingConfig(epsilon=eps, coupling_form=coupling_form)
    sys_p = make_system_from_clone(cfg_engine, coupling_cfg_meas,
                                    state_clones, rng_states, prev_h, base_seed)

    var_M_init = {
        m: compute_observables(getattr(sys_p, f'state_{m}').psi).get('var_M', 0.0)
        for m in ['A', 'B', 'C']
    }
    # Ramp starts FROM var_init(A) (no expansion phase), not from cfg's
    # var_max. This ensures Φ_corr applies contraction pressure from t=0.
    var_max_ramp_start = max(var_M_init['A'], 0.05)

    G_traj = {m: [] for m in ['A', 'B', 'C']}
    delta_traj = {m: [] for m in ['A', 'B', 'C']}
    var_M_traj = {m: [] for m in ['A', 'B', 'C']}
    tau_T_traj = {m: [] for m in ['A', 'B', 'C']}
    var_max_applied_traj = []
    psi_finite_traj = []

    numerical_instability_step = None

    for step in range(contraction_steps):
        var_max_now = max(var_max_ramp_start - alpha * step, var_floor)
        var_max_applied_traj.append(var_max_now)
        # Apply ramped cfg only to engine_A (M_B, M_C nominal)
        sys_p.engine_A.cfg = make_swapped_cfg(
            cfg_engine, var_max=var_max_now, lambda_KNV=lambda_KNV_target,
        )

        try:
            sys_p, _ = step_three_modules(sys_p, coupling_active=True)
        except Exception as e:
            numerical_instability_step = step
            for m in ['A', 'B', 'C']:
                G_traj[m].append(np.nan)
                delta_traj[m].append(np.nan)
                var_M_traj[m].append(np.nan)
                tau_T_traj[m].append(np.nan)
            psi_finite_traj.append(False)
            break

        all_finite = all(
            is_finite_state(getattr(sys_p, f'state_{m}'))
            for m in ['A', 'B', 'C']
        )
        psi_finite_traj.append(all_finite)
        if not all_finite:
            numerical_instability_step = step
            for m in ['A', 'B', 'C']:
                st = getattr(sys_p, f'state_{m}')
                if is_finite_state(st):
                    G_traj[m].append(compute_G_modular(st)['G_proxy_total'])
                    delta_traj[m].append(compute_delta_modular(st)['delta'])
                    var_M_traj[m].append(compute_observables(st.psi).get('var_M', np.nan))
                    tau_T_traj[m].append(compute_modular_contributions(st)['T'])
                else:
                    G_traj[m].append(np.nan)
                    delta_traj[m].append(np.nan)
                    var_M_traj[m].append(np.nan)
                    tau_T_traj[m].append(np.nan)
            break

        for m in ['A', 'B', 'C']:
            st = getattr(sys_p, f'state_{m}')
            G_traj[m].append(compute_G_modular(st)['G_proxy_total'])
            delta_traj[m].append(compute_delta_modular(st)['delta'])
            var_M_traj[m].append(compute_observables(st.psi).get('var_M', 0.0))
            tau_T_traj[m].append(compute_modular_contributions(st)['T'])

    tau_A_arr = np.array(
        [v for v in tau_T_traj['A'] if not np.isnan(v)]
    ).reshape(-1, 1)
    if tau_A_arr.shape[0] >= 3:
        gamma_A = compute_gamma_double_prime(tau_A_arr, cfg_engine.dt).flatten()
        gamma_A_max = float(np.abs(gamma_A).max())
        # Damping criterion: Γ″ is non-dampable only if amplitude clearly
        # GROWS (second half max significantly larger than first half max).
        # A flat or modestly oscillating Γ″ is dampable (or already damped).
        # The threshold 1.5x is arbitrary but corresponds to >50% growth,
        # which is a clear non-damping signal.
        n = len(gamma_A)
        if n >= 4 and gamma_A_max > 1e-9:
            first_half_max = float(np.abs(gamma_A[:n // 2]).max())
            second_half_max = float(np.abs(gamma_A[n // 2:]).max())
            growth_ratio = second_half_max / max(first_half_max, 1e-12)
            gamma_A_dampable = (growth_ratio < 1.5)
            gamma_A_growth_ratio = growth_ratio
        else:
            gamma_A_dampable = True
            gamma_A_growth_ratio = 1.0
    else:
        gamma_A_max = 0.0
        gamma_A_dampable = True
        gamma_A_growth_ratio = 1.0

    var_M_final = {
        m: (var_M_traj[m][-1] if var_M_traj[m] and not np.isnan(var_M_traj[m][-1])
            else var_M_init[m])
        for m in ['A', 'B', 'C']
    }
    G_A_min = float(np.nanmin(G_traj['A'])) if G_traj['A'] else 0.0
    delta_A_max = float(np.nanmax(delta_traj['A'])) if delta_traj['A'] else 0.0

    G_diff_KNV = G_A_min < KNV_THRESHOLDS_6C_B['G_min']
    delta_top_KNV = delta_A_max > KNV_THRESHOLDS_6C_B['delta_crit']
    gamma_morph_KNV = not gamma_A_dampable

    drop_A = (var_M_init['A'] - var_M_final['A']) / max(abs(var_M_init['A']), 1e-12)
    drop_B = (var_M_init['B'] - var_M_final['B']) / max(abs(var_M_init['B']), 1e-12)
    drop_C = (var_M_init['C'] - var_M_final['C']) / max(abs(var_M_init['C']), 1e-12)

    if numerical_instability_step is not None:
        outcome = 'NUMERICAL_INSTABILITY'
    elif G_diff_KNV or delta_top_KNV or gamma_morph_KNV:
        outcome = 'KNV_COLLAPSE'
    elif drop_A < 0.05 and not (drop_B > 0.10 or drop_C > 0.10):
        outcome = 'STRUCTURAL_RESISTANCE'
    elif drop_A >= 0.30:
        if drop_B >= 0.30 or drop_C >= 0.30:
            outcome = 'PROPAGATED_CONTRACTION'
        else:
            outcome = 'LOCAL_CONTRACTION_VIABLE'
    else:
        outcome = 'PARTIAL_CONTRACTION_NON_PROPAGATED'

    return {
        'coupling_form': coupling_form,
        'epsilon': eps,
        'alpha': alpha,
        'var_floor': var_floor,
        'lambda_KNV_target': lambda_KNV_target,
        'var_M_init': var_M_init,
        'var_M_final': var_M_final,
        'drop_A': drop_A, 'drop_B': drop_B, 'drop_C': drop_C,
        'G_A_min': G_A_min,
        'G_A_first': float(G_traj['A'][0]) if G_traj['A'] else 0.0,
        'delta_A_max': delta_A_max,
        'gamma_A_max': gamma_A_max,
        'gamma_A_dampable': gamma_A_dampable,
        'gamma_A_growth_ratio': gamma_A_growth_ratio,
        'G_diff_KNV': G_diff_KNV,
        'delta_top_KNV': delta_top_KNV,
        'gamma_morph_KNV': gamma_morph_KNV,
        'numerical_instability_step': numerical_instability_step,
        'final_var_max_applied': var_max_applied_traj[-1] if var_max_applied_traj else var_max_ramp_start,
        'var_max_ramp_start': var_max_ramp_start,
        'outcome': outcome,
        'n_steps_completed': len([v for v in psi_finite_traj if v]),
    }


def measure_T25_prime_for_form(
    cfg_engine, coupling_form: str, eps: float, base_seed: int,
) -> dict:
    """For one form, sweep (alpha × var_floor × lambda) to find a viable corridor.

    Calibration: alpha is set so the ramp REACHES the floor within
    contraction_steps. Starting var_max = cfg_engine.var_max (e.g. 2.5),
    target var_floor (e.g. 0.30), ramp distance = 2.5 - 0.30 = 2.2.
    Over 200 steps, the rate must be ~0.011/step minimum to actually
    descend below var_init ≈ 0.49 in time. Slower rates are equivalent
    to "no ramp at all" relative to var_init.

    The relevant scale is var_init (~0.49 post-warmup), not var_max_initial
    (2.5). Effective contraction begins only when var_max(t) < var_init.
    """
    var_init_estimate = 0.49  # observed post-warmup, ramp starts here
    contraction_steps = 200

    # Ramp now starts FROM var_init. So var_max(t=0) = var_init, contraction
    # pressure begins immediately. alpha = (var_init - var_floor) / target_step
    # to reach floor at target_step.
    def alpha_to_reach_floor_at(target_step: int, target_floor: float) -> float:
        return (var_init_estimate - target_floor) / max(target_step, 1)

    sweep_configs = [
        # Reach floor 0.40 at step 50: drop of 0.09 over 50 steps → very gentle
        {'alpha': alpha_to_reach_floor_at(50, 0.40),  'var_floor': 0.40, 'lambda_KNV_target': 0.10},
        {'alpha': alpha_to_reach_floor_at(80, 0.30),  'var_floor': 0.30, 'lambda_KNV_target': 0.10},
        {'alpha': alpha_to_reach_floor_at(80, 0.30),  'var_floor': 0.30, 'lambda_KNV_target': 0.30},
        {'alpha': alpha_to_reach_floor_at(120, 0.25), 'var_floor': 0.25, 'lambda_KNV_target': 0.30},
        {'alpha': alpha_to_reach_floor_at(150, 0.20), 'var_floor': 0.20, 'lambda_KNV_target': 0.30},
    ]

    sweep_results = {}
    for cfg in sweep_configs:
        label = f"a{cfg['alpha']:.4f}_floor{cfg['var_floor']}_lam{cfg['lambda_KNV_target']}"
        r = measure_T25_prime_run(
            cfg_engine, coupling_form, eps, base_seed,
            alpha=cfg['alpha'],
            var_floor=cfg['var_floor'],
            lambda_KNV_target=cfg['lambda_KNV_target'],
            contraction_steps=contraction_steps,
        )
        sweep_results[label] = r

    priority = [
        'LOCAL_CONTRACTION_VIABLE',
        'PROPAGATED_CONTRACTION',
        'PARTIAL_CONTRACTION_NON_PROPAGATED',
        'STRUCTURAL_RESISTANCE',
        'KNV_COLLAPSE',
        'NUMERICAL_INSTABILITY',
    ]
    best = None
    for level in priority:
        for label, r in sweep_results.items():
            if r['outcome'] == level:
                best = (label, level)
                break
        if best:
            break

    return {
        'coupling_form': coupling_form,
        'sweep': sweep_results,
        'best_label': best[0] if best else None,
        'best_outcome': best[1] if best else None,
    }


def run_T25_prime_test(base_seed: int = 42) -> dict:
    print("=" * 70)
    print("F6.T25' — Morphodynamic constrained contraction (MCQ-aligned ramp)")
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
        print(f"\n  {form}")
        r = measure_T25_prime_for_form(cfg, form, eps, base_seed)
        per_form[form] = r
        for label, sweep_r in r['sweep'].items():
            print(f"    [{label}] {sweep_r['outcome']:<35s} "
                  f"drop_A={sweep_r['drop_A']:+.3f} "
                  f"G_min={sweep_r['G_A_min']:.4f} "
                  f"d_max={sweep_r['delta_A_max']:.4f} "
                  f"G_KNV={sweep_r['G_diff_KNV']} "
                  f"d_KNV={sweep_r['delta_top_KNV']} "
                  f"morph_KNV={sweep_r['gamma_morph_KNV']} "
                  f"n_inst={sweep_r['numerical_instability_step']}")
        print(f"    → best: {r['best_outcome']} at {r['best_label']}")

    return {
        'test': 'F6_T25_prime_morphodynamic_constrained_contraction',
        'per_form': per_form,
        'base_seed': base_seed,
        'epsilon': eps,
        'note': (
            "Progressive ramp on var_max, MCQ-aligned monitoring of "
            "𝒢_proxy/Δ/Γ″_proxy. Five outcome categories distinguishing "
            "viable contraction (LOCAL/PROPAGATED), structural resistance "
            "(non-failure), KNV collapse (genuine MCQ violation), and "
            "numerical instability (engine breakdown, distinct from KNV). "
            "Classification priority order: LOCAL_VIABLE > PROPAGATED > "
            "PARTIAL > STRUCTURAL_RESISTANCE > KNV_COLLAPSE > "
            "NUMERICAL_INSTABILITY (better outcomes preferred at any "
            "sweep point)."
        ),
    }


if __name__ == "__main__":
    result = run_T25_prime_test(base_seed=42)
    out = Path("/home/claude/mcq_v4/results/phase6c_b/F6_T25_prime_contraction.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print()
    print("Summary by form:")
    for form, r in result['per_form'].items():
        bo = r['best_outcome'] if r['best_outcome'] else 'NO_VALID_OUTCOME'
        bl = r['best_label'] if r['best_label'] else 'n/a'
        print(f"  {form:<32s} → {bo:<35s} at {bl}")
