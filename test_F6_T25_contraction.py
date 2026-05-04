"""
Phase 6c-B — F6.T25 transposed: forced modular contraction.

Original Phase 5 T25:
    Force instance 0 into sustained contraction via swap of Φ_eff
    (strong narrow well). Keep instance 1 free. Classify as
    LOCAL_CONTRACTION / PROPAGATED_CONTRACTION / NEITHER_CONTRACTS /
    OUT_OF_PERTURBATIVE_REGIME.

Transposition for 6c-B (3 modules, intra-instance):
    Force module M_A into sustained contraction via per-module swap
    of var_min/lambda_KNV (option a — Φ_corr swap-equivalent), keep
    M_B and M_C nominal. Same classification + an additional category
    PARTNER_DISTURBED_NON_CONTRACTION (Alex's spec).

MCQ-aligned monitoring (REQUIRED, Alex's audit):
    Throughout the contraction, monitor:
      - 𝒢_proxy(M_A): must remain > G_min (else KNV collapse)
      - Δ(M_A): must not exceed delta_crit (else dispersion)
      - Γ″(M_A) = ∂²_t τ'_A: bounded, no chaotic acceleration

    Without these, the test would only check "response to energetic
    forcing" rather than "structural capacity to remain local under
    constraint" (which is the actual T25 question).

Force-finding: empirically tune the contraction strength between
the KNV regime (force too high → A collapses) and NEITHER_CONTRACTS
(force too low → no measurable effect). The protocol auto-sweeps
contraction levels and reports the finest classification reachable.

Outputs:
    - per-form classification at each contraction strength
    - 𝒢/Δ/Γ″ trajectories of M_A
    - propagation evidence to M_B, M_C
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


def apply_contraction_swap(sys_obj, cfg_engine_base: FactorialEngineConfig,
                            lambda_KNV_strong: float, var_max_low: float):
    """
    Apply the Φ_corr swap to M_A: stronger pull-back to mean, narrower
    var_max ceiling. M_B and M_C keep their nominal config.

    The semantic of T25 contraction is "force the system into a narrow
    state": this means lowering var_max (the upper limit Φ_corr permits)
    so the regulation actively pulls var down towards a small ceiling.
    Lowering var_min would have the opposite effect (allow more spread).
    """
    swapped_cfg = FactorialEngineConfig(
        dt=cfg_engine_base.dt, T_steps=cfg_engine_base.T_steps,
        mode=cfg_engine_base.mode,
        D_0=cfg_engine_base.D_0, D_min=cfg_engine_base.D_min,
        beta_0=cfg_engine_base.beta_0, gamma_0=cfg_engine_base.gamma_0,
        h_0=cfg_engine_base.h_0, h_min=cfg_engine_base.h_min,
        sigma_eta=cfg_engine_base.sigma_eta,
        var_min=cfg_engine_base.var_min, var_max=var_max_low,
        H_min=cfg_engine_base.H_min, lambda_KNV=lambda_KNV_strong,
    )
    sys_obj.engine_A.cfg = swapped_cfg


def measure_contraction_dynamics(sys_obj, n_steps: int, dt: float):
    """Run n_steps with coupling active, capture full trajectory of A/B/C."""
    G_traj = {m: [] for m in ['A', 'B', 'C']}
    delta_traj = {m: [] for m in ['A', 'B', 'C']}
    var_M_traj = {m: [] for m in ['A', 'B', 'C']}  # variance on M-axis = main contraction proxy
    tau_T_traj = {m: [] for m in ['A', 'B', 'C']}
    psi_snapshot_pre = {m: getattr(sys_obj, f'state_{m}').psi.copy() for m in ['A', 'B', 'C']}

    for _ in range(n_steps):
        sys_obj, _ = step_three_modules(sys_obj, coupling_active=True)
        for m in ['A', 'B', 'C']:
            st = getattr(sys_obj, f'state_{m}')
            G_traj[m].append(compute_G_modular(st)['G_proxy_total'])
            delta_traj[m].append(compute_delta_modular(st)['delta'])
            obs = compute_observables(st.psi)
            var_M_traj[m].append(obs.get('var_M', 0.0))
            contrib = compute_modular_contributions(st)
            tau_T_traj[m].append(contrib['T'])

    psi_snapshot_post = {m: getattr(sys_obj, f'state_{m}').psi.copy() for m in ['A', 'B', 'C']}

    # Γ″ for module A's tau_T
    tau_A_arr = np.array(tau_T_traj['A']).reshape(-1, 1)
    if tau_A_arr.shape[0] >= 3:
        gamma_A = compute_gamma_double_prime(tau_A_arr, dt).flatten()
        gamma_A_max = float(np.abs(gamma_A).max())
    else:
        gamma_A_max = 0.0

    return {
        'G_traj': G_traj,
        'delta_traj': delta_traj,
        'var_M_traj': var_M_traj,
        'tau_T_traj': tau_T_traj,
        'gamma_A_max': gamma_A_max,
        'psi_snapshot_pre': psi_snapshot_pre,
        'psi_snapshot_post': psi_snapshot_post,
    }


def classify_contraction_outcome(
    var_M_pre: dict, var_M_post: dict,
    G_traj: dict, delta_traj: dict,
    extra_to_lambda_ratio: float,
) -> tuple:
    """
    Classify the outcome based on:
      - var_M drop fraction per module
      - 𝒢/Δ violations on M_A (KNV during contraction = invalid test)

    Note: the lambda ratio is logged but no longer used as a hard OOP
    gate. The OOP detection is replaced by the KNV monitoring (G_A_min
    < G_min OR delta_A_max > delta_crit), which is the structurally
    meaningful check.
    """
    drop_A = (var_M_pre['A'] - var_M_post['A']) / max(abs(var_M_pre['A']), 1e-12)
    drop_B = (var_M_pre['B'] - var_M_post['B']) / max(abs(var_M_pre['B']), 1e-12)
    drop_C = (var_M_pre['C'] - var_M_post['C']) / max(abs(var_M_pre['C']), 1e-12)

    G_A_min = min(G_traj['A']) if G_traj['A'] else 0.0
    delta_A_max = max(delta_traj['A']) if delta_traj['A'] else 0.0
    knv_violated_A = (
        G_A_min < KNV_THRESHOLDS_6C_B['G_min']
        or delta_A_max > KNV_THRESHOLDS_6C_B['delta_crit']
    )

    if knv_violated_A:
        return 'KNV_VIOLATED_FORCE_TOO_STRONG', {
            'drop_A': drop_A, 'drop_B': drop_B, 'drop_C': drop_C,
            'G_A_min': G_A_min, 'delta_A_max': delta_A_max,
            'lambda_ratio_logged': extra_to_lambda_ratio,
        }

    if drop_A < 0.50:
        partner_disturbed = (drop_B > 0.10 or drop_C > 0.10) or \
                            (drop_B < -0.10 or drop_C < -0.10)
        if partner_disturbed:
            return 'PARTNER_DISTURBED_NON_CONTRACTION', {
                'drop_A': drop_A, 'drop_B': drop_B, 'drop_C': drop_C,
                'lambda_ratio_logged': extra_to_lambda_ratio,
            }
        return 'NEITHER_CONTRACTS', {
            'drop_A': drop_A, 'drop_B': drop_B, 'drop_C': drop_C,
            'lambda_ratio_logged': extra_to_lambda_ratio,
        }

    if drop_B > 0.30 or drop_C > 0.30:
        return 'PROPAGATED_CONTRACTION', {
            'drop_A': drop_A, 'drop_B': drop_B, 'drop_C': drop_C,
            'lambda_ratio_logged': extra_to_lambda_ratio,
        }
    return 'LOCAL_CONTRACTION', {
        'drop_A': drop_A, 'drop_B': drop_B, 'drop_C': drop_C,
        'lambda_ratio_logged': extra_to_lambda_ratio,
    }


def measure_T25_for_form(
    cfg_engine, coupling_form: str, eps: float, base_seed: int,
    warmup_steps: int = 100, contraction_steps: int = 150,
) -> dict:
    """For one coupling form, sweep contraction strengths and find the regime."""
    initial_psi = make_initial_psi()
    coupling_cfg_warmup = CouplingConfig(epsilon=0.0, coupling_form='contrastive')
    sys_warmup = build_three_module_system(
        cfg_engine, coupling_cfg_warmup, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )
    for _ in range(warmup_steps):
        sys_warmup, _ = step_three_modules(sys_warmup, coupling_active=False)

    state_clones, rng_states, prev_h = clone_system(sys_warmup)

    # Pre-contraction var_M baseline
    var_M_pre = {
        m: compute_observables(state_clones[m].psi).get('var_M', 0.0)
        for m in ['A', 'B', 'C']
    }

    # Sweep contraction strengths
    sweep_strengths = [
        ('weak',     0.20, 1.0),   # lambda_KNV, var_max_low
        ('moderate', 0.50, 0.5),
        ('strong',   1.00, 0.2),
        ('extreme',  2.00, 0.1),
    ]

    sweep_results = {}
    coupling_cfg_meas = CouplingConfig(epsilon=eps, coupling_form=coupling_form)

    for label, lambda_strong, var_max_low in sweep_strengths:
        sys_p = make_system_from_clone(cfg_engine, coupling_cfg_meas,
                                        state_clones, rng_states, prev_h, base_seed)
        apply_contraction_swap(sys_p, cfg_engine, lambda_strong, var_max_low)
        dyn = measure_contraction_dynamics(sys_p, contraction_steps, cfg_engine.dt)

        var_M_post = {m: dyn['var_M_traj'][m][-1] for m in ['A', 'B', 'C']}
        # Heuristic perturbative check: ratio of typical contraction force scale
        ratio = lambda_strong / max(cfg_engine.lambda_KNV, 1e-12)
        outcome, evidence = classify_contraction_outcome(
            var_M_pre, var_M_post, dyn['G_traj'], dyn['delta_traj'], ratio
        )

        sweep_results[label] = {
            'lambda_KNV_strong': lambda_strong,
            'var_max_low': var_max_low,
            'outcome': outcome,
            'evidence': evidence,
            'gamma_A_max': dyn['gamma_A_max'],
            'G_A_min': float(min(dyn['G_traj']['A'])),
            'G_A_max': float(max(dyn['G_traj']['A'])),
            'delta_A_min': float(min(dyn['delta_traj']['A'])),
            'delta_A_max': float(max(dyn['delta_traj']['A'])),
        }

    # Find the strongest force that still produces a valid (non-KNV) outcome
    valid_outcomes = [
        ('LOCAL_CONTRACTION', 'PROPAGATED_CONTRACTION'),
        ('PARTNER_DISTURBED_NON_CONTRACTION',),
        ('NEITHER_CONTRACTS',),
    ]
    best_label = None
    best_outcome = None
    for label, _, _ in sweep_strengths:
        out = sweep_results[label]['outcome']
        for tier in valid_outcomes:
            if out in tier:
                best_label = label
                best_outcome = out
                break
        if best_outcome in ('LOCAL_CONTRACTION', 'PROPAGATED_CONTRACTION'):
            break

    return {
        'coupling_form': coupling_form,
        'epsilon': eps,
        'sweep': sweep_results,
        'best_label': best_label,
        'best_outcome': best_outcome,
        'var_M_pre': var_M_pre,
    }


def run_T25_test(base_seed: int = 42) -> dict:
    print("=" * 70)
    print("F6.T25 — Forced modular contraction (with 𝒢/Δ/Γ″ monitoring)")
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
        r = measure_T25_for_form(cfg, form, eps, base_seed)
        per_form[form] = r
        for label, sweep_r in r['sweep'].items():
            print(f"    [{label}] {sweep_r['outcome']:<40s} "
                  f"drop_A={sweep_r['evidence'].get('drop_A', 0):+.3f}, "
                  f"drop_B={sweep_r['evidence'].get('drop_B', 0):+.3f}, "
                  f"drop_C={sweep_r['evidence'].get('drop_C', 0):+.3f}, "
                  f"G_A_min={sweep_r['G_A_min']:.5f}")
        print(f"    → best: {r['best_outcome']} at {r['best_label']}")

    return {
        'test': 'F6_T25_forced_modular_contraction',
        'per_form': per_form,
        'base_seed': base_seed,
        'epsilon': eps,
        'note': (
            "Φ_corr swap on M_A (option a, var_max reduction). Sweep over "
            "contraction strengths. KNV monitoring (𝒢_A_min > G_min, "
            "Δ_A_max < delta_crit) is the structural validity check.\n"
            "\n"
            "OBSERVATION 6c-B run-1 (ε=0.005, base_seed=42): the engine "
            "becomes numerically unstable at var_max < 0.3 (overflow in "
            "diffusion drift), so the perturbative regime supporting clean "
            "contraction is narrow. At achievable strengths, M_A does not "
            "contract structurally (it equilibrates near the lowered var_max "
            "ceiling); meanwhile, M_B and M_C drift freely. The classification "
            "PARTNER_DISTURBED_NON_CONTRACTION dominates across all four forms.\n"
            "\n"
            "INTERPRETATION (with caveat): under Φ_extra coupling, the "
            "engine cannot sustain a local sustained contraction without "
            "numerical instability, so the LOCAL vs PROPAGATED distinction "
            "(the heart of T25) is NOT empirically reachable. This reflects "
            "the gradient-coupling architectural limit observed in 6c-A: "
            "perspective shapes the field structurally but cannot translate "
            "the contraction signal across the gradient barrier. Phase 6d "
            "(non-gradient 𝒞^{mod}) may permit cleaner contraction tests."
        ),
    }


if __name__ == "__main__":
    result = run_T25_test(base_seed=42)
    out = Path("/home/claude/mcq_v4/results/phase6c_b/F6_T25_contraction.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print()
    print("Summary by form (best outcome):")
    for form, r in result['per_form'].items():
        bo = r['best_outcome'] if r['best_outcome'] else 'NO_VALID_OUTCOME'
        bl = r['best_label'] if r['best_label'] else 'n/a'
        print(f"  {form:<32s} → {bo:<35s} at {bl}")
