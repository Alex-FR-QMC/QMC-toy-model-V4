"""
Phase 6c-B — F5 Asymmetry of propagation.

Plan-initial F5 (6c):
    With two modules A and B coupled, perturb A — measure propagation
    to B. Then perturb B — measure propagation to A. The Ch.2/Ch.3
    prediction is that the propagation should be ASYMMETRIC because A
    and B perceive the event in their respective metrics (h_A ≠ h_B).

Falsification: if the return trip is exactly mirror-symmetric to the
forward trip, the perspective has no measurable dynamic effect.

This is the H2/H3 differential at trajectorial scale.

Normalised asymmetry metric (Alex's correction):

    asymmetry = ‖Δψ_B|pert_A − Δψ_A|pert_B‖
                ----------------------------
              (‖Δψ_B|pert_A‖ + ‖Δψ_A|pert_B‖)

Bounded in [0, 1]:
  0 = perfectly symmetric propagation
  1 = orthogonal propagation patterns

The normalisation prevents INV_H from inflating asymmetry by amplitude
amplification (the absolute ‖Δψ‖ scales with 1/h, so an unnormalised
asymmetry would be biased).

Interpretation matrix:

  asymmetry(form) > asymmetry(contrastive) + threshold:
      perspective creates a structurally asymmetric propagation
      (the form's extra effect compared to label-based)

  asymmetry(form) ≤ asymmetry(contrastive):
      no perspective-induced asymmetry beyond what the geometric
      sharing already produces

The contrastive baseline asymmetry is NOT necessarily zero — even
label-based coupling produces some asymmetry from non-identical
ψ_A and ψ_B. F5 is a DIFFERENTIAL test, not absolute.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Optional

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
    """Deep clone (states, RNG, prev_h_fields) so that each branch has identical starting conditions."""
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


def perturb_module_T(sys_obj, target: str, magnitude: float = 0.10):
    """Localised perturbation: add Gaussian bump on T-axis at θ_T = +1, renormalise."""
    state = getattr(sys_obj, f'state_{target}')
    psi = state.psi.copy()
    bump = np.zeros_like(psi)
    bump[3, 2, 2] = magnitude  # θ_T = +1 (index 3)
    psi_new = psi + bump
    psi_new = np.maximum(psi_new, 0.0)
    psi_new = psi_new / psi_new.sum()
    new_state = FactorialState(
        psi=psi_new,
        h_T=state.h_T.copy(),
        h_M=state.h_M.copy(),
        h_I=state.h_I.copy(),
        cfg=state.cfg,
    )
    if target == 'A':
        sys_obj.state_A = new_state
    elif target == 'B':
        sys_obj.state_B = new_state
    else:
        sys_obj.state_C = new_state


def measure_propagation(sys_obj, target: str, k_steps: int = 20):
    """
    Run k_steps with coupling active. Return Δψ for the OTHER module.

    For perturb_A (target='A'): we measure Δψ_B (effect on B from A's perturbation)
    For perturb_B (target='B'): we measure Δψ_A (effect on A from B's perturbation)
    """
    other = 'B' if target == 'A' else 'A'
    psi_other_pre = getattr(sys_obj, f'state_{other}').psi.copy()
    for _ in range(k_steps):
        sys_obj, _ = step_three_modules(sys_obj, coupling_active=True)
    psi_other_post = getattr(sys_obj, f'state_{other}').psi.copy()
    return psi_other_post - psi_other_pre


def measure_F5_for_form(
    cfg_engine, coupling_form: str, eps: float, base_seed: int,
    warmup_steps: int = 100, k_propagation: int = 20,
    perturb_magnitude: float = 0.10,
    sigma_eta_override: Optional[float] = None,
) -> dict:
    """One coupling form: perturb-A vs perturb-B, normalised asymmetry.

    sigma_eta_override (None or 0.0): if 0.0, runs with multiplicative
    noise OFF — used as the F5_NOISE_OFF control to isolate metric/
    structural asymmetry from the stochastic component (Alex's audit).
    """
    initial_psi = make_initial_psi()
    if sigma_eta_override is not None:
        cfg_active = FactorialEngineConfig(
            dt=cfg_engine.dt, T_steps=cfg_engine.T_steps, mode=cfg_engine.mode,
            D_0=cfg_engine.D_0, D_min=cfg_engine.D_min,
            beta_0=cfg_engine.beta_0, gamma_0=cfg_engine.gamma_0,
            h_0=cfg_engine.h_0, h_min=cfg_engine.h_min,
            sigma_eta=sigma_eta_override,
            var_min=cfg_engine.var_min, var_max=cfg_engine.var_max,
            H_min=cfg_engine.H_min, lambda_KNV=cfg_engine.lambda_KNV,
        )
    else:
        cfg_active = cfg_engine

    coupling_cfg_warmup = CouplingConfig(epsilon=0.0, coupling_form='contrastive')
    sys_warmup = build_three_module_system(
        cfg_active, coupling_cfg_warmup, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )
    for _ in range(warmup_steps):
        sys_warmup, _ = step_three_modules(sys_warmup, coupling_active=False)

    state_clones, rng_states, prev_h = clone_system(sys_warmup)
    coupling_cfg_meas = CouplingConfig(epsilon=eps, coupling_form=coupling_form)

    sys_A = make_system_from_clone(cfg_active, coupling_cfg_meas,
                                    state_clones, rng_states, prev_h, base_seed)
    perturb_module_T(sys_A, 'A', perturb_magnitude)
    delta_psi_B_given_A = measure_propagation(sys_A, 'A', k_propagation)

    sys_B = make_system_from_clone(cfg_active, coupling_cfg_meas,
                                    state_clones, rng_states, prev_h, base_seed)
    perturb_module_T(sys_B, 'B', perturb_magnitude)
    delta_psi_A_given_B = measure_propagation(sys_B, 'B', k_propagation)

    norm_B_given_A = float(np.linalg.norm(delta_psi_B_given_A))
    norm_A_given_B = float(np.linalg.norm(delta_psi_A_given_B))
    norm_diff = float(np.linalg.norm(delta_psi_B_given_A - delta_psi_A_given_B))

    if (norm_B_given_A + norm_A_given_B) < 1e-12:
        asymmetry_normalised = 0.0
    else:
        asymmetry_normalised = norm_diff / (norm_B_given_A + norm_A_given_B)

    return {
        'coupling_form': coupling_form,
        'epsilon': eps,
        'k_propagation': k_propagation,
        'perturb_magnitude': perturb_magnitude,
        'sigma_eta_used': cfg_active.sigma_eta,
        'noise_off_control': sigma_eta_override == 0.0,
        'norm_propagation_B_given_A': norm_B_given_A,
        'norm_propagation_A_given_B': norm_A_given_B,
        'norm_difference': norm_diff,
        'asymmetry_normalised': asymmetry_normalised,
    }


def evaluate_F5(per_form_results: dict) -> dict:
    """Differential test vs contrastive baseline."""
    baseline = per_form_results.get('contrastive')
    if baseline is None:
        return {'verdict': 'NO_BASELINE'}
    a_base = baseline['asymmetry_normalised']

    # Threshold: difference > 0.05 of normalised asymmetry
    THRESHOLD = 0.05

    per_form_evaluation = {}
    for form, r in per_form_results.items():
        if form == 'contrastive':
            per_form_evaluation[form] = {'role': 'baseline',
                                         'asymmetry_normalised': a_base}
            continue
        a_form = r['asymmetry_normalised']
        delta = a_form - a_base
        if delta > THRESHOLD:
            verdict = 'PERSPECTIVE_INDUCED_ASYMMETRY'
        elif delta > 0:
            verdict = 'WEAK_PERSPECTIVE_INDUCED_ASYMMETRY'
        else:
            verdict = 'NO_PERSPECTIVE_ASYMMETRY_BEYOND_BASELINE'
        per_form_evaluation[form] = {
            'asymmetry_normalised': a_form,
            'asymmetry_baseline': a_base,
            'delta_vs_baseline': delta,
            'verdict': verdict,
        }
    return per_form_evaluation


def run_F5_test(base_seed: int = 42) -> dict:
    print("=" * 70)
    print("F5 — Propagation asymmetry (normalised, differential vs contrastive)")
    print("=" * 70)

    cfg = FactorialEngineConfig(
        dt=0.05, T_steps=200, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    eps = 0.005
    forms = ['contrastive',
             'perspectival_INV_H',
             'perspectival_H_OPEN',
             'perspectival_MORPHO_ACTIVE']

    print()
    print("[1/2] WITH NOISE (sigma_eta=0.10)")
    print("-" * 70)
    per_form_noisy = {}
    for form in forms:
        print(f"  {form}...", end=" ", flush=True)
        r = measure_F5_for_form(cfg, form, eps, base_seed)
        per_form_noisy[form] = r
        print(f"asymmetry={r['asymmetry_normalised']:.4f}")
    eval_noisy = evaluate_F5(per_form_noisy)

    # Noise-off control branch
    cfg_noise_off = FactorialEngineConfig(
        dt=0.05, T_steps=200, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.0,  # NOISE OFF
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )

    print()
    print("[2/2] NOISE OFF (sigma_eta=0) — control to isolate metric/structural asymmetry")
    print("-" * 70)
    per_form_noiseless = {}
    for form in forms:
        print(f"  {form}...", end=" ", flush=True)
        r = measure_F5_for_form(cfg_noise_off, form, eps, base_seed)
        per_form_noiseless[form] = r
        print(f"asymmetry={r['asymmetry_normalised']:.4f}")
    eval_noiseless = evaluate_F5(per_form_noiseless)

    # Comparative reading: stochastic vs structural component of asymmetry
    print()
    print("Stochastic vs structural decomposition:")
    asymmetry_decomposition = {}
    for form in forms:
        a_noisy = per_form_noisy[form]['asymmetry_normalised']
        a_clean = per_form_noiseless[form]['asymmetry_normalised']
        stochastic_component = a_noisy - a_clean
        asymmetry_decomposition[form] = {
            'asymmetry_with_noise': a_noisy,
            'asymmetry_noise_off': a_clean,
            'stochastic_component_estimate': stochastic_component,
            'structural_component_estimate': a_clean,
        }
        print(f"  {form:<32s} structural={a_clean:.4f}, "
              f"stochastic={stochastic_component:+.4f}")

    return {
        'test': 'F5_propagation_asymmetry',
        'with_noise': {
            'per_form': per_form_noisy,
            'evaluation': eval_noisy,
            'sigma_eta': 0.10,
        },
        'noise_off': {
            'per_form': per_form_noiseless,
            'evaluation': eval_noiseless,
            'sigma_eta': 0.0,
        },
        'asymmetry_decomposition': asymmetry_decomposition,
        'base_seed': base_seed,
        'epsilon': eps,
        'note': (
            "Two-branch test: with-noise (canonical) and noise-off (control). "
            "Stochastic component = asymmetry_with_noise - asymmetry_noise_off. "
            "Only the structural component (noise-off) is attributable to "
            "weights or perspective. The differential vs contrastive baseline "
            "is the perspective-induced excess."
        ),
    }


if __name__ == "__main__":
    result = run_F5_test(base_seed=42)
    out = Path("/home/claude/mcq_v4/results/phase6c_b/F5_propagation_asymmetry.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print()
    print("=" * 70)
    print("Summary — structural asymmetry differential vs baseline:")
    print("=" * 70)
    base_struct = result['asymmetry_decomposition']['contrastive']['structural_component_estimate']
    for form in ['contrastive', 'perspectival_INV_H',
                 'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']:
        s = result['asymmetry_decomposition'][form]['structural_component_estimate']
        if form == 'contrastive':
            print(f"  {form:<32s} structural baseline = {s:.4f}")
        else:
            delta = s - base_struct
            print(f"  {form:<32s} structural = {s:.4f} (Δ vs baseline = {delta:+.4f})")
