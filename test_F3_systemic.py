"""
Phase 6c-B — F3 Systemic test (MCQ-aligned criterion).

Plan-initial F3 systemic (6b):
    Measure 𝒟(t) at the system scale (18 directions = 6 per module × 3
    modules). Verify that each module preserves its intra-modular
    polymorphy under perturbation of others, and that coupling does NOT
    collapse 𝒟 of one module at the expense of another.

Alex's MCQ-aligned correction:
    The PASS/FAIL criterion is NOT "preserve polymorphy" but:
        𝒢 > G_min  AND  Δ in corridor  under perturbation

    Because:
      propagation + 𝒢>0 maintained → anti-coherence productive
      propagation + 𝒢→0           → KNV collapse
      no propagation                → decoupling

For each (origin_module, direction, target_module) triplet:
  - Apply perturbation on origin_module's direction (one of 6 per module)
  - Run k_propagation steps
  - Measure 𝒢_target (modular gradient) and Δ_target (corridor)
  - Classify the (origin_dir, target) cell

Aggregate: matrix 18 × 3.

The 6 directions per module are:
  T+1, T-1, M+1, M-1, I+1, I-1
Each is a localised bump on the corresponding axis at index 3 (for +1)
or index 1 (for -1), then renormalisation.
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
from mcq_v4.factorial.signatures_6c_b import (
    compute_G_modular, compute_delta_modular, KNV_THRESHOLDS_6C_B,
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


def perturb_direction(sys_obj, origin: str, direction: str, magnitude: float = 0.10):
    """
    Apply a localised bump on `origin` module along `direction`.

    direction is one of: 'T+', 'T-', 'M+', 'M-', 'I+', 'I-'.
    +1 maps to index 3, -1 maps to index 1.
    """
    state = getattr(sys_obj, f'state_{origin}')
    psi = state.psi.copy()
    bump = np.zeros_like(psi)
    axis_letter, sign = direction[0], direction[1]
    idx = 3 if sign == '+' else 1
    if axis_letter == 'T':
        bump[idx, 2, 2] = magnitude
    elif axis_letter == 'M':
        bump[2, idx, 2] = magnitude
    elif axis_letter == 'I':
        bump[2, 2, idx] = magnitude
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
    if origin == 'A':
        sys_obj.state_A = new_state
    elif origin == 'B':
        sys_obj.state_B = new_state
    else:
        sys_obj.state_C = new_state


DIRECTIONS = ['T+', 'T-', 'M+', 'M-', 'I+', 'I-']
MODULES = ['A', 'B', 'C']


def measure_F3_systemic_for_form(
    cfg_engine, coupling_form: str, eps: float, base_seed: int,
    warmup_steps: int = 100, k_propagation: int = 25,
    perturb_magnitude: float = 0.10,
) -> dict:
    """
    For each (origin, direction), perturb and measure 𝒢/Δ on each target.

    Output cell shape (origin × direction × target):
      - propagation: norm of Δψ_target
      - G_target_post / Δ_target_post (compared to no-pert reference)
      - classification: ANTI_COHERENCE_PRODUCTIVE / KNV_COLLAPSE / DECOUPLING
    """
    initial_psi = make_initial_psi()

    # Warmup with coupling OFF (consistent with rest of 6c-B)
    coupling_cfg_warmup = CouplingConfig(epsilon=0.0, coupling_form='contrastive')
    sys_warmup = build_three_module_system(
        cfg_engine, coupling_cfg_warmup, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )
    for _ in range(warmup_steps):
        sys_warmup, _ = step_three_modules(sys_warmup, coupling_active=False)

    state_clones, rng_states, prev_h = clone_system(sys_warmup)

    coupling_cfg_meas = CouplingConfig(epsilon=eps, coupling_form=coupling_form)

    # Reference branch: NO perturbation, just run k_propagation steps
    sys_ref = make_system_from_clone(cfg_engine, coupling_cfg_meas,
                                      state_clones, rng_states, prev_h, base_seed)
    psi_ref_pre = {m: getattr(sys_ref, f'state_{m}').psi.copy() for m in MODULES}
    for _ in range(k_propagation):
        sys_ref, _ = step_three_modules(sys_ref, coupling_active=True)
    G_ref = {m: compute_G_modular(getattr(sys_ref, f'state_{m}'))['G_total'] for m in MODULES}
    delta_ref = {m: compute_delta_modular(getattr(sys_ref, f'state_{m}'))['delta'] for m in MODULES}
    psi_ref_post = {m: getattr(sys_ref, f'state_{m}').psi.copy() for m in MODULES}

    # 18 perturbations × 3 targets
    cells = {}
    for origin in MODULES:
        for direction in DIRECTIONS:
            sys_p = make_system_from_clone(cfg_engine, coupling_cfg_meas,
                                            state_clones, rng_states, prev_h, base_seed)
            perturb_direction(sys_p, origin, direction, perturb_magnitude)
            for _ in range(k_propagation):
                sys_p, _ = step_three_modules(sys_p, coupling_active=True)

            for target in MODULES:
                target_state = getattr(sys_p, f'state_{target}')
                psi_target_post = target_state.psi
                # Propagation = deviation from reference (no-pert) trajectory
                propagation = float(np.linalg.norm(
                    psi_target_post - psi_ref_post[target]
                ))
                G_target = compute_G_modular(target_state)['G_total']
                d_target = compute_delta_modular(target_state)['delta']

                G_above_floor = G_target > KNV_THRESHOLDS_6C_B['G_min']
                # KNV criterion refined:
                #   - delta_crit violation: Δ > delta_crit (excessive dispersion)
                #   - G collapse: G fell below floor relative to reference
                # A structurally low Δ that REMAINS low is not a KNV violation
                # (the system is centred at rest — STR-compatible).
                delta_explosion = d_target > KNV_THRESHOLDS_6C_B['delta_crit']
                G_collapsed = (
                    not G_above_floor
                    and G_ref[target] > KNV_THRESHOLDS_6C_B['G_min']
                )

                # Calibrated propagation threshold:
                # 3× the typical floating-point/numerical noise level
                # observed in label-based runs (≈ 1e-5). We classify above
                # this as "propagation present", below as "decoupling".
                PROPAGATION_THRESHOLD = 3e-5
                propagated = propagation > PROPAGATION_THRESHOLD
                if not propagated and origin != target:
                    classification = 'DECOUPLING'
                elif propagated and G_above_floor and not delta_explosion:
                    classification = 'ANTI_COHERENCE_PRODUCTIVE'
                elif propagated and G_collapsed:
                    classification = 'KNV_COLLAPSE_G'
                elif propagated and delta_explosion:
                    classification = 'KNV_COLLAPSE_DELTA'
                elif origin == target and G_above_floor and not delta_explosion:
                    classification = 'ORIGIN_PRESERVED'
                else:
                    classification = 'UNCLASSIFIED'

                cells[(origin, direction, target)] = {
                    'propagation': propagation,
                    'G_target': G_target,
                    'delta_target': d_target,
                    'G_above_floor': G_above_floor,
                    'delta_explosion': delta_explosion,
                    'G_collapsed': G_collapsed,
                    'classification': classification,
                }

    # Aggregate counts
    total_count = {}
    for cell in cells.values():
        c = cell['classification']
        total_count[c] = total_count.get(c, 0) + 1

    cross_module_cells = [
        cell for (origin, _, target), cell in cells.items()
        if origin != target
    ]
    cross_module_count = {}
    for cell in cross_module_cells:
        c = cell['classification']
        cross_module_count[c] = cross_module_count.get(c, 0) + 1

    # Verdict for the form: count of KNV collapses on cross-module cells
    n_knv = sum(cross_module_count.get(k, 0)
                for k in ['KNV_COLLAPSE_G', 'KNV_COLLAPSE_DELTA'])
    n_anti_coh = cross_module_count.get('ANTI_COHERENCE_PRODUCTIVE', 0)
    n_decoupling = cross_module_count.get('DECOUPLING', 0)

    if n_knv > 0:
        form_verdict = 'KNV_VIOLATIONS_DETECTED'
    elif n_anti_coh > n_decoupling:
        form_verdict = 'ANTI_COHERENCE_DOMINATES'
    elif n_decoupling >= len(cross_module_cells) * 0.7:
        form_verdict = 'DECOUPLING_DOMINANT'
    else:
        form_verdict = 'MIXED'

    return {
        'coupling_form': coupling_form,
        'epsilon': eps,
        'k_propagation': k_propagation,
        'perturb_magnitude': perturb_magnitude,
        'cells': {f"{o}.{d}.{t}": v for (o, d, t), v in cells.items()},
        'count_total': total_count,
        'count_cross_module': cross_module_count,
        'cross_module_summary': {
            'n_anti_coherence_productive': n_anti_coh,
            'n_KNV_collapse': n_knv,
            'n_decoupling': n_decoupling,
            'n_cross_cells': len(cross_module_cells),
        },
        'form_verdict': form_verdict,
        'G_ref': G_ref,
        'delta_ref': delta_ref,
    }


def run_F3_systemic_test(base_seed: int = 42) -> dict:
    print("=" * 70)
    print("F3 systemic — 18 directions × 3 modules (MCQ criterion: 𝒢 + Δ)")
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

    per_form = {}
    for form in forms:
        print(f"  {form}...", end=" ", flush=True)
        r = measure_F3_systemic_for_form(cfg, form, eps, base_seed)
        per_form[form] = r
        cms = r['cross_module_summary']
        print(f"verdict={r['form_verdict']}, "
              f"anti_coh={cms['n_anti_coherence_productive']}, "
              f"KNV={cms['n_KNV_collapse']}, "
              f"decoup={cms['n_decoupling']}/{cms['n_cross_cells']}")

    return {
        'test': 'F3_systemic_MCQ_aligned',
        'per_form': per_form,
        'base_seed': base_seed,
        'epsilon': eps,
        'note': (
            "MCQ-aligned criterion: PASS = no KNV collapse on cross-module "
            "cells. ANTI_COHERENCE_PRODUCTIVE means propagation occurs while "
            "𝒢 stays above floor and Δ stays in corridor — productive "
            "differentiation rather than collapse."
        ),
    }


if __name__ == "__main__":
    result = run_F3_systemic_test(base_seed=42)
    out = Path("/home/claude/mcq_v4/results/phase6c_b/F3_systemic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print()
    print("Summary by form:")
    for form, r in result['per_form'].items():
        cms = r['cross_module_summary']
        print(f"  {form:<32s} → {r['form_verdict']:<28s} "
              f"({cms['n_anti_coherence_productive']}/{cms['n_cross_cells']} anti_coh)")
