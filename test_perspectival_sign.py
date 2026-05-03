"""
Preflight 8 — Coupling sign micro-test for Phase 6c PERSPECTIVAL forms.

Verifies the direction of each of the three perspectival coupling forms
in isolation:

  - perspectival_INV_H        (literal Ch.3 §3.2.2 form)
  - perspectival_H_OPEN       (textual interpretation: open-metric)
  - perspectival_MORPHO_ACTIVE (open + active metric)

Setup identical to test_coupling_sign.py: A at θ_T=0, B at θ_T=+1,
diffusion/noise/Phi_intra all OFF, coupling ON at ε=0.5.

Records the direction (REPULSIVE / ATTRACTIVE / NEUTRAL) for each form.

Note (different from preflight 4):
  Preflight 4 ASSERTS that 'positive' and 'contrastive' must be REPULSIVE,
  because we have prior architectural justification.

  Preflight 8 RECORDS the direction of the perspectival forms WITHOUT
  asserting it, because:
   - INV_H may be REPULSIVE or ATTRACTIVE depending on the convention
     tension noted in §3.2.2 (g_k = θ_k/h vs "unfamiliar contributes more").
   - H_OPEN should be ATTRACTIVE-toward-open-zones (by construction),
     but the net displacement of A's centre of mass depends on the
     spatial overlap between A's open zones and B's mass.
   - MORPHO_ACTIVE depends on both forms and ∂_t h, which is zero on
     the first step.

This preflight is INFORMATIVE, not BLOCKING for 6c. The exit code is
non-zero only if all three forms produce COUPLING_NEUTRAL (which would
mean the perspectival mechanism is dead).

Run from project root:
    PYTHONPATH=src python tests/phase6c/test_perspectival_sign.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    FactorialEngineConfig, ModuleConfig, EngineMode, THETA_T,
)
from mcq_v4.factorial.engine import FactorialEngine, make_initial_state
from mcq_v4.factorial.coupling import compute_extra_phi_for_module
from mcq_v4.factorial.three_module_system import CouplingConfig
from mcq_v4.factorial.overlaps import compute_pairwise_R_psi


def test_perspectival_form_with_warmup(coupling_form: str) -> dict:
    """
    Run the test AFTER warming up the metrics differently per module.

    At t=0, h_T=h_M=h_I=h_0 uniformly for all modules. The three
    perspectival forms degenerate to the same expression (since h_field
    is constant and equal across modules — no perspective).

    To see the differentiation between forms, we apply different
    warmup histories to A and B, producing h_A ≠ h_B. Then we reset
    ψ to the target configurations (concentrated at θ_T=0 for A,
    θ_T=+1 for B) while KEEPING the warmed h. Only then does the
    perspectival structure manifest.
    """
    import numpy as np
    from mcq_v4.factorial.engine import FactorialEngine, make_initial_state
    from mcq_v4.factorial.state import FactorialState

    # Module A: warm h with ψ concentrated at θ_T = 0 (sediments h there)
    psi_A_warmup_init = np.zeros((5, 5, 5))
    psi_A_warmup_init[2, 2, 2] = 1.0
    mcfg_A = ModuleConfig(name='A', weights=(1.0, 1.0, 1.0), seed=42)
    state_A_warmup = make_initial_state(psi_A_warmup_init, mcfg_A)
    cfg_warmup = FactorialEngineConfig(
        dt=0.05, T_steps=200, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.0,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    engine_A_warmup = FactorialEngine(cfg_warmup, mcfg_A)
    for _ in range(50):
        state_A_warmup, _ = engine_A_warmup.step(state_A_warmup)

    # Module B: warm h with ψ concentrated at θ_T = +1 (sediments h there)
    psi_B_warmup_init = np.zeros((5, 5, 5))
    psi_B_warmup_init[3, 2, 2] = 1.0
    mcfg_B = ModuleConfig(name='B', weights=(1.0, 1.0, 1.0), seed=43)
    state_B_warmup = make_initial_state(psi_B_warmup_init, mcfg_B)
    engine_B_warmup = FactorialEngine(cfg_warmup, mcfg_B)
    for _ in range(50):
        state_B_warmup, _ = engine_B_warmup.step(state_B_warmup)

    # Now reset ψ to target measurement configurations, KEEPING warmed h
    psi_A_target = np.zeros((5, 5, 5))
    psi_A_target[2, 2, 2] = 0.8
    psi_A_target[1, 2, 2] = 0.1
    psi_A_target[3, 2, 2] = 0.1

    psi_B_target = np.zeros((5, 5, 5))
    psi_B_target[3, 2, 2] = 0.8
    psi_B_target[2, 2, 2] = 0.1
    psi_B_target[4, 2, 2] = 0.1

    state_A = FactorialState(
        psi=psi_A_target,
        h_T=state_A_warmup.h_T.copy(),
        h_M=state_A_warmup.h_M.copy(),
        h_I=state_A_warmup.h_I.copy(),
        cfg=state_A_warmup.cfg,
    )
    state_B = FactorialState(
        psi=psi_B_target,
        h_T=state_B_warmup.h_T.copy(),
        h_M=state_B_warmup.h_M.copy(),
        h_I=state_B_warmup.h_I.copy(),
        cfg=state_B_warmup.cfg,
    )

    # Module C: dummy uniform
    psi_C = np.ones((5, 5, 5)) / 125
    mcfg_C = ModuleConfig(name='C', weights=(1.0, 1.0, 1.0), seed=44)
    state_C = make_initial_state(psi_C, mcfg_C)

    states = {'A': state_A, 'B': state_B, 'C': state_C}

    # Apply a single coupling step (engine OFF, only coupling)
    cfg_test = FactorialEngineConfig(
        dt=0.01, T_steps=1, mode=EngineMode.FULL,
        D_0=0.0, D_min=0.0, beta_0=0.0, gamma_0=0.0,
        h_0=1.0, h_min=0.1, sigma_eta=0.0,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.0,
    )
    coupling_cfg = CouplingConfig(epsilon=0.5, coupling_form=coupling_form)

    R_pairs = compute_pairwise_R_psi(states)

    prev_h_fields = None
    if coupling_form == 'perspectival_MORPHO_ACTIVE':
        from mcq_v4.factorial.coupling import _h_field_product
        # Use slightly different prev to give |∂_t h| something to work with
        prev_h_fields = {
            name: _h_field_product(st) - 0.01 for name, st in states.items()
        }

    extra_A = compute_extra_phi_for_module(
        'A', states, R_pairs, coupling_cfg,
        prev_h_fields=prev_h_fields,
        h_min=cfg_test.h_min, h_0=cfg_test.h_0, dt=cfg_test.dt,
    )

    engine_A_test = FactorialEngine(cfg_test, mcfg_A)
    new_state_A, _ = engine_A_test.step(state_A, phi_extra=extra_A)

    psi_T_before = state_A.psi.sum(axis=(1, 2))
    psi_T_after = new_state_A.psi.sum(axis=(1, 2))
    mean_T_before = float((THETA_T * psi_T_before).sum())
    mean_T_after = float((THETA_T * psi_T_after).sum())
    delta = mean_T_after - mean_T_before

    psi_T_B = states['B'].psi.sum(axis=(1, 2))
    mean_T_B = float((THETA_T * psi_T_B).sum())
    direction_to_B = mean_T_B - mean_T_before

    if abs(delta) < 1e-9:
        verdict = 'COUPLING_NEUTRAL'
    elif delta * direction_to_B > 0:
        verdict = 'COUPLING_ATTRACTIVE'
    else:
        verdict = 'COUPLING_REPULSIVE'

    h_div_T = float(np.sum(np.abs(state_A.h_T - state_B.h_T)))
    h_div_M = float(np.sum(np.abs(state_A.h_M - state_B.h_M)))
    h_div_I = float(np.sum(np.abs(state_A.h_I - state_B.h_I)))
    h_divergence_L1 = h_div_T + h_div_M + h_div_I

    return {
        'coupling_form': coupling_form,
        'h_A_T': state_A.h_T.tolist(),
        'h_B_T': state_B.h_T.tolist(),
        'h_div_T_L1': h_div_T,
        'h_div_M_L1': h_div_M,
        'h_div_I_L1': h_div_I,
        'h_divergence': h_divergence_L1,
        'mean_T_before_A': mean_T_before,
        'mean_T_after_A': mean_T_after,
        'mean_T_B': mean_T_B,
        'delta_mean_T_A': delta,
        'verdict': verdict,
        'phi_extra_max_abs': float(np.abs(extra_A).max()),
    }


def test_perspectival_form(coupling_form: str) -> dict:
    """Run the isolation test for one perspectival coupling form."""

    # A at θ_T = 0, B at θ_T = +1 (offset)
    psi_A = np.zeros((5, 5, 5))
    psi_A[2, 2, 2] = 0.8
    psi_A[1, 2, 2] = 0.1
    psi_A[3, 2, 2] = 0.1

    psi_B = np.zeros((5, 5, 5))
    psi_B[3, 2, 2] = 0.8
    psi_B[2, 2, 2] = 0.1
    psi_B[4, 2, 2] = 0.1

    psi_C = np.ones((5, 5, 5)) / 125

    mcfg_A = ModuleConfig(name='A', weights=(1.0, 1.0, 1.0), seed=42)
    mcfg_B = ModuleConfig(name='B', weights=(1.0, 1.0, 1.0), seed=43)
    mcfg_C = ModuleConfig(name='C', weights=(1.0, 1.0, 1.0), seed=44)

    state_A = make_initial_state(psi_A, mcfg_A)
    states = {
        'A': state_A,
        'B': make_initial_state(psi_B, mcfg_B),
        'C': make_initial_state(psi_C, mcfg_C),
    }

    cfg = FactorialEngineConfig(
        dt=0.01, T_steps=1, mode=EngineMode.FULL,
        D_0=0.0, D_min=0.0, beta_0=0.0, gamma_0=0.0,
        h_0=1.0, h_min=0.1, sigma_eta=0.0,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.0,
    )
    coupling_cfg = CouplingConfig(epsilon=0.5, coupling_form=coupling_form)

    R_pairs = compute_pairwise_R_psi(states)

    # For MORPHO_ACTIVE, we need a non-zero ∂_t h to see any effect.
    # We provide a synthetic prev_h_field that differs from current.
    prev_h_fields = None
    if coupling_form == 'perspectival_MORPHO_ACTIVE':
        # Inject a non-trivial prev h_field (slight perturbation)
        from mcq_v4.factorial.coupling import _h_field_product
        prev_h_fields = {}
        for name, st in states.items():
            current_h = _h_field_product(st)
            # Synthetic: prev was slightly different (h was 0.01 lower everywhere)
            prev_h_fields[name] = current_h - 0.01

    extra_A = compute_extra_phi_for_module(
        'A', states, R_pairs, coupling_cfg,
        prev_h_fields=prev_h_fields,
        h_min=cfg.h_min, h_0=cfg.h_0, dt=cfg.dt,
    )

    engine_A = FactorialEngine(cfg, mcfg_A)
    new_state_A, _ = engine_A.step(state_A, phi_extra=extra_A)

    psi_T_before = state_A.psi.sum(axis=(1, 2))
    psi_T_after = new_state_A.psi.sum(axis=(1, 2))
    mean_T_before = float((THETA_T * psi_T_before).sum())
    mean_T_after = float((THETA_T * psi_T_after).sum())
    delta = mean_T_after - mean_T_before

    psi_T_B = states['B'].psi.sum(axis=(1, 2))
    mean_T_B = float((THETA_T * psi_T_B).sum())

    direction_to_B = mean_T_B - mean_T_before

    if abs(delta) < 1e-9:
        verdict = 'COUPLING_NEUTRAL'
    elif delta * direction_to_B > 0:
        verdict = 'COUPLING_ATTRACTIVE'
    else:
        verdict = 'COUPLING_REPULSIVE'

    return {
        'coupling_form': coupling_form,
        'mean_T_before_A': mean_T_before,
        'mean_T_after_A': mean_T_after,
        'mean_T_B': mean_T_B,
        'delta_mean_T_A': delta,
        'verdict': verdict,
        'phi_extra_max_abs': float(np.abs(extra_A).max()),
        'phi_extra_max_at_B': float(extra_A[3, 2, 2]),
        'phi_extra_at_A': float(extra_A[2, 2, 2]),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PREFLIGHT 8 — Perspectival coupling sign micro-test")
    print("=" * 70)
    print()
    print("Two configurations:")
    print("  (1) PRE-WARMUP: h_A = h_B = h_0 uniformly. Forms degenerate.")
    print("  (2) POST-WARMUP: h_A ≠ h_B from differential history. Forms diverge.")
    print()

    forms = ['perspectival_INV_H', 'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']

    print("=" * 70)
    print("(1) PRE-WARMUP TESTS — h uniform, perspective collapses")
    print("=" * 70)
    print()
    pre_results = {}
    for form in forms:
        r = test_perspectival_form(form)
        pre_results[form] = r
        print(f"  {form:<35s} → {r['verdict']:<22s} "
              f"(Δ = {r['delta_mean_T_A']:+.6e})")

    # Sanity check: under uniform h, all three forms should produce
    # identical Δ (they reduce to the same expression).
    deltas = [r['delta_mean_T_A'] for r in pre_results.values()]
    pre_warmup_collapsed = (max(deltas) - min(deltas) < 1e-12)
    print()
    print(f"  Forms degenerate under uniform h: {pre_warmup_collapsed}")
    print(f"  This is EXPECTED: when h is uniform, perspective is undefined.")

    print()
    print("=" * 70)
    print("(2) POST-WARMUP TESTS — h_A ≠ h_B from differential history")
    print("=" * 70)
    print()
    post_results = {}
    for form in forms:
        r = test_perspectival_form_with_warmup(form)
        post_results[form] = r
        print(f"  {form:<35s} → {r['verdict']:<22s} "
              f"(Δ = {r['delta_mean_T_A']:+.6e}, "
              f"h_div = {r['h_divergence']:.4f})")

    # Now the forms should produce DIFFERENT Δ
    deltas_post = [r['delta_mean_T_A'] for r in post_results.values()]
    forms_distinct_post = (max(deltas_post) - min(deltas_post) > 1e-9)
    print()
    print(f"  Forms produce distinct Δ post-warmup: {forms_distinct_post}")
    if not forms_distinct_post:
        print("  WARNING: forms still degenerate post-warmup. Investigate.")

    # Block only if ALL three are NEUTRAL post-warmup (perspectival mechanism dead)
    all_neutral_post = all(r['verdict'] == 'COUPLING_NEUTRAL'
                           for r in post_results.values())

    print()
    print("=" * 70)
    if all_neutral_post:
        print("FAIL — all three perspectival forms produce zero displacement post-warmup.")
        print("The perspectival mechanism appears dead. Investigate.")
        sys.exit(1)
    elif not forms_distinct_post:
        print("WARNING — forms do not distinguish even post-warmup. Sweep may be inconclusive.")
        sys.exit(0)
    else:
        print("PASS — perspectival forms produce distinct directions post-warmup.")
        print("The sweep in 6c will measure which form produces the strongest signature.")
        sys.exit(0)
