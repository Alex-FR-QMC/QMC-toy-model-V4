"""
Coupling sign micro-test for Phase 6b.

Verifies, for each coupling form ('positive' and 'contrastive'),
the direction of the coupling effect by isolation:

  - module A concentrated at θ_T = 0 (centre)
  - module B concentrated at θ_T = +1 (offset)
  - diffusion OFF (D_0 = 0)
  - noise OFF (sigma_eta = 0)
  - intra Φ_corr OFF (lambda_KNV = 0)
  - inter-modular coupling ON

Measures the change in centre of mass of ψ_A along the T axis after
one coupling step. Classifies as:
  COUPLING_ATTRACTIVE  : A moves toward B
  COUPLING_REPULSIVE   : A moves away from B
  COUPLING_NEUTRAL     : no measurable motion (< 1e-7)

Run from project root:
    PYTHONPATH=src python tests/phase6b/test_coupling_sign.py
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


def test_coupling_sign(coupling_form: str) -> dict:
    """
    Run the isolation test for one coupling form.
    """
    # Init: A at θ_T = 0, B at θ_T = +1 (offset)
    psi_A = np.zeros((5, 5, 5))
    psi_A[2, 2, 2] = 0.8
    psi_A[1, 2, 2] = 0.1
    psi_A[3, 2, 2] = 0.1

    psi_B = np.zeros((5, 5, 5))
    psi_B[3, 2, 2] = 0.8
    psi_B[2, 2, 2] = 0.1
    psi_B[4, 2, 2] = 0.1

    psi_C = np.ones((5, 5, 5)) / 125  # uniform — no role here

    mcfg_A = ModuleConfig(name='A', weights=(1.0, 1.0, 1.0), seed=42)
    mcfg_B = ModuleConfig(name='B', weights=(1.0, 1.0, 1.0), seed=43)
    mcfg_C = ModuleConfig(name='C', weights=(1.0, 1.0, 1.0), seed=44)

    state_A = make_initial_state(psi_A, mcfg_A)
    states = {
        'A': state_A,
        'B': make_initial_state(psi_B, mcfg_B),
        'C': make_initial_state(psi_C, mcfg_C),
    }

    # Engine config: everything off except coupling
    cfg = FactorialEngineConfig(
        dt=0.01, T_steps=1, mode=EngineMode.FULL,
        D_0=0.0, D_min=0.0, beta_0=0.0, gamma_0=0.0,
        h_0=1.0, h_min=0.1, sigma_eta=0.0,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.0,
    )

    coupling_cfg = CouplingConfig(epsilon=0.5, coupling_form=coupling_form)

    R_pairs = compute_pairwise_R_psi(states)
    R_AB = R_pairs[('A', 'B')]

    extra_A = compute_extra_phi_for_module('A', states, R_pairs, coupling_cfg)

    # Step A only
    engine_A = FactorialEngine(cfg, mcfg_A)
    new_state_A, _ = engine_A.step(state_A, phi_extra=extra_A)

    # Centre of mass on T axis
    psi_T_before = state_A.psi.sum(axis=(1, 2))
    psi_T_after = new_state_A.psi.sum(axis=(1, 2))
    mean_T_before = float((THETA_T * psi_T_before).sum())
    mean_T_after = float((THETA_T * psi_T_after).sum())
    delta = mean_T_after - mean_T_before

    psi_T_B = states['B'].psi.sum(axis=(1, 2))
    mean_T_B = float((THETA_T * psi_T_B).sum())

    # Direction toward B
    direction_to_B = mean_T_B - mean_T_before  # positive since B is at +1, A at 0

    if abs(delta) < 1e-7:
        verdict = 'COUPLING_NEUTRAL'
    elif delta * direction_to_B > 0:
        verdict = 'COUPLING_ATTRACTIVE'
    else:
        verdict = 'COUPLING_REPULSIVE'

    return {
        'coupling_form': coupling_form,
        'R_AB': R_AB,
        'mean_T_before_A': mean_T_before,
        'mean_T_after_A': mean_T_after,
        'mean_T_B': mean_T_B,
        'delta_mean_T_A': delta,
        'verdict': verdict,
        'phi_extra_max': float(np.abs(extra_A).max()),
        'phi_extra_at_B_position': float(extra_A[3, 2, 2]),
        'phi_extra_at_A_position': float(extra_A[2, 2, 2]),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("COUPLING SIGN MICRO-TEST — Phase 6b")
    print("=" * 70)

    results = {}
    for form in ['positive', 'contrastive']:
        print(f"\n[{form}]")
        print("-" * 70)
        r = test_coupling_sign(form)
        results[form] = r
        print(f"  R_AB                     = {r['R_AB']:.4f}")
        print(f"  Phi_extra max            = {r['phi_extra_max']:.6f}")
        print(f"  Phi_extra at A's centre  = {r['phi_extra_at_A_position']:.6f}")
        print(f"  Phi_extra at B's centre  = {r['phi_extra_at_B_position']:.6f}")
        print(f"  ψ_A centre of mass on T:")
        print(f"    before = {r['mean_T_before_A']:+.6f}")
        print(f"    after  = {r['mean_T_after_A']:+.6f}")
        print(f"    Δ      = {r['delta_mean_T_A']:+.6f}")
        print(f"  ψ_B centre of mass on T = {r['mean_T_B']:+.6f}")
        print(f"  VERDICT: {r['verdict']}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for form, r in results.items():
        print(f"  {form:<12s} → {r['verdict']}  "
              f"(Δ centre A = {r['delta_mean_T_A']:+.6f})")

    print()
    print("Both forms are documented as non-perspectival baselines for Phase 6c.")
    print("The coupling form chosen for F1/F4/F6 must be loggued in every verdict.")

    # Exit-code policy:
    # The MCQ coupling is by construction differentiating, so both 'positive'
    # and 'contrastive' MUST produce COUPLING_REPULSIVE in this isolation
    # micro-test. Any other verdict means the coupling sign is broken.
    print()
    print("=" * 70)
    expected = {'positive': 'COUPLING_REPULSIVE', 'contrastive': 'COUPLING_REPULSIVE'}
    all_expected = all(results[f]['verdict'] == expected[f] for f in expected)
    if all_expected:
        print("OVERALL: PASS — both coupling forms produce COUPLING_REPULSIVE")
        print("(differentiating direction). The native MCQ coupling sign is correct.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — coupling sign is BROKEN.")
        for f, r in results.items():
            if r['verdict'] != expected[f]:
                print(f"  {f}: got {r['verdict']}, expected {expected[f]}")
        print("Phase 6b coupling does NOT produce the differentiating direction.")
        print("Investigate Φ_corr sign, drift divergence convention, or coupling form.")
        sys.exit(1)
