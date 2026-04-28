"""
ATTRACTIVE_SIGN_CONTROL — annexe micro-test.

This is NOT a Phase 6b model. It is a sign-control reference used only
to verify that the MCQ coupling's anti-fusion behaviour is a property
of the SIGN, not of some other architectural detail.

Setup: identical to test_coupling_sign.py, but with the coupling sign
deliberately inverted (negate phi_extra). Expected verdict:
COUPLING_ATTRACTIVE — module A moves TOWARD module B's centre of mass.

If this control passes (= attractive when sign inverted), it confirms
that the MCQ coupling form, with its native sign, is correctly oriented
toward differentiation rather than convergence.

This file is annexe — its results do NOT enter the Phase 6b verdict.
It is referenced in the README as a methodological control.

Run from project root:
    PYTHONPATH=src python tests/phase6b/annexe_attractive_sign_control.py
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


def attractive_sign_control(coupling_form: str) -> dict:
    """
    Run the same micro-test as test_coupling_sign, but invert the sign
    of phi_extra to confirm sign asymmetry of the result.

    With native MCQ sign: COUPLING_REPULSIVE (anti-fusion).
    With inverted sign:    expected COUPLING_ATTRACTIVE (control).
    """
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
    extra_A = compute_extra_phi_for_module('A', states, R_pairs, coupling_cfg)

    # SIGN INVERSION (this is the only difference from test_coupling_sign)
    extra_A_inverted = -extra_A

    engine_A = FactorialEngine(cfg, mcfg_A)
    new_state_A, _ = engine_A.step(state_A, phi_extra=extra_A_inverted)

    psi_T_before = state_A.psi.sum(axis=(1, 2))
    psi_T_after = new_state_A.psi.sum(axis=(1, 2))
    mean_T_before = float((THETA_T * psi_T_before).sum())
    mean_T_after = float((THETA_T * psi_T_after).sum())
    delta = mean_T_after - mean_T_before

    psi_T_B = states['B'].psi.sum(axis=(1, 2))
    mean_T_B = float((THETA_T * psi_T_B).sum())

    direction_to_B = mean_T_B - mean_T_before

    if abs(delta) < 1e-7:
        verdict = 'COUPLING_NEUTRAL'
    elif delta * direction_to_B > 0:
        verdict = 'COUPLING_ATTRACTIVE'
    else:
        verdict = 'COUPLING_REPULSIVE'

    return {
        'coupling_form': coupling_form,
        'sign_inverted': True,
        'mean_T_before_A': mean_T_before,
        'mean_T_after_A': mean_T_after,
        'mean_T_B': mean_T_B,
        'delta_mean_T_A': delta,
        'verdict': verdict,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("ATTRACTIVE SIGN CONTROL — Phase 6b annexe")
    print("=" * 70)
    print()
    print("Methodological reference. Inverts the sign of phi_extra to verify")
    print("that the native MCQ coupling's REPULSIVE direction is a property")
    print("of the sign, not of an unrelated architectural detail.")
    print()
    print("Expected: with inverted sign, A moves TOWARD B (COUPLING_ATTRACTIVE).")
    print("Expected: this confirms the native MCQ coupling is structurally")
    print("differentiating, not convergent.")
    print()

    all_pass = True
    for form in ['positive', 'contrastive']:
        print(f"[{form}]")
        print("-" * 70)
        r = attractive_sign_control(form)
        print(f"  ψ_A centre on T: {r['mean_T_before_A']:+.6f} → {r['mean_T_after_A']:+.6f}")
        print(f"  Δ = {r['delta_mean_T_A']:+.6f}")
        print(f"  ψ_B centre = {r['mean_T_B']:+.6f}")
        print(f"  Verdict: {r['verdict']}")
        if r['verdict'] != 'COUPLING_ATTRACTIVE':
            all_pass = False
            print(f"  WARNING: expected COUPLING_ATTRACTIVE under sign inversion.")
        print()

    print("=" * 70)
    if all_pass:
        print("SIGN ASYMMETRY CONFIRMED: native MCQ coupling = REPULSIVE,")
        print("inverted sign = ATTRACTIVE. The coupling direction is a property")
        print("of the sign, not an artefact of another mechanism.")
        sys.exit(0)
    else:
        print("UNEXPECTED: sign inversion did not flip the coupling direction.")
        print("Investigate before proceeding.")
        # Methodological annex — non-zero exit triggers a warning in the
        # preflight orchestrator (preflight 5), but does NOT block 6b
        # interpretation. See orchestrator policy.
        sys.exit(1)
