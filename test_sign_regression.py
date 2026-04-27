"""
Sign regression test for Phase 6a.

Verifies that with var < var_min and Φ_corr active, the regulatory drift
moves mass FROM the centre of the perturbed axis TO its neighbours. This
is the anti-collapse signature.

If a future change inverts the sign of:
  - Φ_corr's anti-collapse term, OR
  - the divergence convention in _drift,
this test will catch it.

The test isolates the drift by:
  - disabling diffusion (D_0 = 0 in step computation, and h irrelevant since
    the diffusion term factor is D_eff which goes to zero)
  - disabling noise (sigma_eta = 0)
  - freezing h evolution (beta_0 = 0, gamma_0 = 0)

Run from the project root:
    PYTHONPATH=src python tests/phase6a/test_sign_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make src importable
_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    ModuleConfig, FactorialEngineConfig, EngineMode, FactorialState,
    THETA_T,
)
from mcq_v4.factorial.engine import FactorialEngine, make_initial_state
from mcq_v4.factorial.observables import compute_Phi_corr, compute_observables


def test_drift_sign_anti_collapse() -> dict:
    """
    Micro-test: with ψ concentrated on T axis (var_T < var_min), the drift
    contribution to ∂_t ψ should:
      - be NEGATIVE at the centre (mass leaves)
      - be POSITIVE at the immediate neighbours (mass arrives)

    Equivalently: variance should grow under drift alone.
    """
    mcfg = ModuleConfig(name="A", weights=(1.5, 0.8, 0.7), seed=42)
    cfg = FactorialEngineConfig(
        dt=0.01, T_steps=1, mode=EngineMode.FULL,
        D_0=0.0, D_min=0.0,                      # no diffusion
        beta_0=0.0, gamma_0=0.0,                  # no metric evolution
        h_0=1.0, h_min=0.1,
        sigma_eta=0.0,                            # no noise
        var_min=0.5, var_max=2.5, H_min=0.5,
        lambda_KNV=0.05,
    )

    # Initial ψ: 80% mass at θ_T = 0 (centre), 10% on each neighbour θ_T = ±1
    # Concentrated on M = 2, I = 2 (centre cells).
    psi = np.zeros((5, 5, 5))
    psi[2, 2, 2] = 0.8
    psi[1, 2, 2] = 0.1
    psi[3, 2, 2] = 0.1
    state = make_initial_state(psi, mcfg)

    # Confirm var_T < var_min (so the anti-collapse bump activates)
    obs = compute_observables(state.psi)
    assert obs["var_T"] < cfg.var_min, (
        f"Test setup error: var_T = {obs['var_T']} should be < var_min = {cfg.var_min}"
    )

    # Compute Phi_corr and isolated drift
    Phi = compute_Phi_corr(obs, cfg, EngineMode.FULL)
    engine = FactorialEngine(cfg, mcfg)
    drift_term = engine._drift(state.psi, state.h_T, state.h_M, state.h_I, Phi)

    # Extract the slice along T at (M=2, I=2)
    centre = drift_term[2, 2, 2]
    left = drift_term[1, 2, 2]
    right = drift_term[3, 2, 2]

    # Sign expectations
    centre_decreases = (centre < -1e-6)
    left_increases = (left > 1e-6)
    right_increases = (right > 1e-6)

    # Verdict
    if centre_decreases and left_increases and right_increases:
        outcome = "PASS"
        msg = "Drift IS anti-collapse (centre loses mass, neighbours gain)."
    elif centre > 0 and left < 0 and right < 0:
        outcome = "FAIL_PRO_COLLAPSE"
        msg = (
            "Drift IS pro-collapse: centre gains mass, neighbours lose. "
            "This indicates a sign error in Φ_corr or in the drift divergence "
            "convention. Phase 6a results are NOT physically meaningful "
            "until this is fixed."
        )
    else:
        outcome = "FAIL_MIXED"
        msg = (
            f"Drift does not have a clean anti-collapse signature. "
            f"centre={centre:.4e}, left={left:.4e}, right={right:.4e}. "
            f"Investigate — could indicate sign inconsistency between drift "
            f"and Φ_corr, or numerical issues."
        )

    return {
        "outcome": outcome,
        "message": msg,
        "Phi_T_profile": [float(Phi[i, 2, 2]) for i in range(5)],
        "drift_T_profile": [float(drift_term[i, 2, 2]) for i in range(5)],
        "psi_T_profile": [float(state.psi[i, 2, 2]) for i in range(5)],
        "var_T": float(obs["var_T"]),
        "var_min_threshold": cfg.var_min,
        "centre_dpsi_dt": float(centre),
        "left_dpsi_dt": float(left),
        "right_dpsi_dt": float(right),
    }


def test_diffusion_sign_spread() -> dict:
    """
    Sanity: pure diffusion (no drift, no noise) on init concentrated at centre
    should spread mass outward. ψ at centre should DECREASE, neighbours INCREASE.
    """
    mcfg = ModuleConfig(name="A", weights=(1.0, 1.0, 1.0), seed=42)
    cfg = FactorialEngineConfig(
        dt=0.01, T_steps=1, mode=EngineMode.NO_REGULATION_BASELINE,
        D_0=0.1, D_min=0.001,
        beta_0=0.0, gamma_0=0.0,
        h_0=1.0, h_min=0.1,
        sigma_eta=0.0,
        var_min=0.5, var_max=2.5, H_min=0.5,
        lambda_KNV=0.0,                          # no Φ_corr in baseline anyway
    )

    psi = np.zeros((5, 5, 5))
    psi[2, 2, 2] = 1.0
    state = FactorialState(psi=psi, h_T=np.ones(5), h_M=np.ones(5), h_I=np.ones(5), cfg=mcfg)

    engine = FactorialEngine(cfg, mcfg)
    new_state, _ = engine.step(state)

    centre_after = float(new_state.psi[2, 2, 2])
    neighbour_T_after = float(new_state.psi[1, 2, 2])
    neighbour_M_after = float(new_state.psi[2, 1, 2])
    neighbour_I_after = float(new_state.psi[2, 2, 1])

    if (centre_after < 1.0 - 1e-6 and
        neighbour_T_after > 1e-6 and
        neighbour_M_after > 1e-6 and
        neighbour_I_after > 1e-6):
        outcome = "PASS"
        msg = "Diffusion correctly spreads mass from centre to all 6 neighbours."
    elif centre_after > 1.0 - 1e-6:
        outcome = "FAIL_NO_DIFFUSION"
        msg = "Centre did not lose mass — diffusion ineffective or sign-inverted."
    else:
        outcome = "FAIL_ASYMMETRIC"
        msg = "Some neighbours did not receive mass — asymmetric diffusion."

    return {
        "outcome": outcome,
        "message": msg,
        "centre_after": centre_after,
        "neighbour_T_after": neighbour_T_after,
        "neighbour_M_after": neighbour_M_after,
        "neighbour_I_after": neighbour_I_after,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SIGN REGRESSION TEST — Phase 6a")
    print("=" * 70)

    # Test 1: drift anti-collapse
    print("\n[1/2] Drift sign — anti-collapse with var < var_min")
    print("-" * 70)
    r1 = test_drift_sign_anti_collapse()
    print(f"  Phi profile (T axis at M=2, I=2):")
    for i, p in enumerate(r1["Phi_T_profile"]):
        print(f"    θ_T={THETA_T[i]:+.0f}: Phi = {p:+.6f}")
    print(f"  Drift dψ/dt (T axis at M=2, I=2):")
    for i, d in enumerate(r1["drift_T_profile"]):
        print(f"    θ_T={THETA_T[i]:+.0f}: dψ/dt = {d:+.6f}")
    print(f"\n  Outcome: {r1['outcome']}")
    print(f"  {r1['message']}")

    # Test 2: diffusion sign
    print("\n[2/2] Diffusion sign — spread from concentrated init")
    print("-" * 70)
    r2 = test_diffusion_sign_spread()
    print(f"  centre_after: {r2['centre_after']:.6f} (was 1.0)")
    print(f"  neighbour T:  {r2['neighbour_T_after']:.6f}")
    print(f"  neighbour M:  {r2['neighbour_M_after']:.6f}")
    print(f"  neighbour I:  {r2['neighbour_I_after']:.6f}")
    print(f"\n  Outcome: {r2['outcome']}")
    print(f"  {r2['message']}")

    # Aggregate
    print()
    print("=" * 70)
    if r1["outcome"] == "PASS" and r2["outcome"] == "PASS":
        print("OVERALL: PASS — sign conventions are physically correct.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — sign error detected. See messages above.")
        sys.exit(1)
