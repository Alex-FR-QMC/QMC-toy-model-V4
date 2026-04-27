"""
Non-regression test for the Phase 6b extension to engine.step().

Verifies that adding the optional `phi_extra` argument with default value
None does NOT change any output of the Phase 6a engine. The 6a engine
must be bitwise-identical (same RNG sequence, same psi trajectory, same
h trajectory, same step diagnostics) between:

  - "old" behaviour: engine.step(state)
  - "new" behaviour: engine.step(state, phi_extra=None)

If this test fails, the modification has silently broken 6a results
and Phase 6b must NOT be run until fixed.

Run from project root:
    PYTHONPATH=src python tests/phase6a/test_phi_extra_none_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    ModuleConfig, FactorialEngineConfig, EngineMode,
)
from mcq_v4.factorial.engine import FactorialEngine, make_initial_state


def init_psi_concentrated_T() -> np.ndarray:
    """Same init as test_module_isolated.py for consistency."""
    from mcq_v4.factorial.state import THETA_T
    p_T = np.exp(-(THETA_T ** 2) / 2.0)
    p_T /= p_T.sum()
    psi = np.zeros((5, 5, 5))
    psi[:, 2, 2] = p_T
    psi /= psi.sum()
    return psi


def run_with(phi_extra_arg, T_steps: int = 200) -> dict:
    """
    Run the canonical Phase 6a setup with a specific phi_extra arg.
      phi_extra_arg = 'absent'     → step(state) [default API behaviour]
      phi_extra_arg = 'none'        → step(state, phi_extra=None)
      phi_extra_arg = 'zeros'       → step(state, phi_extra=np.zeros((5,5,5)))
    """
    cfg = FactorialEngineConfig(
        dt=0.05, T_steps=T_steps, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    mcfg = ModuleConfig(name="A", weights=(1.5, 0.8, 0.7), seed=42)
    engine = FactorialEngine(cfg, mcfg)
    state = make_initial_state(init_psi_concentrated_T(), mcfg)

    psi_traj = [state.psi.copy()]
    h_T_traj = [state.h_T.copy()]
    h_M_traj = [state.h_M.copy()]
    h_I_traj = [state.h_I.copy()]
    drifts = []

    for t in range(T_steps):
        if phi_extra_arg == 'absent':
            state, diag = engine.step(state)
        elif phi_extra_arg == 'none':
            state, diag = engine.step(state, phi_extra=None)
        elif phi_extra_arg == 'zeros':
            state, diag = engine.step(state, phi_extra=np.zeros((5, 5, 5)))
        else:
            raise ValueError(f"Unknown arg: {phi_extra_arg}")

        psi_traj.append(state.psi.copy())
        h_T_traj.append(state.h_T.copy())
        h_M_traj.append(state.h_M.copy())
        h_I_traj.append(state.h_I.copy())
        drifts.append(diag.mass_drift_step)

    return {
        'psi_final': psi_traj[-1],
        'psi_traj': np.array(psi_traj),
        'h_T_final': h_T_traj[-1],
        'h_M_final': h_M_traj[-1],
        'h_I_final': h_I_traj[-1],
        'drifts': np.array(drifts),
        'mass_final': float(psi_traj[-1].sum()),
    }


def test_absent_vs_none_bitwise():
    """The two API styles must produce bitwise-identical outputs."""
    r_absent = run_with('absent')
    r_none = run_with('none')

    psi_match = np.array_equal(r_absent['psi_final'], r_none['psi_final'])
    h_T_match = np.array_equal(r_absent['h_T_final'], r_none['h_T_final'])
    h_M_match = np.array_equal(r_absent['h_M_final'], r_none['h_M_final'])
    h_I_match = np.array_equal(r_absent['h_I_final'], r_none['h_I_final'])
    drifts_match = np.array_equal(r_absent['drifts'], r_none['drifts'])
    mass_match = (r_absent['mass_final'] == r_none['mass_final'])

    full_traj_match = np.array_equal(r_absent['psi_traj'], r_none['psi_traj'])

    return {
        'psi_final_match': psi_match,
        'h_T_match': h_T_match,
        'h_M_match': h_M_match,
        'h_I_match': h_I_match,
        'drifts_match': drifts_match,
        'mass_match': mass_match,
        'full_psi_trajectory_match': full_traj_match,
        'all_pass': all([psi_match, h_T_match, h_M_match, h_I_match,
                         drifts_match, mass_match, full_traj_match]),
    }


def test_zeros_vs_none_bitwise():
    """phi_extra=zeros((5,5,5)) should also be bitwise-identical to None."""
    r_zeros = run_with('zeros')
    r_none = run_with('none')

    psi_match = np.array_equal(r_zeros['psi_final'], r_none['psi_final'])
    h_T_match = np.array_equal(r_zeros['h_T_final'], r_none['h_T_final'])
    drifts_match = np.array_equal(r_zeros['drifts'], r_none['drifts'])

    full_traj_match = np.array_equal(r_zeros['psi_traj'], r_none['psi_traj'])

    return {
        'psi_final_match': psi_match,
        'h_T_match': h_T_match,
        'drifts_match': drifts_match,
        'full_psi_trajectory_match': full_traj_match,
        'all_pass': all([psi_match, h_T_match, drifts_match, full_traj_match]),
    }


def test_nonzero_phi_extra_changes_output():
    """
    Sanity: phi_extra non-zero MUST change the output. Otherwise the
    integration is not actually wired in.
    """
    r_none = run_with('none')

    cfg = FactorialEngineConfig(
        dt=0.05, T_steps=200, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    mcfg = ModuleConfig(name="A", weights=(1.5, 0.8, 0.7), seed=42)
    engine = FactorialEngine(cfg, mcfg)
    state = make_initial_state(init_psi_concentrated_T(), mcfg)

    # Non-trivial phi_extra: small positive bump on one cell
    extra = np.zeros((5, 5, 5))
    extra[2, 2, 2] = 0.05  # small positive perturbation

    for t in range(200):
        state, _ = engine.step(state, phi_extra=extra)

    psi_final_with_extra = state.psi.copy()

    differs = not np.allclose(r_none['psi_final'], psi_final_with_extra, atol=1e-10)

    return {
        'psi_differs_with_nonzero_extra': differs,
        'max_abs_diff': float(np.abs(r_none['psi_final'] - psi_final_with_extra).max()),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PHI_EXTRA=NONE NON-REGRESSION TEST — Phase 6b API extension")
    print("=" * 70)

    print("\n[1/3] Default API (no phi_extra arg) vs phi_extra=None")
    print("-" * 70)
    r1 = test_absent_vs_none_bitwise()
    for k, v in r1.items():
        if k == 'all_pass':
            continue
        symbol = "✓" if v else "✗"
        print(f"  {symbol} {k}: {v}")
    overall1 = "PASS" if r1['all_pass'] else "FAIL"
    print(f"  Overall: {overall1}")

    print("\n[2/3] phi_extra=zeros((5,5,5)) vs phi_extra=None")
    print("-" * 70)
    r2 = test_zeros_vs_none_bitwise()
    for k, v in r2.items():
        if k == 'all_pass':
            continue
        symbol = "✓" if v else "✗"
        print(f"  {symbol} {k}: {v}")
    overall2 = "PASS" if r2['all_pass'] else "FAIL"
    print(f"  Overall: {overall2}")

    print("\n[3/3] Sanity: non-zero phi_extra DOES change output")
    print("-" * 70)
    r3 = test_nonzero_phi_extra_changes_output()
    print(f"  psi differs with non-zero extra: {r3['psi_differs_with_nonzero_extra']}")
    print(f"  max abs diff: {r3['max_abs_diff']:.4e}")
    overall3 = "PASS" if r3['psi_differs_with_nonzero_extra'] else "FAIL"
    print(f"  Overall: {overall3}")

    print()
    print("=" * 70)
    if all([r1['all_pass'], r2['all_pass'], r3['psi_differs_with_nonzero_extra']]):
        print("OVERALL: PASS — phi_extra extension is non-regressive and effective.")
        print("Phase 6a engine outputs are unchanged when phi_extra is None or zero.")
        print("Phase 6b development can proceed.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — phi_extra extension has broken Phase 6a or is not wired in.")
        print("Phase 6b development MUST NOT proceed until this is fixed.")
        sys.exit(1)
