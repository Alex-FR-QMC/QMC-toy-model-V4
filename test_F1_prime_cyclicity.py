"""
Phase 6c-B — F1' Cyclicity test.

Plan-initial requirement (6b F1'):
    Run A+B+C coupled. Measure how the dynamics of the three shared
    factors (k_AB, k_BC, k_CA) reflects the cyclic topology.
    Expected: no hierarchical relation between them, but non-trivial
    covariance (cyclic phase shifts possible).

This test is FREE CHARACTERISATION, not PASS/FAIL.

Outputs (per coupling form):
  - cov_matrix          : 3x3 temporal covariance matrix of (k_AB, k_BC, k_CA)
  - corr_matrix         : 3x3 Pearson correlation
  - phase_shifts        : pairwise lag of max cross-correlation in steps
  - dominant_period     : autocorrelation period estimate per factor
  - cyclic_signature    : qualitative classification

Cyclic signature classification:
  CYCLIC_PHASE_SHIFTED  : phase shifts ≈ ±period/3 between cyclically
                          adjacent pairs, suggesting a 2π/3 rotation
                          pattern in the shared-factor dynamics.
  SYNCHRONOUS           : all phase shifts ≈ 0 → modules co-vary in lockstep.
                          (Expected for label-based contrastive at low ε.)
  ANTI_PHASED           : phase shift ≈ ±period/2 between some pairs.
  NO_CLEAR_PERIOD       : autocorrelation does not produce a stable period;
                          regime is non-oscillatory or chaotic.
  NO_COVARIANCE         : pairwise correlations all near zero — modules
                          are decoupled in the shared subspace.

This is the "topological coherence" reading of F1' (cf. plan v6c-B §7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    FactorialEngineConfig, EngineMode, THETA_T,
)
from mcq_v4.factorial.three_module_system import (
    DIFFERENTIATED_WEIGHTS, CouplingConfig, build_three_module_system,
)
from mcq_v4.factorial.coupling import run_three_modules, step_three_modules
from mcq_v4.factorial.tau_prime import compute_tau_prime_3modules
from mcq_v4.factorial.signatures_6c_b import estimate_dominant_period


def make_initial_psi():
    """Same canonical initial conditions as the rest of Phase 6c."""
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


def cross_correlation_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 30) -> tuple:
    """
    Returns (best_lag, best_corr) where best_lag minimises in [-max_lag, max_lag].

    Positive lag means y leads x (i.e. y(t) ≈ x(t + lag)).
    """
    x = x - x.mean()
    y = y - y.mean()
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return 0, 0.0
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag >= 0:
            xs = x[lag:]; ys = y[:n - lag] if n - lag > 0 else y[:0]
        else:
            xs = x[:n + lag] if n + lag > 0 else x[:0]; ys = y[-lag:]
        if len(xs) < 5:
            corrs.append(0.0); continue
        c = float(np.dot(xs, ys) / (nx * ny))
        corrs.append(c)
    corrs = np.array(corrs)
    idx_max = int(np.argmax(np.abs(corrs)))
    return int(lags[idx_max]), float(corrs[idx_max])


def classify_cyclic_signature(
    corr_matrix: np.ndarray,
    phase_shifts: dict,
    dominant_period: int,
) -> str:
    """Qualitative classification of the (k_AB, k_BC, k_CA) dynamics."""
    # Mean off-diagonal correlation magnitude
    off_diag = [corr_matrix[0, 1], corr_matrix[0, 2], corr_matrix[1, 2]]
    max_abs_corr = max(abs(c) for c in off_diag)
    if max_abs_corr < 0.1:
        return 'NO_COVARIANCE'

    if dominant_period <= 5:
        return 'NO_CLEAR_PERIOD'

    # Phase shifts: AB→BC, BC→CA, CA→AB
    shifts = [
        phase_shifts['AB_vs_BC'],
        phase_shifts['BC_vs_CA'],
        phase_shifts['CA_vs_AB'],
    ]
    third_period = dominant_period / 3.0
    half_period = dominant_period / 2.0
    tol_third = max(2, dominant_period // 6)

    n_third_phase = sum(
        1 for s in shifts
        if abs(abs(s) - third_period) < tol_third
    )
    n_half_phase = sum(
        1 for s in shifts
        if abs(abs(s) - half_period) < tol_third
    )
    n_zero_phase = sum(1 for s in shifts if abs(s) < tol_third)

    if n_third_phase >= 2:
        return 'CYCLIC_PHASE_SHIFTED'
    if n_half_phase >= 2:
        return 'ANTI_PHASED'
    if n_zero_phase >= 2:
        return 'SYNCHRONOUS'
    return 'MIXED_NO_DOMINANT_PATTERN'


def measure_F1_prime_for_form(
    cfg_engine,
    coupling_form: str,
    eps: float,
    base_seed: int,
    n_steps: int = 300,
    warmup_steps: int = 50,
) -> dict:
    """Run one form, capture k_AB/k_BC/k_CA trajectories, characterise."""
    initial_psi = make_initial_psi()
    coupling_cfg = CouplingConfig(epsilon=eps, coupling_form=coupling_form)
    sys_obj = build_three_module_system(
        cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )

    # Warmup with coupling OFF (consistent with 6c-A protocol)
    for _ in range(warmup_steps):
        sys_obj, _ = step_three_modules(sys_obj, coupling_active=False)

    # Measurement with coupling ON
    k_AB_traj, k_BC_traj, k_CA_traj = [], [], []
    for _ in range(n_steps):
        sys_obj, _ = step_three_modules(sys_obj, coupling_active=True)
        out = compute_tau_prime_3modules(sys_obj.states)
        k_AB_traj.append(out.k_AB)
        k_BC_traj.append(out.k_BC)
        k_CA_traj.append(out.k_CA)

    k_AB = np.array(k_AB_traj); k_BC = np.array(k_BC_traj); k_CA = np.array(k_CA_traj)

    # Covariance / correlation
    stacked = np.stack([k_AB, k_BC, k_CA], axis=0)
    cov = np.cov(stacked)
    # Pearson correlation
    stds = np.std(stacked, axis=1)
    if (stds < 1e-12).any():
        corr = np.eye(3)
    else:
        corr = cov / np.outer(stds, stds)

    # Period estimate per factor
    p_AB = estimate_dominant_period(k_AB)
    p_BC = estimate_dominant_period(k_BC)
    p_CA = estimate_dominant_period(k_CA)
    dominant_period = int(np.median([p_AB, p_BC, p_CA]))

    # Phase shifts (pairwise lag of max cross-correlation)
    shifts = {}
    lag, c = cross_correlation_lag(k_AB, k_BC); shifts['AB_vs_BC'] = lag; shifts['AB_vs_BC_corr'] = c
    lag, c = cross_correlation_lag(k_BC, k_CA); shifts['BC_vs_CA'] = lag; shifts['BC_vs_CA_corr'] = c
    lag, c = cross_correlation_lag(k_CA, k_AB); shifts['CA_vs_AB'] = lag; shifts['CA_vs_AB_corr'] = c

    sig = classify_cyclic_signature(corr, shifts, dominant_period)

    return {
        'coupling_form': coupling_form,
        'epsilon': eps,
        'n_steps': n_steps,
        'warmup_steps': warmup_steps,
        'cov_matrix': cov.tolist(),
        'corr_matrix': corr.tolist(),
        'phase_shifts': shifts,
        'period_per_factor': {'k_AB': int(p_AB), 'k_BC': int(p_BC), 'k_CA': int(p_CA)},
        'dominant_period': dominant_period,
        'cyclic_signature': sig,
        'k_AB_first10': k_AB[:10].tolist(),
        'k_AB_last10': k_AB[-10:].tolist(),
        'k_AB_mean': float(k_AB.mean()),
        'k_AB_std': float(k_AB.std()),
        'k_BC_mean': float(k_BC.mean()),
        'k_BC_std': float(k_BC.std()),
        'k_CA_mean': float(k_CA.mean()),
        'k_CA_std': float(k_CA.std()),
    }


def run_F1_prime_test(base_seed: int = 42) -> dict:
    print("=" * 70)
    print("F1' — Cyclicity (k_AB / k_BC / k_CA temporal covariance + phase)")
    print("=" * 70)

    cfg = FactorialEngineConfig(
        dt=0.05, T_steps=400, mode=EngineMode.FULL,
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
        r = measure_F1_prime_for_form(cfg, form, eps, base_seed)
        per_form[form] = r
        print(f"period={r['dominant_period']}, "
              f"signature={r['cyclic_signature']}, "
              f"corr_AB_BC={r['corr_matrix'][0][1]:+.3f}, "
              f"shift_AB_BC={r['phase_shifts']['AB_vs_BC']}")

    return {
        'test': 'F1_prime_cyclicity',
        'per_form': per_form,
        'base_seed': base_seed,
        'epsilon': eps,
        'note': "Free characterisation — no PASS/FAIL. Cyclic signature is qualitative.",
    }


if __name__ == "__main__":
    result = run_F1_prime_test(base_seed=42)
    out = Path("/home/claude/mcq_v4/results/phase6c_b/F1_prime_cyclicity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print()
    print("Summary by form:")
    for form, r in result['per_form'].items():
        print(f"  {form:<32s} → {r['cyclic_signature']:<25s} "
              f"period={r['dominant_period']}")
