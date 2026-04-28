"""
Preflight 6 — RNG independence test for Phase 6b.

Verifies three things:
  1. The three FactorialEngines hold DISTINCT RNG objects in memory
     (id-distinctness — protects against shared-by-reference regression).
  2. The three RNGs produce DIFFERENT sequences when sampled
     (functional distinctness — protects against same-seed regression
     even when objects are distinct).
  3. The seed log is properly populated and self-consistent.

If any of these fails, phantom synchronisation via shared noise is
possible, and Phase 6b results are NOT interpretable.

Run from project root:
    PYTHONPATH=src python tests/phase6b/test_rng_independence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import FactorialEngineConfig, EngineMode
from mcq_v4.factorial.three_module_system import (
    DIFFERENTIATED_WEIGHTS, CouplingConfig, build_three_module_system,
)


def _trivial_initial_psi():
    """Identical psi for all modules — isolates RNG effect from IC effect."""
    psi = np.zeros((5, 5, 5))
    psi[2, 2, 2] = 1.0
    return {'A': psi.copy(), 'B': psi.copy(), 'C': psi.copy()}


def test_rng_object_distinctness() -> dict:
    """The three engines must hold rng objects with distinct memory ids."""
    cfg_engine = FactorialEngineConfig(
        dt=0.05, T_steps=1, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    coupling_cfg = CouplingConfig(epsilon=0.0)
    sys = build_three_module_system(cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
                                    _trivial_initial_psi(), base_seed=42)

    rng_ids = {
        'A': id(sys.engine_A.rng),
        'B': id(sys.engine_B.rng),
        'C': id(sys.engine_C.rng),
    }
    n_distinct = len(set(rng_ids.values()))
    pass_distinctness = (n_distinct == 3)

    return {
        'rng_ids': rng_ids,
        'n_distinct_objects': n_distinct,
        'pass': pass_distinctness,
        'message': (
            "Three distinct RNG objects in memory."
            if pass_distinctness
            else "RNG objects are NOT all distinct — phantom synchronisation possible."
        ),
    }


def test_rng_sequence_distinctness() -> dict:
    """
    Even when objects are distinct, they could be seeded identically.
    Functional test: draw N samples from each RNG and verify the
    sequences differ.
    """
    cfg_engine = FactorialEngineConfig(
        dt=0.05, T_steps=1, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    coupling_cfg = CouplingConfig(epsilon=0.0)
    sys = build_three_module_system(cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
                                    _trivial_initial_psi(), base_seed=42)

    n_samples = 100
    samples_A = sys.engine_A.rng.standard_normal(n_samples)
    samples_B = sys.engine_B.rng.standard_normal(n_samples)
    samples_C = sys.engine_C.rng.standard_normal(n_samples)

    AB_distinct = not np.array_equal(samples_A, samples_B)
    BC_distinct = not np.array_equal(samples_B, samples_C)
    AC_distinct = not np.array_equal(samples_A, samples_C)

    # Cross-correlation should be near 0 for independent streams
    corr_AB = float(np.corrcoef(samples_A, samples_B)[0, 1])
    corr_BC = float(np.corrcoef(samples_B, samples_C)[0, 1])
    corr_AC = float(np.corrcoef(samples_A, samples_C)[0, 1])

    pass_distinct = AB_distinct and BC_distinct and AC_distinct
    pass_correlation = all(abs(c) < 0.3 for c in [corr_AB, corr_BC, corr_AC])

    return {
        'AB_sequences_distinct': AB_distinct,
        'BC_sequences_distinct': BC_distinct,
        'AC_sequences_distinct': AC_distinct,
        'correlation_AB': corr_AB,
        'correlation_BC': corr_BC,
        'correlation_AC': corr_AC,
        'pass_distinctness': pass_distinct,
        'pass_correlation': pass_correlation,
        'pass': pass_distinct and pass_correlation,
        'message': (
            f"Three RNG streams are functionally independent "
            f"(correlations |c| < 0.3 for n={n_samples})."
            if (pass_distinct and pass_correlation)
            else "RNG streams show suspicious correlation or identity. Investigate."
        ),
    }


def test_seed_log() -> dict:
    """The seed log must be populated with the expected offsets."""
    cfg_engine = FactorialEngineConfig(
        dt=0.05, T_steps=1, mode=EngineMode.FULL,
        D_0=0.02, D_min=0.002, beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1, sigma_eta=0.10,
        var_min=0.5, var_max=2.5, H_min=0.5, lambda_KNV=0.05,
    )
    coupling_cfg = CouplingConfig(epsilon=0.0)
    sys = build_three_module_system(cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
                                    _trivial_initial_psi(), base_seed=42)

    log = sys.seed_log()
    expected_seeds = {'A': 143, 'B': 244, 'C': 345}
    actual_seeds = {'A': log['seed_A'], 'B': log['seed_B'], 'C': log['seed_C']}

    seeds_match = (actual_seeds == expected_seeds)
    base_seed_match = (log['base_seed'] == 42)
    rngs_independent_flag = log['rngs_independent']

    return {
        'log': log,
        'expected_seeds': expected_seeds,
        'actual_seeds': actual_seeds,
        'seeds_match': seeds_match,
        'base_seed_match': base_seed_match,
        'rngs_independent_flag': rngs_independent_flag,
        'pass': seeds_match and base_seed_match and rngs_independent_flag,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PREFLIGHT 6 — RNG independence test")
    print("=" * 70)

    print("\n[1/3] Object distinctness (id-level)")
    print("-" * 70)
    r1 = test_rng_object_distinctness()
    print(f"  RNG ids: {r1['rng_ids']}")
    print(f"  Distinct: {r1['n_distinct_objects']}/3")
    print(f"  Pass: {r1['pass']} — {r1['message']}")

    print("\n[2/3] Sequence distinctness (functional)")
    print("-" * 70)
    r2 = test_rng_sequence_distinctness()
    print(f"  AB distinct: {r2['AB_sequences_distinct']}")
    print(f"  BC distinct: {r2['BC_sequences_distinct']}")
    print(f"  AC distinct: {r2['AC_sequences_distinct']}")
    print(f"  Correlations: AB={r2['correlation_AB']:.4f}, "
          f"BC={r2['correlation_BC']:.4f}, AC={r2['correlation_AC']:.4f}")
    print(f"  Pass: {r2['pass']}")

    print("\n[3/3] Seed log integrity")
    print("-" * 70)
    r3 = test_seed_log()
    print(f"  expected: {r3['expected_seeds']}")
    print(f"  actual:   {r3['actual_seeds']}")
    print(f"  base_seed correct: {r3['base_seed_match']}")
    print(f"  rngs_independent flag: {r3['rngs_independent_flag']}")
    print(f"  Pass: {r3['pass']}")

    print()
    print("=" * 70)
    if r1['pass'] and r2['pass'] and r3['pass']:
        print("OVERALL: PASS — RNG independence verified at three levels.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — RNG independence is compromised.")
        print("Phase 6b results would suffer from phantom synchronisation.")
        sys.exit(1)
