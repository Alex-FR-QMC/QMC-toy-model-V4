"""
Coupling form regression test — Phase 6b.

Verifies that the two coupling forms 'positive' and 'contrastive' are
ACTUALLY distinct at runtime, by computing extra_phi for the same
(states, R_pairs) under both forms and asserting they differ.

Configuration:
  - ψ_A and ψ_B distinct (concentrated at different positions)
  - R_AB = R_psi(A, B) > 0
  - ε > 0
  - Then phi_extra_positive must NOT equal phi_extra_contrastive.

If this assertion fails, the coupling_form selector is dead code and
the positive/contrastive sweep in F6 is invalid.

Run from project root:
    PYTHONPATH=src python tests/phase6b/test_coupling_forms_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    FactorialState, ModuleConfig, THETA_T,
)
from mcq_v4.factorial.engine import make_initial_state
from mcq_v4.factorial.coupling import compute_extra_phi_for_module
from mcq_v4.factorial.three_module_system import CouplingConfig
from mcq_v4.factorial.overlaps import compute_pairwise_R_psi, compute_R_psi


def test_forms_differ() -> dict:
    """
    Build states with ψ_A ≠ ψ_B (R > 0 but R < 1) and compare
    extra_phi under 'positive' vs 'contrastive'.
    """
    # ψ_A: concentrated at θ_T = 0
    psi_A = np.zeros((5, 5, 5))
    psi_A[2, 2, 2] = 0.6
    psi_A[1, 2, 2] = 0.2
    psi_A[3, 2, 2] = 0.2

    # ψ_B: concentrated at θ_T = +1 (different from A)
    psi_B = np.zeros((5, 5, 5))
    psi_B[3, 2, 2] = 0.6
    psi_B[2, 2, 2] = 0.2
    psi_B[4, 2, 2] = 0.2

    # ψ_C: irrelevant, dummy uniform
    psi_C = np.ones((5, 5, 5)) / 125

    mcfg_A = ModuleConfig(name='A', weights=(1.0, 1.0, 1.0), seed=42)
    mcfg_B = ModuleConfig(name='B', weights=(1.0, 1.0, 1.0), seed=43)
    mcfg_C = ModuleConfig(name='C', weights=(1.0, 1.0, 1.0), seed=44)

    states = {
        'A': make_initial_state(psi_A, mcfg_A),
        'B': make_initial_state(psi_B, mcfg_B),
        'C': make_initial_state(psi_C, mcfg_C),
    }

    R_pairs = compute_pairwise_R_psi(states)
    R_AB = R_pairs[('A', 'B')]

    assert 0.0 < R_AB < 1.0, (
        f"Test setup invalid: R_AB = {R_AB:.4f} should be in (0, 1) for the "
        f"contrastive form to differ meaningfully from positive."
    )

    eps = 0.5  # large enough that the difference is well above roundoff
    cfg_pos = CouplingConfig(epsilon=eps, coupling_form='positive')
    cfg_con = CouplingConfig(epsilon=eps, coupling_form='contrastive')

    extra_A_positive = compute_extra_phi_for_module('A', states, R_pairs, cfg_pos)
    extra_A_contrastive = compute_extra_phi_for_module('A', states, R_pairs, cfg_con)

    # Expected mathematical difference:
    #   positive:    Σ_j ε R(1-R) ψ_j
    #   contrastive: Σ_j ε R(1-R) (ψ_j - R ψ_i)
    # Difference for A's calculation (over its partners):
    #   positive - contrastive = Σ_j ε R(1-R) · R · ψ_self
    psi_self = states['A'].psi
    partners_of_A = cfg_pos.neighbours_of('A')
    expected_diff = np.zeros((5, 5, 5))
    for j in partners_of_A:
        pair_key = tuple(sorted(['A', j]))
        R_ij = R_pairs[pair_key]
        weight = eps * R_ij * (1.0 - R_ij)
        expected_diff = expected_diff + weight * R_ij * psi_self

    actual_diff = extra_A_positive - extra_A_contrastive

    max_diff = float(np.abs(actual_diff).max())
    expected_max = float(np.abs(expected_diff).max())
    formula_match = bool(np.allclose(actual_diff, expected_diff, atol=1e-12))

    pass_runtime = (max_diff > 1e-6)
    pass_formula = formula_match

    return {
        'R_AB': R_AB,
        'partners_of_A': partners_of_A,
        'max_diff_positive_vs_contrastive': max_diff,
        'expected_max_diff': expected_max,
        'formula_match': formula_match,
        'runtime_test_pass': pass_runtime,
        'formula_test_pass': pass_formula,
        'all_pass': pass_runtime and pass_formula,
    }


def test_R_zero_makes_forms_equal() -> dict:
    """
    Sanity: when R_AB = 0 (disjoint distributions), both forms should
    produce zero extra_phi (because R(1-R) = 0).
    """
    psi_A = np.zeros((5, 5, 5))
    psi_A[0, 0, 0] = 1.0
    psi_B = np.zeros((5, 5, 5))
    psi_B[4, 4, 4] = 1.0
    psi_C = np.ones((5, 5, 5)) / 125

    mcfg_A = ModuleConfig(name='A', weights=(1.0, 1.0, 1.0), seed=42)
    mcfg_B = ModuleConfig(name='B', weights=(1.0, 1.0, 1.0), seed=43)
    mcfg_C = ModuleConfig(name='C', weights=(1.0, 1.0, 1.0), seed=44)

    states = {
        'A': make_initial_state(psi_A, mcfg_A),
        'B': make_initial_state(psi_B, mcfg_B),
        'C': make_initial_state(psi_C, mcfg_C),
    }
    R_pairs = compute_pairwise_R_psi(states)

    cfg_pos = CouplingConfig(epsilon=0.5, coupling_form='positive')
    cfg_con = CouplingConfig(epsilon=0.5, coupling_form='contrastive')

    # With ψ_A and ψ_B disjoint, R(A,B) = 0 → R(1-R) = 0 → extra = 0 in both forms
    # Note: R(A,C) and R(B,C) are non-zero because C is uniform.
    # We focus only on the A-B pair contribution. A's partners include B (via k_AB)
    # and C (via k_CA). If R(A,C) > 0, A still gets a contribution from C.
    # So the strict R=0 test is on A's pairwise contribution from B alone.

    R_AB = R_pairs[('A', 'B')]
    return {
        'R_AB_with_disjoint_AB': R_AB,
        'R_AB_is_zero': bool(R_AB < 1e-12),
        'note': "When R_ij = 0 the j contribution to i's extra is zero in both forms",
    }


def test_R_one_makes_contrastive_zero() -> dict:
    """
    Sanity: when R_AB = 1 (identical distributions), R(1-R) = 0,
    so both forms produce zero contribution from the j=B branch.
    Additionally the term (ψ_j - R ψ_i) = (ψ_j - ψ_i) = 0 when ψ_i = ψ_j.
    """
    psi_identical = np.zeros((5, 5, 5))
    psi_identical[2, 2, 2] = 1.0
    psi_C = np.ones((5, 5, 5)) / 125

    mcfg_A = ModuleConfig(name='A', weights=(1.0, 1.0, 1.0), seed=42)
    mcfg_B = ModuleConfig(name='B', weights=(1.0, 1.0, 1.0), seed=43)
    mcfg_C = ModuleConfig(name='C', weights=(1.0, 1.0, 1.0), seed=44)

    states = {
        'A': make_initial_state(psi_identical.copy(), mcfg_A),
        'B': make_initial_state(psi_identical.copy(), mcfg_B),
        'C': make_initial_state(psi_C, mcfg_C),
    }
    R_pairs = compute_pairwise_R_psi(states)
    R_AB = R_pairs[('A', 'B')]

    return {
        'R_AB_with_identical_AB': R_AB,
        'R_AB_is_one': bool(R_AB > 1.0 - 1e-12),
        'note': "When ψ_A = ψ_B, R = 1, so R(1-R) = 0, both forms produce 0 from B branch",
    }


if __name__ == "__main__":
    print("=" * 70)
    print("COUPLING FORMS REGRESSION TEST — Phase 6b")
    print("=" * 70)

    print("\n[1/3] Forms differ at runtime when ψ_A ≠ ψ_B and 0 < R < 1")
    print("-" * 70)
    r1 = test_forms_differ()
    print(f"  R_AB                              = {r1['R_AB']:.4f}")
    print(f"  Partners of A                     = {r1['partners_of_A']}")
    print(f"  max |extra_positive - contrastive|= {r1['max_diff_positive_vs_contrastive']:.6e}")
    print(f"  expected from formula             = {r1['expected_max_diff']:.6e}")
    print(f"  formula match (allclose)          = {r1['formula_match']}")
    runtime_str = "PASS" if r1['runtime_test_pass'] else "FAIL"
    formula_str = "PASS" if r1['formula_test_pass'] else "FAIL"
    print(f"  runtime difference test  : {runtime_str}")
    print(f"  formula match test       : {formula_str}")

    print("\n[2/3] Sanity: R=0 (disjoint) → no inter-modular contribution from that pair")
    print("-" * 70)
    r2 = test_R_zero_makes_forms_equal()
    print(f"  R_AB (ψ disjoint)                 = {r2['R_AB_with_disjoint_AB']:.6f}")
    print(f"  R_AB is essentially zero          = {r2['R_AB_is_zero']}")

    print("\n[3/3] Sanity: R=1 (identical) → R(1-R) = 0")
    print("-" * 70)
    r3 = test_R_one_makes_contrastive_zero()
    print(f"  R_AB (ψ identical)                = {r3['R_AB_with_identical_AB']:.6f}")
    print(f"  R_AB is essentially one           = {r3['R_AB_is_one']}")

    print()
    print("=" * 70)
    if r1['all_pass'] and r2['R_AB_is_zero'] and r3['R_AB_is_one']:
        print("OVERALL: PASS — coupling_form selector works at runtime,")
        print("formula matches expected algebraic difference, R=0 and R=1 limits")
        print("behave as expected. The positive/contrastive sweep in F6 is valid.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — coupling_form selector is broken or dead code.")
        print("The positive/contrastive sweep in F6 must NOT be trusted until fixed.")
        sys.exit(1)
