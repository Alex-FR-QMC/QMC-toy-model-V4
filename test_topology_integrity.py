"""
Preflight 7 — Topology integrity test for Phase 6b.

Verifies the cyclic sharing topology of the three-module system:

  - Three shared factors: k_AB, k_BC, k_CA
  - Three private factors: k_A_private, k_B_private, k_C_private
  - Each shared factor observed by EXACTLY two modules
  - Each module has EXACTLY two neighbours
  - Neighbour relations are symmetric (i is neighbour of j iff j is
    neighbour of i)
  - No self-coupling

If any of these is violated, the topology has degraded from cyclic to
linear or other, and the F1/F4/F6 measurements no longer reflect the
intended architecture.

Run from project root:
    PYTHONPATH=src python tests/phase6b/test_topology_integrity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.three_module_system import CouplingConfig


def test_default_topology() -> dict:
    """The default CouplingConfig must encode the cyclic A-B-C topology."""
    cfg = CouplingConfig()

    # Expected shared factors and donors
    expected_shared = {
        'k_AB': ('A', 'B'),
        'k_BC': ('B', 'C'),
        'k_CA': ('C', 'A'),
    }
    expected_private = {
        'k_A_private': 'A',
        'k_B_private': 'B',
        'k_C_private': 'C',
    }

    # Check shared topology
    actual_shared = cfg.sharing_topology
    shared_match = (actual_shared == expected_shared)
    n_shared = len(actual_shared)
    expected_n_shared = 3

    # Check private factors
    actual_private = cfg.private_factors
    private_match = (actual_private == expected_private)
    n_private = len(actual_private)
    expected_n_private = 3

    # Each shared factor has exactly 2 distinct donors
    each_factor_has_two_donors = all(
        len(donors) == 2 and donors[0] != donors[1]
        for donors in actual_shared.values()
    )

    # Each module has exactly 2 neighbours
    n_neighbours = {
        'A': len(cfg.neighbours_of('A')),
        'B': len(cfg.neighbours_of('B')),
        'C': len(cfg.neighbours_of('C')),
    }
    each_has_two_neighbours = all(n == 2 for n in n_neighbours.values())

    # Neighbour relations are symmetric
    symmetric = all(
        i in cfg.neighbours_of(j) and j in cfg.neighbours_of(i)
        for i in ['A', 'B', 'C']
        for j in cfg.neighbours_of(i)
    )

    # No self-coupling: a module is not its own neighbour
    no_self = all(
        m not in cfg.neighbours_of(m)
        for m in ['A', 'B', 'C']
    )

    # Each module observes 3 factors total: 2 shared + 1 private
    factors_per_module = {m: 0 for m in ['A', 'B', 'C']}
    for factor, (m1, m2) in actual_shared.items():
        factors_per_module[m1] += 1
        factors_per_module[m2] += 1
    for factor, m in actual_private.items():
        factors_per_module[m] += 1
    each_observes_three = all(n == 3 for n in factors_per_module.values())

    pass_overall = (
        shared_match and private_match
        and n_shared == expected_n_shared
        and n_private == expected_n_private
        and each_factor_has_two_donors
        and each_has_two_neighbours
        and symmetric
        and no_self
        and each_observes_three
    )

    return {
        'shared_match': shared_match,
        'private_match': private_match,
        'n_shared': n_shared,
        'n_private': n_private,
        'each_factor_two_donors': each_factor_has_two_donors,
        'n_neighbours_per_module': n_neighbours,
        'each_two_neighbours': each_has_two_neighbours,
        'symmetric': symmetric,
        'no_self_coupling': no_self,
        'factors_per_module': factors_per_module,
        'each_observes_three': each_observes_three,
        'pass': pass_overall,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PREFLIGHT 7 — Topology integrity test")
    print("=" * 70)

    r = test_default_topology()
    print()
    print(f"  shared topology matches expected:        {r['shared_match']}")
    print(f"  private topology matches expected:       {r['private_match']}")
    print(f"  number of shared factors == 3:           {r['n_shared'] == 3}")
    print(f"  number of private factors == 3:          {r['n_private'] == 3}")
    print(f"  each shared factor has 2 distinct donors: {r['each_factor_two_donors']}")
    print(f"  neighbours per module: {r['n_neighbours_per_module']}")
    print(f"  each module has 2 neighbours:            {r['each_two_neighbours']}")
    print(f"  neighbour relations symmetric:           {r['symmetric']}")
    print(f"  no self-coupling:                        {r['no_self_coupling']}")
    print(f"  factors observed per module: {r['factors_per_module']}")
    print(f"  each module observes 3 factors total:    {r['each_observes_three']}")

    print()
    print("=" * 70)
    if r['pass']:
        print("OVERALL: PASS — cyclic A-B-C topology integrity verified.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — topology has degraded from cyclic.")
        print("Phase 6b measurements are NOT reflecting the intended architecture.")
        sys.exit(1)
