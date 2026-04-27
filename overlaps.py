"""
Phase 6b — Overlap measures between modules.

Two overlaps are computed:
  R_psi : Bhattacharyya overlap of distributions in homologous coordinates.
          Drives the coupling strength via the productive-overlap weight
          R(1-R). This is a label-based comparison, NOT metric-aware.
          (Phase 6b approximation; will be replaced/complemented in 6c.)

  R_tau : Functional overlap on shared factors. Compares the modular
          contributions to the same shared factor (e.g. contrib_A^{k_AB}
          vs contrib_B^{k_AB}). DIAGNOSTIC ONLY — not used in coupling.
"""

from __future__ import annotations

import numpy as np

from mcq_v4.factorial.state import FactorialState


def compute_R_psi(state_i: FactorialState, state_j: FactorialState) -> float:
    """
    Bhattacharyya coefficient between two distributions, evaluated in
    homologous label coordinates.

    R_psi = Σ sqrt(ψ_i(θ) · ψ_j(θ))    (with cell-by-cell label-wise comparison)

    Bounded in [0, 1]:
      R_psi → 0 : disjoint distributions
      R_psi → 1 : identical distributions (label-wise)

    APPROXIMATION (Phase 6b): assumes homologous coordinates carry
    comparable meaning across modules. Distributions with same label
    coordinates may correspond to different physical configurations
    when h_A^a ≠ h_B^a — this is the perspective effect that 6c will
    address. Logged as the canonical Phase 6b overlap.
    """
    return float(np.sqrt(np.maximum(state_i.psi * state_j.psi, 0.0)).sum())


def compute_R_tau(contrib_i: float, contrib_j: float, eps: float = 1e-12) -> float:
    """
    Functional overlap between two modular contributions to the same
    shared factor.

    R_tau = 1 - |contrib_i - contrib_j| / (|contrib_i| + |contrib_j| + eps)

    Bounded in [0, 1]:
      R_tau → 1 : both modules contribute identical values (factorial agreement)
      R_tau → 0 : contributions are maximally divergent

    DIAGNOSTIC: not used in coupling. Logged to characterise the gap
    between morphological similarity (R_psi) and factorial similarity (R_tau):

      R_psi high, R_tau low : modules look alike but evaluate the shared
                              factor differently (their metrics differ enough
                              to produce different g_k contributions)
      R_psi low,  R_tau high: modules look distinct but converge on the
                              shared factor value (compensating metric effects)
    """
    num = abs(contrib_i - contrib_j)
    den = abs(contrib_i) + abs(contrib_j) + eps
    return float(1.0 - num / den)


def compute_pairwise_R_psi(states: dict[str, FactorialState]) -> dict[tuple[str, str], float]:
    """
    Compute R_psi for all three pairs (A,B), (B,C), (A,C).
    Returns dict keyed by sorted-pair tuples for canonical access.
    """
    return {
        ('A', 'B'): compute_R_psi(states['A'], states['B']),
        ('B', 'C'): compute_R_psi(states['B'], states['C']),
        ('A', 'C'): compute_R_psi(states['A'], states['C']),
    }


def coupling_weight(R: float) -> float:
    """
    Productive-overlap weight: R(1-R). Vanishes at R=0 and R=1, maximal at R=0.5.
    """
    return float(R * (1.0 - R))
