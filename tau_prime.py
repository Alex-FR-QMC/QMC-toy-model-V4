"""
Phase 6b — τ' ∈ ℝ⁶ aggregation across three modules.

Convention (option β, validated):
  - Each module has its OWN Θ_m and metric h_m.
  - Shared factors observe the T axis; private factors observe the I axis.
  - τ' is a COMPOSITION of modular contributions, NOT an integration over
    a shared space.

  τ'_{k_AB} = contrib_A^T + contrib_B^T          (k_AB partagé par A, B)
  τ'_{k_BC} = contrib_B^T + contrib_C^T
  τ'_{k_CA} = contrib_C^T + contrib_A^T
  τ'_{k_A}  = contrib_A^I                         (private to A)
  τ'_{k_B}  = contrib_B^I
  τ'_{k_C}  = contrib_C^I

with contrib_m^T = Σ_{θ_T} (θ_T / h_m^T(θ_T)) · ψ_m^T_marg(θ_T)
     contrib_m^I = Σ_{θ_I} (θ_I / h_m^I(θ_I)) · ψ_m^I_marg(θ_I)

Note: in this minimal Phase 6b spec, a module produces a single tensional
contribution that goes to BOTH shared factors involving it (e.g. A
contributes the same value to k_AB and k_CA). This is a deliberate
simplification — see notes in spec discussion. If F1 shows the topology
collapses because of this, we can introduce per-edge tensional channels
in a future refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mcq_v4.factorial.state import FactorialState, THETA_T, THETA_I


@dataclass
class FactorialFieldOutput:
    """τ' ∈ ℝ⁶ with full traceability of contributions per donor module."""

    # Aggregate (the τ' itself)
    k_AB: float
    k_BC: float
    k_CA: float
    k_A_private: float
    k_B_private: float
    k_C_private: float

    # Per-donor contributions (for diagnostics, R_tau, F1 reconstruction)
    contributions: dict = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Return τ' as a flat 6-element array."""
        return np.array([
            self.k_AB, self.k_BC, self.k_CA,
            self.k_A_private, self.k_B_private, self.k_C_private,
        ], dtype=float)

    def shared_array(self) -> np.ndarray:
        """Return only the 3 shared components."""
        return np.array([self.k_AB, self.k_BC, self.k_CA], dtype=float)

    def private_array(self) -> np.ndarray:
        """Return only the 3 private components."""
        return np.array([self.k_A_private, self.k_B_private, self.k_C_private], dtype=float)


def compute_modular_contributions(state: FactorialState) -> dict:
    """
    Compute the tensional and interface contributions of one module.

    contrib_T = Σ (θ_T / h_T(θ_T)) · ψ_marg^T(θ_T)
    contrib_I = Σ (θ_I / h_I(θ_I)) · ψ_marg^I(θ_I)

    Returns
    -------
    dict with keys 'T', 'I'.
    """
    psi_T = state.psi.sum(axis=(1, 2))   # marginal on T
    psi_I = state.psi.sum(axis=(0, 1))   # marginal on I

    contrib_T = float(((THETA_T / np.maximum(state.h_T, 1e-12)) * psi_T).sum())
    contrib_I = float(((THETA_I / np.maximum(state.h_I, 1e-12)) * psi_I).sum())

    return {'T': contrib_T, 'I': contrib_I}


def compute_tau_prime_3modules(
    states: dict[str, FactorialState],
) -> FactorialFieldOutput:
    """
    Aggregate τ' ∈ ℝ⁶ from three modules.

    Phase 6b convention (option β): no shared space; each module computes
    its contribution in its own metric, and the aggregation is a sum at
    the factor level — not an integration on a common Θ.
    """
    # Per-module contributions
    contrib = {
        name: compute_modular_contributions(states[name])
        for name in ['A', 'B', 'C']
    }

    return FactorialFieldOutput(
        k_AB=contrib['A']['T'] + contrib['B']['T'],
        k_BC=contrib['B']['T'] + contrib['C']['T'],
        k_CA=contrib['C']['T'] + contrib['A']['T'],
        k_A_private=contrib['A']['I'],
        k_B_private=contrib['B']['I'],
        k_C_private=contrib['C']['I'],
        contributions={
            'k_AB': {'A': contrib['A']['T'], 'B': contrib['B']['T']},
            'k_BC': {'B': contrib['B']['T'], 'C': contrib['C']['T']},
            'k_CA': {'C': contrib['C']['T'], 'A': contrib['A']['T']},
            'k_A_private': {'A': contrib['A']['I']},
            'k_B_private': {'B': contrib['B']['I']},
            'k_C_private': {'C': contrib['C']['I']},
        },
    )


def compute_modular_only_contribution(state: FactorialState, name: str) -> dict:
    """
    Solo contribution of a single module (not aggregated). Used for F1
    trajectorial test where we need contrib_A^solo(t) + contrib_B^solo(t).

    Returns the same dict structure as the per-donor contributions of
    FactorialFieldOutput so they can be summed externally.
    """
    raw = compute_modular_contributions(state)
    return {'T': raw['T'], 'I': raw['I']}
