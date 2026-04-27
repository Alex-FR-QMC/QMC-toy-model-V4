"""
Phase 6b — Coupling computation and step orchestration.

The coupling injects a per-module phi_extra into the existing 6a engine
through the (already wired) optional argument engine.step(state, phi_extra=...).
The 6a engine itself is NOT modified by this module.

Coupling form (perturbative, non-perspectival):

    𝒞_i = Σ_{j ∈ neighbours(i)} ε · R_ij · (1 - R_ij) · ψ_j

where ψ_j is read in homologous coordinates (label-based), and R_ij is
R_psi.

Since 𝒞_i has the dimensions of a Φ_corr field on Θ_i, we inject it as
phi_extra in the engine. The drift term then becomes:

    drift = +∇·(ψ ∇(Φ_corr_intra + 𝒞_i))    [+ sign from J = -ψ ∇Φ]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mcq_v4.factorial.state import FactorialState, StepDiagnostic
from mcq_v4.factorial.three_module_system import ThreeModuleSystem, CouplingConfig
from mcq_v4.factorial.overlaps import (
    compute_pairwise_R_psi, compute_R_tau, coupling_weight,
)
from mcq_v4.factorial.tau_prime import compute_tau_prime_3modules
from mcq_v4.factorial.observables import compute_observables, compute_Phi_corr


# ============================================================================
# Coupling extra-Φ for one module
# ============================================================================

def compute_extra_phi_for_module(
    name: str,
    states: dict[str, FactorialState],
    R_pairs: dict[tuple[str, str], float],
    coupling_cfg: CouplingConfig,
) -> np.ndarray:
    """
    Compute the inter-modular coupling contribution to Φ_corr for `name`.

    𝒞_name = Σ_{j ∈ neighbours} ε · R_{name,j} · (1 - R_{name,j}) · ψ_j

    Returns a (5, 5, 5) array suitable for injection as engine.step(phi_extra).
    """
    eps_coupling = coupling_cfg.epsilon
    partners = coupling_cfg.neighbours_of(name)

    extra = np.zeros((5, 5, 5))
    for j in partners:
        pair_key = tuple(sorted([name, j]))
        R = R_pairs.get(pair_key, 0.0)
        weight = eps_coupling * coupling_weight(R)
        # Add partner's distribution as additive perturbation to Phi_corr
        # in homologous coordinates (Phase 6b approximation)
        extra = extra + weight * states[j].psi

    return extra


# ============================================================================
# Step orchestration
# ============================================================================

def step_three_modules(
    sys: ThreeModuleSystem,
    freeze_h_M: Optional[dict[str, bool]] = None,
    coupling_active: bool = True,
) -> tuple[ThreeModuleSystem, dict]:
    """
    Advance the three-module system by one step.

    coupling_active=False is used as control (no coupling at all,
    even with three modules in the same system).

    Returns
    -------
    new_sys : ThreeModuleSystem
        System with updated states. RNG state preserved in engines.
    step_log : dict
        Per-step diagnostics:
          - tau_prime              : FactorialFieldOutput
          - R_psi                  : pairwise R_psi
          - R_tau                  : pairwise R_tau (on shared factors)
          - per_module_diagnostics : {name: StepDiagnostic}
          - extra_phi_diagnostics  : {name: dict with max_extra, max_phi_intra, ratio}
    """
    if freeze_h_M is None:
        freeze_h_M = {'A': False, 'B': False, 'C': False}

    states = sys.states

    # 1. Compute pairwise R_psi (snapshot before step)
    R_pairs = compute_pairwise_R_psi(states)

    # 2. Compute extra phi per module (or zeros if coupling disabled)
    extras = {}
    for name in ['A', 'B', 'C']:
        if coupling_active:
            extras[name] = compute_extra_phi_for_module(
                name, states, R_pairs, sys.coupling_cfg,
            )
        else:
            extras[name] = None  # bitwise-identical to 6a path

    # 3. Compute Phi_corr_intra per module (for diagnostic ratio reporting)
    phi_intra_max = {}
    for name in ['A', 'B', 'C']:
        obs = compute_observables(states[name].psi)
        phi_intra = compute_Phi_corr(obs, sys.cfg_engine, sys.cfg_engine.mode)
        phi_intra_max[name] = float(np.abs(phi_intra).max())

    # 4. Step each module's engine independently
    new_states = {}
    diagnostics = {}
    extra_diagnostics = {}
    for name in ['A', 'B', 'C']:
        new_state, diag = sys.engines[name].step(
            states[name],
            freeze_h_M=freeze_h_M[name],
            phi_extra=extras[name],
        )
        new_states[name] = new_state
        diagnostics[name] = diag

        # Extra-phi diagnostic: ratio |extra| / |Phi_intra|
        if extras[name] is None:
            max_extra = 0.0
            ratio = 0.0
        else:
            max_extra = float(np.abs(extras[name]).max())
            ratio = max_extra / max(phi_intra_max[name], 1e-12)
        extra_diagnostics[name] = {
            'max_phi_extra': max_extra,
            'max_phi_intra': phi_intra_max[name],
            'ratio_extra_to_intra': ratio,
        }

    # 5. Aggregate τ' on the new states
    tau_prime = compute_tau_prime_3modules(new_states)

    # 6. R_tau diagnostic on new contributions
    R_tau_pairs = {
        ('A', 'B'): compute_R_tau(
            tau_prime.contributions['k_AB']['A'],
            tau_prime.contributions['k_AB']['B'],
        ),
        ('B', 'C'): compute_R_tau(
            tau_prime.contributions['k_BC']['B'],
            tau_prime.contributions['k_BC']['C'],
        ),
        ('A', 'C'): compute_R_tau(
            tau_prime.contributions['k_CA']['C'],
            tau_prime.contributions['k_CA']['A'],
        ),
    }

    new_sys = sys.replace_states(new_states)

    step_log = {
        'tau_prime': tau_prime,
        'R_psi': R_pairs,
        'R_tau': R_tau_pairs,
        'per_module_diagnostics': diagnostics,
        'extra_phi_diagnostics': extra_diagnostics,
        'coupling_active': coupling_active,
    }

    return new_sys, step_log


# ============================================================================
# Multi-step driver
# ============================================================================

def run_three_modules(
    sys: ThreeModuleSystem,
    n_steps: int,
    freeze_h_M: Optional[dict[str, bool]] = None,
    coupling_active: bool = True,
) -> tuple[list[ThreeModuleSystem], list[dict]]:
    """
    Run the system for `n_steps`. Returns the trajectory of systems and
    the step logs.
    """
    history = [sys]
    logs = []

    for t in range(n_steps):
        sys, log = step_three_modules(sys, freeze_h_M=freeze_h_M,
                                      coupling_active=coupling_active)
        history.append(sys)
        logs.append(log)

    return history, logs
