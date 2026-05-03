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

def _h_field_product(state: FactorialState) -> np.ndarray:
    """
    Lift the three marginal h_a(θ_a) into a (5,5,5) conformal field by
    outer product:  h_field(t, m, i) = h_T(t) · h_M(m) · h_I(i)

    This is the natural conformal lifting consistent with the engine's
    separable treatment of axes (each axis has its own scalar conformal
    factor, with no cross-terms). Documented as a Phase 6c convention —
    a tensor metric would replace this in a later phase.
    """
    return (state.h_T[:, None, None]
            * state.h_M[None, :, None]
            * state.h_I[None, None, :])


def _h_open_field(state: FactorialState, h_min: float, h_0: float) -> np.ndarray:
    """
    Compute the 'open metric' field h_open ∈ [0, 1].

    h_open(θ) = (h_field(θ) - h_min^3) / (h_0^3 - h_min^3)

    The cube of h_min/h_0 is the natural normaliser for the product
    field (since h_field is bounded in [h_min^3, h_0^3] when each
    marginal is in [h_min, h_0]).

    Reading: h_open ≈ 1 in regions of high open-ness (unfamiliar to
    the module), h_open ≈ 0 in deeply sedimented regions.

    Note (cf. README 6c): the literal Ch.3 §3.2.2 formula g_k = θ_k/h
    and the textual interpretation "regions where the metric is dilated
    contribute more strongly" are in tension under our convention
    (h_min = familiar/sedimented, h_0 = unfamiliar/open). This h_open
    field instantiates the textual interpretation, where INV_H below
    instantiates the literal formula. The sweep tests both.
    """
    h_field = _h_field_product(state)
    h_min_cube = h_min ** 3
    h_0_cube = h_0 ** 3
    return (h_field - h_min_cube) / max(h_0_cube - h_min_cube, 1e-12)


def _compute_novelty_and_form(
    state: FactorialState,
    coupling_cfg: 'CouplingConfig',
    prev_h_field: Optional[np.ndarray] = None,
    h_min: float = 0.1,
    h_0: float = 1.0,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Compute the perspectival novelty/form scalar field for one module.

    Both `novelty_j^{h_j}` (in j's metric) and `form_i^{h_i}` (in i's
    metric) share the same structural formula — only the underlying
    state changes. This homology preserves the structure of Ch.3 §3.1.III:

        𝒞_ij ∝ R_ij · (novelty_j^{h_j} − R_ij · form_i^{h_i})

    Three candidates (selected by coupling_cfg.coupling_form):

      'perspectival_INV_H'        : ψ / h_field
            literal Ch.3 §3.2.2 form (g_k = θ_k/h reduced to ψ/h);
            under our convention, may amplify familiar regions

      'perspectival_H_OPEN'       : ψ · h_open
            instantiates "unfamiliar contributes more strongly";
            amplifies open-metric regions

      'perspectival_MORPHO_ACTIVE': ψ · h_open · |∂_t h_field|
            amplifies open-AND-actively-changing regions;
            requires prev_h_field for finite-difference ∂_t h

    Returns a (5, 5, 5) array.
    """
    form = coupling_cfg.coupling_form
    psi = state.psi
    h_field = _h_field_product(state)

    if form == 'perspectival_INV_H':
        return psi / np.maximum(h_field, 1e-12)

    if form == 'perspectival_H_OPEN':
        h_open = _h_open_field(state, h_min, h_0)
        return psi * h_open

    if form == 'perspectival_MORPHO_ACTIVE':
        h_open = _h_open_field(state, h_min, h_0)
        if prev_h_field is None:
            # First step: no previous h, treat |∂_t h| as zero.
            # This means MORPHO_ACTIVE coupling is zero on the first step.
            dh_dt_abs = np.zeros_like(h_field)
        else:
            dh_dt_abs = np.abs(h_field - prev_h_field) / max(dt, 1e-12)
        return psi * h_open * dh_dt_abs

    raise ValueError(f"Unknown perspectival form: {form!r}")


def compute_extra_phi_for_module(
    name: str,
    states: dict[str, FactorialState],
    R_pairs: dict[tuple[str, str], float],
    coupling_cfg: CouplingConfig,
    prev_h_fields: Optional[dict[str, np.ndarray]] = None,
    h_min: float = 0.1,
    h_0: float = 1.0,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Compute the inter-modular coupling contribution to Φ_corr for `name`.

    Five forms supported (selected by coupling_cfg.coupling_form):

      Phase 6b (non-perspectival, label-based):

        'positive':
          phi_extra_i = Σ_j ε · R_ij(1-R_ij) · ψ_j

        'contrastive':
          phi_extra_i = Σ_j ε · R_ij(1-R_ij) · (ψ_j − R_ij · ψ_i)

      Phase 6c (perspectival, metric-aware via h_j and h_i):

        'perspectival_INV_H':
          phi_extra_i = Σ_j ε · R_ij(1-R_ij) · (ψ_j/h_j  −  R_ij · ψ_i/h_i)

        'perspectival_H_OPEN':
          phi_extra_i = Σ_j ε · R_ij(1-R_ij) · (ψ_j·h_open_j  −  R_ij · ψ_i·h_open_i)

        'perspectival_MORPHO_ACTIVE':
          phi_extra_i = Σ_j ε · R_ij(1-R_ij) · (
                ψ_j·h_open_j·|∂_t h_j|
              − R_ij · ψ_i·h_open_i·|∂_t h_i|
          )

    Both sides of the contrastive subtraction are STRUCTURALLY HOMOLOGOUS —
    novelty_j and form_i use the same formula in their own metric. This
    preserves the structure of Ch.3 §3.1.III.

    For MORPHO_ACTIVE, prev_h_fields[name] must contain the (5,5,5)
    h_field of the previous step (or None for the first step).

    Returns a (5, 5, 5) array suitable for engine.step(phi_extra=...).
    """
    eps_coupling = coupling_cfg.epsilon
    partners = coupling_cfg.neighbours_of(name)
    psi_self = states[name].psi
    form = coupling_cfg.coupling_form

    is_perspectival = form.startswith('perspectival_')

    if is_perspectival:
        if prev_h_fields is None:
            prev_h_fields = {}
        # Pre-compute form/novelty field for self (used in the contrastive term)
        form_self = _compute_novelty_and_form(
            states[name], coupling_cfg,
            prev_h_field=prev_h_fields.get(name),
            h_min=h_min, h_0=h_0, dt=dt,
        )

    extra = np.zeros((5, 5, 5))
    for j in partners:
        pair_key = tuple(sorted([name, j]))
        R = R_pairs.get(pair_key, 0.0)
        weight = eps_coupling * coupling_weight(R)

        if form == 'positive':
            contribution = weight * states[j].psi
        elif form == 'contrastive':
            # Phase 6b: subtract the 'already-known to self' (label-based)
            contribution = weight * (states[j].psi - R * psi_self)
        elif is_perspectival:
            # Phase 6c: novelty in j's metric MINUS R · form in i's metric
            novelty_j = _compute_novelty_and_form(
                states[j], coupling_cfg,
                prev_h_field=prev_h_fields.get(j),
                h_min=h_min, h_0=h_0, dt=dt,
            )
            contribution = weight * (novelty_j - R * form_self)
        else:
            raise ValueError(
                f"Unknown coupling form: {form!r}. Use 'positive', 'contrastive', "
                f"'perspectival_INV_H', 'perspectival_H_OPEN', or 'perspectival_MORPHO_ACTIVE'."
            )

        extra = extra + contribution

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
    # Pass prev_h_fields for perspectival forms (used by MORPHO_ACTIVE).
    prev_h_fields = sys.prev_h_fields  # may be None on first step
    extras = {}
    for name in ['A', 'B', 'C']:
        if coupling_active:
            extras[name] = compute_extra_phi_for_module(
                name, states, R_pairs, sys.coupling_cfg,
                prev_h_fields=prev_h_fields,
                h_min=sys.cfg_engine.h_min,
                h_0=sys.cfg_engine.h_0,
                dt=sys.cfg_engine.dt,
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

        # Extra-phi diagnostic: report TWO ratios for robustness.
        # ratio_to_intra: extra / Phi_intra (can blow up when intra→0)
        # ratio_to_lambda: extra / lambda_KNV (stable reference scale of regulation)
        if extras[name] is None:
            max_extra = 0.0
            ratio_intra = 0.0
            ratio_lambda = 0.0
        else:
            max_extra = float(np.abs(extras[name]).max())
            # Avoid blowing up when phi_intra is near zero (var in viable corridor)
            phi_intra_floor = max(phi_intra_max[name], 1e-3)
            ratio_intra = max_extra / phi_intra_floor
            # Stable reference: extra normalised by the regulation scale lambda_KNV
            ratio_lambda = max_extra / max(sys.cfg_engine.lambda_KNV, 1e-12)
        extra_diagnostics[name] = {
            'max_phi_extra': max_extra,
            'max_phi_intra': phi_intra_max[name],
            'ratio_extra_to_intra': ratio_intra,
            'ratio_extra_to_lambda_KNV': ratio_lambda,
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

    # Snapshot the current h_fields (BEFORE step) as "prev" for next step.
    # MORPHO_ACTIVE will compute |∂_t h| as |h_field(now) - h_field(prev)| at next call.
    new_prev_h_fields = {
        name: _h_field_product(states[name])  # current = pre-step
        for name in ['A', 'B', 'C']
    }

    new_sys = sys.replace_states(new_states, new_prev_h_fields=new_prev_h_fields)

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
