"""
Phase 6c-B — F6.T25' Morphodynamic constrained contraction (MCQ-aligned).

Replaces F6.T25 (brute swap version) which conflated KNV violations
with numerical instability and never reached the regime
`var_max < var_init AND stable`.

T25' protocol (Alex's spec):

Phase 1 — Warmup
    System 3 modules stabilised, coupling OFF, n=100 steps.

Phase 2 — Progressive contraction ramp on M_A
    var_max(t) = max(var_init - α·(t-t₀), var_floor)
    α small (0.001–0.01 per step), var_floor ≥ 0.2
    No brutal swap → Γ″ numerical divergence avoided.

Phase 3 — MCQ monitoring (REQUIRED)
    For module A (and B, C for propagation):
      - 𝒢_proxy(A): from compute_G_modular (note: ψ-roughness proxy,
                     not full 𝒢 = ‖∂τ′/∂Γ_meta‖)
      - Δ(A): from compute_delta_modular
      - Γ″(A) proxy: second time derivative of contrib_T (not full
                     ‖Γ″‖ on Γ_meta — proxy)
      - var_M(A): the contraction observable

Classification (5 categories — Alex's spec):

    LOCAL_CONTRACTION_VIABLE
        var_M(A) ↓ significantly
        AND 𝒢_proxy(A) > G_min (no diff. KNV)
        AND Δ(A) < Δ_crit (no top. KNV)
        AND Γ″ proxy bounded and damped (no morph. KNV)
        AND propagation to B,C below threshold

    PROPAGATED_CONTRACTION
        var_M(A,B,C) ↓ together AND MCQ conditions met

    STRUCTURAL_RESISTANCE      ← KEY new category
        var_M(A) does not decrease despite progressive constraint
        AND no KNV violation
        Reading: system actively resists contraction without breaking 𝒱
        This is NOT failure — it is a viable steady response.

    KNV_COLLAPSE
        Genuine KNV violation: 𝒢_proxy → 0 OR Δ exceeds Δ_crit
        OR Γ″ proxy unbounded (non-dampable)

    NUMERICAL_INSTABILITY      ← separate from KNV
        NaN / overflow / numerical divergence
        Distinct from morphogenic non-damping. Engine breakdown,
        not an MCQ-meaningful regime.

References (cf. /mnt/project/QMC_Chap1.pdf, Executive Summary):
    §1.5 — three KNV foundations: differential 𝒢→0, topological Δ
           outside corridor, morphogenic Γ″ non-dampable.
    §1.7 — 𝓔_QMC = 𝒱 ∩ 𝒟_QMC: viability is necessary but not sufficient;
           dynamic existence requires also Var(Δφ_{ij}) > 0,
           0 < Ξ_obs < Ξ_crit, ℒ < ℒ_crit.
    §1.4 — RTS regime: dτ'/dt ≈ 0 AND Var(τ') > 0 AND Γ' ≠ 0 —
           the signature of viable stability under constraint
           (relevant for STRUCTURAL_RESISTANCE).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    FactorialEngineConfig, EngineMode, FactorialState, THETA_T,
)
from mcq_v4.factorial.three_module_system import (
    DIFFERENTIATED_WEIGHTS, CouplingConfig, build_three_module_system,
)
from mcq_v4.factorial.coupling import step_three_modules
from mcq_v4.factorial.observables import compute_observables
from mcq_v4.factorial.tau_prime import compute_modular_contributions
from mcq_v4.factorial.signatures_6c_b import (
    compute_G_modular, compute_delta_modular, compute_delta_shape,
    compute_gamma_signals, KNV_THRESHOLDS_6C_B,
)


def make_initial_psi():
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


def clone_system(sys_obj):
    state_clones = {
        name: FactorialState(
            psi=getattr(sys_obj, f'state_{name}').psi.copy(),
            h_T=getattr(sys_obj, f'state_{name}').h_T.copy(),
            h_M=getattr(sys_obj, f'state_{name}').h_M.copy(),
            h_I=getattr(sys_obj, f'state_{name}').h_I.copy(),
            cfg=getattr(sys_obj, f'state_{name}').cfg,
        )
        for name in ['A', 'B', 'C']
    }
    rng_states = {
        name: copy.deepcopy(getattr(sys_obj, f'engine_{name}').rng.bit_generator.state)
        for name in ['A', 'B', 'C']
    }
    prev_h = copy.deepcopy(sys_obj.prev_h_fields) if sys_obj.prev_h_fields else None
    return state_clones, rng_states, prev_h


def make_system_from_clone(cfg_engine, coupling_cfg, state_clones,
                           rng_states, prev_h, base_seed):
    initial_psi = {n: state_clones[n].psi.copy() for n in ['A', 'B', 'C']}
    new_sys = build_three_module_system(
        cfg_engine, coupling_cfg, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )
    for name in ['A', 'B', 'C']:
        st = state_clones[name]
        new_st = FactorialState(
            psi=st.psi.copy(), h_T=st.h_T.copy(),
            h_M=st.h_M.copy(), h_I=st.h_I.copy(),
            cfg=st.cfg,
        )
        if name == 'A':
            new_sys.state_A = new_st
        elif name == 'B':
            new_sys.state_B = new_st
        else:
            new_sys.state_C = new_st
    for name in ['A', 'B', 'C']:
        getattr(new_sys, f'engine_{name}').rng.bit_generator.state = \
            copy.deepcopy(rng_states[name])
    new_sys.prev_h_fields = copy.deepcopy(prev_h) if prev_h else None
    return new_sys


def make_swapped_cfg(base_cfg: FactorialEngineConfig, var_max: float,
                     lambda_KNV: float) -> FactorialEngineConfig:
    return FactorialEngineConfig(
        dt=base_cfg.dt, T_steps=base_cfg.T_steps, mode=base_cfg.mode,
        D_0=base_cfg.D_0, D_min=base_cfg.D_min,
        beta_0=base_cfg.beta_0, gamma_0=base_cfg.gamma_0,
        h_0=base_cfg.h_0, h_min=base_cfg.h_min,
        sigma_eta=base_cfg.sigma_eta,
        var_min=base_cfg.var_min, var_max=var_max,
        H_min=base_cfg.H_min, lambda_KNV=lambda_KNV,
    )


def is_finite_state(state: FactorialState) -> bool:
    return (
        np.all(np.isfinite(state.psi))
        and np.all(np.isfinite(state.h_T))
        and np.all(np.isfinite(state.h_M))
        and np.all(np.isfinite(state.h_I))
    )


def measure_T25_prime_run(
    cfg_engine: FactorialEngineConfig,
    coupling_form: str,
    eps: float,
    base_seed: int,
    alpha: float,
    var_floor: float,
    lambda_KNV_target: float,
    warmup_steps: int = 100,
    contraction_steps: int = 200,
) -> dict:
    """
    One T25' run with progressive ramp + 5 corrections (session 4):

      (1) Δ_shape monitoring in addition to Δ_centred (catches var_M
          changes that don't shift E_M).
      (2) Multi-signal Γ″: var_M, h_M_mean, tau_vec — morph_KNV ssi
          ≥ 2 non-dampable.
      (3) Multi-temporal drop: final_drop, min_drop (max contraction
          ever reached), sustained_drop_fraction.
      (4) EXPANSIVE_RESPONSE separated from STRUCTURAL_RESISTANCE.
      (5) var_init_A computed dynamically from post-warmup state of
          THIS run (no hardcoded estimate).

    Ramp now starts from var_init_A (no expansion-then-contraction
    phase).
    """
    initial_psi = make_initial_psi()

    coupling_cfg_warmup = CouplingConfig(epsilon=0.0, coupling_form='contrastive')
    sys_warmup = build_three_module_system(
        cfg_engine, coupling_cfg_warmup, DIFFERENTIATED_WEIGHTS,
        initial_psi, base_seed=base_seed,
    )
    for _ in range(warmup_steps):
        sys_warmup, _ = step_three_modules(sys_warmup, coupling_active=False)

    state_clones, rng_states, prev_h = clone_system(sys_warmup)
    coupling_cfg_meas = CouplingConfig(epsilon=eps, coupling_form=coupling_form)

    # CONTROL BRANCH: same warmup state, same RNG, NO swap applied to engine_A
    # This baseline captures the natural drift of var_M(A) under the
    # nominal dynamics. Without this, "drop" is conflated with the
    # natural expansion of var_M observed in 6a (≈ +156% in 200 steps).
    sys_ctrl = make_system_from_clone(cfg_engine, coupling_cfg_meas,
                                       state_clones, rng_states, prev_h, base_seed)

    # MEASUREMENT BRANCH: with progressive swap on engine_A
    sys_p = make_system_from_clone(cfg_engine, coupling_cfg_meas,
                                    state_clones, rng_states, prev_h, base_seed)

    # (5) Compute var_init dynamically from THIS run's post-warmup state
    var_M_init = {
        m: compute_observables(getattr(sys_p, f'state_{m}').psi).get('var_M', 0.0)
        for m in ['A', 'B', 'C']
    }
    var_max_ramp_start = max(var_M_init['A'], 0.05)

    # Trajectories — all multi-signal
    G_traj = {m: [] for m in ['A', 'B', 'C']}
    delta_centred_traj = {m: [] for m in ['A', 'B', 'C']}
    delta_shape_traj = {m: [] for m in ['A', 'B', 'C']}
    var_M_traj = {m: [] for m in ['A', 'B', 'C']}
    var_T_traj = {m: [] for m in ['A', 'B', 'C']}
    var_I_traj = {m: [] for m in ['A', 'B', 'C']}
    tau_T_traj = {m: [] for m in ['A', 'B', 'C']}
    tau_I_traj = {m: [] for m in ['A', 'B', 'C']}
    h_M_mean_traj = {m: [] for m in ['A', 'B', 'C']}
    var_max_applied_traj = []
    psi_finite_traj = []

    # Control branch (no swap) — track baseline var_M(A)
    var_M_A_ctrl_traj = []

    numerical_instability_step = None

    for step in range(contraction_steps):
        var_max_now = max(var_max_ramp_start - alpha * step, var_floor)
        var_max_applied_traj.append(var_max_now)
        sys_p.engine_A.cfg = make_swapped_cfg(
            cfg_engine, var_max=var_max_now, lambda_KNV=lambda_KNV_target,
        )

        try:
            sys_p, _ = step_three_modules(sys_p, coupling_active=True)
            sys_ctrl, _ = step_three_modules(sys_ctrl, coupling_active=True)
        except Exception:
            numerical_instability_step = step
            psi_finite_traj.append(False)
            break

        all_finite = all(
            is_finite_state(getattr(sys_p, f'state_{m}'))
            for m in ['A', 'B', 'C']
        )
        psi_finite_traj.append(all_finite)
        if not all_finite:
            numerical_instability_step = step
            break

        for m in ['A', 'B', 'C']:
            st = getattr(sys_p, f'state_{m}')
            G_traj[m].append(compute_G_modular(st)['G_proxy_total'])
            delta_centred_traj[m].append(compute_delta_modular(st)['delta'])
            shape = compute_delta_shape(st)
            delta_shape_traj[m].append(shape['delta_shape'])
            var_M_traj[m].append(shape['var_M'])
            var_T_traj[m].append(shape['var_T'])
            var_I_traj[m].append(shape['var_I'])
            contrib = compute_modular_contributions(st)
            tau_T_traj[m].append(contrib['T'])
            tau_I_traj[m].append(contrib['I'])
            h_M_mean_traj[m].append(float(np.mean(st.h_M)))

        # Control: capture var_M(A) of the no-swap branch
        var_M_A_ctrl_traj.append(
            compute_observables(sys_ctrl.state_A.psi).get('var_M', 0.0)
        )

    # (3) Multi-temporal drop on var_M(A) — RELATIVE to control branch
    # which captures the natural drift of var_M(A) without contraction.
    # Without this baseline, "drop" was conflated with the natural
    # expansion of var_M observed in 6a (~+156% in 200 steps).
    var_M_A = np.array(var_M_traj['A'])
    var_M_A_ctrl = np.array(var_M_A_ctrl_traj)
    n_match = min(len(var_M_A), len(var_M_A_ctrl))
    var_M_A = var_M_A[:n_match]
    var_M_A_ctrl = var_M_A_ctrl[:n_match]

    if n_match == 0 or var_M_init['A'] < 1e-12:
        final_drop_vs_init = 0.0
        final_drop_vs_ctrl = 0.0
        min_drop_vs_init = 0.0
        min_drop_vs_ctrl = 0.0
        sustained_drop_fraction = 0.0
    else:
        # Drops vs init
        drops_vs_init = (var_M_init['A'] - var_M_A) / var_M_init['A']
        final_drop_vs_init = float(drops_vs_init[-1])
        min_drop_vs_init = float(drops_vs_init.max())  # max contraction = max drop value

        # Drops vs control (the structurally meaningful one)
        ctrl_safe = np.maximum(var_M_A_ctrl, 1e-12)
        drops_vs_ctrl = (var_M_A_ctrl - var_M_A) / ctrl_safe
        final_drop_vs_ctrl = float(drops_vs_ctrl[-1])
        min_drop_vs_ctrl = float(drops_vs_ctrl.max())

        # Sustained: fraction of steps where drop_vs_ctrl >= 0.30
        threshold_drop = 0.30
        sustained_drop_fraction = float(np.mean(drops_vs_ctrl >= threshold_drop))

    # B/C drops vs init (no per-module ctrl branch — they evolve same in both)
    final_drop_B = ((var_M_init['B'] - var_M_traj['B'][-1]) / var_M_init['B']
                    if var_M_traj['B'] and var_M_init['B'] > 1e-12 else 0.0)
    final_drop_C = ((var_M_init['C'] - var_M_traj['C'][-1]) / var_M_init['C']
                    if var_M_traj['C'] and var_M_init['C'] > 1e-12 else 0.0)

    # KNV checks
    G_A_min = float(np.nanmin(G_traj['A'])) if G_traj['A'] else 0.0
    delta_centred_A_max = float(np.nanmax(delta_centred_traj['A'])) if delta_centred_traj['A'] else 0.0
    delta_shape_A_max = float(np.nanmax(delta_shape_traj['A'])) if delta_shape_traj['A'] else 0.0

    G_diff_KNV = G_A_min < KNV_THRESHOLDS_6C_B['G_min']
    # Δ-topological KNV: either centred Δ OR Δ_shape exceeds delta_crit
    delta_top_KNV = (
        delta_centred_A_max > KNV_THRESHOLDS_6C_B['delta_crit']
        or delta_shape_A_max > KNV_THRESHOLDS_6C_B['delta_crit']
    )

    # (2) Multi-signal Γ″
    tau_vec_A = np.column_stack([tau_T_traj['A'], tau_I_traj['A']]) if tau_T_traj['A'] else np.zeros((0, 2))
    gamma_diag = compute_gamma_signals({
        'var_M': var_M_traj['A'],
        'h_M_mean': h_M_mean_traj['A'],
        'tau_vec': tau_vec_A,
    }, dt=cfg_engine.dt)
    gamma_morph_KNV = gamma_diag['aggregate_morph_KNV']

    # ─────────────────────────────────────────────────────────────────
    # (B) Local diagnostics — atomic state flags, no aggregate verdict
    # ─────────────────────────────────────────────────────────────────
    DELTA_SHAPE_EXCURSION_BAND = (0.30, 0.35)  # provisional, not calibrated
    DELTA_SHAPE_KNV_NET = 0.35                 # > this = clearly outside any
                                                # plausible Δ_shape corridor
    diagnostics = {
        # Absolute contraction (vs initial state) — would mean var_M
        # actually decreases below its starting value
        'absolute_contraction_present': bool(final_drop_vs_init >= 0.30),
        'absolute_contraction_min_present': bool(min_drop_vs_init >= 0.30),

        # Relative contraction (vs control branch) — means swap suppressed
        # the natural expansion. Not the same thing as absolute contraction.
        'relative_contraction_final_present': bool(final_drop_vs_ctrl >= 0.30),
        'relative_contraction_min_present': bool(min_drop_vs_ctrl >= 0.30),
        'relative_contraction_sustained_50pct': bool(sustained_drop_fraction >= 0.50),
        'relative_contraction_sustained_30pct': bool(sustained_drop_fraction >= 0.30),

        # Expansion of A vs ctrl (swap caused MORE expansion than baseline)
        'expansion_vs_ctrl_present': bool(final_drop_vs_ctrl <= -0.30),
        'expansion_extreme_vs_ctrl': bool(final_drop_vs_ctrl <= -1.0),

        # Bifurcation: transient contraction then expansion
        'bifurcation_transient_then_expansion': bool(
            min_drop_vs_ctrl >= 0.20 and final_drop_vs_ctrl <= -0.10
        ),

        # Δ states — distinguishing centred vs shape (critical for T25')
        'delta_centred_in_corridor': bool(
            delta_centred_A_max <= KNV_THRESHOLDS_6C_B['delta_crit']
        ),
        'delta_shape_in_corridor': bool(
            delta_shape_A_max <= DELTA_SHAPE_EXCURSION_BAND[0]  # 0.30
        ),
        'delta_shape_boundary_excursion': bool(
            DELTA_SHAPE_EXCURSION_BAND[0] < delta_shape_A_max
            <= DELTA_SHAPE_EXCURSION_BAND[1]
        ),
        'delta_shape_clearly_outside': bool(
            delta_shape_A_max > DELTA_SHAPE_KNV_NET
        ),
        'delta_centred_shape_diverge': bool(
            abs(delta_centred_A_max - delta_shape_A_max) > 0.10
        ),

        # 𝒢 state
        'G_proxy_alive': bool(G_A_min > KNV_THRESHOLDS_6C_B['G_min']),

        # Γ″ state
        'gamma_dampable_var_M': gamma_diag['per_signal'].get('var_M', {}).get('dampable', True),
        'gamma_dampable_h_M': gamma_diag['per_signal'].get('h_M_mean', {}).get('dampable', True),
        'gamma_dampable_tau_vec': gamma_diag['per_signal'].get('tau_vec', {}).get('dampable', True),
        'gamma_n_non_dampable': gamma_diag['n_non_dampable_signals'],

        # Partner state — DIRECTIONALLY split (Alex's fix, session 5)
        # final_drop_X >= 0.30  → partner X contracts
        # final_drop_X <= -0.30 → partner X expands
        # |final_drop_X| in [0.10, 0.30] → partner X disturbed (non-contracting)
        'partner_B_contracts': bool(final_drop_B >= 0.30),
        'partner_B_expands': bool(final_drop_B <= -0.30),
        'partner_B_disturbed_no_contraction': bool(
            0.10 <= abs(final_drop_B) < 0.30
        ),
        'partner_C_contracts': bool(final_drop_C >= 0.30),
        'partner_C_expands': bool(final_drop_C <= -0.30),
        'partner_C_disturbed_no_contraction': bool(
            0.10 <= abs(final_drop_C) < 0.30
        ),

        # Numerical state
        'numerical_instability': bool(numerical_instability_step is not None),
    }

    partners_contract = (diagnostics['partner_B_contracts']
                         or diagnostics['partner_C_contracts'])
    partners_expand = (diagnostics['partner_B_expands']
                       or diagnostics['partner_C_expands'])
    partners_disturbed_no_contraction = (
        diagnostics['partner_B_disturbed_no_contraction']
        or diagnostics['partner_C_disturbed_no_contraction']
    )

    # ─────────────────────────────────────────────────────────────────
    # (C) Regimes detected — NON-EXCLUSIVE multi-label.
    # Critical: NO regime is labelled "VIABLE" if a KNV signal is
    # co-present in this run. Viability is a global property — partial
    # axis success while another axis violates the corridor is not
    # viability, it is partial axis contraction.
    # ─────────────────────────────────────────────────────────────────
    regimes = []

    # KNV layer
    if diagnostics['numerical_instability']:
        regimes.append('NUMERICAL_INSTABILITY')

    if not diagnostics['G_proxy_alive']:
        regimes.append('KNV_DIFFERENTIAL_G_COLLAPSE')

    if not diagnostics['delta_centred_in_corridor']:
        regimes.append('KNV_TOPOLOGICAL_DELTA_CENTRED')

    if diagnostics['delta_shape_clearly_outside']:
        regimes.append('KNV_TOPOLOGICAL_DELTA_SHAPE')
    elif diagnostics['delta_shape_boundary_excursion']:
        # Provisional label — Δ_shape corridor not yet calibrated
        regimes.append('DELTA_SHAPE_BOUNDARY_EXCURSION')

    if diagnostics['gamma_n_non_dampable'] >= 2:
        regimes.append('KNV_MORPHOGENIC')

    # KNV co-presence guard for "VIABLE" labels
    knv_co_present = any(
        r.startswith('KNV_') or r == 'NUMERICAL_INSTABILITY'
        for r in regimes
    )

    # Contraction layer
    # Absolute contraction (var_M actually decreased below init)
    if diagnostics['absolute_contraction_present']:
        if partners_contract and not knv_co_present:
            regimes.append('PROPAGATED_CONTRACTION_VIABLE')
        elif partners_contract and knv_co_present:
            regimes.append('PROPAGATED_CONTRACTION_WITH_KNV_CO_PRESENT')
        elif not partners_contract and not partners_expand and not knv_co_present:
            regimes.append('LOCAL_CONTRACTION_VIABLE')
        elif not partners_contract and not partners_expand and knv_co_present:
            regimes.append('LOCAL_CONTRACTION_WITH_KNV_CO_PRESENT')
        elif partners_expand:
            regimes.append('PARTIAL_AXIS_CONTRACTION_NON_GLOBAL_VIABILITY')

    # Relative contraction without absolute contraction (= suppression of
    # natural expansion drift, NOT actual contraction)
    if (diagnostics['relative_contraction_final_present']
            and not diagnostics['absolute_contraction_present']):
        regimes.append('RELATIVE_CONTRACTION_VS_CONTROL_ONLY')

    # Transient contraction (visible early but not sustained)
    if (diagnostics['relative_contraction_min_present']
            and not diagnostics['relative_contraction_sustained_50pct']):
        regimes.append('TRANSIENT_CONTRACTION_NON_SUSTAINED')

    # Bifurcation pattern
    if diagnostics['bifurcation_transient_then_expansion']:
        regimes.append('BIFURCATION_CONTRACTION_TO_EXPANSION')

    # Expansion patterns
    if diagnostics['expansion_vs_ctrl_present']:
        regimes.append('A_EXPANSION_VS_CTRL')
    if diagnostics['expansion_extreme_vs_ctrl']:
        regimes.append('A_EXPANSION_EXTREME')

    # Partner pattern (cross-cutting)
    if partners_expand and not partners_contract:
        regimes.append('PARTNER_EXPANSION_UNDER_A_CONSTRAINT')
    if partners_disturbed_no_contraction and not partners_contract and not partners_expand:
        regimes.append('PARTNER_DISTURBED_NON_CONTRACTION')

    # Resistance: only if neither contraction nor expansion of A registered
    if (abs(final_drop_vs_ctrl) < 0.10
            and not diagnostics['relative_contraction_min_present']):
        if not partners_contract and not partners_expand and not partners_disturbed_no_contraction:
            regimes.append('STRUCTURAL_RESISTANCE_OSCILLATORY')

    if not regimes:
        regimes.append('UNCLASSIFIED')

    # ─────────────────────────────────────────────────────────────────
    # (D) Tensions observed — structured contradictions to NOT collapse
    # ─────────────────────────────────────────────────────────────────
    tensions = []
    if diagnostics['relative_contraction_min_present'] and \
       diagnostics['delta_shape_clearly_outside']:
        tensions.append({
            'kind': 'relative_contraction_present_but_delta_shape_KNV',
            'description': (
                "var_M shows relative contraction vs control "
                "(min_drop_vs_ctrl >= 0.30) but Δ_shape clearly exits any "
                "plausible corridor (>0.35). Contraction signal on var_M "
                "axis is real but accompanied by morphology drift."
            ),
            'min_drop_vs_ctrl': min_drop_vs_ctrl,
            'delta_shape_max': delta_shape_A_max,
        })

    if diagnostics['relative_contraction_min_present'] and \
       diagnostics['delta_shape_boundary_excursion']:
        tensions.append({
            'kind': 'relative_contraction_with_delta_shape_at_boundary',
            'description': (
                "var_M shows relative contraction vs control AND Δ_shape "
                "is in the 0.30-0.35 boundary band. Δ_shape corridor not "
                "yet calibrated — boundary signal noted but not declared "
                "KNV."
            ),
            'min_drop_vs_ctrl': min_drop_vs_ctrl,
            'delta_shape_max': delta_shape_A_max,
        })

    if diagnostics['delta_centred_shape_diverge']:
        tensions.append({
            'kind': 'delta_centred_vs_shape_divergence',
            'description': (
                "Δ_centred and Δ_shape report different morphology states. "
                "Δ_centred measures spread of MEANS (E_T, E_M, E_I); Δ_shape "
                "measures spread of VARIANCES. A symmetric var_M change with "
                "stable mean shows up only on Δ_shape — exactly what T25' is "
                "designed to detect."
            ),
            'delta_centred_max': delta_centred_A_max,
            'delta_shape_max': delta_shape_A_max,
        })

    if diagnostics['relative_contraction_min_present'] and \
       diagnostics['gamma_n_non_dampable'] == 1:
        tensions.append({
            'kind': 'contraction_present_with_partial_morph_alarm',
            'description': (
                "var_M relative contraction registered AND one Γ″ signal "
                "non-dampable (but below the 2-signal aggregate threshold). "
                "Morphogenic stress on a single channel — informative but "
                "not declared KNV."
            ),
            'gamma_n_non_dampable': gamma_diag['n_non_dampable_signals'],
        })

    if diagnostics['bifurcation_transient_then_expansion']:
        tensions.append({
            'kind': 'transient_contraction_followed_by_expansion',
            'description': (
                "Initial contraction phase visible (min_drop_vs_ctrl > 0.20) "
                "is reversed by end (final_drop_vs_ctrl < -0.10). The system "
                "responds to the swap then escapes — a bifurcation pattern, "
                "not a stable regime."
            ),
            'min_drop_vs_ctrl': min_drop_vs_ctrl,
            'final_drop_vs_ctrl': final_drop_vs_ctrl,
        })

    # NEW tension: relative contraction WITHOUT absolute contraction
    # = the swap suppresses the natural expansion drift but does not
    # actually contract var_M below its initial value
    if (diagnostics['relative_contraction_final_present']
            and not diagnostics['absolute_contraction_present']):
        tensions.append({
            'kind': 'relative_contraction_without_absolute_contraction',
            'description': (
                "var_M(swap) is below var_M(ctrl) but ABOVE var_M_init. "
                "The swap suppresses the natural expansion drift of 6a "
                "without actually contracting var_M. Modulation, not "
                "contraction. This is the central question raised by T25': "
                "does the test measure a true contraction capacity, or only "
                "a modulation of natural expansion under constraint?"
            ),
            'final_drop_vs_init': final_drop_vs_init,
            'final_drop_vs_ctrl': final_drop_vs_ctrl,
        })

    # NEW tension: A is constrained but partners EXPAND
    if partners_expand:
        tensions.append({
            'kind': 'partner_expansion_under_A_constraint',
            'description': (
                "Module A under contraction swap, but B and/or C show "
                "var_M expansion (final_drop <= -0.30). The constraint on "
                "A's morphology displaces variance to partner modules — "
                "the system as a whole does not contract; it redistributes."
            ),
            'final_drop_B': final_drop_B,
            'final_drop_C': final_drop_C,
        })

    # ─────────────────────────────────────────────────────────────────
    # (E) Open hypotheses — three competing readings
    # ─────────────────────────────────────────────────────────────────
    open_hypotheses = {
        'H1_numerical_residual': {
            'plausible_if': [
                'Δ_shape values exceed natural psi-grid scale (>10)',
                'overflow warnings during run',
                'final_drop_vs_ctrl ∈ {NaN, very large negative}',
            ],
            'flag_active': bool(
                delta_shape_A_max > 10.0 or final_drop_vs_ctrl < -10.0
                or numerical_instability_step is not None
            ),
        },
        'H2_parametric_under_exploration': {
            'plausible_if': [
                'no LOCAL_CONTRACTION_VIABLE found in any sweep config',
                'sweep range did not push var_floor low enough OR '
                'lambda_KNV high enough to overcome natural drift',
            ],
            # Marked active for the calling layer to set after the sweep
            'flag_active': None,
        },
        'H3_structural_limit': {
            'plausible_if': [
                'across all (alpha, var_floor, lambda) tested, var_M(swap) '
                'is never sustained below var_M(ctrl) by the threshold',
                'AND the swap DOES produce an effect (Δ_shape or KNV signal)',
                'AND increasing lambda only triggers KNV without contraction',
            ],
            'flag_active': None,
        },
    }

    return {
        'coupling_form': coupling_form,
        'epsilon': eps,
        'alpha': alpha,
        'var_floor': var_floor,
        'lambda_KNV_target': lambda_KNV_target,

        # ── (A) Raw signals — full trajectories ──
        'signals_raw': {
            'var_M_A_traj': [float(v) for v in var_M_traj['A']],
            'var_M_A_ctrl_traj': [float(v) for v in var_M_A_ctrl_traj],
            'var_M_B_traj': [float(v) for v in var_M_traj['B']],
            'var_M_C_traj': [float(v) for v in var_M_traj['C']],
            'delta_centred_A_traj': [float(v) for v in delta_centred_traj['A']],
            'delta_shape_A_traj': [float(v) for v in delta_shape_traj['A']],
            'G_proxy_A_traj': [float(v) for v in G_traj['A']],
            'tau_T_A_traj': [float(v) for v in tau_T_traj['A']],
            'tau_I_A_traj': [float(v) for v in tau_I_traj['A']],
            'h_M_mean_A_traj': [float(v) for v in h_M_mean_traj['A']],
            'var_max_applied_traj': [float(v) for v in var_max_applied_traj],
        },
        'signals_summary': {
            'var_M_init_A': var_M_init['A'],
            'var_M_final_A': float(var_M_traj['A'][-1]) if var_M_traj['A'] else var_M_init['A'],
            'var_M_min_A': float(var_M_A.min()) if var_M_A.size > 0 else var_M_init['A'],
            'var_M_max_A': float(var_M_A.max()) if var_M_A.size > 0 else var_M_init['A'],
            'var_M_A_ctrl_final': float(var_M_A_ctrl[-1]) if var_M_A_ctrl.size > 0 else var_M_init['A'],
            'final_drop_vs_init_A': final_drop_vs_init,
            'final_drop_vs_ctrl_A': final_drop_vs_ctrl,
            'min_drop_vs_init_A': min_drop_vs_init,
            'min_drop_vs_ctrl_A': min_drop_vs_ctrl,
            'sustained_drop_fraction_A': sustained_drop_fraction,
            'final_drop_B': final_drop_B,
            'final_drop_C': final_drop_C,
            'G_A_min': G_A_min,
            'G_A_max': float(np.nanmax(G_traj['A'])) if G_traj['A'] else 0.0,
            'delta_centred_A_max': delta_centred_A_max,
            'delta_shape_A_max': delta_shape_A_max,
            'gamma_diag': gamma_diag,
            'numerical_instability_step': numerical_instability_step,
            'final_var_max_applied': var_max_applied_traj[-1] if var_max_applied_traj else var_max_ramp_start,
            'var_max_ramp_start': var_max_ramp_start,
            'n_steps_completed': len([v for v in psi_finite_traj if v]),
        },

        # ── (B) Local diagnostics — atomic flags ──
        'diagnostics_local': diagnostics,

        # ── (C) Regimes detected — non-exclusive ──
        'regimes_detected': regimes,

        # ── (D) Tensions — contradictions preserved ──
        'tensions_observed': tensions,

        # ── (E) Open hypotheses ──
        'open_hypotheses': open_hypotheses,
    }


def measure_T25_prime_for_form(
    cfg_engine, coupling_form: str, eps: float, base_seed: int,
) -> dict:
    """For one form, sweep (alpha × var_floor × lambda) and AGGREGATE
    regimes/tensions across the sweep WITHOUT collapsing to a single
    best-outcome label.

    Calibration (Alex's 5 corrections): var_init_A is computed dynamically
    via a probe run, then alpha is derived to reach var_floor at well-
    defined steps starting FROM var_init.
    """
    contraction_steps = 200

    # PROBE: minimal run to capture true var_init_A
    probe = measure_T25_prime_run(
        cfg_engine, coupling_form, eps, base_seed,
        alpha=0.0, var_floor=10.0, lambda_KNV_target=cfg_engine.lambda_KNV,
        contraction_steps=2,
    )
    var_init_A = probe['signals_summary']['var_M_init_A']

    def alpha_to_reach_floor_at(target_step: int, target_floor: float) -> float:
        return (var_init_A - target_floor) / max(target_step, 1)

    sweep_configs = [
        {'alpha': alpha_to_reach_floor_at(50, max(var_init_A * 0.80, 0.05)),
         'var_floor': max(var_init_A * 0.80, 0.05), 'lambda_KNV_target': 0.10},
        {'alpha': alpha_to_reach_floor_at(80, max(var_init_A * 0.60, 0.05)),
         'var_floor': max(var_init_A * 0.60, 0.05), 'lambda_KNV_target': 0.10},
        {'alpha': alpha_to_reach_floor_at(80, max(var_init_A * 0.60, 0.05)),
         'var_floor': max(var_init_A * 0.60, 0.05), 'lambda_KNV_target': 0.30},
        {'alpha': alpha_to_reach_floor_at(120, max(var_init_A * 0.50, 0.05)),
         'var_floor': max(var_init_A * 0.50, 0.05), 'lambda_KNV_target': 0.30},
        {'alpha': alpha_to_reach_floor_at(150, max(var_init_A * 0.40, 0.05)),
         'var_floor': max(var_init_A * 0.40, 0.05), 'lambda_KNV_target': 0.30},
    ]

    sweep_results = {}
    for cfg in sweep_configs:
        label = (f"a{cfg['alpha']:.4f}_floor{cfg['var_floor']:.3f}"
                 f"_lam{cfg['lambda_KNV_target']}")
        r = measure_T25_prime_run(
            cfg_engine, coupling_form, eps, base_seed,
            alpha=cfg['alpha'],
            var_floor=cfg['var_floor'],
            lambda_KNV_target=cfg['lambda_KNV_target'],
            contraction_steps=contraction_steps,
        )
        sweep_results[label] = r

    # ── Aggregate across sweep — NO collapse to single label ──
    all_regimes_seen = set()
    sweep_regime_map = {}
    for label, r in sweep_results.items():
        sweep_regime_map[label] = set(r['regimes_detected'])
        all_regimes_seen.update(r['regimes_detected'])

    all_tensions = []
    for label, r in sweep_results.items():
        for t in r['tensions_observed']:
            all_tensions.append({**t, 'sweep_config': label})

    # Strict viability: absolute contraction AND no co-present KNV
    any_viable_local_or_propagated = any(
        ('LOCAL_CONTRACTION_VIABLE' in r['regimes_detected']
         or 'PROPAGATED_CONTRACTION_VIABLE' in r['regimes_detected'])
        for r in sweep_results.values()
    )
    any_relative_contraction_only = any(
        'RELATIVE_CONTRACTION_VS_CONTROL_ONLY' in r['regimes_detected']
        for r in sweep_results.values()
    )
    any_transient_contraction = any(
        'TRANSIENT_CONTRACTION_NON_SUSTAINED' in r['regimes_detected']
        for r in sweep_results.values()
    )
    any_partial_axis = any(
        'PARTIAL_AXIS_CONTRACTION_NON_GLOBAL_VIABILITY' in r['regimes_detected']
        for r in sweep_results.values()
    )
    any_partner_expansion = any(
        'PARTNER_EXPANSION_UNDER_A_CONSTRAINT' in r['regimes_detected']
        for r in sweep_results.values()
    )
    any_swap_effect = any(
        len(set(r['regimes_detected']) - {'UNCLASSIFIED'}) > 0
        for r in sweep_results.values()
    )
    any_KNV_when_lambda_increased = any(
        any(reg.startswith('KNV_') for reg in r['regimes_detected'])
        and r['lambda_KNV_target'] >= 0.30
        for r in sweep_results.values()
    )

    # H2: parametric under-exploration. Active iff we did NOT find a
    # viable contraction in the sweep range. The fact that we found
    # relative-only contraction or partial-axis contraction does NOT
    # rule out H2 — wider sweep might still find a viable point.
    h2_active = (not any_viable_local_or_propagated)

    # H3: structural limit. Active iff swap had effect AND increasing
    # lambda triggers KNV AND no viable contraction found. Necessary
    # but not sufficient — wider sweep could still rule it out.
    h3_active = (
        not any_viable_local_or_propagated
        and any_swap_effect
        and any_KNV_when_lambda_increased
    )

    return {
        'coupling_form': coupling_form,
        'var_init_A_probe': var_init_A,
        'sweep_results': sweep_results,
        'aggregate': {
            'all_regimes_seen': sorted(all_regimes_seen),
            'sweep_regime_map': {k: sorted(v) for k, v in sweep_regime_map.items()},
            'any_viable_local_or_propagated': any_viable_local_or_propagated,
            'any_relative_contraction_only': any_relative_contraction_only,
            'any_transient_contraction': any_transient_contraction,
            'any_partial_axis_contraction': any_partial_axis,
            'any_partner_expansion': any_partner_expansion,
            'any_swap_effect': any_swap_effect,
            'any_KNV_when_lambda_increased': any_KNV_when_lambda_increased,
        },
        'all_tensions_across_sweep': all_tensions,
        'open_hypotheses_status': {
            'H1_numerical_residual': {
                'flag_active': any(
                    r['open_hypotheses']['H1_numerical_residual']['flag_active']
                    for r in sweep_results.values()
                ),
            },
            'H2_parametric_under_exploration': {
                'flag_active': h2_active,
                'note': (
                    "H2 active iff no LOCAL/PROPAGATED_CONTRACTION_VIABLE "
                    "found in the sweep. Relative-only contraction or "
                    "partial-axis contraction do NOT rule out H2 — wider "
                    "parametric range could still find a viable point."
                ),
            },
            'H3_structural_limit': {
                'flag_active': h3_active,
                'note': (
                    "H3 active iff (no viable contraction) AND swap had "
                    "effect AND increasing lambda triggers KNV. Necessary "
                    "but not sufficient — wider sweep could rule it out."
                ),
            },
        },
    }


def run_T25_prime_test(base_seed: int = 42) -> dict:
    print("=" * 70)
    print("F6.T25' — Morphodynamic constrained contraction")
    print("Output structure per form: (A) signals (B) diagnostics")
    print("                          (C) regimes (non-exclusive)")
    print("                          (D) tensions (E) open hypotheses")
    print("=" * 70)

    cfg = FactorialEngineConfig(
        dt=0.05, T_steps=300, mode=EngineMode.FULL,
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
        print(f"\n{'─' * 70}")
        print(f"  Form: {form}")
        print(f"{'─' * 70}")
        r = measure_T25_prime_for_form(cfg, form, eps, base_seed)
        per_form[form] = r
        agg = r['aggregate']

        print(f"\n  var_init_A (probed) = {r['var_init_A_probe']:.4f}")

        # Per-config compact view (signals A + diagnostics B condensed)
        print(f"\n  Per-config signals + diagnostics:")
        for label, sweep_r in r['sweep_results'].items():
            sigs = sweep_r['signals_summary']
            d = sweep_r['diagnostics_local']
            print(f"  [{label}]")
            print(f"    (A) drops_vs_init: final={sigs['final_drop_vs_init_A']:+.3f}  "
                  f"min={sigs['min_drop_vs_init_A']:+.3f}")
            print(f"        drops_vs_ctrl: final={sigs['final_drop_vs_ctrl_A']:+.3f}  "
                  f"min={sigs['min_drop_vs_ctrl_A']:+.3f}  "
                  f"sustained={sigs['sustained_drop_fraction_A']:.2f}")
            print(f"        partner drops final: B={sigs['final_drop_B']:+.3f} "
                  f"C={sigs['final_drop_C']:+.3f}")
            print(f"        Δ_centred_max={sigs['delta_centred_A_max']:.4f}  "
                  f"Δ_shape_max={sigs['delta_shape_A_max']:.4f}  "
                  f"G_min={sigs['G_A_min']:.4f}")
            print(f"    (B) abs_contr={d['absolute_contraction_present']}  "
                  f"rel_contr_final={d['relative_contraction_final_present']}  "
                  f"rel_contr_min={d['relative_contraction_min_present']}  "
                  f"rel_sust50={d['relative_contraction_sustained_50pct']}")
            print(f"        A_expand={d['expansion_vs_ctrl_present']}  "
                  f"bifurc={d['bifurcation_transient_then_expansion']}")
            print(f"        partner_B(c/e/d)={d['partner_B_contracts']}/{d['partner_B_expands']}/{d['partner_B_disturbed_no_contraction']}  "
                  f"partner_C(c/e/d)={d['partner_C_contracts']}/{d['partner_C_expands']}/{d['partner_C_disturbed_no_contraction']}")
            print(f"        Δ_centred_in_corr={d['delta_centred_in_corridor']}  "
                  f"Δ_shape: in_corr={d['delta_shape_in_corridor']} "
                  f"boundary_excur={d['delta_shape_boundary_excursion']} "
                  f"clearly_outside={d['delta_shape_clearly_outside']}")
            print(f"        G_alive={d['G_proxy_alive']}  "
                  f"gamma_n_non_damp={d['gamma_n_non_dampable']}/3")
            print(f"    (C) regimes: {sweep_r['regimes_detected']}")

        # (C) Aggregate regimes seen across the sweep
        print(f"\n  (C-aggregate) ALL regimes seen across sweep:")
        for reg in agg['all_regimes_seen']:
            print(f"      • {reg}")

        # (D) Tensions
        if r['all_tensions_across_sweep']:
            print(f"\n  (D) Tensions observed (NOT collapsed):")
            for t in r['all_tensions_across_sweep']:
                print(f"      • [{t['sweep_config']}] {t['kind']}")
        else:
            print(f"\n  (D) No structural tensions detected.")

        # (E) Open hypotheses
        oh = r['open_hypotheses_status']
        print(f"\n  (E) Open hypotheses status:")
        print(f"      H1 (numerical residual)         : "
              f"{'ACTIVE' if oh['H1_numerical_residual']['flag_active'] else 'inactive'}")
        print(f"      H2 (parametric under-exploration): "
              f"{'ACTIVE' if oh['H2_parametric_under_exploration']['flag_active'] else 'inactive'}")
        print(f"      H3 (structural limit)            : "
              f"{'ACTIVE' if oh['H3_structural_limit']['flag_active'] else 'inactive'}")

    # ─────────────────────────────────────────────────────────────────
    # Cross-form aggregate — what differs between forms?
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  CROSS-FORM DIFFERENTIATION")
    print(f"{'═' * 70}")
    cross_form = {}
    for form in forms:
        agg = per_form[form]['aggregate']
        cross_form[form] = {
            'regimes_seen': agg['all_regimes_seen'],
            'any_VIABLE_LOCAL_OR_PROPAGATED': agg['any_viable_local_or_propagated'],
            'any_RELATIVE_ONLY': agg['any_relative_contraction_only'],
            'any_TRANSIENT': agg['any_transient_contraction'],
            'any_PARTIAL_AXIS': agg['any_partial_axis_contraction'],
            'any_PARTNER_EXPANSION': agg['any_partner_expansion'],
        }
        print(f"  {form:<32s}: {len(cross_form[form]['regimes_seen']):>2} regimes")
        print(f"    VIABLE_local_or_propagated={agg['any_viable_local_or_propagated']}, "
              f"RELATIVE_ONLY={agg['any_relative_contraction_only']}, "
              f"PARTIAL_AXIS={agg['any_partial_axis_contraction']}")
        print(f"    PARTNER_EXPANSION={agg['any_partner_expansion']}, "
              f"TRANSIENT={agg['any_transient_contraction']}")

    return {
        'test': 'F6_T25_prime_morphodynamic_constrained_contraction',
        'per_form': per_form,
        'cross_form_summary': cross_form,
        'base_seed': base_seed,
        'epsilon': eps,
        'note': (
            "Three-layer output (signals/diagnostics/regimes) without "
            "single-label collapse. Tensions are preserved as structured "
            "contradictions. H1/H2/H3 hypotheses tracked; THIS RUN DOES "
            "NOT CONCLUDE on Φ_extra. The verdict on perspective coupling "
            "requires (a) wider parametric sweep to rule out H2, (b) "
            "examination of which regimes differ between forms — that is "
            "where the perspective signal would manifest."
        ),
    }


if __name__ == "__main__":
    result = run_T25_prime_test(base_seed=42)
    out = Path("/home/claude/mcq_v4/results/phase6c_b/F6_T25_prime_contraction.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
