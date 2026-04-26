"""
FactorialEngine — master equation on Θ = T × M × I.

Implements the geometric continuity equation of Ch.3 §3.1.2 in the
factorial representation, with marginal h approximation (Phase 6a).

  ∂_t ψ = ∇·(D_eff/h^2 · ∇ψ)              (α: diffusion in geometry)
        - ∇·(ψ · ∇Φ_corr)                  (β: regulation)
        - ∇·J_noise                        (multiplicative flux noise)

  ∂_t h_a(θ_a) = -β_a · ψ_marg_a(θ_a) · h_a(θ_a)
                + γ_a · h_a(θ_a) · (1 - h_a(θ_a)/h_0)

with β_a = β_0 · w_a, γ_a = γ_0 / w_a per module weights.

Boundary conditions: reflective (Neumann) on all axes for ψ.
Mass conservation: by flux divergence form (exact to roundoff before clipping).
Positivity: by exceptional clipping (logged, not renormalised).

The engine supports three modes:
  FULL        — α + β (canonical)
  ALPHA_ONLY  — diffusion + drift, Φ_corr ≡ 0
  BETA_ONLY   — diffusion isotropic (no h-modulation) + Φ_corr active
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mcq_v4.factorial.state import (
    FactorialState, ModuleConfig, FactorialEngineConfig, EngineMode,
    StepDiagnostic, initial_h,
    THETA_T, THETA_M, THETA_I,
)
from mcq_v4.factorial.observables import (
    compute_observables, compute_D_eff, compute_Phi_corr,
)


class FactorialEngine:
    """
    Master equation engine on Θ = T × M × I.

    The engine is per-module: one engine instance corresponds to one module.
    Phase 6a uses one isolated engine. Phase 6b will couple multiple engines.

    Parameters
    ----------
    cfg : FactorialEngineConfig
        Numerical configuration (dt, D_0, ..., mode).
    module_cfg : ModuleConfig
        Module-specific configuration (weights, name, seed).
    """

    def __init__(
        self,
        cfg: FactorialEngineConfig,
        module_cfg: ModuleConfig,
    ):
        self.cfg = cfg
        self.module_cfg = module_cfg
        self.rng = np.random.default_rng(module_cfg.seed)

        # Pre-compute per-axis sedimentation and erosion rates
        w_T, w_M, w_I = module_cfg.weights
        self.beta_T = cfg.beta_0 * w_T
        self.beta_M = cfg.beta_0 * w_M
        self.beta_I = cfg.beta_0 * w_I
        self.gamma_T = cfg.gamma_0 / w_T
        self.gamma_M = cfg.gamma_0 / w_M
        self.gamma_I = cfg.gamma_0 / w_I

    # ------------------------------------------------------------------
    # Diffusion (α component)
    # ------------------------------------------------------------------

    def _diffuse(
        self, psi: np.ndarray, h_T: np.ndarray, h_M: np.ndarray, h_I: np.ndarray,
        D_eff: float, mode: EngineMode,
    ) -> np.ndarray:
        """
        Discrete Laplacian on T × M × I in the marginal geometry.

        For each axis a:
          flux_a(edge) = D_eff / h_a^2(edge) · (psi_right - psi_left)

        Reflective BC: zero flux on domain boundary edges.

        In BETA_ONLY mode, h is replaced by h_0 (isotropic diffusion).
        In ALPHA_ONLY and FULL modes, h modulates diffusion as usual.
        """
        # In BETA_ONLY mode, use isotropic baseline h
        if mode == EngineMode.BETA_ONLY:
            inv_h2_T = np.ones_like(h_T) / (self.cfg.h_0 ** 2)
            inv_h2_M = np.ones_like(h_M) / (self.cfg.h_0 ** 2)
            inv_h2_I = np.ones_like(h_I) / (self.cfg.h_0 ** 2)
        else:
            inv_h2_T = 1.0 / np.maximum(h_T ** 2, 1e-12)
            inv_h2_M = 1.0 / np.maximum(h_M ** 2, 1e-12)
            inv_h2_I = 1.0 / np.maximum(h_I ** 2, 1e-12)

        # Edge-averaged 1/h^2 (interior edges have indices 0..3 in the 5-level axis)
        edge_T = 0.5 * (inv_h2_T[:-1] + inv_h2_T[1:])  # shape (4,)
        edge_M = 0.5 * (inv_h2_M[:-1] + inv_h2_M[1:])
        edge_I = 0.5 * (inv_h2_I[:-1] + inv_h2_I[1:])

        # Differences psi_right - psi_left on interior edges
        # T-axis: shape (4, 5, 5)
        d_psi_T = psi[1:, :, :] - psi[:-1, :, :]
        d_psi_M = psi[:, 1:, :] - psi[:, :-1, :]
        d_psi_I = psi[:, :, 1:] - psi[:, :, :-1]

        # Diffusive fluxes on interior edges
        flux_T_int = D_eff * edge_T[:, None, None] * d_psi_T  # (4, 5, 5)
        flux_M_int = D_eff * edge_M[None, :, None] * d_psi_M
        flux_I_int = D_eff * edge_I[None, None, :] * d_psi_I

        # Pad with zero fluxes on boundary edges → shape (6, 5, 5) etc.
        flux_T = np.concatenate([np.zeros((1, 5, 5)), flux_T_int, np.zeros((1, 5, 5))], axis=0)
        flux_M = np.concatenate([np.zeros((5, 1, 5)), flux_M_int, np.zeros((5, 1, 5))], axis=1)
        flux_I = np.concatenate([np.zeros((5, 5, 1)), flux_I_int, np.zeros((5, 5, 1))], axis=2)

        # Divergence: flux_out - flux_in per cell
        div_T = flux_T[1:] - flux_T[:-1]  # shape (5, 5, 5)
        div_M = flux_M[:, 1:] - flux_M[:, :-1]
        div_I = flux_I[:, :, 1:] - flux_I[:, :, :-1]

        return div_T + div_M + div_I

    # ------------------------------------------------------------------
    # Drift (β component — also active in ALPHA_ONLY for the V_𝒩 part,
    # but Phase 6a has no V_𝒩 yet, so drift comes only from Φ_corr,
    # which is zero in ALPHA_ONLY mode)
    # ------------------------------------------------------------------

    def _drift(
        self, psi: np.ndarray, h_T: np.ndarray, h_M: np.ndarray, h_I: np.ndarray,
        Phi_corr: np.ndarray,
    ) -> np.ndarray:
        """
        Drift term: -∇·(ψ · ∇Φ_corr) in the marginal geometry.

        Geometric gradient on each axis: ∂_a Φ / h_a(edge).
        Edge-averaged ψ for the flux: 0.5 · (ψ_left + ψ_right).
        """
        # Gradient of Phi_corr on interior edges, weighted by 1/h_edge
        h_T_edge = 0.5 * (h_T[:-1] + h_T[1:])  # (4,)
        h_M_edge = 0.5 * (h_M[:-1] + h_M[1:])
        h_I_edge = 0.5 * (h_I[:-1] + h_I[1:])

        d_Phi_T = Phi_corr[1:, :, :] - Phi_corr[:-1, :, :]  # (4, 5, 5)
        d_Phi_M = Phi_corr[:, 1:, :] - Phi_corr[:, :-1, :]
        d_Phi_I = Phi_corr[:, :, 1:] - Phi_corr[:, :, :-1]

        grad_Phi_T = d_Phi_T / np.maximum(h_T_edge[:, None, None], 1e-12)
        grad_Phi_M = d_Phi_M / np.maximum(h_M_edge[None, :, None], 1e-12)
        grad_Phi_I = d_Phi_I / np.maximum(h_I_edge[None, None, :], 1e-12)

        # Edge-averaged psi
        psi_T_edge = 0.5 * (psi[:-1, :, :] + psi[1:, :, :])  # (4, 5, 5)
        psi_M_edge = 0.5 * (psi[:, :-1, :] + psi[:, 1:, :])
        psi_I_edge = 0.5 * (psi[:, :, :-1] + psi[:, :, 1:])

        # Drift fluxes: -ψ · ∇Φ
        flux_T_int = -psi_T_edge * grad_Phi_T
        flux_M_int = -psi_M_edge * grad_Phi_M
        flux_I_int = -psi_I_edge * grad_Phi_I

        # Pad with zero fluxes on boundary
        flux_T = np.concatenate([np.zeros((1, 5, 5)), flux_T_int, np.zeros((1, 5, 5))], axis=0)
        flux_M = np.concatenate([np.zeros((5, 1, 5)), flux_M_int, np.zeros((5, 1, 5))], axis=1)
        flux_I = np.concatenate([np.zeros((5, 5, 1)), flux_I_int, np.zeros((5, 5, 1))], axis=2)

        # Divergence
        div_T = flux_T[1:] - flux_T[:-1]
        div_M = flux_M[:, 1:] - flux_M[:, :-1]
        div_I = flux_I[:, :, 1:] - flux_I[:, :, :-1]

        return div_T + div_M + div_I

    # ------------------------------------------------------------------
    # Multiplicative noise (flux-conservative)
    # ------------------------------------------------------------------

    def _noise_term(
        self, psi: np.ndarray, h_T: np.ndarray, h_M: np.ndarray, h_I: np.ndarray,
        D_eff: float, mode: EngineMode,
    ) -> np.ndarray:
        """
        Conservative multiplicative noise as flux divergence.

        On each interior edge:
          J_a^noise = σ · D_eff · sqrt(ψ_left · ψ_right) · ξ / h_a(edge)

        with ξ ~ N(0, 1) iid per edge per axis per time step.

        The flux-divergence form preserves total mass exactly (to roundoff
        before clipping). The geometric mean preserves multiplicativity.

        In BETA_ONLY mode, h is replaced by h_0 in the noise (consistent
        with isotropic diffusion).
        """
        sigma = self.cfg.sigma_eta

        if mode == EngineMode.BETA_ONLY:
            inv_h_T_edge = np.ones(4) / self.cfg.h_0
            inv_h_M_edge = np.ones(4) / self.cfg.h_0
            inv_h_I_edge = np.ones(4) / self.cfg.h_0
        else:
            inv_h_T_edge = 0.5 * (1.0 / np.maximum(h_T[:-1], 1e-12)
                                  + 1.0 / np.maximum(h_T[1:], 1e-12))
            inv_h_M_edge = 0.5 * (1.0 / np.maximum(h_M[:-1], 1e-12)
                                  + 1.0 / np.maximum(h_M[1:], 1e-12))
            inv_h_I_edge = 0.5 * (1.0 / np.maximum(h_I[:-1], 1e-12)
                                  + 1.0 / np.maximum(h_I[1:], 1e-12))

        # Geometric mean of psi on interior edges
        psi_edge_T = np.sqrt(np.maximum(psi[:-1, :, :] * psi[1:, :, :], 0))
        psi_edge_M = np.sqrt(np.maximum(psi[:, :-1, :] * psi[:, 1:, :], 0))
        psi_edge_I = np.sqrt(np.maximum(psi[:, :, :-1] * psi[:, :, 1:], 0))

        # Random fluxes on interior edges
        xi_T = self.rng.standard_normal(size=(4, 5, 5))
        xi_M = self.rng.standard_normal(size=(5, 4, 5))
        xi_I = self.rng.standard_normal(size=(5, 5, 4))

        J_T_int = sigma * D_eff * psi_edge_T * inv_h_T_edge[:, None, None] * xi_T
        J_M_int = sigma * D_eff * psi_edge_M * inv_h_M_edge[None, :, None] * xi_M
        J_I_int = sigma * D_eff * psi_edge_I * inv_h_I_edge[None, None, :] * xi_I

        # Pad with zero fluxes on boundary
        J_T = np.concatenate([np.zeros((1, 5, 5)), J_T_int, np.zeros((1, 5, 5))], axis=0)
        J_M = np.concatenate([np.zeros((5, 1, 5)), J_M_int, np.zeros((5, 1, 5))], axis=1)
        J_I = np.concatenate([np.zeros((5, 5, 1)), J_I_int, np.zeros((5, 5, 1))], axis=2)

        # Divergence (with negative sign as in continuity equation)
        div_T = J_T[1:] - J_T[:-1]
        div_M = J_M[:, 1:] - J_M[:, :-1]
        div_I = J_I[:, :, 1:] - J_I[:, :, :-1]

        return -(div_T + div_M + div_I)

    # ------------------------------------------------------------------
    # Metric update (sedimentation + erosion, marginal)
    # ------------------------------------------------------------------

    def _update_metric(
        self,
        psi: np.ndarray,
        h_T: np.ndarray, h_M: np.ndarray, h_I: np.ndarray,
        dt: float, freeze_h_M: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update marginal h_a using marginal projection of psi.

        ∂_t h_a(θ_a) = -β_a · ψ_a_marg(θ_a) · h_a(θ_a)
                      + γ_a · h_a(θ_a) · (1 - h_a(θ_a)/h_0)
        """
        psi_T = psi.sum(axis=(1, 2))
        psi_M = psi.sum(axis=(0, 2))
        psi_I = psi.sum(axis=(0, 1))

        h_0 = self.cfg.h_0
        dh_T = -self.beta_T * psi_T * h_T + self.gamma_T * h_T * (1 - h_T / h_0)
        dh_M = -self.beta_M * psi_M * h_M + self.gamma_M * h_M * (1 - h_M / h_0)
        dh_I = -self.beta_I * psi_I * h_I + self.gamma_I * h_I * (1 - h_I / h_0)

        h_T_new = np.clip(h_T + dt * dh_T, self.cfg.h_min, self.cfg.h_0)
        h_I_new = np.clip(h_I + dt * dh_I, self.cfg.h_min, self.cfg.h_0)

        if freeze_h_M:
            h_M_new = h_M.copy()
        else:
            h_M_new = np.clip(h_M + dt * dh_M, self.cfg.h_min, self.cfg.h_0)

        return h_T_new, h_M_new, h_I_new

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: FactorialState,
        dt: Optional[float] = None,
        freeze_h_M: bool = False,
    ) -> tuple[FactorialState, StepDiagnostic]:
        """
        Advance the state by one time step.

        Order:
          1. Compute observables (entropy, variances, marginals)
          2. Compute D_eff from observables
          3. Compute Φ_corr from observables (zero if mode == ALPHA_ONLY)
          4. Compute ∂ψ/∂t = diffusion + drift + noise
          5. Update ψ with explicit Euler
          6. Clip to enforce positivity (no renormalisation — log clips)
          7. Update marginal metric (skip h_M if frozen)

        Returns
        -------
        state : FactorialState
            New state after one time step.
        diag : StepDiagnostic
            Per-step numerical diagnostics (clip events, mass drift).
        """
        if dt is None:
            dt = self.cfg.dt
        mode = self.cfg.mode

        # 1. Observables
        obs = compute_observables(state.psi)

        # 2. D_eff
        D_eff = compute_D_eff(obs, self.cfg)

        # 3. Φ_corr
        Phi_corr = compute_Phi_corr(obs, self.cfg, mode)

        # 4. dψ/dt
        d_diffuse = self._diffuse(state.psi, state.h_T, state.h_M, state.h_I, D_eff, mode)
        d_drift = self._drift(state.psi, state.h_T, state.h_M, state.h_I, Phi_corr)
        d_noise = self._noise_term(state.psi, state.h_T, state.h_M, state.h_I, D_eff, mode)
        dpsi = d_diffuse + d_drift + d_noise

        # 5. Update ψ
        psi_pre_clip = state.psi + dt * dpsi

        # 6. Clip + log
        negative_mask = psi_pre_clip < 0
        n_clip = int(negative_mask.sum())
        if n_clip > 0:
            max_clip = float(-psi_pre_clip[negative_mask].min())
            clipped_mass = float(-psi_pre_clip[negative_mask].sum())
        else:
            max_clip = 0.0
            clipped_mass = 0.0

        psi_new = np.clip(psi_pre_clip, 0.0, None)

        # 7. Metric update
        h_T_new, h_M_new, h_I_new = self._update_metric(
            state.psi, state.h_T, state.h_M, state.h_I, dt, freeze_h_M,
        )

        new_state = FactorialState(
            psi=psi_new, h_T=h_T_new, h_M=h_M_new, h_I=h_I_new,
            cfg=state.cfg,
        )

        diag = StepDiagnostic(
            n_clip_events=n_clip,
            max_clip_magnitude=max_clip,
            clipped_mass=clipped_mass,
            mass_after_step=float(psi_new.sum()),
            mass_drift_step=abs(1.0 - float(psi_new.sum())),
        )

        return new_state, diag


# Helper to build initial states ------------------------------------------------

def make_initial_state(
    psi_init: np.ndarray, module_cfg: ModuleConfig,
) -> FactorialState:
    """Build a FactorialState from an initial ψ; metric initialised to h_0."""
    h_T_init, h_M_init, h_I_init = initial_h()
    # Normalise psi
    psi_normed = psi_init / max(psi_init.sum(), 1e-12)
    return FactorialState(
        psi=psi_normed,
        h_T=h_T_init, h_M=h_M_init, h_I=h_I_init,
        cfg=module_cfg,
    )
