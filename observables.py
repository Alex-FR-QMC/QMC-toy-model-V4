"""
Observables of the distributional state and computation of the regulatory
field Φ_corr (β component).

All observables are computed on the discrete cells of Θ = T × M × I with
uniform measure (dθ = 1 per cell). The marginal projections psi_T, psi_M,
psi_I are sums over the other two axes — exactly preserving normalisation
when ψ is normalised.
"""

from __future__ import annotations

import numpy as np

from mcq_v4.factorial.state import (
    FactorialEngineConfig, EngineMode,
    THETA_T, THETA_M, THETA_I,
)


def compute_observables(psi: np.ndarray) -> dict:
    """
    Returns global observables of the distributional state.

    Keys:
      psi_T, psi_M, psi_I  : marginal distributions (shape (5,))
      mean_T, mean_M, mean_I : marginal expectations
      var_T, var_M, var_I  : marginal variances
      H_global             : Shannon entropy on the joint distribution
      H_T, H_M, H_I        : marginal entropies
    """
    # Marginals (Σ over the other two axes)
    psi_T = psi.sum(axis=(1, 2))
    psi_M = psi.sum(axis=(0, 2))
    psi_I = psi.sum(axis=(0, 1))

    # Means
    mass = float(psi.sum())
    if mass < 1e-12:
        # Degenerate run — return zeros
        return {
            'psi_T': psi_T, 'psi_M': psi_M, 'psi_I': psi_I,
            'mean_T': 0.0, 'mean_M': 0.0, 'mean_I': 0.0,
            'var_T': 0.0, 'var_M': 0.0, 'var_I': 0.0,
            'H_global': 0.0, 'H_T': 0.0, 'H_M': 0.0, 'H_I': 0.0,
            'mass': mass,
        }

    mean_T = float((THETA_T * psi_T).sum() / max(psi_T.sum(), 1e-12))
    mean_M = float((THETA_M * psi_M).sum() / max(psi_M.sum(), 1e-12))
    mean_I = float((THETA_I * psi_I).sum() / max(psi_I.sum(), 1e-12))

    # Variances (in level units)
    var_T = float(((THETA_T - mean_T) ** 2 * psi_T).sum() / max(psi_T.sum(), 1e-12))
    var_M = float(((THETA_M - mean_M) ** 2 * psi_M).sum() / max(psi_M.sum(), 1e-12))
    var_I = float(((THETA_I - mean_I) ** 2 * psi_I).sum() / max(psi_I.sum(), 1e-12))

    # Entropies
    eps = 1e-12
    H_global = float(-(psi * np.log(psi + eps)).sum())
    H_T = float(-(psi_T * np.log(psi_T + eps)).sum())
    H_M = float(-(psi_M * np.log(psi_M + eps)).sum())
    H_I = float(-(psi_I * np.log(psi_I + eps)).sum())

    return {
        'psi_T': psi_T, 'psi_M': psi_M, 'psi_I': psi_I,
        'mean_T': mean_T, 'mean_M': mean_M, 'mean_I': mean_I,
        'var_T': var_T, 'var_M': var_M, 'var_I': var_I,
        'H_global': H_global, 'H_T': H_T, 'H_M': H_M, 'H_I': H_I,
        'mass': mass,
    }


def compute_D_eff(obs: dict, cfg: FactorialEngineConfig) -> float:
    """
    Effective diffusion coefficient.

    D_eff = D_0 · g_Ω(observables), with floor D_min.

    g_Ω boosts D when global entropy is low (counter informational collapse).
    Smooth function — no thresholds, no switching.
    """
    H = obs['H_global']
    if H >= cfg.H_min:
        boost = 1.0
    else:
        # Smooth boost: linearly increasing as H drops below H_min
        deficit = (cfg.H_min - H) / max(cfg.H_min, 1e-12)
        boost = 1.0 + 2.0 * deficit
    return max(cfg.D_min, cfg.D_0 * boost)


def compute_Phi_corr(
    obs: dict, cfg: FactorialEngineConfig, mode: EngineMode,
) -> np.ndarray:
    """
    Regulatory potential Φ_corr on Θ = T × M × I, shape (5, 5, 5).

    Per axis a:
      - if var_a < var_min  : repulsive bump at the current mean (smooth)
                              → drift pushes mass outward
      - if var_a > var_max  : confining quadratic toward the mean
                              → drift pushes mass inward
      - else                : zero contribution (interior of corridor)

    Returns a scalar field whose negative gradient enters the drift term.

    In ALPHA_ONLY mode, Φ_corr ≡ 0 (regulation disabled).
    In NO_REGULATION_BASELINE mode, Φ_corr ≡ 0 (regulation disabled).
    In BETA_ISOTROPIC mode, Φ_corr is computed normally (regulation kept).
    In BETA_PURE mode, Φ_corr is computed normally (regulation is the only mechanism).
    In FULL mode, Φ_corr is computed normally.
    """
    if mode in (EngineMode.ALPHA_ONLY, EngineMode.NO_REGULATION_BASELINE):
        return np.zeros((5, 5, 5))

    # Build coordinate grids
    TT, MM, II = np.meshgrid(THETA_T, THETA_M, THETA_I, indexing='ij')

    Phi = np.zeros((5, 5, 5))

    for var_a, mean_a, theta_grid in [
        (obs['var_T'], obs['mean_T'], TT),
        (obs['var_M'], obs['mean_M'], MM),
        (obs['var_I'], obs['mean_I'], II),
    ]:
        if var_a < cfg.var_min:
            # Anti-collapse: POSITIVE bump at mean → repulsive force
            # Drift is -∇·(ψ ∇Φ), so -∇Φ points AWAY from the maximum of Φ.
            # A positive Gaussian bump at the mean creates a hill — the
            # gradient points outward → -∇Φ points outward → mass repelled.
            # (The previous version used a NEGATIVE bump which created a well
            # at the mean, attracting mass and making the term pro-collapse.)
            deficit = (cfg.var_min - var_a) / max(cfg.var_min, 1e-12)
            Phi += cfg.lambda_KNV * deficit * np.exp(
                -((theta_grid - mean_a) ** 2) / 1.0
            )
        elif var_a > cfg.var_max:
            # Anti-dispersion: positive quadratic → confining force
            excess = (var_a - cfg.var_max) / max(cfg.var_max, 1e-12)
            Phi += cfg.lambda_KNV * excess * 0.5 * (theta_grid - mean_a) ** 2

    return Phi


def compute_tau_prime_modular(
    psi: np.ndarray, h_T: np.ndarray, h_I: np.ndarray,
) -> dict:
    """
    Modular factorial production τ' for an isolated module.

    Convention (cf. squelette §3):
      tau_T : factor associated with axis T
              τ'_T = Σ_θ_T (θ_T / h_T(θ_T)) · ψ_T(θ_T)
      tau_I : factor associated with axis I (private to the module)
              τ'_I = Σ_θ_I (θ_I / h_I(θ_I)) · ψ_I(θ_I)

    Note: M does not produce a direct τ' component. M is the mediator —
    its effect on τ' is indirect (via the dynamics).

    Phase 6a uses h marginal: g_T depends only on h_T(θ_T), not on θ_M, θ_I.
    """
    psi_T = psi.sum(axis=(1, 2))
    psi_I = psi.sum(axis=(0, 1))

    tau_T = float(((THETA_T / np.maximum(h_T, 1e-12)) * psi_T).sum())
    tau_I = float(((THETA_I / np.maximum(h_I, 1e-12)) * psi_I).sum())

    return {'tau_T': tau_T, 'tau_I': tau_I}
