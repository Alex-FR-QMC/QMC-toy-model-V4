"""
State containers and configurations for the factorial engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# Coordinates of the three factorial axes
# T (tensional): centred at 0, allows positive/negative directions
# M (morphodynamic): non-negative, depth of sedimentation
# I (interface): centred at 0, predictive-evaluative orientation
THETA_T = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
THETA_M = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
THETA_I = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# Index of the "centre" cell on each axis (used in initial conditions)
T_CENTRE_IDX = 2
M_CENTRE_IDX = 2
I_CENTRE_IDX = 2


class EngineMode(Enum):
    """
    Decomposition mode for ablation studies.

    FULL        : α + β. Full diffusion in geometry + Φ_corr regulation.
                  This is the canonical Phase 6a engine.
    ALPHA_ONLY  : α only. 3D geometric diffusion + drift, but Φ_corr ≡ 0.
                  Tests whether tripartition can emerge from simple
                  factorial diffusion alone.
    BETA_ONLY   : β only. Φ_corr active, but diffusion is isotropic
                  (D_eff acts uniformly, not modulated by h on each axis).
                  Tests whether the regulatory loop alone produces
                  tripartition without the geometric diffusion structure.
    """
    FULL = "FULL"
    ALPHA_ONLY = "ALPHA_ONLY"
    BETA_ONLY = "BETA_ONLY"


@dataclass
class ModuleConfig:
    """
    Per-module configuration.

    Parameters
    ----------
    name : str
        Module identifier ("A", "B", "C", ...).
    weights : tuple
        (w_T, w_M, w_I) modulating sedimentation/erosion rates per axis.
        β_a = β_0 · w_a (faster sedimentation on dominant axis).
        γ_a = γ_0 / w_a (slower erosion on dominant axis).
        Phase 6a default for module A: (1.5, 0.8, 0.7) — moderate
        tensional dominance.
    seed : int
        RNG seed for this module's stochastic flux.
    levels_T, levels_M, levels_I : int
        Number of discrete levels per axis. Default 5 (matches THETA_*).
    """
    name: str = "A"
    weights: tuple = (1.5, 0.8, 0.7)  # default: tensional dominance for A
    seed: int = 42
    levels_T: int = 5
    levels_M: int = 5
    levels_I: int = 5

    def __post_init__(self):
        assert self.levels_T == 5, "Phase 6a: 5 levels per axis fixed"
        assert self.levels_M == 5, "Phase 6a: 5 levels per axis fixed"
        assert self.levels_I == 5, "Phase 6a: 5 levels per axis fixed"
        w_T, w_M, w_I = self.weights
        assert all(w > 0.4 for w in self.weights), \
            f"All weights must be > 0.4 (no functional privation); got {self.weights}"


@dataclass
class FactorialEngineConfig:
    """
    Engine-level numerical configuration.

    Time integration:
      dt        : explicit Euler time step
      T_steps   : total number of steps

    Diffusion (α component):
      D_0       : base diffusion coefficient
      D_min     : non-closure floor (D_eff never drops below)

    Sedimentation / erosion:
      beta_0    : base sedimentation rate (modulated by w_a per module)
      gamma_0   : base erosion rate (modulated by 1/w_a)
      h_0       : baseline (undeformed) metric value
      h_min     : regularisation floor for h

    Noise:
      sigma_eta : amplitude of multiplicative noise

    Regulation (β component):
      var_min   : corridor lower bound on per-axis variance
      var_max   : corridor upper bound
      H_min     : entropic floor (informational collapse)
      lambda_KNV: amplitude of Φ_corr proximity functionals

    Mode:
      mode      : EngineMode.FULL | ALPHA_ONLY | BETA_ONLY
    """
    dt: float = 0.05
    T_steps: int = 200
    D_0: float = 0.02
    D_min: float = 0.002
    beta_0: float = 0.4
    gamma_0: float = 0.08
    h_0: float = 1.0
    h_min: float = 0.1
    sigma_eta: float = 0.10  # smaller than V3 (0.15) — finer factorial domain
    var_min: float = 0.5
    var_max: float = 2.5
    H_min: float = 0.5
    lambda_KNV: float = 0.3
    mode: EngineMode = EngineMode.FULL


@dataclass
class FactorialState:
    """
    Complete state of one module: distribution + marginal metric per axis.

    Attributes
    ----------
    psi : np.ndarray, shape (5, 5, 5)
        Distribution over Θ = T × M × I. Indices [θ_T, θ_M, θ_I].
        Conserved with Σ ψ ≈ 1 (mass drift logged separately).
    h_T : np.ndarray, shape (5,)
        Marginal metric on T-axis. Bounded in [h_min, h_0].
    h_M, h_I : same.
    cfg : ModuleConfig
        Module configuration (weights, name).
    """
    psi: np.ndarray
    h_T: np.ndarray
    h_M: np.ndarray
    h_I: np.ndarray
    cfg: ModuleConfig

    def copy(self) -> "FactorialState":
        return FactorialState(
            psi=self.psi.copy(),
            h_T=self.h_T.copy(),
            h_M=self.h_M.copy(),
            h_I=self.h_I.copy(),
            cfg=self.cfg,
        )

    def total_mass(self) -> float:
        return float(self.psi.sum())


@dataclass
class StepDiagnostic:
    """Per-step numerical diagnostics."""
    n_clip_events: int
    max_clip_magnitude: float
    clipped_mass: float
    mass_after_step: float
    mass_drift_step: float

    def empty(self) -> bool:
        return self.n_clip_events == 0


def initial_h() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initial undeformed metric: h = h_0 = 1.0 on all axes."""
    return np.ones(5), np.ones(5), np.ones(5)
