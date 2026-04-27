"""
Phase 6b — Three-module system on factorial domains.

Three modules M_A, M_B, M_C, each with its own FactorialState on its own
Θ_m (option β: homologous-but-not-identical axes; shared factors are
labels of aggregation, not common dimensions).

================================================================
APPROXIMATION_NOTES_6B (must appear in every Phase 6b verdict)
================================================================

The Phase 6b coupling is NON-PERSPECTIVAL by design:

  - R_psi compares ψ_A and ψ_B via homologous label coordinates,
    NOT through metric-aware projection in each other's metric.
  - The coupling 𝒞_{i,j} ∝ ε · R_ij(1-R_ij) · ψ_j uses ψ_j as read
    in the (homologous) coordinates of i, not as evaluated in j's
    own metric.
  - The native perspective-dependent coupling of Ch.3 §3.1.3.III
    (Σ R_ij · (novelty of j in j's metric − R_ij · form of i in i's
    metric)) is DEFERRED to Phase 6c.

Phase 6b therefore characterises the factorial perturbative coupling
as a baseline. It is NOT a validation of native MCQ^N. The phase_status
field of every verdict reflects this.

================================================================
Seeds convention
================================================================

Each module has its OWN independent RNG stream to avoid phantom
synchronisation via shared noise:

    seed_A = base_seed + 101
    seed_B = base_seed + 202
    seed_C = base_seed + 303

Comparative runs (ablation, ε sweep) keep the same offsets but may
vary base_seed for multi-seed robustness studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mcq_v4.factorial.state import (
    FactorialState, FactorialEngineConfig, ModuleConfig, EngineMode,
    StepDiagnostic, THETA_T, THETA_M, THETA_I,
)
from mcq_v4.factorial.engine import FactorialEngine, make_initial_state


# ============================================================================
# Coupling configuration
# ============================================================================

@dataclass
class CouplingConfig:
    """
    Phase 6b coupling configuration. The coupling is intentionally
    non-perspectival; see APPROXIMATION_NOTES_6B in the module docstring.
    """
    epsilon: float = 0.005
    """Coupling strength. To be characterised by ε sweep, not pre-chosen."""

    # Sharing topology (cyclic) — each shared factor is observed by exactly 2 modules
    sharing_topology: dict = field(default_factory=lambda: {
        'k_AB': ('A', 'B'),
        'k_BC': ('B', 'C'),
        'k_CA': ('C', 'A'),
    })
    """Shared factors. Each key maps to (donor_1, donor_2)."""

    # Private factors — each observed by exactly 1 module (axis I)
    private_factors: dict = field(default_factory=lambda: {
        'k_A_private': 'A',
        'k_B_private': 'B',
        'k_C_private': 'C',
    })
    """Private factors per module."""

    phase_status: str = (
        "Phase 6b — factorial perturbative coupling baseline. "
        "NON-PERSPECTIVAL. Not MCQ^N native validation."
    )

    def neighbours_of(self, module_name: str) -> list[str]:
        """Return modules that share at least one factor with the given module."""
        partners = set()
        for factor, (m1, m2) in self.sharing_topology.items():
            if m1 == module_name and m2 != module_name:
                partners.add(m2)
            elif m2 == module_name and m1 != module_name:
                partners.add(m1)
        return sorted(partners)


# ============================================================================
# Three-module system
# ============================================================================

@dataclass
class ThreeModuleSystem:
    """
    Container for three coupled (or uncoupled) modules.

    Each module has its own FactorialState, its own FactorialEngine
    (with its own RNG), and the system shares a single
    FactorialEngineConfig + CouplingConfig.

    The three engines are initialised once and reused across steps;
    they hold the per-module RNG state.
    """
    state_A: FactorialState
    state_B: FactorialState
    state_C: FactorialState

    cfg_engine: FactorialEngineConfig
    coupling_cfg: CouplingConfig

    # Engines (mutable — hold RNG state)
    engine_A: FactorialEngine = None
    engine_B: FactorialEngine = None
    engine_C: FactorialEngine = None

    # Seed bookkeeping (logged)
    base_seed: int = 42
    seed_A: int = field(init=False)
    seed_B: int = field(init=False)
    seed_C: int = field(init=False)

    def __post_init__(self):
        self.seed_A = self.base_seed + 101
        self.seed_B = self.base_seed + 202
        self.seed_C = self.base_seed + 303

    @property
    def states(self) -> dict[str, FactorialState]:
        return {'A': self.state_A, 'B': self.state_B, 'C': self.state_C}

    @property
    def engines(self) -> dict[str, FactorialEngine]:
        return {'A': self.engine_A, 'B': self.engine_B, 'C': self.engine_C}

    def replace_states(self, new_states: dict[str, FactorialState]) -> 'ThreeModuleSystem':
        """Return a new system with updated states (engines reused)."""
        return ThreeModuleSystem(
            state_A=new_states['A'],
            state_B=new_states['B'],
            state_C=new_states['C'],
            cfg_engine=self.cfg_engine,
            coupling_cfg=self.coupling_cfg,
            engine_A=self.engine_A,
            engine_B=self.engine_B,
            engine_C=self.engine_C,
            base_seed=self.base_seed,
        )

    def verify_independent_rngs(self) -> bool:
        """
        Sanity check: the three engines must have DISTINCT RNG objects
        (not shared by reference). Protects against silent regressions.
        """
        rng_ids = [id(self.engine_A.rng), id(self.engine_B.rng), id(self.engine_C.rng)]
        return len(set(rng_ids)) == 3

    def seed_log(self) -> dict:
        """Return a dict of seed information for verdict logging."""
        return {
            'base_seed': self.base_seed,
            'seed_A': self.seed_A,
            'seed_B': self.seed_B,
            'seed_C': self.seed_C,
            'derivation': 'base_seed + (101, 202, 303)',
            'rationale': (
                "Independent RNG streams per module to avoid phantom synchronisation "
                "via shared noise. Same offsets across comparative conditions to "
                "isolate the effect under study from stochastic variability."
            ),
            'rngs_independent': self.verify_independent_rngs(),
        }


# ============================================================================
# Module configurations
# ============================================================================

DIFFERENTIATED_WEIGHTS = {
    'A': (1.5, 0.8, 0.7),    # tensional dominance
    'B': (0.7, 1.6, 0.7),    # morphodynamic dominance
    'C': (0.8, 0.7, 1.5),    # interface dominance
}

IDENTICAL_WEIGHTS = {
    'A': (1.0, 1.0, 1.0),
    'B': (1.0, 1.0, 1.0),
    'C': (1.0, 1.0, 1.0),
}


def build_module_configs(
    weights_dict: dict[str, tuple[float, float, float]],
    base_seed: int = 42,
) -> dict[str, ModuleConfig]:
    """Build per-module configurations with independent seeds."""
    seed_offsets = {'A': 101, 'B': 202, 'C': 303}
    return {
        name: ModuleConfig(
            name=name,
            weights=weights_dict[name],
            seed=base_seed + seed_offsets[name],
        )
        for name in ['A', 'B', 'C']
    }


def build_three_module_system(
    cfg_engine: FactorialEngineConfig,
    coupling_cfg: CouplingConfig,
    weights_dict: dict[str, tuple[float, float, float]],
    initial_psi_dict: dict[str, np.ndarray],
    base_seed: int = 42,
) -> ThreeModuleSystem:
    """
    Construct a fresh ThreeModuleSystem with independent RNGs and
    per-module initial conditions.
    """
    mod_cfgs = build_module_configs(weights_dict, base_seed)

    states = {
        name: make_initial_state(initial_psi_dict[name], mod_cfgs[name])
        for name in ['A', 'B', 'C']
    }

    engines = {
        name: FactorialEngine(cfg_engine, mod_cfgs[name])
        for name in ['A', 'B', 'C']
    }

    sys = ThreeModuleSystem(
        state_A=states['A'], state_B=states['B'], state_C=states['C'],
        cfg_engine=cfg_engine,
        coupling_cfg=coupling_cfg,
        engine_A=engines['A'], engine_B=engines['B'], engine_C=engines['C'],
        base_seed=base_seed,
    )

    assert sys.verify_independent_rngs(), (
        "Failed sanity check: the three engines do not have distinct RNG objects. "
        "Phase 6b must NOT proceed — phantom synchronisation is possible."
    )

    return sys
