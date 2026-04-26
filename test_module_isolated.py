"""
Phase 6a test harness — module M_A isolated.

Runs:
  - Three engine modes (FULL, ALPHA_ONLY, BETA_ONLY)
  - For each mode: free run + frozen-h_M run for F2'
  - F2 (tripartition) on the FULL mode free run
  - F3 (polymorphism) on the FULL mode free run
  - MCF intra: two FULL mode runs from very different initial conditions

Cascade invalidation: if any run is NUMERICAL_INVALID, the F2/F2'/F3
results from that run are not interpreted.

Usage:
  python tests/phase6a/test_module_isolated.py
  → writes results to /home/claude/mcq_v4/results/phase6a/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make src importable
_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

from mcq_v4.factorial.state import (
    ModuleConfig, FactorialEngineConfig, FactorialState, EngineMode,
    THETA_T, THETA_M, THETA_I,
)
from mcq_v4.factorial.engine import FactorialEngine, make_initial_state
from mcq_v4.factorial.metrics import (
    metric_invariants, metric_F2, metric_F2_prime, metric_F3_temporal,
    metric_MCF_intra, synthesize_mediation,
)


# ============================================================================
# Initial conditions
# ============================================================================

def init_psi_concentrated_T() -> np.ndarray:
    """
    F2 init: ψ Gaussian on T axis (centred at θ_T = 0, σ = 1),
    point-mass at M = 2 (centre) and I = 2 (centre).
    """
    p_T = np.exp(-(THETA_T ** 2) / 2.0)
    p_T /= p_T.sum()

    psi = np.zeros((5, 5, 5))
    psi[:, 2, 2] = p_T  # all T levels at M=2, I=2
    psi /= psi.sum()
    return psi


def init_psi_uniform() -> np.ndarray:
    """MCF alternative: uniform distribution on Θ."""
    psi = np.ones((5, 5, 5))
    psi /= psi.sum()
    return psi


def init_psi_concentrated_corner() -> np.ndarray:
    """MCF alternative: concentrated at one corner of Θ."""
    psi = np.zeros((5, 5, 5))
    psi[0, 0, 0] = 1.0
    return psi


# ============================================================================
# Single-run helpers
# ============================================================================

def run_simulation(
    engine_cfg: FactorialEngineConfig,
    module_cfg: ModuleConfig,
    psi_init: np.ndarray,
    freeze_h_M: bool = False,
) -> tuple[list[FactorialState], list]:
    """
    Run a single simulation for engine_cfg.T_steps steps.

    Returns
    -------
    history : list of FactorialState (length T_steps + 1)
    diagnostics : list of StepDiagnostic (length T_steps)
    """
    engine = FactorialEngine(engine_cfg, module_cfg)
    state = make_initial_state(psi_init, module_cfg)

    history = [state]
    diagnostics = []
    for t in range(engine_cfg.T_steps):
        state, diag = engine.step(state, freeze_h_M=freeze_h_M)
        history.append(state)
        diagnostics.append(diag)

    return history, diagnostics


# ============================================================================
# Phase 6a runner
# ============================================================================

def run_phase_6a() -> dict:
    """
    Run the complete Phase 6a test suite.
    """
    base_engine_cfg_kwargs = dict(
        dt=0.05, T_steps=200,
        D_0=0.02, D_min=0.002,
        beta_0=0.4, gamma_0=0.08,
        h_0=1.0, h_min=0.1,
        sigma_eta=0.10,
        var_min=0.5, var_max=2.5,
        H_min=0.5, lambda_KNV=0.3,
    )
    module_cfg = ModuleConfig(name="A", weights=(1.5, 0.8, 0.7), seed=42)

    psi_init = init_psi_concentrated_T()

    results_per_mode: dict[str, dict] = {}

    # ----- Loop over engine modes -----
    for mode in [EngineMode.FULL, EngineMode.ALPHA_ONLY, EngineMode.BETA_ONLY]:
        cfg_mode = FactorialEngineConfig(**base_engine_cfg_kwargs, mode=mode)

        # Free run (h_M evolves freely)
        history_free, diag_free = run_simulation(cfg_mode, module_cfg, psi_init, freeze_h_M=False)
        inv_free = metric_invariants(history_free, diag_free, cfg_mode)

        # Frozen run (h_M held at h_M(0))
        history_frozen, diag_frozen = run_simulation(cfg_mode, module_cfg, psi_init, freeze_h_M=True)
        inv_frozen = metric_invariants(history_frozen, diag_frozen, cfg_mode)

        if not (inv_free['run_interpretable'] and inv_frozen['run_interpretable']):
            results_per_mode[mode.value] = {
                'status': 'NUMERICAL_INVALID',
                'invariants': {'free': inv_free, 'frozen': inv_frozen},
                'F2': None, 'F2_prime': None, 'F3': None,
                'recommendation': (
                    f"Mass drift exceeded {inv_free['thresholds']['warning']:.0e} "
                    f"in mode {mode.value}. Reduce dt (current {cfg_mode.dt}) by "
                    f"factor 2 or sigma_eta (current {cfg_mode.sigma_eta}) by factor 2 "
                    f"and relaunch."
                ),
            }
            continue

        # All metrics for this mode
        f2 = metric_F2(history_free)
        f2p = metric_F2_prime(history_free, history_frozen, cfg_mode)
        f3 = metric_F3_temporal(history_free, sample_every=10)

        results_per_mode[mode.value] = {
            'status': 'INTERPRETABLE',
            'invariants': {'free': inv_free, 'frozen': inv_frozen},
            'F2': f2,
            'F2_prime': f2p,
            'F3': f3,
        }

    # ----- MCF intra (only FULL mode) -----
    cfg_full = FactorialEngineConfig(**base_engine_cfg_kwargs, mode=EngineMode.FULL)
    psi_alt = init_psi_uniform()

    history_mcf_1, diag_mcf_1 = run_simulation(cfg_full, module_cfg, psi_init)
    history_mcf_2, diag_mcf_2 = run_simulation(
        cfg_full,
        ModuleConfig(name="A", weights=(1.5, 0.8, 0.7), seed=43),  # different seed for MCF
        psi_alt,
    )

    inv_mcf_1 = metric_invariants(history_mcf_1, diag_mcf_1, cfg_full)
    inv_mcf_2 = metric_invariants(history_mcf_2, diag_mcf_2, cfg_full)

    if inv_mcf_1['run_interpretable'] and inv_mcf_2['run_interpretable']:
        mcf_result = metric_MCF_intra(history_mcf_1, history_mcf_2)
        mcf_status = 'INTERPRETABLE'
    else:
        mcf_result = None
        mcf_status = 'NUMERICAL_INVALID'

    # ----- Cross-mode mediation synthesis -----
    mediation = synthesize_mediation(results_per_mode)

    # ----- Global verdict -----
    invalid_modes = [m for m, r in results_per_mode.items() if r['status'] == 'NUMERICAL_INVALID']
    if invalid_modes:
        global_status = 'PARTIAL_INVALID'
    elif mcf_status != 'INTERPRETABLE':
        global_status = 'PARTIAL_INVALID_MCF'
    else:
        global_status = 'INTERPRETABLE'

    return {
        'phase': '6a',
        'description': 'Single isolated module M_A on factorial domain Θ = T × M × I',
        'global_status': global_status,
        'engine_config': {**base_engine_cfg_kwargs},
        'module_config': {
            'name': module_cfg.name,
            'weights': list(module_cfg.weights),
            'seed': module_cfg.seed,
        },
        'modes': results_per_mode,
        'mediation_synthesis': mediation,
        'mcf_intra': {
            'status': mcf_status,
            'result': mcf_result,
            'invariants_run1': inv_mcf_1,
            'invariants_run2': inv_mcf_2,
        },
        'invalid_modes': invalid_modes,
    }


# ============================================================================
# Pretty-printing
# ============================================================================

def _print_section(title: str, indent: int = 0):
    pad = '  ' * indent
    print(f"\n{pad}{'=' * (60 - len(pad))}")
    print(f"{pad}{title}")
    print(f"{pad}{'=' * (60 - len(pad))}")


def print_phase_6a_report(verdict: dict):
    print()
    print("#" * 70)
    print("# PHASE 6a — Module M_A isolated, factorial domain Θ = T × M × I")
    print("#" * 70)

    print(f"\nGlobal status: {verdict['global_status']}")
    print(f"Module: {verdict['module_config']['name']}, "
          f"weights = {verdict['module_config']['weights']}")
    print(f"Engine: dt = {verdict['engine_config']['dt']}, "
          f"T_steps = {verdict['engine_config']['T_steps']}")

    # Per mode
    for mode_name, r in verdict['modes'].items():
        _print_section(f"Mode: {mode_name}")
        print(f"  Status: {r['status']}")

        if r['status'] != 'INTERPRETABLE':
            print(f"  Recommendation: {r.get('recommendation', '<none>')}")
            inv = r['invariants']['free']
            print(f"  Mass drift (free run): {inv['mass_drift_final']:.2e} "
                  f"({inv['mass_drift_classification']})")
            inv2 = r['invariants']['frozen']
            print(f"  Mass drift (frozen run): {inv2['mass_drift_final']:.2e} "
                  f"({inv2['mass_drift_classification']})")
            continue

        # Invariants
        inv_free = r['invariants']['free']
        inv_frozen = r['invariants']['frozen']
        print(f"  Mass drift (free):   {inv_free['mass_drift_final']:.2e} "
              f"[{inv_free['mass_drift_classification']}], "
              f"{inv_free['total_clip_events']} clips")
        print(f"  Mass drift (frozen): {inv_frozen['mass_drift_final']:.2e} "
              f"[{inv_frozen['mass_drift_classification']}], "
              f"{inv_frozen['total_clip_events']} clips")

        # F2
        f2 = r['F2']
        print(f"  F2 (tripartition): {f2['outcome']} ({f2['signal_quality']})")
        print(f"    var_T: {f2['var_T_init']:.3f} → {f2['var_T_final']:.3f}")
        print(f"    var_M: {f2['var_M_init']:.3f} → {f2['var_M_final']:.3f} "
              f"(growth = {f2['growth_M']:.3f})")
        print(f"    var_I: {f2['var_I_init']:.3f} → {f2['var_I_final']:.3f} "
              f"(growth = {f2['growth_I']:.3f})")

        # F2'
        f2p = r['F2_prime']
        print(f"  F2' (h_M frozen): {f2p['outcome']} ({f2p['signal_quality']})")
        print(f"    rel_diff: max = {f2p['rel_diff_max']:.4f} "
              f"(T = {f2p['rel_diff_T']:.4f}, I = {f2p['rel_diff_I']:.4f})")

        # F3
        f3 = r['F3']
        print(f"  F3 (polymorphism): {f3['outcome']} ({f3['signal_quality']})")
        print(f"    cardinality: min = {f3['min_cardinality']}, "
              f"max = {f3['max_cardinality']}")
        print(f"    identity changes: {f3['identity_changes']}")

    # Mediation synthesis
    _print_section("Cross-mode mediation synthesis")
    med = verdict['mediation_synthesis']
    print(f"  Verdict: {med['verdict']}")
    print(f"  Per mode: {med['outcomes_per_mode']}")
    if 'interpretation' in med:
        print(f"  Interpretation: {med['interpretation']}")
    if 'caveat' in med:
        print(f"  Caveat: {med['caveat']}")

    # MCF
    _print_section("MCF intra-modular")
    mcf = verdict['mcf_intra']
    print(f"  Status: {mcf['status']}")
    if mcf['result'] is not None:
        r = mcf['result']
        print(f"  Outcome: {r['outcome']}")
        print(f"  Max relative diff: {r['max_relative_diff']:.4f} "
              f"(threshold = {r['threshold']:.4f})")
        print(f"  Per-observable: {r['relative_diffs']}")


# ============================================================================
# Save / serialize verdict to JSON
# ============================================================================

def _make_json_safe(obj):
    """Recursively convert numpy types and tuples to JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_verdict(verdict: dict, output_path: Path):
    """Save verdict as JSON, with numpy types converted to Python."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe = _make_json_safe(verdict)
    with open(output_path, "w") as f:
        json.dump(safe, f, indent=2)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Running Phase 6a — single isolated module M_A...")
    verdict = run_phase_6a()

    out_dir = Path("/home/claude/mcq_v4/results/phase6a")
    save_verdict(verdict, out_dir / "verdict_phase6a.json")
    print(f"\nVerdict saved to: {out_dir / 'verdict_phase6a.json'}")

    print_phase_6a_report(verdict)
