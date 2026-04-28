"""
Phase 6b preflight orchestrator.

Runs the 7 preflight tests in order before any 6b verdict can be
interpreted:

  Preflight 0 — Phase 6a verdict canonical (no regression of 6a)
                Tolerant: not bitwise-identical, but outcomes unchanged
                and key metrics within tolerance.

  Preflight 1 — sign regression 6a (anti-collapse drift, diffusion spread)
  Preflight 2 — phi_extra=None bitwise non-regression
  Preflight 3 — coupling_forms_regression (positive ≠ contrastive at runtime)
  Preflight 4 — coupling_sign_microtest (REPULSIVE for both forms)
  Preflight 5 — attractive_sign_control annexe (warning only)
  Preflight 6 — RNG independence (objects + sequences + seeds)
  Preflight 7 — topology integrity (cyclic A-B-C)

Status policy:
  - If preflights 0,1,2,3,6,7 all PASS → status OK (verdict 6b interpretable)
  - If preflight 4 fails              → status COUPLING_SIGN_INVALID
  - If any of 0,1,2,3,6,7 fails       → status NUMERICAL_OR_IMPLEMENTATION_INVALID
  - Preflight 5 result is logged as methodological warning only
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TESTS_PHASE6A = _PROJECT_ROOT / "tests" / "phase6a"
_TESTS_PHASE6B = _PROJECT_ROOT / "tests" / "phase6b"

# Preflight 0 tolerances and canonical values
# These values reflect the closed Phase 6a verdict. If they need to be
# updated (e.g. after a deliberate calibration change to 6a), update them
# here AND document the change in the README.
_P0_TOLERANCES = {
    'mass_drift_max': 1e-4,                 # mass drift must stay ACCEPTABLE
    'F2_prime_FULL_rel_diff_min': 0.05,     # FULL must still PASS (≥ 5%)
    'F2_prime_ALPHA_rel_diff_min': 0.05,    # ALPHA_ONLY must still PASS
    'F2_prime_FULL_lt_ALPHA': True,         # β attenuation pattern must persist
    # F2 (tripartition) — should PASS in modes that have regulation+diffusion
    'F2_required_modes': ['FULL', 'ALPHA_ONLY'],
    'F2_required_outcome': 'PASS',
    'F2_growth_M_min': 0.3,                 # σ_M growth threshold (matches F2 spec)
    # F3a (tensional-visible polymorphism)
    # The canonical 6a verdict reports FAIL_NO_PRODUCTIVE_FORGETTING with
    # max_cardinality=4 because the marginal-h setup makes M directions
    # structurally invisible to τ' (documented architectural artefact).
    # We assert this canonical state to detect any divergence from it.
    'F3a_canonical_outcome_FULL': 'FAIL_NO_PRODUCTIVE_FORGETTING',
    'F3a_canonical_max_cardinality_FULL': 4,
    # F3b (internal morphodynamic transformability) — should produce
    # at least WEAK_DIVERGENCE (i.e. div_var_M not vanishing) on FULL mode.
    'F3b_required_outcomes_FULL': {'WEAK_DIVERGENCE', 'PASS'},
    'F3b_div_var_M_min_FULL': 0.005,        # at least 0.5% divergence
    # MCF intra — must still converge cleanly
    'MCF_intra_required_outcome': 'PASS',
    'MCF_intra_max_rel_diff_max': 0.05,     # 5% tolerance (canonical is ~0.0001)
}


# ============================================================================
# Preflight 0 — 6a canonical verdict
# ============================================================================

def preflight_0_6a_canonical() -> dict:
    """
    Run the 6a test_module_isolated runner, parse its JSON verdict, and
    check that key outcomes are unchanged within tolerance.
    """
    runner = _TESTS_PHASE6A / "test_module_isolated.py"
    proc = subprocess.run(
        [sys.executable, str(runner)],
        cwd=str(_PROJECT_ROOT),
        env={**__import__('os').environ, 'PYTHONPATH': str(_PROJECT_ROOT / "src")},
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        return {
            'pass': False,
            'reason': f"test_module_isolated.py exited with code {proc.returncode}",
            'stderr_tail': proc.stderr[-500:] if proc.stderr else '',
        }

    # Parse the JSON verdict
    import json
    verdict_path = _PROJECT_ROOT / "results" / "phase6a" / "verdict_phase6a.json"
    try:
        with open(verdict_path) as f:
            verdict = json.load(f)
    except Exception as e:
        return {'pass': False, 'reason': f"Could not load 6a verdict JSON: {e}"}

    issues = []

    # Check global status
    if verdict.get('global_status') != 'INTERPRETABLE':
        issues.append(f"global_status = {verdict.get('global_status')}, expected INTERPRETABLE")

    # Check mediation verdict
    mediation = verdict.get('mediation_synthesis', {}).get('verdict', '')
    if mediation != 'M_MEDIATES_VIA_ALPHA_BETA_ATTENUATES':
        issues.append(f"mediation verdict = {mediation}, expected M_MEDIATES_VIA_ALPHA_BETA_ATTENUATES")

    # Check FULL mode F2'
    full_f2p = verdict.get('modes', {}).get('FULL', {}).get('F2_prime', {})
    full_outcome = full_f2p.get('outcome', '')
    full_rel_diff = full_f2p.get('rel_diff_max', 0.0)
    if full_outcome != 'PASS':
        issues.append(f"FULL F2' outcome = {full_outcome}, expected PASS")
    if full_rel_diff < _P0_TOLERANCES['F2_prime_FULL_rel_diff_min']:
        issues.append(f"FULL F2' rel_diff = {full_rel_diff:.4f} < "
                      f"{_P0_TOLERANCES['F2_prime_FULL_rel_diff_min']}")

    # Check ALPHA_ONLY F2'
    alpha_f2p = verdict.get('modes', {}).get('ALPHA_ONLY', {}).get('F2_prime', {})
    alpha_outcome = alpha_f2p.get('outcome', '')
    alpha_rel_diff = alpha_f2p.get('rel_diff_max', 0.0)
    if alpha_outcome != 'PASS':
        issues.append(f"ALPHA_ONLY F2' outcome = {alpha_outcome}, expected PASS")
    if alpha_rel_diff < _P0_TOLERANCES['F2_prime_ALPHA_rel_diff_min']:
        issues.append(f"ALPHA_ONLY F2' rel_diff = {alpha_rel_diff:.4f} < "
                      f"{_P0_TOLERANCES['F2_prime_ALPHA_rel_diff_min']}")

    # Check FULL < ALPHA_ONLY (β attenuation pattern)
    if _P0_TOLERANCES['F2_prime_FULL_lt_ALPHA']:
        if not (full_rel_diff < alpha_rel_diff):
            issues.append(f"β-attenuation pattern broken: FULL ({full_rel_diff:.4f}) "
                          f"≥ ALPHA_ONLY ({alpha_rel_diff:.4f}). Expected FULL < ALPHA_ONLY.")

    # F2 (tripartition) — required modes must PASS
    for mode_name in _P0_TOLERANCES['F2_required_modes']:
        m = verdict.get('modes', {}).get(mode_name, {})
        f2 = m.get('F2', {})
        f2_out = f2.get('outcome', '')
        if f2_out != _P0_TOLERANCES['F2_required_outcome']:
            issues.append(f"{mode_name} F2 outcome = {f2_out}, "
                          f"expected {_P0_TOLERANCES['F2_required_outcome']}")
        growth_M = f2.get('growth_M', 0.0)
        if growth_M < _P0_TOLERANCES['F2_growth_M_min']:
            issues.append(f"{mode_name} F2 growth_M = {growth_M:.4f} < "
                          f"{_P0_TOLERANCES['F2_growth_M_min']} "
                          f"(tripartition redistribution insufficient)")

    # F3a — assert canonical (architectural artefact) values for FULL
    full_f3a = verdict.get('modes', {}).get('FULL', {}).get('F3a', {})
    f3a_outcome = full_f3a.get('outcome', '')
    f3a_max_card = full_f3a.get('max_cardinality', -1)
    if f3a_outcome != _P0_TOLERANCES['F3a_canonical_outcome_FULL']:
        issues.append(f"FULL F3a outcome = {f3a_outcome}, "
                      f"expected canonical {_P0_TOLERANCES['F3a_canonical_outcome_FULL']} "
                      f"(this is the documented marginal-h architectural artefact; "
                      f"any other value indicates an unintended change to F3a or "
                      f"the metric/coupling pathway).")
    if f3a_max_card != _P0_TOLERANCES['F3a_canonical_max_cardinality_FULL']:
        issues.append(f"FULL F3a max_cardinality = {f3a_max_card}, "
                      f"expected {_P0_TOLERANCES['F3a_canonical_max_cardinality_FULL']} "
                      f"(M directions structurally invisible in marginal-h setup).")

    # F3b — internal morphodynamic transformability must produce a measurable signal
    full_f3b = verdict.get('modes', {}).get('FULL', {}).get('F3b_M', {})
    f3b_outcome = full_f3b.get('outcome', '')
    f3b_div = full_f3b.get('div_var_M', 0.0)
    allowed_f3b = _P0_TOLERANCES['F3b_required_outcomes_FULL']
    if f3b_outcome not in allowed_f3b:
        issues.append(f"FULL F3b outcome = {f3b_outcome}, "
                      f"expected one of {allowed_f3b}")
    if f3b_div < _P0_TOLERANCES['F3b_div_var_M_min_FULL']:
        issues.append(f"FULL F3b div_var_M = {f3b_div:.4f} < "
                      f"{_P0_TOLERANCES['F3b_div_var_M_min_FULL']} "
                      f"(M not transformable internally — gap F3a vs F3b lost)")

    # MCF intra
    mcf = verdict.get('mcf_intra', {})
    mcf_status = mcf.get('status', '')
    if mcf_status != 'INTERPRETABLE':
        issues.append(f"MCF intra status = {mcf_status}, expected INTERPRETABLE")
    mcf_result = mcf.get('result', {}) or {}
    mcf_outcome = mcf_result.get('outcome', '')
    if mcf_outcome != _P0_TOLERANCES['MCF_intra_required_outcome']:
        issues.append(f"MCF intra outcome = {mcf_outcome}, "
                      f"expected {_P0_TOLERANCES['MCF_intra_required_outcome']}")
    mcf_max_rel = mcf_result.get('max_relative_diff', 1.0)
    if mcf_max_rel > _P0_TOLERANCES['MCF_intra_max_rel_diff_max']:
        issues.append(f"MCF intra max_relative_diff = {mcf_max_rel:.4f} > "
                      f"{_P0_TOLERANCES['MCF_intra_max_rel_diff_max']}")

    # Check mass drift across modes (use 'free' invariants)
    for mode in ['FULL', 'ALPHA_ONLY', 'NO_REGULATION_BASELINE']:
        m = verdict.get('modes', {}).get(mode, {})
        inv = m.get('invariants', {}).get('free', {})
        drift = inv.get('mass_drift_final', 1.0)
        if drift > _P0_TOLERANCES['mass_drift_max']:
            issues.append(f"{mode} mass_drift = {drift:.2e} > "
                          f"{_P0_TOLERANCES['mass_drift_max']}")

    return {
        'pass': len(issues) == 0,
        'issues': issues,
        'mediation_verdict': mediation,
        'FULL_F2_prime': {'outcome': full_outcome, 'rel_diff': full_rel_diff},
        'ALPHA_F2_prime': {'outcome': alpha_outcome, 'rel_diff': alpha_rel_diff},
        'FULL_F2': verdict.get('modes', {}).get('FULL', {}).get('F2', {}).get('outcome'),
        'FULL_F3a': f3a_outcome,
        'FULL_F3b': f3b_outcome,
        'MCF_outcome': mcf_outcome,
        'MCF_max_rel_diff': mcf_max_rel,
    }


# ============================================================================
# Preflights 1–7 — run as subprocesses
# ============================================================================

def _run_test_subprocess(test_path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=str(_PROJECT_ROOT),
        env={**__import__('os').environ, 'PYTHONPATH': str(_PROJECT_ROOT / "src")},
        capture_output=True, text=True, timeout=180,
    )
    return {
        'pass': (proc.returncode == 0),
        'returncode': proc.returncode,
        'stdout_tail': proc.stdout[-300:] if proc.stdout else '',
        'stderr_tail': proc.stderr[-300:] if proc.stderr else '',
    }


def preflight_1_sign_regression() -> dict:
    return _run_test_subprocess(_TESTS_PHASE6A / "test_sign_regression.py")


def preflight_2_phi_extra_none() -> dict:
    return _run_test_subprocess(_TESTS_PHASE6A / "test_phi_extra_none_regression.py")


def preflight_3_coupling_forms() -> dict:
    return _run_test_subprocess(_TESTS_PHASE6B / "test_coupling_forms_regression.py")


def preflight_4_coupling_sign() -> dict:
    return _run_test_subprocess(_TESTS_PHASE6B / "test_coupling_sign.py")


def preflight_5_attractive_control() -> dict:
    return _run_test_subprocess(_TESTS_PHASE6B / "annexe_attractive_sign_control.py")


def preflight_6_rng_independence() -> dict:
    return _run_test_subprocess(_TESTS_PHASE6B / "test_rng_independence.py")


def preflight_7_topology_integrity() -> dict:
    return _run_test_subprocess(_TESTS_PHASE6B / "test_topology_integrity.py")


# ============================================================================
# Orchestrator
# ============================================================================

def run_preflight_suite(verbose: bool = True) -> dict:
    """
    Run all 7 preflights in order and return a consolidated status.

    Returns
    -------
    dict with:
      - status: 'OK' / 'NUMERICAL_OR_IMPLEMENTATION_INVALID' /
                 'COUPLING_SIGN_INVALID'
      - results: dict of per-preflight results
      - warnings: list of methodological warnings (preflight 5 if it fails)
      - blocking_failures: list of preflight names that block interpretation
    """
    if verbose:
        print("Running Phase 6b preflight suite...")

    results = {}
    blocking_failures = []
    warnings = []

    # Preflight 0
    if verbose:
        print("  [0] 6a canonical verdict ...", end=" ", flush=True)
    r = preflight_0_6a_canonical()
    results['preflight_0_6a_canonical'] = r
    if not r['pass']:
        blocking_failures.append('preflight_0_6a_canonical')
    if verbose:
        if r['pass']:
            print("PASS")
            print(f"      mediation: {r.get('mediation_verdict', '?')}")
            print(f"      FULL F2'   : {r.get('FULL_F2_prime', {})}")
            print(f"      ALPHA F2'  : {r.get('ALPHA_F2_prime', {})}")
            print(f"      FULL F2    : {r.get('FULL_F2', '?')}")
            print(f"      FULL F3a   : {r.get('FULL_F3a', '?')}")
            print(f"      FULL F3b   : {r.get('FULL_F3b', '?')}")
            print(f"      MCF        : {r.get('MCF_outcome', '?')} "
                  f"(max_rel_diff={r.get('MCF_max_rel_diff', '?')})")
        else:
            print(f"FAIL ({len(r.get('issues', []))} issues)")
            for issue in r.get('issues', []):
                print(f"      - {issue}")

    # Preflight 1
    if verbose:
        print("  [1] sign regression 6a ...", end=" ", flush=True)
    r = preflight_1_sign_regression()
    results['preflight_1_sign_regression'] = r
    if not r['pass']:
        blocking_failures.append('preflight_1_sign_regression')
    if verbose:
        print("PASS" if r['pass'] else "FAIL")

    # Preflight 2
    if verbose:
        print("  [2] phi_extra=None bitwise ...", end=" ", flush=True)
    r = preflight_2_phi_extra_none()
    results['preflight_2_phi_extra_none'] = r
    if not r['pass']:
        blocking_failures.append('preflight_2_phi_extra_none')
    if verbose:
        print("PASS" if r['pass'] else "FAIL")

    # Preflight 3
    if verbose:
        print("  [3] coupling forms regression ...", end=" ", flush=True)
    r = preflight_3_coupling_forms()
    results['preflight_3_coupling_forms'] = r
    if not r['pass']:
        blocking_failures.append('preflight_3_coupling_forms')
    if verbose:
        print("PASS" if r['pass'] else "FAIL")

    # Preflight 4 — separate status if it fails
    if verbose:
        print("  [4] coupling sign micro-test ...", end=" ", flush=True)
    r = preflight_4_coupling_sign()
    results['preflight_4_coupling_sign'] = r
    p4_fail = not r['pass']
    if verbose:
        print("PASS" if r['pass'] else "FAIL → COUPLING_SIGN_INVALID")

    # Preflight 5 — annex, warning only
    if verbose:
        print("  [5] attractive sign control (annex) ...", end=" ", flush=True)
    r = preflight_5_attractive_control()
    results['preflight_5_attractive_control'] = r
    if not r['pass']:
        warnings.append("attractive_sign_control failed: methodological reference broken, "
                        "but does not block 6b interpretation.")
    if verbose:
        print("PASS" if r['pass'] else "FAIL (warning only)")

    # Preflight 6
    if verbose:
        print("  [6] RNG independence ...", end=" ", flush=True)
    r = preflight_6_rng_independence()
    results['preflight_6_rng_independence'] = r
    if not r['pass']:
        blocking_failures.append('preflight_6_rng_independence')
    if verbose:
        print("PASS" if r['pass'] else "FAIL")

    # Preflight 7
    if verbose:
        print("  [7] topology integrity ...", end=" ", flush=True)
    r = preflight_7_topology_integrity()
    results['preflight_7_topology_integrity'] = r
    if not r['pass']:
        blocking_failures.append('preflight_7_topology_integrity')
    if verbose:
        print("PASS" if r['pass'] else "FAIL")

    # Determine final status
    if blocking_failures:
        status = 'NUMERICAL_OR_IMPLEMENTATION_INVALID'
    elif p4_fail:
        status = 'COUPLING_SIGN_INVALID'
    else:
        status = 'OK'

    return {
        'status': status,
        'results': results,
        'warnings': warnings,
        'blocking_failures': blocking_failures,
        'coupling_sign_failed': p4_fail,
    }


if __name__ == "__main__":
    suite = run_preflight_suite(verbose=True)
    print()
    print("=" * 70)
    print(f"PREFLIGHT SUITE STATUS: {suite['status']}")
    print("=" * 70)
    if suite['blocking_failures']:
        print(f"Blocking failures: {suite['blocking_failures']}")
    if suite['warnings']:
        print(f"Warnings: {suite['warnings']}")

    sys.exit(0 if suite['status'] == 'OK' else 1)
