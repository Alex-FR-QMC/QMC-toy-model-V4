"""
Phase 6c-B — Bloc 3: Multi-seed robustness.

Re-runs each Bloc 1 test for seeds {42, 123, 2024} and reports for
each key signal:
  - median value across seeds
  - max-min dispersion
  - robustness verdict: SIGNAL_SURVIVES vs WITHIN_RNG_NOISE

Reasoning (Alex's session 7 priority):
  Early-window signals in T27' are weak (+1/+2 steps). Before
  integrating them into structural readings, we need to verify they
  survive seed variation. If dispersion > signal magnitude on any
  given test, that signal is RNG noise, not a genuine MCQ signature.

Seed escalation rule:
  If verdict varies by more than 1 regime category between seeds,
  add 2 more seeds {7, 999} for a 5-seed extended check.

Tests covered:
  - F1' cyclicity (cyclic signature stability)
  - F3 systemic (regime stability per form)
  - F5 propagation asymmetry (asymmetry differential vs baseline)
  - F6 T25' (regime presence per form)
  - F6 T27' (early-window timing per form)

Output: a single consolidated JSON summarising robustness per test
per form across the seed set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))

# Import the per-test runners
sys.path.insert(0, str(Path(__file__).parent))
from test_F1_prime_cyclicity import run_F1_prime_test
from test_F3_systemic import run_F3_systemic_test
from test_F5_propagation_asymmetry import run_F5_test
from test_F6_T25_prime_contraction import run_T25_prime_test
from test_F6_T27_prime_plurality import run_T27_test


SEEDS_BASE = [42, 123, 2024]
SEEDS_EXTENDED = [42, 123, 2024, 7, 999]


# ============================================================================
# Robustness extractors per test
# ============================================================================

def extract_F1_signals_per_seed(per_seed_results: dict) -> dict:
    """For each form, extract cyclic signature label, period, corr_AB_BC."""
    out = {}
    forms = ['contrastive', 'perspectival_INV_H',
             'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']
    for form in forms:
        out[form] = {
            'cyclic_signature_per_seed': {},
            'period_per_seed': {},
            'corr_AB_BC_per_seed': {},
        }
        for seed, full in per_seed_results.items():
            r = full['per_form'][form]
            sig_field = r.get('cyclic_signature')
            if isinstance(sig_field, dict):
                label = sig_field.get('label')
            else:
                label = sig_field
            out[form]['cyclic_signature_per_seed'][seed] = label
            out[form]['period_per_seed'][seed] = r.get('dominant_period')
            corr = r.get('corr_matrix', [[None]*3]*3)
            out[form]['corr_AB_BC_per_seed'][seed] = (
                corr[0][1] if (corr and corr[0]) else None
            )
        # Robustness summary
        labels = list(out[form]['cyclic_signature_per_seed'].values())
        periods = [p for p in out[form]['period_per_seed'].values() if p is not None]
        corrs = [c for c in out[form]['corr_AB_BC_per_seed'].values() if c is not None]
        n_distinct_labels = len(set(labels))
        out[form]['signature_stable'] = bool(n_distinct_labels == 1)
        out[form]['period_median'] = float(np.median(periods)) if periods else None
        out[form]['period_dispersion'] = (
            float(max(periods) - min(periods)) if len(periods) > 1 else 0.0
        )
        out[form]['corr_AB_BC_median'] = float(np.median(corrs)) if corrs else None
        out[form]['corr_AB_BC_dispersion'] = (
            float(max(corrs) - min(corrs)) if len(corrs) > 1 else 0.0
        )
    return out


def extract_F3_signals_per_seed(per_seed_results: dict) -> dict:
    """Per form: form_verdict per seed, count of anti-coh / KNV / decoupling cells."""
    out = {}
    forms = ['contrastive', 'perspectival_INV_H',
             'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']
    for form in forms:
        out[form] = {
            'verdict_per_seed': {},
            'n_anti_coherence_per_seed': {},
            'n_KNV_per_seed': {},
            'n_decoupling_per_seed': {},
        }
        for seed, full in per_seed_results.items():
            r = full['per_form'][form]
            out[form]['verdict_per_seed'][seed] = r.get('form_verdict')
            cms = r.get('cross_module_summary', {})
            out[form]['n_anti_coherence_per_seed'][seed] = cms.get(
                'n_anti_coherence_productive', 0)
            out[form]['n_KNV_per_seed'][seed] = cms.get('n_KNV_collapse', 0)
            out[form]['n_decoupling_per_seed'][seed] = cms.get('n_decoupling', 0)
        verdicts = list(out[form]['verdict_per_seed'].values())
        out[form]['verdict_stable'] = bool(len(set(verdicts)) == 1)
        ncs = list(out[form]['n_anti_coherence_per_seed'].values())
        out[form]['n_anti_coh_median'] = float(np.median(ncs)) if ncs else 0.0
        out[form]['n_anti_coh_dispersion'] = (
            max(ncs) - min(ncs) if len(ncs) > 1 else 0
        )
    return out


def extract_F5_signals_per_seed(per_seed_results: dict) -> dict:
    """Per form: structural asymmetry (noise off) and stochastic component."""
    out = {}
    forms = ['contrastive', 'perspectival_INV_H',
             'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']
    for form in forms:
        out[form] = {
            'asymmetry_noise_off_per_seed': {},
            'stochastic_component_per_seed': {},
        }
        for seed, full in per_seed_results.items():
            decomp = full['asymmetry_decomposition'][form]
            out[form]['asymmetry_noise_off_per_seed'][seed] = decomp['asymmetry_noise_off']
            out[form]['stochastic_component_per_seed'][seed] = decomp['stochastic_component_estimate']
        a_off = list(out[form]['asymmetry_noise_off_per_seed'].values())
        out[form]['asymmetry_noise_off_median'] = float(np.median(a_off)) if a_off else None
        out[form]['asymmetry_noise_off_dispersion'] = (
            float(max(a_off) - min(a_off)) if len(a_off) > 1 else 0.0
        )
    base = out['contrastive']['asymmetry_noise_off_per_seed']
    for form in forms:
        if form == 'contrastive':
            continue
        diffs_per_seed = {}
        for seed, val in out[form]['asymmetry_noise_off_per_seed'].items():
            diffs_per_seed[seed] = val - base[seed]
        out[form]['delta_vs_contrastive_per_seed'] = diffs_per_seed
        ds = list(diffs_per_seed.values())
        out[form]['delta_vs_contrastive_median'] = float(np.median(ds))
        out[form]['delta_vs_contrastive_dispersion'] = (
            float(max(ds) - min(ds)) if len(ds) > 1 else 0.0
        )
        out[form]['delta_sign_stable'] = bool(
            all(d > 0 for d in ds) or all(d < 0 for d in ds) or all(d == 0 for d in ds)
        )
    return out


def extract_T25_signals_per_seed(per_seed_results: dict) -> dict:
    """Per form: any_viable, any_partner_expansion, any_relative_only stability."""
    out = {}
    forms = ['contrastive', 'perspectival_INV_H',
             'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']
    for form in forms:
        out[form] = {
            'any_viable_per_seed': {},
            'any_relative_only_per_seed': {},
            'any_partner_expansion_per_seed': {},
            'all_regimes_seen_per_seed': {},
        }
        for seed, full in per_seed_results.items():
            agg = full['per_form'][form]['aggregate']
            out[form]['any_viable_per_seed'][seed] = agg['any_viable_local_or_propagated']
            out[form]['any_relative_only_per_seed'][seed] = agg['any_relative_contraction_only']
            out[form]['any_partner_expansion_per_seed'][seed] = agg['any_partner_expansion']
            out[form]['all_regimes_seen_per_seed'][seed] = agg['all_regimes_seen']
        # Stability checks
        viable = list(out[form]['any_viable_per_seed'].values())
        rel = list(out[form]['any_relative_only_per_seed'].values())
        pe = list(out[form]['any_partner_expansion_per_seed'].values())
        out[form]['any_viable_stable'] = bool(len(set(viable)) == 1)
        out[form]['any_relative_only_stable'] = bool(len(set(rel)) == 1)
        out[form]['any_partner_expansion_stable'] = bool(len(set(pe)) == 1)

        # Compute regime overlap across seeds
        all_seed_sets = [set(s) for s in out[form]['all_regimes_seen_per_seed'].values()]
        if all_seed_sets:
            common = set.intersection(*all_seed_sets)
            union_all = set.union(*all_seed_sets)
            out[form]['regimes_in_all_seeds'] = sorted(common)
            out[form]['regimes_in_some_seed'] = sorted(union_all - common)
        else:
            out[form]['regimes_in_all_seeds'] = []
            out[form]['regimes_in_some_seed'] = []
    return out


def extract_T27_signals_per_seed(per_seed_results: dict) -> dict:
    """Per form: long-run regime + early-window timings stability."""
    out = {}
    forms = ['contrastive', 'perspectival_INV_H',
             'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']
    for form in forms:
        out[form] = {
            'long_run_regimes_per_seed': {},
            'time_to_local_fusion_per_seed': {},
            'time_to_local_fusion_baseline_per_seed': {},
            'time_to_union_collapse_per_seed': {},
            'time_to_union_collapse_baseline_per_seed': {},
            'max_abs_dev_per_seed': {},
        }
        for seed, full in per_seed_results.items():
            r = full['per_form'][form]
            out[form]['long_run_regimes_per_seed'][seed] = r['regimes_detected']
            ew = r['early_window']
            out[form]['time_to_local_fusion_per_seed'][seed] = ew['time_to_local_fusion']
            out[form]['time_to_local_fusion_baseline_per_seed'][seed] = ew['time_to_local_fusion_baseline']
            out[form]['time_to_union_collapse_per_seed'][seed] = ew['time_to_union_collapse']
            out[form]['time_to_union_collapse_baseline_per_seed'][seed] = ew['time_to_union_collapse_baseline']
            out[form]['max_abs_dev_per_seed'][seed] = ew['early_window_max_abs_dev']

        # Long-run regime stability — sets per seed
        rsets = [set(s) for s in out[form]['long_run_regimes_per_seed'].values()]
        if rsets:
            common = set.intersection(*rsets)
            out[form]['regimes_in_all_seeds'] = sorted(common)

        # Early-window timing differential robustness:
        # for each module and each seed, compute meas_time - baseline_time
        # then check whether the SIGN is consistent across seeds
        timing_diffs_per_seed_per_module = {m: {} for m in ['A', 'B', 'C']}
        union_K1_diffs_per_seed = {}
        for seed in out[form]['time_to_local_fusion_per_seed']:
            tlf = out[form]['time_to_local_fusion_per_seed'][seed]
            tlf_b = out[form]['time_to_local_fusion_baseline_per_seed'][seed]
            for m in ['A', 'B', 'C']:
                if tlf.get(m) is not None and tlf_b.get(m) is not None:
                    timing_diffs_per_seed_per_module[m][seed] = tlf[m] - tlf_b[m]
            tuc1 = out[form]['time_to_union_collapse_per_seed'][seed].get(1)
            tucb1 = out[form]['time_to_union_collapse_baseline_per_seed'][seed].get(1)
            if tuc1 is not None and tucb1 is not None:
                union_K1_diffs_per_seed[seed] = tuc1 - tucb1

        out[form]['timing_diff_per_module_per_seed'] = timing_diffs_per_seed_per_module
        out[form]['union_K1_diff_per_seed'] = union_K1_diffs_per_seed

        # Robustness verdict per module
        robust = {}
        for m in ['A', 'B', 'C']:
            diffs = list(timing_diffs_per_seed_per_module[m].values())
            if len(diffs) == 0:
                robust[m] = {'verdict': 'NO_DATA'}
                continue
            n_pos = sum(1 for d in diffs if d > 0)
            n_neg = sum(1 for d in diffs if d < 0)
            n_zero = sum(1 for d in diffs if d == 0)
            median_diff = float(np.median(diffs))
            max_abs = float(max(abs(d) for d in diffs))
            min_abs = float(min(abs(d) for d in diffs))
            dispersion = max_abs - min_abs
            # Signal survives if all-non-zero same sign
            signal_present_all_seeds = (n_zero == 0)
            sign_stable = (n_pos == len(diffs) or n_neg == len(diffs))
            if signal_present_all_seeds and sign_stable and abs(median_diff) >= 1:
                verdict = 'SIGNAL_SURVIVES'
            elif sign_stable and median_diff != 0:
                verdict = 'SIGNAL_PRESENT_BUT_INCONSISTENT_PER_SEED'
            elif n_zero == len(diffs):
                verdict = 'NO_SIGNAL_ALL_SEEDS'
            else:
                verdict = 'WITHIN_RNG_NOISE'
            robust[m] = {
                'verdict': verdict,
                'diffs': diffs,
                'median': median_diff,
                'dispersion_max_minus_min': dispersion,
                'n_pos': n_pos, 'n_neg': n_neg, 'n_zero': n_zero,
            }
        out[form]['robustness_per_module'] = robust

        # Same for union K=1
        u_diffs = list(union_K1_diffs_per_seed.values())
        if u_diffs:
            n_pos = sum(1 for d in u_diffs if d > 0)
            n_neg = sum(1 for d in u_diffs if d < 0)
            n_zero = sum(1 for d in u_diffs if d == 0)
            median_diff = float(np.median(u_diffs))
            sign_stable = (n_pos == len(u_diffs) or n_neg == len(u_diffs))
            if (n_zero == 0) and sign_stable and abs(median_diff) >= 1:
                u_verdict = 'SIGNAL_SURVIVES'
            elif n_zero == len(u_diffs):
                u_verdict = 'NO_SIGNAL_ALL_SEEDS'
            else:
                u_verdict = 'WITHIN_RNG_NOISE'
            out[form]['robustness_union_K1'] = {
                'verdict': u_verdict, 'diffs': u_diffs,
                'median': median_diff, 'n_pos': n_pos,
                'n_neg': n_neg, 'n_zero': n_zero,
            }
        else:
            out[form]['robustness_union_K1'] = {'verdict': 'NO_DATA'}
    return out


# ============================================================================
# Orchestrator
# ============================================================================

def run_bloc3(seeds: list = None) -> dict:
    if seeds is None:
        seeds = SEEDS_BASE
    print(f"\n{'#' * 70}")
    print(f"# Bloc 3 — Multi-seed robustness")
    print(f"# Seeds: {seeds}")
    print(f"{'#' * 70}\n")

    f1_per_seed, f3_per_seed, f5_per_seed, t25_per_seed, t27_per_seed = {}, {}, {}, {}, {}

    for seed in seeds:
        print(f"\n[seed={seed}] F1' cyclicity")
        f1_per_seed[seed] = run_F1_prime_test(base_seed=seed)
        print(f"\n[seed={seed}] F3 systemic")
        f3_per_seed[seed] = run_F3_systemic_test(base_seed=seed)
        print(f"\n[seed={seed}] F5 asymmetry")
        f5_per_seed[seed] = run_F5_test(base_seed=seed)
        print(f"\n[seed={seed}] T25' contraction")
        t25_per_seed[seed] = run_T25_prime_test(base_seed=seed)
        print(f"\n[seed={seed}] T27' plurality")
        t27_per_seed[seed] = run_T27_test(base_seed=seed)

    # Extract robustness per test
    print(f"\n{'═' * 70}")
    print("  ROBUSTNESS EXTRACTION")
    print(f"{'═' * 70}\n")
    f1_rob = extract_F1_signals_per_seed(f1_per_seed)
    f3_rob = extract_F3_signals_per_seed(f3_per_seed)
    f5_rob = extract_F5_signals_per_seed(f5_per_seed)
    t25_rob = extract_T25_signals_per_seed(t25_per_seed)
    t27_rob = extract_T27_signals_per_seed(t27_per_seed)

    return {
        'seeds': seeds,
        'F1_robustness': f1_rob,
        'F3_robustness': f3_rob,
        'F5_robustness': f5_rob,
        'T25_robustness': t25_rob,
        'T27_robustness': t27_rob,
        # Raw per-seed for inspection (large; consider trimming if needed)
        # 'raw_per_seed': {
        #     'F1': f1_per_seed, 'F3': f3_per_seed, 'F5': f5_per_seed,
        #     'T25': t25_per_seed, 'T27': t27_per_seed,
        # },
    }


def print_robustness_summary(result: dict):
    forms = ['contrastive', 'perspectival_INV_H',
             'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']
    print(f"\n{'═' * 70}")
    print("  ROBUSTNESS SUMMARY ACROSS SEEDS")
    print(f"{'═' * 70}\n")

    print("F1' cyclicity:")
    for form in forms:
        r = result['F1_robustness'][form]
        print(f"  {form:<32s}: stable_label={r['signature_stable']}  "
              f"period_med={r['period_median']}  "
              f"period_disp={r['period_dispersion']}  "
              f"corr_AB_BC_med={r['corr_AB_BC_median']:.3f}  "
              f"corr_disp={r['corr_AB_BC_dispersion']:.3f}")

    print("\nF3 systemic:")
    for form in forms:
        r = result['F3_robustness'][form]
        verdicts = list(r['verdict_per_seed'].values())
        print(f"  {form:<32s}: stable={r['verdict_stable']}  "
              f"verdicts={verdicts}  "
              f"n_anti_coh_med={r['n_anti_coh_median']:.0f}")

    print("\nF5 asymmetry (Δ vs contrastive baseline, noise-off):")
    for form in forms:
        r = result['F5_robustness'][form]
        if form == 'contrastive':
            print(f"  {form:<32s}: baseline asym_med={r['asymmetry_noise_off_median']:.4f}")
        else:
            print(f"  {form:<32s}: Δ_med={r['delta_vs_contrastive_median']:+.4f}  "
                  f"Δ_disp={r['delta_vs_contrastive_dispersion']:.4f}  "
                  f"sign_stable={r['delta_sign_stable']}")

    print("\nT25' contraction (regimes-set stability):")
    for form in forms:
        r = result['T25_robustness'][form]
        print(f"  {form:<32s}: viable_stable={r['any_viable_stable']}  "
              f"rel_only_stable={r['any_relative_only_stable']}  "
              f"partner_exp_stable={r['any_partner_expansion_stable']}")
        print(f"    regimes in ALL seeds: {r['regimes_in_all_seeds']}")
        if r['regimes_in_some_seed']:
            print(f"    regimes in SOME seed only: {r['regimes_in_some_seed']}")

    print("\nT27' early-window robustness (timing diff vs baseline):")
    for form in forms:
        r = result['T27_robustness'][form]
        rob = r['robustness_per_module']
        print(f"  {form:<32s}:")
        for m in ['A', 'B', 'C']:
            v = rob[m]
            if v.get('verdict') == 'NO_DATA':
                continue
            print(f"    module {m}: verdict={v['verdict']}  "
                  f"diffs={v['diffs']}  "
                  f"median={v['median']:+.1f}  "
                  f"disp={v['dispersion_max_minus_min']:.0f}")
        u = r['robustness_union_K1']
        if u.get('verdict') != 'NO_DATA':
            print(f"    union_K1: verdict={u['verdict']}  "
                  f"diffs={u['diffs']}  median={u['median']:+.1f}")


if __name__ == "__main__":
    result = run_bloc3(SEEDS_BASE)
    out = Path("/home/claude/mcq_v4/results/phase6c_b/bloc3_multi_seed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print_robustness_summary(result)
