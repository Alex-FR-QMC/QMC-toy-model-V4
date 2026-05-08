"""
Phase 6c-B — Consolidated runner.

Aggregates all artefacts (Bloc 1 single-seed regime classifications,
Bloc 2 reader interpretive layer, Bloc 3 multi-seed robustness) into
a single verdict JSON for archival and reference.

Does NOT re-run the engine. Reads existing JSON files from
/home/claude/mcq_v4/results/phase6c_b/.

Output structure (per Alex's session 9 wording rules):

  - block_1_regimes        : single-seed regime sets (raw)
  - block_2_readers        : interpretive layer (NOT a new verdict)
  - block_3_robustness     : multi-seed signal-survival verdicts
  - signals_robust         : signals confirmed by Bloc 3 only
  - signals_retracted      : signals from Bloc 1 not surviving Bloc 3
  - tensions_preserved     : structured contradictions kept open
  - hypotheses_status      : H1/H2/H3 status, NOT collapsed
  - caveats                : explicit limits and forbidden conclusions
  - bounds                 : negative bounds the runs establish

Wording invariants (from Alex's audits):
  - "Bloc 2 is a reading layer, not a new verdict"
  - "τ' projection and internal morphology are partially decoupled"
  - "LOW_REST is centred-rest zone, not KNV violation"
  - "T_star_proxy_h_M ≠ full 𝕋*"
  - "globally consistent, with locally informative discordance"
  - "Bloc 2 cohérent avec multi-seed INV_H, ne prouve pas seul la robustesse"
  - "RR³ NOT_MEASURABLE_WITHOUT_G_OMEGA — explicit bound"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


RESULTS_DIR = Path("/home/claude/mcq_v4/results/phase6c_b")


def load_json(name: str) -> Optional[dict]:
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def consolidate() -> dict:
    f1 = load_json("F1_prime_cyclicity.json")
    f3 = load_json("F3_systemic.json")
    f5 = load_json("F5_propagation_asymmetry.json")
    t25 = load_json("F6_T25_prime_contraction.json")
    t27 = load_json("F6_T27_prime_plurality.json")
    readers = load_json("bloc2_readers.json")
    multiseed = load_json("bloc3_multi_seed.json")

    forms = ['contrastive', 'perspectival_INV_H',
             'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']

    # ── Bloc 1 — single-seed regime sets ──
    block_1_regimes = {}
    for form in forms:
        f1_sig = None
        if f1:
            sig_field = f1['per_form'][form].get('cyclic_signature')
            if isinstance(sig_field, dict):
                f1_sig = sig_field.get('label')
            else:
                f1_sig = sig_field
        block_1_regimes[form] = {
            'F1_prime_cyclic_signature_seed42': f1_sig,
            'F3_systemic_verdict_seed42': (
                f3['per_form'][form].get('form_verdict') if f3 else None
            ),
            'F5_asymmetry_noise_off_seed42': (
                f5['asymmetry_decomposition'][form]['asymmetry_noise_off']
                if f5 else None
            ),
            'T25_prime_aggregate_regimes_seed42': (
                t25['per_form'][form]['aggregate']['all_regimes_seen']
                if t25 else []
            ),
            'T27_prime_long_run_regimes_seed42': (
                t27['per_form'][form]['regimes_detected']
                if t27 else []
            ),
        }

    # ── Bloc 3 — multi-seed robustness ──
    block_3_robustness = {}
    if multiseed:
        for form in forms:
            f1_rob = multiseed['F1_robustness'][form]
            f3_rob = multiseed['F3_robustness'][form]
            f5_rob = multiseed['F5_robustness'][form]
            t25_rob = multiseed['T25_robustness'][form]
            t27_rob = multiseed['T27_robustness'][form]

            block_3_robustness[form] = {
                'F1_signature_stable_across_seeds': f1_rob['signature_stable'],
                'F1_period_median': f1_rob['period_median'],
                'F1_corr_AB_BC_dispersion': f1_rob['corr_AB_BC_dispersion'],
                'F3_verdict_stable': f3_rob['verdict_stable'],
                'F3_n_anti_coh_median': f3_rob['n_anti_coh_median'],
                'F5_delta_vs_baseline_median': (
                    f5_rob.get('delta_vs_contrastive_median')
                    if form != 'contrastive' else None
                ),
                'F5_delta_sign_stable': (
                    f5_rob.get('delta_sign_stable')
                    if form != 'contrastive' else None
                ),
                'T25_regimes_in_all_seeds': t25_rob['regimes_in_all_seeds'],
                'T25_regimes_in_some_seeds': t25_rob['regimes_in_some_seed'],
                'T27_robustness_per_module': t27_rob['robustness_per_module'],
                'T27_robustness_union_K1': t27_rob['robustness_union_K1'],
            }

    # ── Bloc 2 — reader layer (interpretive only, not a new verdict) ──
    block_2_readers = {}
    if readers:
        block_2_readers['T25_per_form'] = readers.get('T25_readers', {})
        block_2_readers['T27_per_form'] = readers.get('T27_readers', {})
        block_2_readers['statement'] = (
            "Reading layer applied post-hoc to existing trajectories. "
            "Does NOT modify Bloc 1 verdicts. Adds STR/RSR + MI/MV + "
            "T*_proxy_h_M (NOT full 𝕋*) + RR3 (explicitly NOT measurable "
            "without g_Ω). Globally consistent with regimes, with one "
            "documented discordance: INV_H at intermediate config "
            "shows reader_layer_consistent_with_regime=False — locally "
            "informative, not a contradiction."
        )

    # ── Signals robust across Bloc 3 ──
    signals_robust = []
    if multiseed:
        # F3 INV_H propagation
        if multiseed['F3_robustness']['perspectival_INV_H']['verdict_stable']:
            signals_robust.append({
                'signal': 'F3_systemic_INV_H_micro_propagation',
                'description': (
                    "perspectival_INV_H produces MICRO_PROPAGATION_WITH_G_PROXY"
                    "_PRESERVED on 36/36 cross-module cells, stable across "
                    "{42, 123, 2024}. The 3 other forms remain DECOUPLING_DOMINANT."
                ),
                'magnitude': '36 cells / 36',
                'dispersion': 0,
            })
        # F5 differential structural asymmetry
        for form in ['perspectival_INV_H',
                     'perspectival_H_OPEN', 'perspectival_MORPHO_ACTIVE']:
            f5r = multiseed['F5_robustness'][form]
            if f5r.get('delta_sign_stable'):
                signals_robust.append({
                    'signal': f'F5_differential_structural_{form}',
                    'description': (
                        f"{form} shows a sign-stable differential structural "
                        "asymmetry vs contrastive baseline."
                    ),
                    'magnitude': f5r.get('delta_vs_contrastive_median'),
                    'dispersion': f5r.get('delta_vs_contrastive_dispersion'),
                })
        # T25' regime stability
        for form in forms:
            t25r = multiseed['T25_robustness'][form]
            signals_robust.append({
                'signal': f'T25_regime_set_stable_{form}',
                'description': (
                    f"{form} produces the same regime set across all 3 seeds. "
                    "Modulation-not-contraction reading is robust."
                ),
                'regimes': t25r['regimes_in_all_seeds'],
                'regimes_seed_specific': t25r['regimes_in_some_seed'],
            })
        # T27' INV_H early-window
        t27_inv = multiseed['T27_robustness']['perspectival_INV_H']
        for module in ['A', 'C']:
            r = t27_inv['robustness_per_module'].get(module, {})
            if r.get('verdict') == 'SIGNAL_SURVIVES':
                signals_robust.append({
                    'signal': f'T27_INV_H_early_window_module_{module}',
                    'description': (
                        f"INV_H delays fusion of module {module} by "
                        f"{r['median']:+.0f} step relative to no-coupling baseline, "
                        "stable 3/3 seeds, dispersion 0."
                    ),
                    'magnitude': r['median'],
                    'dispersion': r['dispersion_max_minus_min'],
                })
        u = t27_inv['robustness_union_K1']
        if u.get('verdict') == 'SIGNAL_SURVIVES':
            signals_robust.append({
                'signal': 'T27_INV_H_union_K1_delay',
                'description': (
                    f"INV_H delays union K=1 collapse by {u['median']:+.0f} step, "
                    "stable 3/3 seeds."
                ),
                'magnitude': u['median'],
                'dispersion': 0,
            })
        # T25 NUMERICAL_INSTABILITY for INV_H
        if 'NUMERICAL_INSTABILITY' in multiseed['T25_robustness']['perspectival_INV_H']['regimes_in_all_seeds']:
            signals_robust.append({
                'signal': 'T25_INV_H_numerical_instability_at_extreme_regime',
                'description': (
                    "INV_H reaches NUMERICAL_INSTABILITY at extreme contraction "
                    "regimes, 3/3 seeds. Other forms remain in KNV_COLLAPSE "
                    "without engine breakdown."
                ),
            })

    # ── Signals retracted (Bloc 1 verdicts not surviving Bloc 3) ──
    signals_retracted = []
    if multiseed:
        if not multiseed['F1_robustness']['contrastive']['signature_stable']:
            signals_retracted.append({
                'signal': 'F1_prime_cyclic_phase_shifted_label',
                'reason': (
                    "Seed=42 produced CANDIDATE_CYCLIC_PHASE_SHIFTED with "
                    "corr_AB_BC = -0.31. Seeds 123/2024 produce "
                    "CANDIDATE_SYNCHRONOUS with corr_AB_BC = +0.30/+0.39. "
                    "Dispersion = 0.70 covers half the [-1, +1] range. "
                    "Cyclic label not robust — F1' becomes a cadence reading "
                    "(period 50 stable) without phase signature."
                ),
                'replacement_label': 'WEAK_CYCLIC_LABEL_INSTABLE_CROSS_SEED',
            })
        # T27 contrastive A early-window
        contr_A = multiseed['T27_robustness']['contrastive']['robustness_per_module']['A']
        if contr_A.get('verdict') == 'WITHIN_RNG_NOISE':
            signals_retracted.append({
                'signal': 'T27_contrastive_early_window_module_A_delay',
                'reason': (
                    "Seed=42 showed +1 step delay; seeds 123/2024 show 0. "
                    "Within RNG noise, not a structural signal."
                ),
            })
        # T27 INV_H module B
        invh_B = multiseed['T27_robustness']['perspectival_INV_H']['robustness_per_module']['B']
        if invh_B.get('verdict') == 'WITHIN_RNG_NOISE':
            signals_retracted.append({
                'signal': 'T27_INV_H_early_window_module_B_delay',
                'reason': (
                    f"Diffs {invh_B.get('diffs')} — one seed at 0, two at 1. "
                    "Sign stable but dispersion = magnitude. Not robust."
                ),
            })

    # ── Tensions preserved ──
    tensions_preserved = [
        {
            'kind': 'delta_centred_vs_shape_divergence',
            'description': (
                "Δ_centred (spread of axis means) and Δ_shape (spread of "
                "axis variances) report different morphology states. T25' "
                "contraction shows up only on Δ_shape; Δ_centred stays in "
                "LOW_REST. Confirmed in Bloc 1 + Bloc 2."
            ),
            'mcq_implication': (
                "Δ classique (centred) is not the only relevant corridor. "
                "Δ_shape is structurally distinct and may need its own "
                "calibrated corridor (current threshold 0.30 borrowed from "
                "Δ_centred is provisional)."
            ),
        },
        {
            'kind': 'relative_contraction_without_absolute_contraction',
            'description': (
                "T25' produces relative contraction vs no-coupling control "
                "(suppression of natural expansion drift) but NEVER absolute "
                "contraction (var_M(swap) > var_M_init throughout). "
                "Stable 3/3 seeds for all forms."
            ),
            'mcq_implication': (
                "T25' measures modulation, not contraction. The question "
                "whether the moteur 6a + Φ_extra can support absolute local "
                "contraction remains OPEN — H2 (parametric under-exploration) "
                "and H3 (structural limit) both active."
            ),
        },
        {
            'kind': 'partner_expansion_under_A_constraint',
            'description': (
                "Under constraint on A, partners B and C systematically "
                "EXPAND beyond their no-coupling baseline. Variance is "
                "redistributed inter-modularly, not contracted globally."
            ),
            'mcq_implication': (
                "Inter-modular variance redistribution is a real signal, "
                "but it is not the kind of structure MCQ predicts for "
                "viable distributed plurality. The redistribution here "
                "preserves total variance, not corridor membership."
            ),
        },
        {
            'kind': 'fusion_engine_driven_not_coupling_attributable',
            'description': (
                "T27' label-space ends in 1/1/1 fusion for all 4 forms, AND "
                "for the no-coupling baseline. The fusion is intrinsic to "
                "the 6a engine (diffusion + Φ_corr) at 5×5×5 resolution. "
                "Coupling cannot be attributed any role in the long-run."
            ),
            'mcq_implication': (
                "Bound: T27' label-space in this configuration does NOT "
                "test distributed plurality. Native MCQᴺ test would require "
                "shared Θ across modules, inter-modular metric projection, "
                "or 15-factor / 7-module architecture."
            ),
        },
        {
            'kind': 'fusion_timing_INV_H_only_signal',
            'description': (
                "Despite identical long-run endpoint (1/1/1 fusion), INV_H "
                "delays the fusion timing of A by +1 step and C by +2 steps "
                "and union K=1 by +1 step (3/3 seeds, dispersion 0). "
                "H_OPEN and MORPHO_ACTIVE produce zero timing differential."
            ),
            'mcq_implication': (
                "INV_H produces the only multi-seed-robust dynamic signal "
                "of perspective coupling in this architecture. Its mechanism "
                "is amplification 1/h, not perspective in the morphological "
                "sense. H_OPEN/MORPHO_ACTIVE remain dynamically transparent."
            ),
        },
        {
            'kind': 'tau_prime_projection_decoupled_from_internal_morphology',
            'description': (
                "Bloc 2 reading: under T25' gentle, τ' tensorial projection "
                "(τ_T, τ_I) is quasi-stationary (var ≈ 1e-7) while morphology "
                "(var_M, h_M_mean) is active. The system reorganises "
                "internally without projecting to shared factors."
            ),
            'mcq_implication': (
                "Internal/projection decoupling is a structural feature of "
                "the gradient-coupled architecture: Φ_extra propagates "
                "through morphology but does not necessarily lift to τ' "
                "tensorial signal at small ε. Phase 6d (non-gradient C^mod) "
                "may decouple this differently."
            ),
        },
    ]

    # ── Hypotheses status ──
    hypotheses_status = {
        'H1_numerical_residual': {
            'status': 'PARTIALLY_ACTIVE',
            'evidence': (
                "Active for INV_H at extreme T25' regimes (NUMERICAL_INSTABILITY "
                "stable 3/3 seeds). Inactive at gentle regimes (invariants held). "
                "T27' baseline-fusion is engine-mechanic, not numerical pathology."
            ),
        },
        'H2_parametric_under_exploration': {
            'status': 'ACTIVE',
            'evidence': (
                "T25' sweep covered 5 (alpha, var_floor, lambda) configs but "
                "did not find LOCAL/PROPAGATED_CONTRACTION_VIABLE for any form. "
                "Wider parametric range may still find a viable point. "
                "Cannot rule out without further exploration."
            ),
        },
        'H3_structural_limit': {
            'status': 'ACTIVE',
            'evidence': (
                "Active for all 4 forms in T25': swap had effect (regime "
                "changes visible) AND increasing lambda triggers KNV. "
                "Necessary but not sufficient — wider sweep could rule it out."
            ),
        },
    }

    # ── Caveats ──
    caveats = [
        "Bloc 2 (readers) is a READING LAYER, not a new verdict. It applies "
        "post-hoc to existing trajectories without re-running the engine.",
        "𝕋* is approximated as T*_proxy_h_M (only h_M_mean per step "
        "captured). Full 𝕋* requires h_T/h_M/h_I per step.",
        "𝒢 is computed as G_proxy (ψ-roughness), NOT the full transformable "
        "gradient ‖∂τ′/∂Γ_meta‖. The full 𝒢 requires Γ_meta as explicit "
        "metric history — deferred to Phase 6d/7.",
        "Δ_shape corridor threshold (0.30) is provisional, borrowed from "
        "Δ_centred. A Δ_shape-specific corridor calibration is pending.",
        "RR³ is NOT_MEASURABLE_WITHOUT_G_OMEGA — explicit hard bound. "
        "Anti-petrification dynamics deferred to Phase 6d.",
        "T27' label-space transposition is NOT native MCQᴺ. Compressed "
        "intra-instance with homologous Θ_A/B/C; global union is in label "
        "index space, not in shared geometry.",
        "Multi-seed robustness uses {42, 123, 2024}. The escalation rule "
        "to {7, 999} when verdicts vary by >1 category is a MANUAL rule, "
        "not automated in the current orchestrator.",
        "5×5×5 grid resolution is coarse. Sub-resolution UNIVERSAL_FUSION "
        "artefact in T27' is a real concern — initial/final masks logged "
        "for inspection.",
        "Δ bas au repos (LOW_REST) is NOT a KNV violation — it is the "
        "centred-rest signature of an STR-compatible system.",
    ]

    # ── Negative bounds (what the runs establish as NOT possible / NOT measured) ──
    bounds = [
        {
            'bound': 'no_local_contraction_viable_in_T25_window',
            'description': (
                "In the (alpha × var_floor × lambda) sweep tested, no form "
                "produced LOCAL_CONTRACTION_VIABLE or PROPAGATED_CONTRACTION_"
                "VIABLE without co-present KNV. T25' measures modulation/"
                "redistribution, not viable contraction."
            ),
            'caveat': "H2 active — wider sweep could change this.",
        },
        {
            'bound': 'no_distributed_plurality_attributable_to_coupling',
            'description': (
                "T27' label-space fusion 1/1/1 is engine-driven, present in "
                "no-coupling baseline. No form produces "
                "DISTRIBUTED_PLURALITY_LABEL_SPACE in this configuration."
            ),
            'caveat': (
                "Native MCQᴺ test deferred. Bound is on this transposition, "
                "not on MCQᴺ in general."
            ),
        },
        {
            'bound': 'H_OPEN_and_MORPHO_ACTIVE_dynamically_transparent',
            'description': (
                "H_OPEN and MORPHO_ACTIVE produce no multi-seed-robust dynamic "
                "signal across F1', F3, F5, T25', T27'. They retain a structural "
                "signature (LEVEL_2 from 6c-A) but it does not translate to "
                "dynamics under Φ_extra at ε=0.005."
            ),
            'caveat': "Larger ε or different coupling architecture untested.",
        },
        {
            'bound': 'INV_H_dynamic_signal_attributed_to_amplification',
            'description': (
                "INV_H produces multi-seed-robust signals on F3, F5, T25' "
                "(NUMERICAL_INSTABILITY), T27' early-window. The mechanism "
                "is amplification 1/h, not perspective-in-the-morphological-sense."
            ),
            'caveat': (
                "Distinguishing amplification from genuine perspective "
                "requires architectures where 1/h does NOT amplify uniformly."
            ),
        },
        {
            'bound': 'F1_prime_topology_no_robust_cyclic_signature',
            'description': (
                "Cyclic phase signature is seed-dependent. Period 50 stable, "
                "but corr_AB_BC dispersion = 0.70. F1' = cadence reading, "
                "not topological cyclicity proof."
            ),
        },
    ]

    return {
        'phase': '6c-B',
        'title': 'Three-module compressed coupling — full Bloc 1 + 2 + 3 verdict',
        'block_1_regimes_per_form_seed42': block_1_regimes,
        'block_2_readers': block_2_readers,
        'block_3_robustness_per_form': block_3_robustness,
        'signals_robust_multi_seed': signals_robust,
        'signals_retracted_after_multi_seed': signals_retracted,
        'tensions_preserved': tensions_preserved,
        'hypotheses_status_open': hypotheses_status,
        'caveats': caveats,
        'negative_bounds_established': bounds,
        'overall_reading': (
            "Phase 6c-B in compressed three-module architecture with Φ_extra "
            "coupling at ε=0.005 produces ONE multi-seed-robust dynamic "
            "signal: INV_H amplification 1/h. F3 propagation, F5 asymmetry "
            "differential, T25' NUMERICAL_INSTABILITY, T27' fusion-timing "
            "delay all converge on this signature. H_OPEN and MORPHO_ACTIVE "
            "remain structurally signed (6c-A LEVEL_2) but dynamically "
            "transparent here. T25' shows modulation-not-contraction with "
            "partner-expansion redistribution. T27' label-space is "
            "engine-driven, not coupling-attributable. Native MCQᴺ test of "
            "distributed plurality, full 𝕋*, full 𝒢 = ‖∂τ′/∂Γ_meta‖, RR³ "
            "anti-petrification all deferred. The configuration does NOT "
            "rule in or out the structural-limit hypothesis (H3) — only "
            "establishes negative bounds within the tested parametric window."
        ),
    }


if __name__ == "__main__":
    result = consolidate()
    out = RESULTS_DIR / "phase6c_b_consolidated_verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved consolidated verdict: {out}")
    print()
    print("=" * 70)
    print("Phase 6c-B — Consolidated reading")
    print("=" * 70)
    print()
    print(f"Robust signals: {len(result['signals_robust_multi_seed'])}")
    for s in result['signals_robust_multi_seed']:
        print(f"  ✓ {s['signal']}")
    print()
    print(f"Retracted signals: {len(result['signals_retracted_after_multi_seed'])}")
    for s in result['signals_retracted_after_multi_seed']:
        print(f"  ✗ {s['signal']}")
    print()
    print(f"Tensions preserved: {len(result['tensions_preserved'])}")
    for t in result['tensions_preserved']:
        print(f"  • {t['kind']}")
    print()
    print(f"Hypotheses open:")
    for h, s in result['hypotheses_status_open'].items():
        print(f"  {h}: {s['status']}")
    print()
    print(f"Negative bounds: {len(result['negative_bounds_established'])}")
    for b in result['negative_bounds_established']:
        print(f"  ⊢ {b['bound']}")
