# Phase 6c-B — Three-module compressed coupling: structural tests

## Scope

Phase 6c-B extends Phase 6c-A by introducing a structured battery of
tests on the three-module compressed system (modules A, B, C with
homologous-but-not-shared Θ_A/B/C, coupled via Φ_extra at ε=0.005,
four coupling forms: `contrastive`, `perspectival_INV_H`,
`perspectival_H_OPEN`, `perspectival_MORPHO_ACTIVE`).

The phase is **not** a native MCQᴺ test: it is a compressed
intra-instance transposition with documented limits.

## Architecture

### What this phase IS

- Five structural tests (F1', F3, F5, F6.T25', F6.T27') applied per
  coupling form
- A reading-layer (Bloc 2) that applies STR/RSR/MI/MV/T*_proxy/RR³
  diagnostics post-hoc to existing trajectories
- Multi-seed robustness check (Bloc 3) on `{42, 123, 2024}` with
  manual escalation rule to `{7, 999}` if verdict varies by >1
  category (not automated in the orchestrator)
- A consolidated verdict aggregating signals/tensions/bounds without
  collapsing to a single label

### What this phase IS NOT

- Not a native MCQᴺ test of distributed plurality (T27' is label-space
  transposition, not geometric)
- Not a measurement of full 𝒢 (we use G_proxy = ψ-roughness; the full
  𝒢 = ‖∂τ′/∂Γ_meta‖ requires Γ_meta as explicit metric history,
  deferred to Phase 6d/7)
- Not a measurement of full 𝕋* (we use T*_proxy_h_M; full 𝕋* requires
  h_T/h_M/h_I per step trajectories)
- Not a measurement of RR³ (NOT_MEASURABLE_WITHOUT_G_OMEGA — explicit
  hard bound; requires g_Ω modulation of D_eff which is in Phase 6d)
- Not a calibrated test of Δ_shape (corridor threshold 0.30 is
  provisional, borrowed from Δ_centred)

## Files

```
tests/phase6c_b/
  test_F1_prime_cyclicity.py            — temporal covariance of shared factors
  test_F3_systemic.py                    — 18×3 perturbation propagation grid
  test_F5_propagation_asymmetry.py       — normalised asymmetry, noise-off control
  test_F6_T25_prime_contraction.py       — progressive contraction ramp + control
  test_F6_T27_prime_plurality.py         — label-space distributed plurality
  bloc2_readers.py                        — STR/RSR/MI/MV/T*/RR³ on existing trajectories
  bloc3_multi_seed.py                     — robustness across {42, 123, 2024}
  test_three_modules_6c_b.py             — consolidated verdict (no engine re-run)

src/mcq_v4/factorial/
  signatures_6c_b.py                     — KNV thresholds, G_modular, Δ_centred,
                                           Δ_shape, Γ″ multi-signal, readers

results/phase6c_b/
  F1_prime_cyclicity.json
  F3_systemic.json
  F5_propagation_asymmetry.json
  F6_T25_prime_contraction.json
  F6_T27_prime_plurality.json
  bloc2_readers.json
  bloc3_multi_seed.json
  phase6c_b_consolidated_verdict.json    — single-source-of-truth output
```

## Reading rules (verbatim from session audits)

These rules govern how results are read and written:

1. **Bloc 2 is a reading layer, not a new verdict.** Readers apply
   post-hoc to trajectories without re-running the engine. They add
   interpretation, not new facts.
2. **τ′ projection and internal morphology are partially decoupled.**
   Under T25' gentle, τ_T/τ_I are quasi-stationary while var_M/h_M
   are active. The system reorganises internally without projecting
   to the shared-factor channel.
3. **LOW_REST is centred-rest zone, not KNV violation.** Δ_centred
   below `delta_min` while never having been higher = STR-compatible
   resting state, not corridor failure.
4. **VIABLE labels are forbidden when KNV is co-present.** Labels
   like `LOCAL_CONTRACTION_VIABLE` are gated on the absence of any
   `KNV_*` regime in the same run.
5. **Partner directions are split.** `partner_X_contracts` ≠
   `partner_X_expands` ≠ `partner_X_disturbed_no_contraction`.
   Avoid `abs()` on directional drops.
6. **Absolute and relative contractions are different objects.**
   `final_drop_vs_init` (vs initial state) and `final_drop_vs_ctrl`
   (vs no-coupling baseline) capture different dynamics. T25' gentle
   shows relative contraction without absolute contraction.
7. **`G_proxy` ≠ full 𝒢.** Always label `G_proxy_total` in code and
   "G_proxy" in prose when discussing the modular gradient.
8. **Δ_shape thresholds are provisional.** Below 0.30 = corridor;
   0.30–0.35 = `DELTA_SHAPE_BOUNDARY_EXCURSION`; above 0.35 = clearly
   outside (provisional KNV signal pending calibration).
9. **Tensions are preserved, not resolved.** Multi-label regimes are
   the norm. A run can simultaneously show `RELATIVE_CONTRACTION_VS_
   CONTROL_ONLY ∧ PARTNER_EXPANSION_UNDER_A_CONSTRAINT ∧ DELTA_SHAPE_
   BOUNDARY_EXCURSION`.
10. **No conclusions on Φ_extra without ruling out H2 (parametric
    under-exploration) and H1 (numerical residual).** Both remain
    active in this phase.

## Per-test summary

### F1' Cyclicity (free characterisation)

Measures temporal covariance, correlation, and phase shifts of the
three shared factors (k_AB, k_BC, k_CA). Period 50 steps stable
across seeds; cyclic phase signature is **NOT** robust (label varies
between seeds; corr_AB_BC dispersion = 0.70).

**Reading:** F1' is a cadence reading (𝕋(t) period stability),
**not** a cyclicity proof. Initial seed=42 `CANDIDATE_CYCLIC_PHASE_
SHIFTED` label is retracted as RNG-dependent.

### F3 Systemic (MCQ-aligned criterion)

18 directions × 3 modules. Criterion: anti-coherence productive iff
propagation present AND G_proxy above floor AND Δ not exceeding
delta_crit.

**Robust signal (3/3 seeds):** INV_H produces `MICRO_PROPAGATION_
WITH_G_PROXY_PRESERVED` on 36/36 cross-module cells, propagation ~×13
the other forms. The 3 other forms are `DECOUPLING_DOMINANT`.

### F5 Asymmetry of propagation (with noise-off control)

Normalised asymmetry ‖Δψ_B|A − Δψ_A|B‖ / (‖.‖+‖.‖). Two passes:
canonical noise + noise-off control to isolate structural component.

**Robust signal (3/3 seeds):**
- INV_H: Δ_struct = -0.0091 (REDUCES asymmetry vs contrastive)
- H_OPEN: Δ_struct = +0.0004 (effectively zero)
- MORPHO_ACTIVE: Δ_struct = +0.0004 (effectively zero)

Sign-stable across seeds, dispersion 0. Stochastic component ~0.003
for all forms (negligible).

### F6.T25' Morphodynamic constrained contraction

Progressive ramp on var_max(t) starting from var_M_init(A). Sweep
over (alpha, var_floor, lambda_KNV). Multi-temporal drops measured
both vs init and vs no-coupling control. KNV-aligned monitoring
(G_proxy, Δ_centred, Δ_shape, Γ″ multi-signal).

**Robust regime set (3/3 seeds, all forms):**

```
RELATIVE_CONTRACTION_VS_CONTROL_ONLY
PARTNER_EXPANSION_UNDER_A_CONSTRAINT
DELTA_SHAPE_BOUNDARY_EXCURSION (gentle config)
A_EXPANSION_VS_CTRL                (other configs)
A_EXPANSION_EXTREME                (extreme configs)
BIFURCATION_CONTRACTION_TO_EXPANSION
KNV_TOPOLOGICAL_DELTA_CENTRED      (extreme configs)
KNV_TOPOLOGICAL_DELTA_SHAPE        (extreme configs)
KNV_MORPHOGENIC                    (extreme configs)
TRANSIENT_CONTRACTION_NON_SUSTAINED
```

INV_H additionally: `NUMERICAL_INSTABILITY` at extreme regimes (3/3
seeds — the only form that reaches engine breakdown).

**Reading:** T25' measures **modulation/redistribution under
constraint**, not contraction. var_M(swap) stays above var_M_init
throughout. The swap suppresses natural expansion drift relative to
control, while partners B and C expand. No `LOCAL_CONTRACTION_VIABLE`
or `PROPAGATED_CONTRACTION_VIABLE` is produced for any form. H2 and
H3 both remain active.

### F6.T27' Distributed plurality (LABEL-SPACE TRANSPOSITION)

Each module starts with a different geometric multi-modal pattern
(cardinally identical 3/3/3 due to threshold; geometrically
differentiated; initial union = 7). 6-connected mode counts.

**Long-run verdict (3/3 seeds, all forms):** `ENGINE_DRIVEN_FUSION_
NOT_COUPLING_ATTRIBUTABLE`. The no-coupling baseline ALSO ends in
1/1/1 fusion. The 6a engine destroys plurality on its own at 5×5×5
resolution.

**Early-window robust signal (3/3 seeds, INV_H only):**
- module A: +1 step delay vs baseline (dispersion 0)
- module C: +2 steps delay vs baseline (dispersion 0)
- union K=1: +1 step delay (dispersion 0)

INV_H is the **only** coupling form that delays fusion timing. C
(the most geometrically differentiated module) shows the largest
INV_H delay.

H_OPEN and MORPHO_ACTIVE produce **strictly identical** trajectories
to the no-coupling baseline.

## Bloc 2 readings (interpretive layer)

Applied to T25' trajectories:

- **STR/RSR on τ′_A:** `TRANSITIONING_ACTIVE` for all 4 forms in
  gentle config. τ′ projection bouge faiblement; ni stationnaire ni
  amorti.
- **MI(Δ_centred):** `MAINTAINED` (corridor 46% + low_rest 55%).
  Système majoritairement en repos centré.
- **MI(Δ_shape):** `MAINTAINED` (corridor 99%). Excursions Δ_shape
  ponctuelles, pas soutenues.
- **T*_proxy_h_M:** `MOVING_h_M`, cadence ≈ 0.012/step. Métrique
  morphologique active.
- **RR³:** `NOT_MEASURABLE_WITHOUT_G_OMEGA`. Hard bound.
- **Consistency:** globally consistent with regimes, with one
  documented discordance (INV_H intermediate config) — locally
  informative, not a contradiction.

## Robust signals (all 3/3 seeds)

12 multi-seed-robust signals identified. The hierarchy:

1. **F3 INV_H propagation** (36/36 cells) — strongest signal
2. **T25' regime stability** for all 4 forms (modulation
   reading is robust)
3. **T27' INV_H early-window** (A: +1, C: +2, union K=1: +1)
4. **F5 sign-stable structural differential** for INV_H/H_OPEN/MORPHO
5. **T25' INV_H NUMERICAL_INSTABILITY** at extremes

## Retracted signals (Bloc 1 not surviving Bloc 3)

- F1' `CANDIDATE_CYCLIC_PHASE_SHIFTED` (seed=42 only)
- T27' contrastive early-window A delay (seed=42 only)
- T27' INV_H module B delay (1 seed at 0, 2 at 1; dispersion = signal)

## Hypotheses open at end of phase

| Hypothesis | Status |
|---|---|
| H1 numerical residual | PARTIALLY_ACTIVE (extreme T25' regimes only) |
| H2 parametric under-exploration | ACTIVE (sweep covered 5 configs only) |
| H3 structural limit | ACTIVE (necessary conditions met, not sufficient) |

H2 and H3 cannot be discriminated within this phase.

## Negative bounds established

The runs establish what is NOT possible/measured in this configuration:

1. No `LOCAL_CONTRACTION_VIABLE` in T25' window (H2 caveat applies)
2. No distributed plurality attributable to coupling in T27' (Native
   MCQᴺ test deferred — bound is on this transposition)
3. H_OPEN and MORPHO_ACTIVE are dynamically transparent under Φ_extra
   at ε=0.005 (larger ε untested)
4. INV_H signal is amplification 1/h, not perspective-in-the-
   morphological-sense (distinguishing requires architectures where
   1/h does not amplify uniformly)
5. F1' has no robust cyclic signature (period stable, phase not)

## Tensions preserved

Six structural tensions kept open for future investigation:

- `delta_centred_vs_shape_divergence` — different morphology states
  reported by Δ_centred vs Δ_shape
- `relative_contraction_without_absolute_contraction` — modulation,
  not contraction
- `partner_expansion_under_A_constraint` — variance redistributed,
  not contracted globally
- `fusion_engine_driven_not_coupling_attributable` — T27' bound
- `fusion_timing_INV_H_only_signal` — only INV_H produces
  multi-seed-robust dynamic differential
- `tau_prime_projection_decoupled_from_internal_morphology` —
  internal/projection decoupling under gradient coupling

## What deferred for Phase 6d/7

- Full 𝒢 = ‖∂τ′/∂Γ_meta‖ via explicit Γ_meta tracking
- Full 𝕋* on h_T/h_M/h_I per step
- RR³ via g_Ω modulation of D_eff
- Native MCQᴺ test of distributed plurality (15-factor / 7-module or
  shared-Θ architecture)
- Δ_shape calibrated corridor
- Non-gradient C^mod coupling architecture (alternative to Φ_extra)

## How to reproduce

```bash
cd /home/claude/mcq_v4

# Run individual tests (single seed = 42)
PYTHONPATH=src python tests/phase6c_b/test_F1_prime_cyclicity.py
PYTHONPATH=src python tests/phase6c_b/test_F3_systemic.py
PYTHONPATH=src python tests/phase6c_b/test_F5_propagation_asymmetry.py
PYTHONPATH=src python tests/phase6c_b/test_F6_T25_prime_contraction.py
PYTHONPATH=src python tests/phase6c_b/test_F6_T27_prime_plurality.py

# Apply reading layer (Bloc 2)
PYTHONPATH=src python tests/phase6c_b/bloc2_readers.py

# Run multi-seed robustness (Bloc 3) — ~10 min
PYTHONPATH=src python tests/phase6c_b/bloc3_multi_seed.py

# Build consolidated verdict (no engine re-run)
PYTHONPATH=src python tests/phase6c_b/test_three_modules_6c_b.py
```

The consolidated verdict is at
`results/phase6c_b/phase6c_b_consolidated_verdict.json`.
