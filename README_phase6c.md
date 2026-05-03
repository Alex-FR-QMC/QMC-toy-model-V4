# MCQ Toy V4 — Phase 6c-A

Perspectival coupling baseline — gradient approximation (Φ_extra architecture).

Phase 6c-A introduces three perspective-dependent coupling forms that
extend the Phase 6b non-perspectival baseline. All three forms are
injected via the same Φ_extra mechanism inherited from 6b — the
non-gradient native coupling (𝒞^{mod} as a separate term in ∂_t ψ
outside Φ_eff) is deferred to Phase 6d.

## Status

**CLOSED** — runner executes end-to-end, verdict saved to
`results/phase6c/verdict_phase6c_a.json`, all preflights pass, three
hypotheses (H1/H2/H3) evaluated with explicit criteria, three-level
synthesis applied per form.

## Architecture (vs Phase 6b)

What changes:
- Three new coupling forms added to `compute_extra_phi_for_module`:
  `perspectival_INV_H`, `perspectival_H_OPEN`, `perspectival_MORPHO_ACTIVE`
- Each form computes a `novelty_j^{h_j}` field in j's metric and a
  `form_i^{h_i}` field in i's metric, structurally homologous on both
  sides of the contrastive subtraction
- `ThreeModuleSystem.prev_h_fields` stores a sliding window of the last
  k=5 h_fields per module, used by MORPHO_ACTIVE for `|∂_t h|_eff`
- Engine, preflights 0-7, sign micro-test, RNG independence, topology
  integrity, ε sweep machinery — **all unchanged**

What stays:
- Φ_extra injection mechanism (couplage as scalar potential)
- Cyclic A↔B↔C topology
- Differentiated weights driving morphodynamic differentiation
- Engine 6a: bitwise non-regression vs 6b confirmed by preflights

### Coupling forms

For each module pair (i, j), the perspectival coupling injected into i's
Φ_extra is:

```
positive (6b):
  phi_extra_i = Σ_j ε R_ij(1-R_ij) · ψ_j

contrastive (6b):
  phi_extra_i = Σ_j ε R_ij(1-R_ij) · (ψ_j − R_ij ψ_i)

perspectival_INV_H (6c):
  phi_extra_i = Σ_j ε R_ij(1-R_ij) · (ψ_j/h_j − R_ij ψ_i/h_i)

perspectival_H_OPEN (6c):
  phi_extra_i = Σ_j ε R_ij(1-R_ij) · (ψ_j·h_open_j − R_ij ψ_i·h_open_i)
  with h_open = (h_field - h_min^3)/(h_0^3 - h_min^3) ∈ [0,1]

perspectival_MORPHO_ACTIVE (6c):
  phi_extra_i = Σ_j ε R_ij(1-R_ij) · (
      ψ_j·h_open_j·|∂_t h_j|_eff
    − R_ij ψ_i·h_open_i·|∂_t h_i|_eff
  )
  with |∂_t h|_eff averaged over a k=5-step window
```

R_ij stays label-based (Bhattacharyya on ψ in homologous coordinates) to
preserve the differential measurement vs 6b. R_metric in cosine
similarity is logged in parallel as the H2 driver.

### h_field lifting (separable convention)

The three marginal h_a(θ_a) (each shape (5,)) are combined into a
(5,5,5) field via outer product:

```
h_field(t,m,i) = h_T(t) · h_M(m) · h_I(i)
```

This is the natural conformal lifting consistent with the engine's
separable axes treatment. Documented as a 6c convention; a tensor
metric would replace this in a later phase. Note that bound is
[h_min^3, h_0^3].

## Tension Ch.3 §3.2.2 (documented)

The Ch.3 §3.2.2 formula reads `g_k(θ, h_i^α) = θ_k / h_i^α(θ)` and
states textually: *"Regions where the metric is dilated (h_i^α ≈ h_0
— unfamiliar, not yet sedimented) contribute more strongly to the
field."*

Under our engine's convention (`dh = -β·ψ_marg·h + γ·h·(1 - h/h_0)`),
sedimentation drives `h → h_min` in frequently-visited regions and
erosion drives `h → h_0` elsewhere. So **`h ≈ h_0` is unfamiliar** and
**`h ≈ h_min` is familiar/sedimented** — consistent with the textual
interpretation. But the literal formula `θ_k / h` then has `1/h` LARGE
where h is small (familiar) and `1/h` SMALL where h is large (unfamiliar)
— **the OPPOSITE of "unfamiliar contributes more strongly"**.

There are two ways to resolve this:
- the formula is symbolic notation and the operative reading is `θ_k · (1 - h/h_0)` — instantiated as **H_OPEN**
- or the convention on h is itself reversed — would require flipping signs on β/γ in the engine

Phase 6c-A does NOT resolve this textually. It tests both the literal
formula (INV_H) and the textual interpretation (H_OPEN) empirically.
The mini-sweep reveals which produces a consistent perspective signature
under our concrete instantiation.

## Preflight suite (8 preflights)

The 7 preflights of Phase 6b are inherited unchanged. Preflight 8 is
added:

| # | Preflight | Status if fails |
|---|-----------|-----------------|
| 0-7 | (same as 6b) | NUMERICAL_OR_IMPLEMENTATION_INVALID / COUPLING_SIGN_INVALID / warning only |
| 8 | perspectival sign micro-test (all three forms) | preflight_8_failed (informative; non-zero exit blocks runner) |

Preflight 8 verifies that each perspectival form produces a measurable
non-zero displacement under controlled conditions. Pre-warmup test:
forms degenerate (h uniform → no perspective). Post-warmup test:
forms diverge (h_A ≠ h_B → distinct phi_extra patterns).

## Warmup protocol

```
n_warmup_steps:        100
coupling_during_warmup: OFF       (option A — pure intra-modular dynamics)
base_seed:             42
h_div_floor:           1e-3       (absolute floor for warmup adequacy)
epsilon_struct:        0.2 × max(h_div_warmup)   (relative threshold for h_div_ratio logging)
```

Coupling OFF during warmup ensures that the post-warmup state is the
**same starting point for all coupling form measurements**, so the
6b vs 6c differential is purely attributable to coupling form (not
to history of coupling).

After warmup, the system state, RNG bit_generator state, and
prev_h_fields window are all cloned. Each coupling form runs from the
clone, ensuring deterministic comparability.

## Three coupling-effect levels

The synthesis applies the following classification, designed to
distinguish three distinct regimes of perspective signal:

```
LEVEL_1_NO_PERSPECTIVE
  H2 NOT_SUPPORTED — ΔR_perspectival under threshold.
  System operationally label-based.

LEVEL_2_PARTIAL_STRUCTURAL_SIGNAL_ONLY
  H2 SUPPORTED but extras_diag fails (pattern dissimilarity or
  temporal stability under thresholds).

LEVEL_2_STRUCTURAL_PERSPECTIVE_NO_DYNAMIC_EFFECT
  H2 SUPPORTED + extras REAL_PERSPECTIVE + (H1, H3) NOT_SUPPORTED.
  Perspective shapes the coupling structurally but doesn't translate
  into measurable mass transport (Φ-saturation).

LEVEL_3_WEAK_STRUCTURAL_DYNAMIC
  H1 SUPPORTED but H2 only WEAKLY_SUPPORTED.
  Dynamic effect on F1 is real, but the structural perspective signal
  is weak — likely numerical amplification of a pattern close to
  baseline.

LEVEL_3_DYNAMICAL_EFFECT
  H2 SUPPORTED + extras REAL_PERSPECTIVE + (H1 or H3) SUPPORTED.
  Full structural perspective + measurable dynamic effect.

CHANNEL_NOT_ALIVE caveat (MORPHO_ACTIVE only):
  If diagnose_morpho_active_channel returns aggregate != ALIVE, the
  level is suffixed with _CHANNEL_NOT_ALIVE.
```

## Hypotheses (with explicit criteria)

```
H1 — F1 trajectorial differential
  rel_diff_pure(form) >= H1_factor * rel_diff_pure(baseline)
    AND rel_diff_pure(form) >= H1_F1_absolute_min (1e-3)
  H1_factor = 2.0; absolute floor prevents ratio-of-near-zero traps.
  Verdicts: SUPPORTED / WEAKLY_SUPPORTED / RATIO_ONLY_WEAK /
            NOT_SUPPORTED / INCONCLUSIVE_BOTH_ZERO /
            SUPPORTED_FORM_HAS_INTERFERENCE_BASELINE_DOES_NOT

H2 — Structural perspective via ΔR_perspectival
  mean ΔR_persp(t) >= H2_delta_R_min (0.05)
  ΔR_persp = |R_metric(form) - R_metric(contrastive)|, both in cosine.
  IMPORTANT: ΔR_psi_vs_metric (legacy) is logged but NOT used here, to
  avoid Bhattacharyya/cosine metric conflation.
  Verdicts: SUPPORTED / WEAKLY_SUPPORTED / NOT_SUPPORTED

H3 — F4 metric divergence trajectory
  final_rel >= 0.10 OR L1_traj_rel >= 0.10 (relative to baseline)
  Trajectory criterion catches transient effects missed by final-state.
  Verdicts: SUPPORTED / WEAKLY_SUPPORTED / NOT_SUPPORTED

H4 — Compression T/M/I kills perspective
  All of H1, H2, H3 NOT_SUPPORTED AND warmup_ok (max h_div >= floor)
  → SUPPORTED_COMPRESSION_KILLS_PERSPECTIVE
  → otherwise NOT_SUPPORTED_PERSPECTIVE_HAS_SIGNAL or
                INCONCLUSIVE_INSUFFICIENT_WARMUP
```

## Closed-state results (default config: ε=0.005, base_seed=42)

### Preflight: 8/8 PASS

### Per-form synthesis

| Form | F1 (rel_diff) | H1 | ΔR_persp mean | H2 | F4 traj_rel | H3 | extras | morpho channel | Level |
|------|---------------|----|----|----|----|----|----|----|------|
| `contrastive` | 0.00015 | (baseline) | (baseline) | — | (baseline) | — | (baseline) | — | (baseline) |
| `perspectival_INV_H` | 0.00273 | **SUPPORTED** (×18.3) | 0.0255 | WEAKLY | 0.0001 | NOT | REAL | n/a | **LEVEL_3_WEAK_STRUCTURAL_DYNAMIC** |
| `perspectival_H_OPEN` | 0.00001 | NOT (×0.08, abs_ko) | **0.1147** | **SUPPORTED** | 0.0000 | NOT | REAL | n/a | **LEVEL_2_STRUCTURAL_PERSPECTIVE_NO_DYNAMIC_EFFECT** |
| `perspectival_MORPHO_ACTIVE` | 0.00000 | NOT (×0, abs_ko) | **0.2010** | **SUPPORTED** | 0.0000 | NOT | REAL | **ALIVE** (mean_dh=3.2e-2) | **LEVEL_2_STRUCTURAL_PERSPECTIVE_NO_DYNAMIC_EFFECT** |

### Three regimes — no smoothing

The result honestly distinguishes three distinct attractors of perspective
signal:

1. **Numerical amplification (INV_H)**: the literal `1/h` formula inflates
   the coupling amplitude in sedimented regions. F1 differential reaches
   ×18.3 vs baseline (above abs_min and above ratio threshold). But the
   perspectival pattern (ΔR_persp = 0.025) remains close to the label
   baseline. The dynamic effect is real but reflects amplification more
   than perspective shaping. Classified `LEVEL_3_WEAK_STRUCTURAL_DYNAMIC`
   to flag this distinction explicitly.

2. **Structural perspective without dynamic translation (H_OPEN, MORPHO_ACTIVE)**:
   ΔR_persp = 0.115 and 0.201 respectively, well above threshold. Pattern
   dissimilarity 0.40 and 0.73, indicating clearly distinct field shapes.
   But F1 stays at noise floor and F4 trajectory is identical to baseline.
   The perspective shapes the coupling structure but the gradient
   transport mechanism absorbs the signal without measurable mass effect.
   For MORPHO_ACTIVE, the channel ALIVE diagnosis confirms this is not an
   artefact of vanishing |∂_t h|: the morphogenic activity is present
   (mean 3.2e-2, max ~9e-2), the perspective signal is structurally
   maximal — but the gradient transport saturates.

3. **Phi_extra absorption (all three forms, H3 NOT_SUPPORTED)**:
   F4 metric divergence trajectories are nearly identical for baseline
   and all forms. The perspective changes the spatial pattern of
   phi_extra but the cumulative effect on metric evolution is
   indistinguishable. This is the empirical signature of "coupling as
   scalar potential cannot translate structural perspective into
   morphodynamic differentiation".

### Validates the gradient-coupling limitation

The result empirically supports Alex's structural insight:

> The Φ_extra architecture (coupling as scalar potential) cannot
> translate perspectival structure into measurable dynamic effect.
> A non-gradient term — `∂_t ψ = ∇·(ψ ∇Φ_eff) + 𝒞^{mod}` — is needed.

H_OPEN and MORPHO_ACTIVE produce maximal structural signal (ΔR_persp,
pattern dissimilarity) without dynamic translation, **even with
channel ALIVE**. The bottleneck is not in the perspectival formula
but in the gradient-transport mechanism. Phase 6d will test this.

## Caveats

1. **Thresholds H1_abs_min=1e-3, H2_min=0.05, H3_rel=0.10 are a priori,
   not empirically calibrated.** A second pass could derive them from
   the observed distribution of effects across multiple seeds.

2. **F1 mini-sweep ε∈{0.01, 0.05, 0.1} not run.** INV_H's H1 SUPPORTED
   verdict at ε=0.005 should ideally be confirmed at other ε values
   before claiming LEVEL_3_WEAK_STRUCTURAL_DYNAMIC robustly.

3. **Multi-seed robustness not assessed.** All measurements use
   base_seed=42. A multi-seed pass would reveal variance.

4. **Compressed factor space (T/M/I).** As documented, T/M/I are
   aggregations of the 15 constitutive factors of MCQ. The perspective
   tested here is on the compressed projection. Whether the dynamic
   effect "switches on" with the full 15-factor field is an open
   question for Phase 6d/7.

## Mandatory verdict fields

Every Phase 6c-A verdict contains (asserted before save):
- `phase` = "6c-A"
- `phase_status` (NON_PERSPECTIVAL_BASELINE_BUT_NOT_NATIVE_MCQ_N statement)
- `preflight_status`
- `seeds` (with rngs_independent=True flag)
- `warmup` (with sufficiency check)
- `forms_tested` (with F1, F4 trajectory, ΔR_persp history,
                  extras_diagnostics, morpho_active_channel where relevant)
- `hypotheses` (H1/H2/H3/H4 + synthesis_3_levels per form)
- `synthesis` (level per form)
- `global_status`

`assert_mandatory_fields` is called before JSON save; missing fields
raise.

## Files

```
src/mcq_v4/factorial/
├── coupling.py                    # 5 forms (extended with perspectivals)
├── three_module_system.py         # ThreeModuleSystem with prev_h_fields window
├── diagnostics_6c.py              # 6c-specific diagnostics + hypothesis evaluators

tests/phase6c/
├── test_perspectival_sign.py      # preflight 8
└── test_three_modules_6c.py       # main runner

results/phase6c/
└── verdict_phase6c_a.json         # final verdict
```

## Phase 6d — what comes next

Phase 6d will move 𝒞^{mod} OUT of Φ_extra:

```
∂_t ψ = ∇·(ψ ∇Φ_eff) + 𝒞^{mod}
```

with 𝒞^{mod} as a direct term in the time evolution, NOT a
gradient-derivable potential. This allows:

- Non-irrotational currents (∇×J ≠ 0)
- Local mass creation/destruction (renormalisation will be needed)
- Asymmetric structural modifications between i and j from symmetric
  coupling (perspective-driven)

Expected: H_OPEN and MORPHO_ACTIVE — currently LEVEL_2 — should produce
measurable dynamic effects (LEVEL_3) when their structural signal is
no longer absorbed by gradient transport. INV_H should produce more
nuanced results since its amplification is not its main feature in
the non-gradient regime.

τ' will also become operative (vector input to Φ_eff, D_eff, J_noise
projections) in 6d, partially addressing the compression T/M/I
limitation.
