# MCQ Toy V4 — Phase 6b

Three-module factorial perturbative coupling — **non-perspectival baseline**.

Phase 6b adds inter-modular interaction to the Phase 6a factorial engine,
without altering the engine itself. Three modules M_A, M_B, M_C run in
parallel on their own factorial domains Θ_m, exchange coupling through
a perturbative term injected into Φ_corr, and produce a 6-dimensional
field τ' ∈ ℝ⁶.

Phase 6b is **NOT** a validation of native MCQ^N. The coupling is
intentionally non-perspectival, used as a baseline against which Phase
6c will measure the contribution of perspective-dependent novelty.

## Status

**CLOSED** — runner executes end-to-end, verdict saved to
`results/phase6b/verdict_phase6b.json`, all preflights pass.

## Architecture

### Three independent modules

Three `FactorialState` instances on three independent Θ_m = T × M × I.
Each module has:
- its own ψ_m ∈ ℝ^{5×5×5} normalised
- its own marginal metric h_m^T, h_m^M, h_m^I
- its own RNG stream (seed = base_seed + offset; A=+101, B=+202, C=+303)
- its own pondération:
  - M_A = (1.5, 0.8, 0.7) — tensional dominance
  - M_B = (0.7, 1.6, 0.7) — morphodynamic dominance
  - M_C = (0.8, 0.7, 1.5) — interface dominance

The 6a engine runs unchanged on each module — preserved by the
`phi_extra=None` regression test (bitwise identity).

### Sharing topology (option β — modular axes, not shared)

Each module's axes T/M/I are **homologous but not identical** to the
others. Shared factors are labels of aggregation, not common dimensions.

```
Shared factors (each observed by 2 modules on T axis):
  k_AB = contrib_A^T + contrib_B^T
  k_BC = contrib_B^T + contrib_C^T
  k_CA = contrib_C^T + contrib_A^T

Private factors (each observed by 1 module on I axis):
  k_A_private = contrib_A^I
  k_B_private = contrib_B^I
  k_C_private = contrib_C^I
```

τ' ∈ ℝ⁶ is a **composition of modular contributions**, not an
integration over a shared space. Each module computes its contribution
in its own metric.

### Coupling form

Perturbative non-perspectival coupling, two forms (both selectable):

```
positive    : phi_extra_i = Σ_j ε · R_ij(1-R_ij) · ψ_j
contrastive : phi_extra_i = Σ_j ε · R_ij(1-R_ij) · (ψ_j - R_ij · ψ_i)   [default]
```

With:
- R_ij = Bhattacharyya overlap of ψ_i, ψ_j in homologous coordinates (label-based)
- R(1-R) = productive-overlap weight; vanishes at R=0 and R=1
- ψ_j read in homologous coordinates (Phase 6b approximation)

The coupling is **anti-fusion by construction**: the coupling sign
micro-test confirms COUPLING_REPULSIVE for both forms in isolation.
Conceptual reading (Alex's correction): inter-modular novelty is
**converted into a gradient of differentiation**, not "fled".

The native MCQ coupling (Ch.3 §3.1.3.III) — `R_ij · (novelty of j in
j's metric − R_ij · form of i in i's metric)` — is deferred to Phase 6c.

### Two overlaps

- **R_psi** (Bhattacharyya): drives the coupling. Label-based comparison.
- **R_tau** (functional): diagnostic only. Compares modular contributions
  to the same shared factor. R_psi high + R_tau low → modules look alike
  but evaluate the shared factor differently because of metric differences.

## Preflight suite

7 preflights run before any verdict interpretation. The runner fails
fast on blocking failures.

| # | Preflight | Status if fails |
|---|-----------|-----------------|
| 0 | 6a canonical verdict (mediation, F2, F3a artefact, F3b, MCF, mass drift) | NUMERICAL_OR_IMPLEMENTATION_INVALID |
| 1 | sign regression 6a (drift anti-collapse + diffusion spread) | NUMERICAL_OR_IMPLEMENTATION_INVALID |
| 2 | phi_extra=None bitwise non-regression | NUMERICAL_OR_IMPLEMENTATION_INVALID |
| 3 | coupling_forms_regression (positive ≠ contrastive at runtime + algebraic match) | NUMERICAL_OR_IMPLEMENTATION_INVALID |
| 4 | coupling_sign_microtest (REPULSIVE for both forms) | COUPLING_SIGN_INVALID |
| 5 | attractive_sign_control annexe (inverted sign → ATTRACTIVE) | warning only |
| 6 | RNG independence (object ids + functional sequences + seed log) | NUMERICAL_OR_IMPLEMENTATION_INVALID |
| 7 | topology integrity (cyclic A-B-C) | NUMERICAL_OR_IMPLEMENTATION_INVALID |

## Metrics

### F1 — Trajectorial interference (3-branch)

Compares τ'_{AB}(t) across:
1. Solo separated: A alone, B alone (two independent engines)
2. Solo parallel: A+B+C in same system, ε=0
3. Coupled: A+B+C in same system, ε > 0

Reports:
- `rel_diff_pure` = ‖coupled − solo_parallel‖ / amplitude
- `rel_diff_setup` = ‖solo_parallel − solo_separated‖ / amplitude
  (must be ~0 if isolation is clean)

### F2-bis — Tripartition per module under coupling

F2 (var_M, var_I growth from concentrated init) applied to each module
within the coupled system. Tests whether coupling preserves intra-modular
factorial dynamics.

### F2'-bis — h_M frozen test per module under coupling

For each module M, runs the coupled system with h_M frozen for that
module (others normal) and compares τ' trajectories. Tests whether the
α-channel mediation persists in the coupled regime.

### F3a / F3b per module under coupling

F3a (tensional-visible polymorphism via δψ→δτ') and F3b (internal
morphodynamic transformability via δψ_M → future evolution) applied
per module under active coupling. F3b uses RNG state synchronisation
between reference and perturbed branches so divergence is purely from
the perturbation.

### F4 — Modular differentiation (priority h_a, secondary R_psi/R_tau)

Mean relative divergence of metric profiles h_a between module pairs.
Reading: ψ similar + h different = real morphodynamic differentiation.

### F5 — Weight ablation

Compares F4 metric divergence under differentiated weights vs identical
weights (1,1,1) for all modules. Same base_seed for both conditions to
isolate the effect of weights from stochastic variability.

### F6 — ε sweep (PRIMARY RESULT)

Sweeps ε ∈ {0.001, 0.005, 0.01, 0.05, 0.1} for both coupling forms
(contrastive default, positive secondary). For each (ε, form):
- regime classification (BUFFERED / PERTURBATIVE / OVERCOUPLED) based on
  max ratio of |phi_extra| / lambda_KNV (stable reference)
- F4 metric divergence
- mass drift per module

The result of Phase 6b is the **map of regimes**, not a chosen ε.

## Closed-state results

### Preflight: 7/7 PASS

| # | Preflight | Result |
|---|-----------|--------|
| 0 | 6a canonical verdict | PASS — mediation `M_MEDIATES_VIA_ALPHA_BETA_ATTENUATES`, F2 PASS, F3a `FAIL_NO_PRODUCTIVE_FORGETTING` (canonical artefact), F3b WEAK_DIVERGENCE, MCF max_rel_diff = 7.4e-5 |
| 1-7 | other preflights | all PASS |

### F1 trajectorial — at ε = 0.005, contrastive

```
outcome:        ADDITIVE (NOISY)
rel_diff_pure:  1.52e-4  (below WEAK threshold 1e-2)
rel_diff_setup: 0.0      (system isolation clean — strict zero)
```

### F2-bis per module — all PASS

| Module | growth_M | growth_I | Outcome |
|--------|----------|----------|---------|
| A | 0.924 | 0.859 | PASS |
| B | 0.726 | 0.859 | PASS |
| C | 0.859 | 0.678 | PASS |

### F2'-bis per module — mediation persists, partially attenuated

| Module | rel_diff | Outcome |
|--------|----------|---------|
| A | 0.0313 | WEAK_MEDIATION |
| B | 0.0231 | WEAK_MEDIATION |
| C | 0.0877 | PASS |

The h_M mediation found in 6a (FULL=0.075, ALPHA_ONLY=0.10) is
partially masked by the coupling. C remains best-mediated, likely
because it dominates on I and is less perturbed by T-axis sharing.

### F3a / F3b per module — opacity preserved

F3a all `FAIL_NO_PRODUCTIVE_FORGETTING` with max_card=4 (canonical
marginal-h artefact, identical to 6a behaviour).
F3b all `WEAK_DIVERGENCE` with div_var_M ∈ [1.6e-2, 3.0e-2].

The factorial opacity (F3a invisible, F3b transformable) persists
under coupling.

### F4 — PASS_METRIC_DIFFERENTIATION

```
metric_divergence_mean: 0.336
R_psi means: A-B = 0.926, B-C = 0.918, A-C = 0.928   (very high)
R_tau means: A-B = 0.168, B-C = 0.223, A-C = 0.523   (low to medium)
```

ψ similar across modules but h profiles strongly distinct — exactly
the morphodynamic differentiation signature predicted by the spec.

### F5 — WEIGHTS_DRIVE_DIFFERENTIATION

| Configuration | metric_divergence | F4 outcome |
|---------------|-------------------|------------|
| Differentiated | 0.336 | PASS |
| Identical | 0.124 | WEAK |
| Ratio identical/differentiated | 0.371 | — |

Identical weights reduce differentiation to 37% of the differentiated
value. Pondérations are the primary driver. No total fusion (consistent
with the prudent formulation).

### F6 — Regime map (PRIMARY RESULT)

| ε | Form | Regime | F4 metric_div | Mass drift max |
|---|------|--------|---------------|----------------|
| 0.001 | contrastive | BUFFERED | 0.250 | 2.2e-16 |
| 0.005 | contrastive | BUFFERED | 0.250 | 2.2e-16 |
| 0.010 | contrastive | BUFFERED | 0.250 | 4.4e-16 |
| 0.050 | contrastive | PERTURBATIVE | 0.251 | 4.4e-16 |
| 0.100 | contrastive | PERTURBATIVE | 0.251 | 2.2e-16 |
| 0.001 | positive | BUFFERED | 0.250 | 4.4e-16 |
| 0.005 | positive | BUFFERED | 0.250 | 3.3e-16 |
| 0.010 | positive | BUFFERED | 0.250 | 4.4e-16 |
| 0.050 | positive | PERTURBATIVE | 0.251 | 2.2e-16 |
| 0.100 | positive | PERTURBATIVE | 0.252 | 3.3e-16 |

No OVERCOUPLED regime observed up to ε=0.1. Mass drift at floating-point
roundoff throughout the sweep, both forms.

### Synthesis verdict

```
phase_status: NON-PERSPECTIVAL baseline. Not MCQ^N native validation.
preflight_status: OK
global_status: INTERPRETABLE
synthesis.overall_verdict: ADDITIVE_NO_INTERFERENCE
synthesis.differentiating_coupling_confirmed: True
```

## Important caveat — F1 was measured at ε=0.005 only

**F1 trajectorial interference was measured only at the default
ε=0.005, NOT swept across the F6 ε grid.**

The F6 sweep characterises:
- regime (BUFFERED / PERTURBATIVE / OVERCOUPLED) via the
  |extra|/lambda_KNV ratio
- F4 metric differentiation
- mass drift / numerical stability per module

But F6 does NOT compute F1 at each ε. So it is **incorrect** to write
"no interference up to ε=0.1". The accurate statement is:

> At ε=0.005, the coupling is additive in the F1 sense (rel_diff_pure
> ≈ 1.5e-4, below the WEAK threshold). The ε sweep shows the system
> stays BUFFERED to PERTURBATIVE up to ε=0.1, with no overcoupling and
> no numerical degradation. The trajectorial interference F1 has not
> been swept across ε.

A mini F1-sweep over ε ∈ {0.01, 0.05, 0.1} could be added before Phase
6c to characterise the exact baseline boundary, but is not blocking
for closing 6b.

## Conceptual reading

At perturbative ε (≤ 0.1, both forms), the simplified non-perspectival
coupling:
- preserves the intra-modular factorial tripartition (F2-bis PASS)
- partially attenuates h_M mediation through the α-channel (F2'-bis WEAK to PASS)
- preserves factorial opacity (F3a invisible, F3b transformable)
- maintains modular differentiation driven by weights (F4 PASS, F5 ratio 0.37)
- stays numerically clean (mass drift at floating-point roundoff)
- is anti-fusion by construction (COUPLING_REPULSIVE confirmed)
- does NOT produce measurable trajectorial interference at ε=0.005 (F1 ADDITIVE)

This is the documented baseline. Phase 6c will introduce
perspective-dependent native coupling on top of this baseline,
allowing measurement of what perspective-dependence adds.

## Mandatory verdict fields

Every Phase 6b verdict contains (asserted before save):
- `phase` = "6b"
- `phase_status` = NON-PERSPECTIVAL baseline statement
- `preflight_status`
- `seeds` (with rngs_independent=True flag verified)
- `mass_drift_per_module` (A, B, C)
- `mass_drift_total`
- `global_status`

If any field is missing, `assert_mandatory_fields()` raises and the
JSON save is refused.

## Files

```
src/mcq_v4/factorial/
├── three_module_system.py   # ThreeModuleSystem, CouplingConfig, build helper
├── overlaps.py              # R_psi, R_tau, coupling_weight
├── tau_prime.py             # FactorialFieldOutput, modular contributions
├── coupling.py              # compute_extra_phi, step_three_modules
└── metrics_6b.py            # F1, F4, F5, F6, ratio_diagnostics, synthesis

tests/phase6b/
├── preflight_suite.py                       # orchestrates 7 preflights
├── test_three_modules.py                    # main runner
├── test_coupling_sign.py                    # preflight 4
├── test_coupling_forms_regression.py        # preflight 3
├── test_rng_independence.py                 # preflight 6
├── test_topology_integrity.py               # preflight 7
└── annexe_attractive_sign_control.py        # preflight 5 (annex)

results/phase6b/
└── verdict_phase6b.json                     # final verdict
```

## Phase 6c — what comes next

Phase 6c will replace the non-perspectival coupling form with the native
formula (Ch.3 §3.1.3.III):

```
𝒞_i ∝ Σ_j R_ij · (novelty_of_j(in j's metric) − R_ij · form_of_i(in i's metric))
```

The "novelty" term is evaluated through a metric-aware projection that
weights ψ_j by entropy contributions in j's metric, not by raw amplitude.
The expectation: 6c should produce measurable F1 trajectorial
interference at ε levels where 6b is additive, and the differential
signal between 6b and 6c at matched ε should be the perspective effect.
