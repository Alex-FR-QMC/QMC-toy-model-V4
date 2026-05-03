# MCQ Toy V4 — Phase 6a

Single isolated module M_A on factorial domain Θ = T × M × I.
First reconstruction of the MCQ architecture on a discrete factorial space,
no longer using the 2D spatial grid as primary support.

## Architecture

### Domain

Θ_m = T × M × I, three axes interpreted by tripartition (Ch.2 §2.2.8):
- **T** (tensional): θ_T ∈ {-2, -1, 0, +1, +2} — module's contribution to the field
- **M** (morphodynamic): θ_M ∈ {0, 1, 2, 3, 4} — depth of sedimentation
- **I** (interface): θ_I ∈ {-2, -1, 0, +1, +2} — predictive-evaluative orientation

5 levels per axis → 125 cells per module. ψ_m ∈ ℝ^{5×5×5}, normalised.

### Module M_A

Phase 6a uses one isolated module:
- weights = (w_T=1.5, w_M=0.8, w_I=0.7) — tensional dominance
- β_a = β_0 · w_a (faster sedimentation on dominant axis)
- γ_a = γ_0 / w_a (slower erosion on dominant axis)

### Metric

Marginal diagonal: h_T(θ_T), h_M(θ_M), h_I(θ_I) — three 1D arrays of length 5
per module. h ∈ [h_min, h_0] = [0.1, 1.0].

**Approximation** (cf. APPROXIMATION_NOTES in `__init__.py`): Phase 6a uses
h marginal. Two cells with the same θ_T but different (θ_M, θ_I) contribute
identically to τ'_T at the level of g_k. Full per-cell metric h_a(θ_T, θ_M, θ_I)
is deferred (debt D3 in the squelette).

## Engine

Master equation in flux-divergence form:

    ∂_t ψ = -∇·J_diff  -∇·J_drift  -∇·J_noise

with currents:

    J_diff   = -D_eff / h^2 · ∇ψ                    (Fick's law in geometry)
    J_drift  = -ψ · ∇Φ_corr                         (mass flows from high Φ toward low Φ)
    J_noise  = σ · D_eff · sqrt(ψ_L · ψ_R) · ξ / h_a(edge)

Substituting:

    ∂_t ψ = ∇·(D_eff/h^2 ∇ψ)         (α: geometric diffusion)
          + ∇·(ψ ∇Φ_corr)            (β: regulation drift; +sign from J=-ψ∇Φ)
          - ∇·J_noise                (multiplicative flux noise)

Note the **positive sign** on the regulation term: it follows from
J_drift = -ψ ∇Φ. A positive Gaussian bump in Φ_corr at the mean
(activated when var < var_min) drives mass outward — anti-collapse.
See `_drift` docstring for full derivation.

    ∂_t h_a(θ_a) = -β_a · ψ_marg_a(θ_a) · h_a(θ_a)
                  + γ_a · h_a(θ_a) · (1 - h_a(θ_a)/h_0)

Reflective boundary conditions on all axes.

## Numerical scheme

### Conservative flux noise

Multiplicative noise as flux divergence (preserves mass exactly to roundoff):

    J_a^noise(edge) = σ · D_eff · sqrt(ψ_left · ψ_right) · ξ / h_a(edge)

with ξ ~ N(0, 1) iid per edge per axis per step. The geometric mean
sqrt(ψ_left · ψ_right) preserves multiplicativity (vanishes where ψ
vanishes on either side).

### Positivity by exceptional clipping

ψ ← max(ψ, 0) without renormalisation. Mass drift is logged at each step
and classified at end of run:

    mass_drift_final < 1e-4    : ACCEPTABLE
    [1e-4, 1e-3)               : NUMERICAL_WARNING (interpretable, borderline)
    ≥ 1e-3                     : NUMERICAL_INVALID (results NOT interpreted)

When NUMERICAL_INVALID, F2/F2'/F3 results from that mode are not interpreted.

## Engine modes (decomposition)

Five modes for ablation studies:

| Mode | α (geometric diff.) | β (Φ_corr) | Noise | Stable? |
|------|---------------------|------------|-------|---------|
| `FULL` | h-modulated | active | active | ✓ |
| `ALPHA_ONLY` | h-modulated | OFF (=0) | active | ✓ |
| `BETA_ISOTROPIC` | isotropic (h=h_0) | active | active | ✓ (after both sign fixes) |
| `BETA_PURE` | OFF | active | OFF | ✓ (after both sign fixes) |
| `NO_REGULATION_BASELINE` | isotropic (h=h_0) | OFF | active | ✓ |

All five modes are numerically stable after the two sign corrections
(Φ_corr sign + drift divergence convention). See "BETA modes instability"
below for the historical narrative.

## Calibration of λ_KNV

The amplitude of Φ_corr proximity functionals (`lambda_KNV`) controls the
strength of the anti-collapse repulsive bumps and anti-dispersion confining
quadratics. Initial Phase 6a value was λ_KNV = 0.3.

### The two sign errors

Phase 6a development encountered TWO sign bugs that had to be discovered
and corrected separately. Each was caught by independent micro-test.

**Bug 1 — Φ_corr anti-collapse term sign.**
The first implementation built the anti-collapse bump as a *negative well*
at the mean (`Phi -= ...exp(...)`). Under `J = -ψ ∇Φ`, the resulting drift
attracted mass toward the mean — pro-collapse, not anti-collapse.
**Fix:** built Φ_corr as a *positive bump* at the mean (`Phi += ...`).

**Bug 2 — drift divergence convention.**
After Bug 1 was fixed, micro-test still showed pro-collapse behaviour: the
drift contribution to ∂_t ψ was POSITIVE at the centre and NEGATIVE at
neighbours, opposite of what the corrected Φ should produce. Investigation
revealed that `_drift` returned `flux[i+1] - flux[i]` (forward divergence)
while the continuity equation requires `-∇·J = -(flux[i+1] - flux[i]) =
flux[i] - flux[i+1]`. The function `_diffuse` had a compensating sign
flip in its flux convention (J_diff = +D·∇ψ instead of -D·∇ψ), so the
two errors cancelled there but not in `_drift`.
**Fix:** `_drift` now returns `-(flux[i+1] - flux[i])`, with explicit
docstring stating that the sign convention differs from `_diffuse`
because `J_drift` already carries the physical minus sign.

Both fixes are protected by the sign regression test
(`tests/phase6a/test_sign_regression.py`), which verifies that:
1. With var<var_min, Φ_corr produces a positive bump and the drift
   makes mass leave the centre and arrive at the neighbours;
2. With diffusion alone on a delta-init, mass spreads outward to all
   six neighbours.

### Calibration scan

After sign correction, the original λ_KNV = 0.3 produced massive numerical
instability (mass_drift = 1.26 in FULL, 4.93e+02 in BETA_PURE). Systematic
scan at dt = 0.05, T_steps = 200:

| λ_KNV | mass_drift | clip events |
|-------|------------|-------------|
| 0.300 | 1.26e+00 | 60 |
| 0.100 | 6.11e-02 | 50 |
| **0.050** | **4.44e-16** | **0** |
| 0.030 | 1.11e-16 | 0 |
| 0.010 | 2.22e-16 | 0 |

λ_KNV = 0.05 is the largest value that produces clean numerical behaviour
(0 clip events, drift at floating-point roundoff). Adopted as Phase 6a
default. Higher values would require finer dt or implicit time stepping.

## Tests

### F2 — Tripartition operationnelle

Init: ψ concentrated on T (Gaussian on T, point at M=2, I=2).
Measure: σ_M(t), σ_I(t) growth from initial 0.

Outcome: PASS / BORDERLINE / FAIL.
PASS criteria: σ_M(t_final) and σ_I(t_final) > 0.3 each, σ_T preserved ≥ 50%.

### F2' — h_M frozen mediation test

Two parallel runs (same init, same seed) — one with h_M evolving freely,
one with h_M frozen at h_M(0). Compare τ'(t) trajectories.

Three outcomes:
- **PASS**: rel_diff ≥ 0.05 — h_M effectively mediates τ'
- **WEAK_MEDIATION**: 0.01 ≤ rel_diff < 0.05 — borderline
- **MARGINAL_APPROXIMATION_LIMIT**: rel_diff < 0.01 — indistinguishable

The third outcome is **AMBIGUOUS** — could indicate (a) tripartition is
inert in this engine, OR (b) marginal h approximation is too weak to expose
the mediation. Disambiguation requires Phase 6a-bis with full h_a.

### F3a — Tensional-visible polymorphism

Discrete perturbation on 6 directions (T±, M±, I±) at sampled steps.
Direction (a, sign) is in 𝒟(t) iff the response on τ' exceeds G_min.

|𝒟(t)| ≤ 4 by construction in marginal-h setup: M perturbations produce
zero δτ' (the marginal projections ψ_T, ψ_I are invariant under mass
shifts within an M-slab). This is a structural property, not a failure
mode. F3a therefore measures *tensional-visible* transformability — what
the system "can see itself transform" through the field τ'. It is not the
full polymorphism 𝒟(t) of Ch.1 §1.3.7.

### F3b — Internal morphodynamic transformability

Complement to F3a. Two parallel runs from a developed state, one perturbed
on a chosen axis. Measure trajectory divergence in observable space (var_M,
var_I, H, h_M) over a horizon.

Outcome: PASS / WEAK_DIVERGENCE / INVISIBLE.
Validated empirically: M-perturbation produces ~3% divergence in var_M
within 50 steps in FULL mode — M is dynamically transformable internally
even though invisible to F3a.

The gap between F3a (M invisible) and F3b (M produces dynamics) measures
exactly the **factorial opacity** of Ch.3 §3.5 cross-tension 18 — what the
system cannot observe of itself through τ'.

### MCF intra-modulaire

Cold-start convergence: two runs with same module (same seed) and very
different initial ψ (concentrated T vs uniform). Should converge to similar
stationary observables.

| T_steps | max relative diff |
|---------|-------------------|
| 200 | 86.7% |
| 500 | 23.2% |
| 1000 | 1.7% |
| **2000** | **0.01%** ✓ |
| 5000 | 0.00% |

T_steps = 2000 reaches < 0.01% — three orders of magnitude below the 5%
threshold. MCF intra-modular is empirically validated.

## BETA modes instability — RESOLVED (was sign artefact)

In an intermediate state of Phase 6a development (after the Φ_corr sign
fix but BEFORE the drift divergence fix), `BETA_ISOTROPIC` and `BETA_PURE`
showed mass drift that did NOT vanish in dt → 0 limit:

| dt (intermediate state) | drift BETA_ISOTROPIC | drift BETA_PURE |
|----|----------------------|--------------------|
| 0.050 | 1.64e-01 | 1.69e+00 |
| 0.001 | 1.64e-01 | 1.69e+00 |

This was originally interpreted as a structural property of the dynamics
("β requires α for stability"). However, after the second sign correction
(the drift divergence convention in `_drift`), all BETA modes are
numerically stable with mass_drift at floating-point roundoff (4e-16)
and zero clip events.

The instability was an artefact of the *intermediate* engine state, where
the Φ_corr sign was correct but the drift divergence convention was
inverted. With both signs correct, the dynamics in BETA modes is
physically well-posed.

This finding is preserved as a methodological lesson: numerical
"capacity results" must be re-validated whenever any sign or convention
change is made to the engine. The sign regression test
(`test_sign_regression.py`) catches such regressions automatically.

## Status (Phase 6a, after both sign corrections)

| Test | FULL | ALPHA_ONLY | BETA_ISOTROPIC | BETA_PURE | NO_REG_BASELINE |
|------|------|------------|-----------------|-----------|-----------------|
| Invariants | ✓ 4e-16 | ✓ 2e-16 | ✓ 4e-16 | ✓ 4e-16 | ✓ 4e-16 |
| F2 (var_M growth) | 0.92 PASS | 0.90 PASS | 0.55 PASS | 0.32 PASS | 0.39 PASS |
| F2' rel_diff | 0.075 PASS | 0.100 PASS | 0.007 MARG | 0.000 MARG | 0.000 MARG |
| F3a (visible) | INVISIBLE_M | INVISIBLE_M | INVISIBLE_M | INVISIBLE_M | INVISIBLE_M |
| F3b on M | WEAK 3.3% | WEAK 3.0% | WEAK 3.5% | WEAK 4.0% | WEAK 4.0% |
| MCF intra | T=2000: 0.01% PASS | (FULL only) | (FULL only) | (FULL only) | (FULL only) |

All five modes are NUMERICALLY STABLE after the sign corrections — mass
drift at floating-point roundoff, zero clip events. The earlier
NUMERICAL_INVALID classifications of BETA_ISOTROPIC and BETA_PURE were
artefacts of the Φ_corr sign bug (the inverted regulation produced
divergent dynamics). The corrected anti-collapse drift is well-behaved
in all modes.

## Synthesis verdict

**M_MEDIATES_VIA_ALPHA_BETA_ATTENUATES**: α is the sole mediator of h_M
on τ'; β attenuates α in FULL mode (does not amplify it).

Empirically:

| Mode | F2' rel_diff | F2' outcome |
|------|--------------|-------------|
| ALPHA_ONLY | 0.1003 | PASS |
| FULL (α + β) | 0.0746 | PASS (but lower than ALPHA_ONLY) |
| BETA_ISOTROPIC | 0.0065 | MARGINAL_APPROXIMATION_LIMIT |
| BETA_PURE | 0.0000 | MARGINAL_APPROXIMATION_LIMIT |
| NO_REG_BASELINE | 0.0000 | MARGINAL_APPROXIMATION_LIMIT |

Reading: the regulation drift Φ_corr does not transmit h_M's effect to
τ' — it does not pass F2' in any of the three β-active modes that
isolate it. When combined with α (in FULL), the regulation pushes mass
outward in the same way the geometric diffusion does, but in a way
that masks the differential sensitivity of τ' to h_M. Hence
rel_diff(FULL) < rel_diff(ALPHA_ONLY).

This is NOT antagonism (β does not invert the effect), but
ATTENUATION (β dampens what α would expose alone). It is a
non-trivial finding about the interaction between two mechanisms
that are conceptually independent in the spec.

### History of incorrect readings (Phase 6a development)

The mediation reading was inverted three times during development before
reaching the current correct one:

1. **Pre-correction** (Φ_corr negative bump = pro-collapse error):
   FULL = 4.21% < ALPHA_ONLY = 10.03% → "β dampens α" (accidentally correct
   reading from a wrong mechanism).
2. **After Φ_corr sign fix** (but drift divergence still inverted):
   FULL = 15.03% > ALPHA_ONLY = 10.03% → "β amplifies α" (incorrect, due
   to the second sign bug compensating in a misleading way).
3. **After Φ_corr + drift divergence fix** (current, verified by
   tests/phase6a/test_sign_regression.py):
   FULL = 7.46% < ALPHA_ONLY = 10.03% → "β attenuates α" (correct).

The correct reading is preserved by the sign regression test, which any
future change to Φ_corr or _drift must continue to satisfy.

## Debts (deferred to later phases)

- **D1**: 7 modules architecture (Phase 7+)
- **D2**: 15 named factors mapping (Phase 7+)
- **D3**: full per-cell tensorial metric h_{ab}(θ) — would disambiguate
  the MARGINAL_APPROXIMATION_LIMIT outcome of F2' if observed (Phase 6a-bis)
- **D4**: perspectival coupling (Phase 6c)
- **D5**: multi-instance MCQ^N (Phase 7+)
- **D6**: tri-scale cadence derivation (post-Phase 6)
- **D7**: RR³ and 𝓜_Ω in factorial formulation (during 6a refinement)
- **D8**: empirical cognitive correspondence (out of scope, internal falsification only)

## Files

- `src/mcq_v4/factorial/state.py` — `ModuleConfig`, `FactorialEngineConfig`, `FactorialState`, `EngineMode`
- `src/mcq_v4/factorial/observables.py` — `compute_observables`, `compute_D_eff`, `compute_Phi_corr`, `compute_tau_prime_modular`
- `src/mcq_v4/factorial/engine.py` — `FactorialEngine`
- `src/mcq_v4/factorial/metrics.py` — F2, F2', F3a, F3b, MCF, invariants, mediation synthesis
- `tests/phase6a/test_module_isolated.py` — runner producing verdict_phase6a.json
- `results/phase6a/verdict_phase6a.json` — output (post-run)
