"""
Test 6d-α micro-étape 4a-β — 𝔊^ero seul (logistique), pas de ψ.

Plan validé Alex (post-4a-α) :
- 𝔊^ero seul, 𝔊^sed désactivé (β=0)
- Comparer engine vs solution logistique analytique exacte
- h(t) = h₀·h_init / (h_init + (h₀ - h_init)·exp(-γ·t))

Critères :
- engine vs analytique logistique à erreur Euler théorique
- positivité h MESURÉE (logistique reste positive si h_init > 0)
- pas de conservation Σh

Configurations :
1. h_init < h₀ : croissance asymptotique vers h₀ (équilibre stable)
2. h_init > h₀ : décroissance asymptotique vers h₀
3. h_init = 0 : équilibre instable, h doit rester 0 exactement

Notes méthodologiques :
- 4a-β reste un régime monotone (pas de compétition entre opérateurs).
- Le vrai changement qualitatif attendu en 4a-γ (𝔊^sed + 𝔊^ero combinés).
- Discipline : si quelque chose résiste empiriquement, RÉVISER avant
  d'étendre.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from mcq_v4.factorial_6d import (  # noqa: E402
    State6dMinimal,
    N_AXIS,
    DX,
)
from mcq_v4.factorial_6d.h_dynamics import (  # noqa: E402
    simulate_h_only,
    solution_ero_logistic,
)


def test_4a_beta_one(
    gamma: float,
    h_init: float,
    h0_target: float,
    label: str,
) -> dict:
    """Test logistique pour un (h_init, h₀, γ) donné."""
    print(f"\n{'='*60}")
    print(f"Test 4a-β : {label}")
    print(f"γ = {gamma}, h_init = {h_init}, h0_target = {h0_target}")
    print(f"{'='*60}")

    # h initial uniforme
    h_initial_field = np.full((N_AXIS, N_AXIS, N_AXIS), h_init)
    # ψ absent (mais doit exister comme tableau)
    psi_zero = np.zeros((N_AXIS, N_AXIS, N_AXIS))

    state_init = State6dMinimal(
        psi=psi_zero.copy(),
        h=h_initial_field.copy(),
        h0=h0_target,
        h_min=0.1,
    )

    # Temps caractéristique : τ_ero = 1/γ
    tau_ero = 1.0 / gamma if gamma > 0 else float("inf")
    t_target = 5.0 * tau_ero
    dt = 0.1 * tau_ero
    n_steps = int(np.ceil(t_target / dt))

    print(f"τ_ero = 1/γ = {tau_ero:.4f}")
    print(f"t_target = 5·τ_ero = {t_target:.4f}")
    print(f"dt = {dt:.4f}, n_steps = {n_steps}")

    # Simulation : 𝔊^ero activé, 𝔊^sed désactivé
    state_final, logs = simulate_h_only(
        state_init=state_init,
        psi_fixed=psi_zero,
        beta=0.0,
        gamma=gamma,
        h0_target=h0_target,
        n_steps=n_steps,
        dt=dt,
        include_sed=False,
        include_ero=True,
        log_every=max(1, n_steps // 30),
    )

    # Solution analytique
    h_analytic = []
    for t in logs["t"]:
        h_t = solution_ero_logistic(h_init, gamma, h0_target, t)
        h_analytic.append(h_t)
    h_analytic = np.array(h_analytic)

    # Comparaisons
    h_engine_center = np.array(logs["h_center"])
    diff_center = float(np.max(np.abs(h_engine_center - h_analytic)))

    # Uniformité h
    h_uniformity_gap = max(
        b - a for a, b in zip(logs["h_min"], logs["h_max"])
    )

    h_min_atteint = min(logs["h_min"])

    # Trace
    print(f"\n=== Trace h(t) ===")
    print(f"  {'t':>10} {'h_analytic':>12} {'h_engine':>12} {'diff':>10}")
    indices = [0, len(logs['t']) // 4, len(logs['t']) // 2, 3 * len(logs['t']) // 4, -1]
    for idx in indices:
        if idx < 0:
            idx = len(logs['t']) + idx
        t = logs['t'][idx]
        ha = h_analytic[idx]
        he = h_engine_center[idx]
        d = abs(ha - he)
        print(f"  {t:>10.4f} {ha:>12.6f} {he:>12.6f} {d:>10.4e}")

    # Erreur Euler théorique : pour logistique, le taux max est γ/4 (à h=h₀/2)
    # Erreur ~ 0.5·(γ/4)²·dt²·t·h₀ en magnitude
    erreur_euler = 0.5 * ((gamma / 4) ** 2) * dt * t_target * h0_target

    print(f"\n=== Résultats ===")
    print(f"diff engine vs analytique (centre) : {diff_center:.4e}")
    print(f"uniformité h (gap max-min)         : {h_uniformity_gap:.4e}")
    print(f"h_min atteint                      : {h_min_atteint:.6f}")
    print(f"  h_init = {h_init:.4f}")
    print(f"  h_final analytique = {h_analytic[-1]:.6f}")
    print(f"  h_final engine     = {h_engine_center[-1]:.6f}")
    print(f"Erreur Euler théorique attendue ~ {erreur_euler:.4e}")

    # VERDICT
    verdict = "PASS"
    reasons = []

    seuil = max(erreur_euler * 10, 1e-10)
    if diff_center > seuil:
        verdict = "FAIL" if diff_center > seuil * 10 else "REVISION"
        reasons.append(f"Engine vs analytique : diff={diff_center:.4e} > {seuil:.4e}")

    # Uniformité h : 𝔊^ero est local, donc avec h_init uniforme,
    # h reste uniforme à machine precision
    if h_uniformity_gap > 1e-12:
        verdict = "FAIL" if h_uniformity_gap > 1e-9 else "REVISION"
        reasons.append(f"Uniformité h cassée : gap={h_uniformity_gap:.4e}")

    # Positivité h : la logistique avec h_init > 0 reste positive,
    # avec h_init = 0 reste 0 exact.
    if h_init > 0 and h_min_atteint < 0:
        verdict = "FAIL"
        reasons.append(f"Positivité h violée : h_min={h_min_atteint:.4e}")

    # Cas h_init = 0 : doit rester strictement 0 partout
    if h_init == 0.0 and abs(h_min_atteint) > 1e-300:
        verdict = "FAIL"
        reasons.append(f"h_init=0 mais h a évolué : h_min={h_min_atteint:.4e}")

    print(f"\n=== VERDICT ===")
    if verdict == "PASS":
        print(f"  PASS ✓ pour {label}")
    else:
        print(f"  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "label": label,
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "gamma": gamma,
            "h_init": h_init,
            "h0_target": h0_target,
            "tau_ero": tau_ero,
            "dt": dt,
            "n_steps": n_steps,
            "t_target": t_target,
        },
        "metrics": {
            "diff_engine_vs_analytic": diff_center,
            "erreur_euler_attendue": float(erreur_euler),
            "h_uniformity_gap": float(h_uniformity_gap),
            "h_min_atteint": float(h_min_atteint),
            "h_final_analytic": float(h_analytic[-1]),
            "h_final_engine": float(h_engine_center[-1]),
        },
    }


def run_step_4a_beta() -> dict:
    """Exécute les 3 cas logistiques."""
    configs = [
        (0.5, 0.2, 1.0, "1_croissance_vers_h0"),     # h_init < h₀
        (0.5, 1.5, 1.0, "2_decroissance_vers_h0"),   # h_init > h₀
        (0.5, 0.0, 1.0, "3_equilibre_instable_h0"),  # h_init = 0
    ]

    results = {}
    for gamma, h_init, h0, label in configs:
        result = test_4a_beta_one(gamma, h_init, h0, label)
        results[label] = result

    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*60}")
    print(f"VERDICT GLOBAL 4a-β : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    return {"global_verdict": global_verdict, "cases": results}


if __name__ == "__main__":
    summary = run_step_4a_beta()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_4a_beta_ero_only.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
