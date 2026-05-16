"""
Test 6d-α micro-étape 4a-α — 𝔊^sed seul, ψ fixe uniforme constante.

Plan validé Alex :
- Premier contact empirique avec h(t).
- ψ FIXE en paramètre (pas variable d'état).
- 𝔊^sed seul (𝔊^ero désactivé).
- Comparer engine vs solution analytique h(t) = h_init · exp(-β·ψ₀·t).

Critères :
- engine vs analytique exponentielle exacte à erreur Euler théorique
  O(β·ψ·dt²·t)
- positivité h MESURÉE (devrait tenir car exponentielle)
- pas de conservation Σh à postuler (h n'est pas une probabilité)

Discipline méthodologique post-3b :
- Test minimal et isolé avant extension.
- Si quelque chose casse ou résiste, RÉVISER la mini-spec §4a avant
  d'étendre vers 4a-β / 4a-γ / 4a-δ.

Configurations testées :
1. β·ψ modéré : décroissance lente, h reste loin de h_min
2. β·ψ fort : décroissance rapide, h s'approche de h_min
3. ψ inhomogène (gaussien centré) : h baisse seulement où ψ est présent
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
    make_gaussian_state,
    N_AXIS,
    DX,
)
from mcq_v4.factorial_6d.h_dynamics import (  # noqa: E402
    simulate_h_only,
    solution_sed_uniform,
)


def test_4a_alpha_uniform(
    beta: float,
    psi_uniform_value: float,
    h0_init: float,
    label: str,
) -> dict:
    """
    Test 4a-α avec ψ uniforme constante.

    Critère central : h(t) = h_init·exp(-β·ψ·t) partout (uniforme reste
    uniforme parce que 𝔊^sed est local cellulaire sans couplage spatial).
    """
    print(f"\n{'='*60}")
    print(f"Test 4a-α : {label}")
    print(f"β = {beta}, ψ_uniform = {psi_uniform_value}, h0_init = {h0_init}")
    print(f"{'='*60}")

    # ψ uniforme partout
    psi_fixed = np.full((N_AXIS, N_AXIS, N_AXIS), psi_uniform_value)
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0_init)

    state_init = State6dMinimal(
        psi=psi_fixed.copy(),
        h=h_init.copy(),
        h0=h0_init,
        h_min=0.1,
    )

    # Temps caractéristique : τ_sed = 1/(β·ψ)
    tau_sed = 1.0 / (beta * psi_uniform_value)
    # On simule sur 5·τ pour voir une décroissance significative
    t_target = 5.0 * tau_sed

    # CFL : dt < 1/(β·ψ_max) = τ_sed pour stabilité explicite
    dt = 0.1 * tau_sed  # 10% du temps caractéristique
    n_steps = int(np.ceil(t_target / dt))

    print(f"τ_sed = 1/(β·ψ) = {tau_sed:.4f}")
    print(f"t_target = 5·τ_sed = {t_target:.4f}")
    print(f"dt = {dt:.4f}, n_steps = {n_steps}")

    # Simulation
    state_final, logs = simulate_h_only(
        state_init=state_init,
        psi_fixed=psi_fixed,
        beta=beta,
        gamma=0.0,  # 𝔊^ero désactivé via γ=0
        h0_target=h0_init,
        n_steps=n_steps,
        dt=dt,
        include_sed=True,
        include_ero=False,
        log_every=max(1, n_steps // 30),
    )

    # Solution analytique aux temps de log
    h_analytic = []
    for t in logs["t"]:
        h_t = solution_sed_uniform(h0_init, beta, psi_uniform_value, t)
        h_analytic.append(h_t)
    h_analytic = np.array(h_analytic)

    # Comparaison engine vs analytique
    h_engine_center = np.array(logs["h_center"])
    h_engine_corner = np.array(logs["h_corner"])
    h_engine_mean = np.array(logs["h_mean"])

    diff_center = np.max(np.abs(h_engine_center - h_analytic))
    diff_corner = np.max(np.abs(h_engine_corner - h_analytic))
    diff_mean = np.max(np.abs(h_engine_mean - h_analytic))

    # Aussi : ψ doit rester strictement constant (pas de dynamique ψ)
    psi_drift = max(
        abs(logs["psi_min"][-1] - psi_uniform_value),
        abs(logs["psi_max"][-1] - psi_uniform_value),
        abs(logs["psi_mean"][-1] - psi_uniform_value),
    )

    # Positivité h
    h_min_atteint = min(logs["h_min"])

    # Uniformité de h(t) : doit rester uniforme partout (ψ uniforme)
    # max - min de h à chaque step
    h_uniformity_max_gap = max(
        b - a for a, b in zip(logs["h_min"], logs["h_max"])
    )

    print(f"\n=== Résultats ===")
    print(f"Engine vs analytique (centre)  : diff_max = {diff_center:.4e}")
    print(f"Engine vs analytique (coin)    : diff_max = {diff_corner:.4e}")
    print(f"Engine vs analytique (moyenne) : diff_max = {diff_mean:.4e}")
    print(f"ψ drift (doit être 0)          : {psi_drift:.4e}")
    print(f"h_min atteint                  : {h_min_atteint:.6f}")
    print(f"Uniformité h (max - min)       : {h_uniformity_max_gap:.4e}")
    print(f"  h_init  = {h0_init:.4f}")
    print(f"  h_final analytique = {h_analytic[-1]:.4f}")
    print(f"  h_final engine     = {h_engine_center[-1]:.4f}")

    # Erreur Euler théorique pour exponentielle : O((β·ψ·dt)²·t/2) sur la valeur
    erreur_euler_attendue = 0.5 * (beta * psi_uniform_value * dt) ** 2 * t_target * h0_init
    print(f"\n  Erreur Euler théorique attendue ~ {erreur_euler_attendue:.4e}")

    # Trace
    print(f"\n=== Trace h(t) ===")
    print(f"  {'t':>10} {'h_analytic':>12} {'h_engine_center':>15} {'diff':>10}")
    indices = [0, len(logs['t']) // 4, len(logs['t']) // 2, 3 * len(logs['t']) // 4, -1]
    for idx in indices:
        if idx < 0:
            idx = len(logs['t']) + idx
        t = logs['t'][idx]
        ha = h_analytic[idx]
        he = h_engine_center[idx]
        d = abs(ha - he)
        print(f"  {t:>10.4f} {ha:>12.6f} {he:>15.6f} {d:>10.4e}")

    # VERDICT
    print(f"\n=== VERDICT ===")
    verdict = "PASS"
    reasons = []

    # Engine vs analytique : tolérance basée sur erreur Euler théorique
    seuil = max(erreur_euler_attendue * 10, 1e-10)
    if diff_center > seuil:
        verdict = "FAIL" if diff_center > seuil * 10 else "REVISION"
        reasons.append(
            f"Engine vs analytique : diff={diff_center:.4e} > {seuil:.4e}"
        )

    # Uniformité h : avec ψ uniforme et 𝔊^sed local cellulaire,
    # h doit rester uniforme à machine precision
    if h_uniformity_max_gap > 1e-12:
        verdict = "FAIL" if h_uniformity_max_gap > 1e-9 else "REVISION"
        reasons.append(
            f"Uniformité h cassée : gap={h_uniformity_max_gap:.4e}"
        )

    # ψ doit rester strictement constant
    if psi_drift > 1e-14:
        verdict = "FAIL"
        reasons.append(f"ψ a changé (devrait être fixe) : drift={psi_drift:.4e}")

    # Positivité h : exponentielle stricte, doit rester > 0
    if h_min_atteint < 0:
        verdict = "FAIL"
        reasons.append(f"Positivité h violée : h_min = {h_min_atteint:.4e}")

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
            "beta": beta,
            "psi_uniform": psi_uniform_value,
            "h0_init": h0_init,
            "tau_sed": tau_sed,
            "dt": dt,
            "n_steps": n_steps,
            "t_target": t_target,
        },
        "metrics": {
            "diff_engine_vs_analytic_center": float(diff_center),
            "diff_engine_vs_analytic_corner": float(diff_corner),
            "diff_engine_vs_analytic_mean": float(diff_mean),
            "erreur_euler_attendue": float(erreur_euler_attendue),
            "psi_drift": float(psi_drift),
            "h_min_atteint": float(h_min_atteint),
            "h_uniformity_max_gap": float(h_uniformity_max_gap),
            "h_final_analytic": float(h_analytic[-1]),
            "h_final_engine_center": float(h_engine_center[-1]),
        },
    }


def test_4a_alpha_inhomogeneous(
    beta: float,
    sigma_psi: float,
    h0_init: float,
    label: str,
) -> dict:
    """
    Test 4a-α avec ψ inhomogène (gaussienne centrée).
    h(θ, t) = h0_init · exp(-β · ψ(θ) · t)  partout indépendamment

    Comme 𝔊^sed est local cellulaire (pas de couplage spatial),
    chaque cellule évolue indépendamment selon son ψ_local.
    """
    print(f"\n{'='*60}")
    print(f"Test 4a-α : {label}")
    print(f"β = {beta}, ψ gaussien σ={sigma_psi}, h0_init = {h0_init}")
    print(f"{'='*60}")

    # ψ gaussien normalisé
    state_template = make_gaussian_state(sigma_0=sigma_psi)
    psi_fixed = state_template.psi.copy()
    psi_max = float(psi_fixed.max())
    psi_min = float(psi_fixed.min())

    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0_init)

    state_init = State6dMinimal(
        psi=psi_fixed.copy(),
        h=h_init.copy(),
        h0=h0_init,
        h_min=0.1,
    )

    # τ_sed local = 1/(β·ψ_local). Temps rapide aux cellules à ψ max.
    tau_sed_min = 1.0 / (beta * psi_max)
    t_target = 5.0 * tau_sed_min

    dt = 0.1 * tau_sed_min
    n_steps = int(np.ceil(t_target / dt))

    print(f"ψ range : [{psi_min:.6f}, {psi_max:.6f}]")
    print(f"τ_sed (au max ψ) = {tau_sed_min:.4f}")
    print(f"t_target = {t_target:.4f}, dt = {dt:.4f}, n_steps = {n_steps}")

    state_final, logs = simulate_h_only(
        state_init=state_init,
        psi_fixed=psi_fixed,
        beta=beta,
        gamma=0.0,
        h0_target=h0_init,
        n_steps=n_steps,
        dt=dt,
        include_sed=True,
        include_ero=False,
        log_every=max(1, n_steps // 30),
    )

    # Solution analytique cellule par cellule (𝔊^sed est local)
    # h(θ, t_final) = h0_init · exp(-β · ψ(θ) · t_final)
    h_analytic_final = h0_init * np.exp(-beta * psi_fixed * n_steps * dt)
    h_engine_final = state_final.h

    diff_max = float(np.max(np.abs(h_engine_final - h_analytic_final)))
    diff_at_max_psi = float(np.abs(
        h_engine_final[np.unravel_index(psi_fixed.argmax(), psi_fixed.shape)]
        - h_analytic_final[np.unravel_index(psi_fixed.argmax(), psi_fixed.shape)]
    ))
    diff_at_min_psi = float(np.abs(
        h_engine_final[np.unravel_index(psi_fixed.argmin(), psi_fixed.shape)]
        - h_analytic_final[np.unravel_index(psi_fixed.argmin(), psi_fixed.shape)]
    ))

    h_min_atteint = min(logs["h_min"])

    # Erreur Euler théorique : max sur cellules
    erreur_euler_max = 0.5 * (beta * psi_max * dt) ** 2 * t_target * h0_init

    print(f"\n=== Résultats ===")
    print(f"diff max engine vs analytique (toutes cellules) : {diff_max:.4e}")
    print(f"diff au max ψ (décroissance la plus rapide)     : {diff_at_max_psi:.4e}")
    print(f"diff au min ψ (décroissance la plus lente)      : {diff_at_min_psi:.4e}")
    print(f"h_min atteint global : {h_min_atteint:.6f}")
    print(f"Erreur Euler attendue ~ {erreur_euler_max:.4e}")

    # h aux cellules : analytique vs engine (centre, coin)
    psi_center = psi_fixed[2, 2, 2]
    psi_corner = psi_fixed[0, 0, 0]
    h_analytic_center = h0_init * np.exp(-beta * psi_center * t_target)
    h_analytic_corner = h0_init * np.exp(-beta * psi_corner * t_target)
    print(f"\n  Au centre (ψ={psi_center:.6f}) :")
    print(f"    h_analytic = {h_analytic_center:.6f}, h_engine = {state_final.h[2,2,2]:.6f}")
    print(f"  Au coin   (ψ={psi_corner:.6f}) :")
    print(f"    h_analytic = {h_analytic_corner:.6f}, h_engine = {state_final.h[0,0,0]:.6f}")

    # VERDICT
    verdict = "PASS"
    reasons = []
    seuil = max(erreur_euler_max * 10, 1e-10)
    if diff_max > seuil:
        verdict = "FAIL" if diff_max > seuil * 10 else "REVISION"
        reasons.append(f"Engine vs analytique : diff={diff_max:.4e} > {seuil:.4e}")

    if h_min_atteint < 0:
        verdict = "FAIL"
        reasons.append(f"Positivité h violée : h_min={h_min_atteint:.4e}")

    if verdict == "PASS":
        print(f"\n  PASS ✓ pour {label}")
    else:
        print(f"\n  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "label": label,
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "beta": beta,
            "sigma_psi": sigma_psi,
            "h0_init": h0_init,
            "tau_sed_at_max_psi": tau_sed_min,
            "psi_max": psi_max,
            "psi_min": psi_min,
            "dt": dt,
            "n_steps": n_steps,
        },
        "metrics": {
            "diff_engine_vs_analytic_max": diff_max,
            "diff_at_max_psi": diff_at_max_psi,
            "diff_at_min_psi": diff_at_min_psi,
            "h_min_atteint": float(h_min_atteint),
            "erreur_euler_attendue": float(erreur_euler_max),
        },
    }


def run_step_4a_alpha() -> dict:
    """Exécute test 4a-α en 3 configs."""
    configs_uniform = [
        # (beta, psi_value, h0_init, label)
        (1.0, 0.008, 1.0, "1_uniform_betapsi_modere"),  # ψ_typical, β=1
        (10.0, 0.008, 1.0, "2_uniform_betapsi_fort"),   # décroissance rapide
    ]

    results = {}
    for beta, psi_val, h0_init, label in configs_uniform:
        result = test_4a_alpha_uniform(beta, psi_val, h0_init, label)
        results[label] = result

    # Cas inhomogène
    result_inhom = test_4a_alpha_inhomogeneous(
        beta=5.0, sigma_psi=1.5, h0_init=1.0,
        label="3_inhomogene_gaussien"
    )
    results["3_inhomogene_gaussien"] = result_inhom

    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*60}")
    print(f"VERDICT GLOBAL 4a-α : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    return {"global_verdict": global_verdict, "cases": results}


if __name__ == "__main__":
    summary = run_step_4a_alpha()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_4a_alpha_sed_only.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
