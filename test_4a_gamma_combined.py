"""
Test 6d-α micro-étape 4a-γ — 𝔊^sed + 𝔊^ero combinés, ψ fixe uniforme.

Plan validé Alex :
- 5 configs βψ/γ ∈ {0.1, 0.5, 0.9, 1.0, 1.5}
- Comparer engine vs solution analytique exacte de la TRAJECTOIRE
  (pas seulement le point fixe)
- Verrouiller la bifurcation locale r = γ - β·ψ

Trois régimes distincts à valider séparément :
- r > 0 (βψ/γ < 1) : sous-critique stable, h → K = h₀·(1-βψ/γ) > 0
- r = 0 (βψ/γ = 1) : critique, h décroît POLYNOMIALEMENT vers 0
- r < 0 (βψ/γ > 1) : sur-critique, h → 0 exponentiellement

Critères :
- engine vs analytique trajectoire complète à erreur Euler théorique
- engine vs analytique POINT FIXE à temps long (sous-critique seulement)
- uniformité h préservée (les deux générateurs sont locaux)
- positivité h MESURÉE

Caveats hérités 4a-α/β :
- h = 0 reste un point fixe numériquement, mais son statut change :
  - instable si βψ < γ
  - critique/marginal si βψ = γ
  - stable si βψ > γ
- la trajectoire vers h=0 dans le cas sur-critique est ce qui valide
  empiriquement la stabilité dynamique de ce point fixe.
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
)
from mcq_v4.factorial_6d.h_dynamics import (  # noqa: E402
    simulate_h_only,
    solution_combined_trajectory,
    solution_combined_pointfix,
)


def test_4a_gamma_one(
    ratio_betapsi_over_gamma: float,
    h_init: float,
    h0_target: float,
    gamma: float,
    label: str,
) -> dict:
    """Test 4a-γ pour un ratio βψ/γ donné, ψ uniforme constante."""
    print(f"\n{'='*60}")
    print(f"Test 4a-γ : {label}")
    print(f"βψ/γ = {ratio_betapsi_over_gamma}, h_init = {h_init}, h0 = {h0_target}, γ = {gamma}")
    print(f"{'='*60}")

    # Choix β et ψ tels que β·ψ = ratio·γ
    beta_psi = ratio_betapsi_over_gamma * gamma
    beta = 1.0
    psi_uniform_value = beta_psi  # avec β=1, ψ = βψ

    r = gamma - beta_psi
    K_pointfix = h0_target * (1.0 - beta_psi / gamma)  # capacité effective
    h_pointfix_attendu = max(K_pointfix, 0.0)

    print(f"  r = γ - βψ = {r:.4f}")
    print(f"  K (capacité effective) = h₀·(1-βψ/γ) = {K_pointfix:.4f}")
    print(f"  h_pointfix attendu = {h_pointfix_attendu:.4f}")
    print(f"  Régime : ", end="")
    if abs(r) < 1e-9:
        print("CRITIQUE (décroissance polynomiale)")
    elif r > 0:
        print(f"SOUS-CRITIQUE STABLE (h → K = {K_pointfix:.3f})")
    else:
        print(f"SUR-CRITIQUE (h → 0 exponentiellement)")

    psi_fixed = np.full((N_AXIS, N_AXIS, N_AXIS), psi_uniform_value)
    h_initial_field = np.full((N_AXIS, N_AXIS, N_AXIS), h_init)

    state_init = State6dMinimal(
        psi=psi_fixed.copy(),
        h=h_initial_field.copy(),
        h0=h0_target,
        h_min=0.1,
    )

    # Temps caractéristique :
    # - sous-critique : 1/r (vers K à exp(-rt))
    # - critique : pas de temps exponentiel, mais 1/(γ·h_init/h₀) donne échelle initiale
    # - sur-critique : 1/|r| (vers 0 à exp(rt) avec r<0)
    if abs(r) > 1e-9:
        tau_char = 1.0 / abs(r)
    else:
        tau_char = h0_target / (gamma * h_init)  # échelle initiale critique

    # Pour le cas critique, il faut un temps beaucoup plus long pour voir
    # la décroissance polynomiale converger un peu
    t_target_factor = 5.0 if abs(r) > 1e-9 else 20.0
    t_target = t_target_factor * tau_char

    # CFL : taux maximal combiné = β·ψ + γ (taux logistique max au pire)
    rate_max = beta_psi + gamma
    dt = 0.05 / rate_max  # facteur 0.05 conservateur
    n_steps = int(np.ceil(t_target / dt))

    print(f"  τ_caractéristique = {tau_char:.4f}, t_target = {t_target:.4f}")
    print(f"  dt = {dt:.4f}, n_steps = {n_steps}")

    state_final, logs = simulate_h_only(
        state_init=state_init,
        psi_fixed=psi_fixed,
        beta=beta,
        gamma=gamma,
        h0_target=h0_target,
        n_steps=n_steps,
        dt=dt,
        include_sed=True,
        include_ero=True,
        log_every=max(1, n_steps // 30),
    )

    # Solution analytique aux temps loggés
    h_analytic = np.array([
        solution_combined_trajectory(h_init, beta, psi_uniform_value, gamma, h0_target, t)
        for t in logs["t"]
    ])
    h_engine_center = np.array(logs["h_center"])

    diff_trajectoire = float(np.max(np.abs(h_engine_center - h_analytic)))

    # Comparaison spécifique au temps final
    h_final_engine = h_engine_center[-1]
    h_final_analytic = h_analytic[-1]
    diff_finale = float(abs(h_final_engine - h_final_analytic))

    # Uniformité h
    h_uniformity_gap = max(
        b - a for a, b in zip(logs["h_min"], logs["h_max"])
    )

    h_min_atteint = min(logs["h_min"])

    # Erreur Euler théorique
    # Pour logistique, taux max = γ/4 (à h = h₀/2). Avec sédimentation,
    # taux additif β·ψ. Erreur ~ 0.5·(rate_max)²·dt²·t·h₀
    erreur_euler_attendue = 0.5 * rate_max * rate_max * dt * t_target * h0_target

    # Trace
    print(f"\n=== Trace h(t) ===")
    print(f"  {'t':>10} {'h_analytic':>12} {'h_engine':>12} {'diff':>10}")
    indices = [0, len(logs['t']) // 4, len(logs['t']) // 2,
               3 * len(logs['t']) // 4, -1]
    for idx in indices:
        if idx < 0:
            idx = len(logs['t']) + idx
        t = logs['t'][idx]
        ha = h_analytic[idx]
        he = h_engine_center[idx]
        d = abs(ha - he)
        print(f"  {t:>10.4f} {ha:>12.6f} {he:>12.6f} {d:>10.4e}")

    print(f"\n=== Résultats ===")
    print(f"diff max engine vs analytique (trajectoire) : {diff_trajectoire:.4e}")
    print(f"diff finale engine vs analytique            : {diff_finale:.4e}")
    print(f"uniformité h (gap max - min)                : {h_uniformity_gap:.4e}")
    print(f"h_min atteint                               : {h_min_atteint:.6e}")
    print(f"erreur Euler théorique attendue            : {erreur_euler_attendue:.4e}")
    print(f"h_final analytic = {h_final_analytic:.6e}")
    print(f"h_final engine   = {h_final_engine:.6e}")
    if abs(r) > 1e-9 and r > 0:
        # cas sous-critique, comparer aussi à K
        diff_to_K = abs(h_final_engine - K_pointfix)
        print(f"h_final engine vs K (point fixe sous-critique) : {diff_to_K:.4e}")

    # VERDICT
    verdict = "PASS"
    reasons = []
    seuil = max(erreur_euler_attendue * 10, 1e-8)

    if diff_trajectoire > seuil:
        verdict = "FAIL" if diff_trajectoire > seuil * 10 else "REVISION"
        reasons.append(
            f"Engine vs analytique trajectoire : diff={diff_trajectoire:.4e} > {seuil:.4e}"
        )

    if h_uniformity_gap > 1e-12:
        verdict = "FAIL" if h_uniformity_gap > 1e-9 else "REVISION"
        reasons.append(f"Uniformité h cassée : gap={h_uniformity_gap:.4e}")

    if h_init > 0 and h_min_atteint < -1e-12:
        verdict = "FAIL"
        reasons.append(f"Positivité h violée : h_min={h_min_atteint:.4e}")

    print(f"\n=== VERDICT ===")
    if verdict == "PASS":
        print(f"  PASS ✓ pour {label}")
    else:
        print(f"  Verdict : {verdict}")
        for r_msg in reasons:
            print(f"  - {r_msg}")

    return {
        "label": label,
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "ratio_betapsi_over_gamma": ratio_betapsi_over_gamma,
            "beta": beta,
            "psi_uniform": psi_uniform_value,
            "gamma": gamma,
            "h_init": h_init,
            "h0_target": h0_target,
            "r": r,
            "K_pointfix": K_pointfix,
            "dt": dt,
            "n_steps": n_steps,
            "t_target": t_target,
            "tau_char": tau_char,
        },
        "metrics": {
            "diff_engine_vs_analytic_trajectoire": diff_trajectoire,
            "diff_engine_vs_analytic_finale": diff_finale,
            "erreur_euler_attendue": float(erreur_euler_attendue),
            "h_uniformity_gap": float(h_uniformity_gap),
            "h_min_atteint": float(h_min_atteint),
            "h_final_engine": float(h_final_engine),
            "h_final_analytic": float(h_final_analytic),
        },
    }


def run_step_4a_gamma() -> dict:
    """5 configs autour de la bifurcation βψ/γ = 1."""
    h_init = 0.5
    h0_target = 1.0
    gamma = 1.0

    configs = [
        (0.1, "1_sous_critique_betapsiOverGamma_0.1"),
        (0.5, "2_sous_critique_betapsiOverGamma_0.5"),
        (0.9, "3_sous_critique_proche_critique_0.9"),
        (1.0, "4_critique_betapsi_equal_gamma"),
        (1.5, "5_sur_critique_betapsiOverGamma_1.5"),
    ]

    results = {}
    for ratio, label in configs:
        result = test_4a_gamma_one(ratio, h_init, h0_target, gamma, label)
        results[label] = result

    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*60}")
    print(f"VERDICT GLOBAL 4a-γ : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    # Tableau récapitulatif des régimes
    print(f"\n{'='*60}")
    print(f"TABLEAU RÉCAPITULATIF DES RÉGIMES")
    print(f"{'='*60}")
    print(f"{'βψ/γ':>6} {'r':>8} {'K':>8} {'h_final_engine':>15} {'diff':>10}")
    for label, r_dict in results.items():
        p = r_dict["params"]
        m = r_dict["metrics"]
        print(f"{p['ratio_betapsi_over_gamma']:>6.1f} {p['r']:>+8.3f} "
              f"{p['K_pointfix']:>+8.3f} {m['h_final_engine']:>15.4e} "
              f"{m['diff_engine_vs_analytic_trajectoire']:>10.2e}")

    return {"global_verdict": global_verdict, "cases": results}


if __name__ == "__main__":
    summary = run_step_4a_gamma()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_4a_gamma_combined.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
