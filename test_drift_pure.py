"""
Test 6d-α §2.2 — drift pur dans potentiel quadratique.

Plan validé Alex :
- Convention de signe : Option 1 (descente dans Φ → masse converge
  vers minimum de Φ)
- Pré-requis bloquant : micro-test de signe PASS
- k ∈ {0.1, 0.5}
- Φ centré (2,2,2) puis décentré (1, 2, 3)
- h uniforme = 1 strict
- Référence matricielle L_drift_3D explicite

Critères PASS :
- micro-test de signe PASS (drift attractif)
- Σψ = 1 ± 1e-10
- min ψ ≥ -1e-12
- engine vs L_drift Euler à machine precision (1e-12)
- engine vs semi-discret à erreur Euler théorique
- COM converge vers θ_0 (drift attractif effectif)
- Var décroît au cours du temps (concentration sur θ_0)
- (continuum exponential decay Var = Var_0 · exp(-2kt) :
  diagnostic non-bloquant)
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
from mcq_v4.factorial_6d.drift import (  # noqa: E402
    make_quadratic_potential,
    simulate_drift,
    grad_Phi_max,
    cfl_dt_drift,
)
from mcq_v4.factorial_6d.reference_drift_3d import (  # noqa: E402
    build_L_drift_3d,
    euler_drift_3d,
    semi_discrete_drift_3d,
    verify_L_drift_properties,
)


def micro_test_signe(k: float = 0.1) -> dict:
    """
    Micro-test de signe préalable : ψ concentrée à x=3,
    θ_0 = (2,2,2). COM_x doit DIMINUER après 1 step.

    Bloquant : si échec, ne pas lancer les tests matriciels.
    """
    print("=== MICRO-TEST DE SIGNE ===")
    N = N_AXIS
    psi = np.zeros((N, N, N))
    psi[3, 2, 2] = 1.0
    h = np.ones((N, N, N))

    state_init = State6dMinimal(psi=psi.copy(), h=h.copy(), h0=1.0, h_min=0.1)
    Phi = make_quadratic_potential(k=k, center=(2.0, 2.0, 2.0))

    gp_max = grad_Phi_max(Phi)
    dt_cfl = cfl_dt_drift(1.0, gp_max)
    dt = 0.5 * dt_cfl

    state_after, logs = simulate_drift(state_init, Phi, n_steps=1, dt=dt, log_every=1)

    cx_before = logs["com_x"][0]
    cx_after = logs["com_x"][-1]
    delta = cx_after - cx_before

    print(f"  ψ initial : delta à (3, 2, 2)")
    print(f"  Φ : ½·{k}·‖θ-(2,2,2)‖²")
    print(f"  dt = {dt}")
    print(f"  COM_x avant : {cx_before:.6f}")
    print(f"  COM_x après : {cx_after:.6f}")
    print(f"  Δ COM_x     : {delta:+.6f}")

    if delta < -1e-9:
        verdict = "PASS"
        print(f"  ✓ PASS : drift attractif vers centre")
    elif delta > 1e-9:
        verdict = "FAIL"
        print(f"  ✗ FAIL : drift RÉPULSIF (signe inversé)")
    else:
        verdict = "FAIL"
        print(f"  ✗ FAIL : pas de mouvement détectable")

    return {
        "verdict": verdict,
        "delta_com_x": float(delta),
    }


def test_drift_one_case(
    k: float,
    center_Phi: tuple[float, float, float],
    label: str,
) -> dict:
    """
    Test diffusion avec un potentiel quadratique donné.
    """
    print(f"\n{'='*60}")
    print(f"Cas : {label}")
    print(f"k = {k}, center_Phi = {center_Phi}")
    print(f"{'='*60}")

    # Paramètres
    sigma_0 = 1.5 * DX
    h = np.ones((N_AXIS, N_AXIS, N_AXIS))

    # Potentiel
    Phi = make_quadratic_potential(k=k, center=center_Phi)
    gp_max = grad_Phi_max(Phi)
    dt_cfl = cfl_dt_drift(1.0, gp_max)
    dt = 0.5 * dt_cfl

    # Temps de simulation : on veut convergence visible
    # τ relaxation continuum ≈ 1/k. On simule t_target ≈ 5·τ = 5/k
    t_target = 5.0 / k
    n_steps = int(np.ceil(t_target / dt))

    print(f"Paramètres : k={k}, sigma_0={sigma_0}")
    print(f"  grad_Phi_max = {gp_max:.4f}, dt_cfl = {dt_cfl:.4f}")
    print(f"  dt = {dt:.4f}, n_steps = {n_steps}, t_target = {t_target:.4f}")

    # État initial : gaussienne centrée sur le centre géométrique (2,2,2)
    # PAS sur le centre de Φ — on veut voir la convergence
    state_template = make_gaussian_state(
        sigma_0=sigma_0, center=(2.0, 2.0, 2.0), h_uniform=1.0
    )
    state_init = State6dMinimal(
        psi=state_template.psi.copy(),
        h=h.copy(),
        h0=1.0,
        h_min=0.1,
    )

    var_x_0, var_y_0, var_z_0 = state_init.variance_per_axis()
    com_x_0, com_y_0, com_z_0 = state_init.center_of_mass()
    print(f"\nInitialisation:")
    print(f"  Σψ = {state_init.total_mass():.10f}")
    print(f"  Var(0) = ({var_x_0:.4f}, {var_y_0:.4f}, {var_z_0:.4f})")
    print(f"  COM(0) = ({com_x_0:.4f}, {com_y_0:.4f}, {com_z_0:.4f})")

    # Propriétés L_drift
    L_drift = build_L_drift_3d(Phi, h)
    props = verify_L_drift_properties(L_drift)
    print(f"\nPropriétés L_drift_3D :")
    print(f"  asymétrique : {not props['is_symmetric']} (asym={props['max_asymmetry']:.2e})")
    print(f"  conservation (col sums = 0) : {props['cols_sum_to_zero']} (max={props['max_col_sum']:.2e})")
    print(f"  M-matrix off-diag ≥ 0 : {props['off_diag_nonneg']}")
    print(f"  spectral radius : {props['spectral_radius']:.4f}")
    print(f"  valeur propre max (devrait être 0) : {props['real_max_eigenvalue']:.4e}")

    # Simulation
    state_final, logs = simulate_drift(
        state_init=state_init, Phi=Phi, n_steps=n_steps, dt=dt, log_every=max(1, n_steps // 50)
    )

    # Référence Euler 3D matricielle
    psi_ref_euler = euler_drift_3d(state_init.psi, Phi, h, dt, n_steps)
    # Référence semi-discrète (coûteux pour grand t)
    psi_ref_exact = semi_discrete_drift_3d(state_init.psi, Phi, h, n_steps * dt)

    diff_euler = float(np.max(np.abs(state_final.psi - psi_ref_euler)))
    diff_exact = float(np.max(np.abs(state_final.psi - psi_ref_exact)))

    print(f"\n=== Comparaison engine vs références ===")
    print(f"  diff max engine vs L_drift Euler : {diff_euler:.4e}")
    print(f"  diff max engine vs semi-discret  : {diff_exact:.4e}")

    # Invariants
    mass_max_drift = max(abs(m - 1.0) for m in logs["total_mass"])
    min_psi_global = min(logs["min_psi"])

    # COM convergence vers center_Phi
    com_x_arr = np.array(logs["com_x"])
    com_y_arr = np.array(logs["com_y"])
    com_z_arr = np.array(logs["com_z"])
    com_x_final = com_x_arr[-1]
    com_y_final = com_y_arr[-1]
    com_z_final = com_z_arr[-1]
    com_distance_final = float(np.sqrt(
        (com_x_final - center_Phi[0])**2
        + (com_y_final - center_Phi[1])**2
        + (com_z_final - center_Phi[2])**2
    ))
    com_distance_initial = float(np.sqrt(
        (com_x_0 - center_Phi[0])**2
        + (com_y_0 - center_Phi[1])**2
        + (com_z_0 - center_Phi[2])**2
    ))
    com_convergence_ratio = (
        com_distance_final / com_distance_initial
        if com_distance_initial > 1e-9 else None
    )

    print(f"\n=== Convergence COM vers center_Phi {center_Phi} ===")
    print(f"  Distance initiale : {com_distance_initial:.4f}")
    print(f"  Distance finale   : {com_distance_final:.4f}")
    if com_convergence_ratio is not None:
        print(f"  Ratio final/initial : {com_convergence_ratio:.4f}")

    # Var décroissance
    var_x_arr = np.array(logs["var_x"])
    var_y_arr = np.array(logs["var_y"])
    var_z_arr = np.array(logs["var_z"])
    var_total_arr = var_x_arr + var_y_arr + var_z_arr
    var_decrease = var_total_arr[-1] < var_total_arr[0]
    var_ratio = float(var_total_arr[-1] / var_total_arr[0])

    print(f"\n=== Décroissance variance totale ===")
    print(f"  Var_total(0) = {var_total_arr[0]:.6f}")
    print(f"  Var_total(t_final) = {var_total_arr[-1]:.6f}")
    print(f"  ratio = {var_ratio:.4f} (doit être < 1)")

    # Variance traces
    print(f"\n=== Var trace (échantillons) ===")
    print(f"  {'t':>8} {'Var_x':>10} {'Var_y':>10} {'Var_z':>10} {'COM_x':>8} {'COM_y':>8} {'COM_z':>8}")
    indices_print = [0, len(logs["t"]) // 4, len(logs["t"]) // 2, 3 * len(logs["t"]) // 4, -1]
    for idx in indices_print:
        if idx < 0:
            idx = len(logs["t"]) + idx
        t = logs["t"][idx]
        vx = logs["var_x"][idx]
        vy = logs["var_y"][idx]
        vz = logs["var_z"][idx]
        cx = logs["com_x"][idx]
        cy = logs["com_y"][idx]
        cz = logs["com_z"][idx]
        print(f"  {t:>8.4f} {vx:>10.6f} {vy:>10.6f} {vz:>10.6f} {cx:>8.4f} {cy:>8.4f} {cz:>8.4f}")

    # Diagnostic continuum (non bloquant)
    # Var(t) = Var_0 · exp(-2kt) en continuum infini
    var_continuum_final = var_total_arr[0] * np.exp(-2 * k * t_target)
    continuum_var_gap = (
        abs(var_total_arr[-1] - var_continuum_final) / var_continuum_final * 100
        if var_continuum_final > 1e-12 else float("nan")
    )
    print(f"\n=== Diagnostic continuum (NON BLOQUANT) ===")
    print(f"  Var_total(t_target) continuum infini = {var_continuum_final:.6f}")
    print(f"  Var_total(t_target) engine          = {var_total_arr[-1]:.6f}")
    print(f"  continuum_var_gap = {continuum_var_gap:.2f}%")
    print(f"  (sur grille Neumann discrète, écart attendu)")

    # VERDICT
    print(f"\n=== VERDICT ===")
    verdict = "PASS"
    reasons = []

    if mass_max_drift > 1e-10:
        verdict = "FAIL"
        reasons.append(f"Conservation masse: drift={mass_max_drift:.2e}")

    if min_psi_global < -1e-12:
        verdict = "FAIL"
        reasons.append(f"Positivité: min ψ={min_psi_global:.4e}")

    if diff_euler > 1e-12:
        verdict = "FAIL" if diff_euler > 1e-9 else "REVISION"
        reasons.append(
            f"Engine vs L_drift Euler: diff={diff_euler:.4e} > 1e-12"
        )

    # Tolérance souple pour semi-discret (erreur Euler attendue O(dt·t))
    erreur_euler_attendue = dt * t_target * 0.5  # ~ dt·spectral_radius·t
    seuil_semidiscret = max(erreur_euler_attendue * 10, 1e-2)
    if diff_exact > seuil_semidiscret:
        verdict = "REVISION"
        reasons.append(
            f"Engine vs semi-discret: diff={diff_exact:.4e} > {seuil_semidiscret:.4e}"
        )

    if not var_decrease:
        verdict = "FAIL"
        reasons.append(f"Variance ne décroît pas : ratio={var_ratio:.4f}")

    if com_convergence_ratio is not None and com_convergence_ratio > 0.5:
        # Si distance finale > 50% distance initiale, convergence insuffisante
        verdict = "FAIL" if com_convergence_ratio > 0.8 else "REVISION"
        reasons.append(
            f"Convergence COM insuffisante : ratio={com_convergence_ratio:.4f}"
        )

    if verdict == "PASS":
        print(f"  PASS ✓ pour cas {label}")
    else:
        print(f"  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "label": label,
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "k": k, "center_Phi": list(center_Phi),
            "sigma_0": sigma_0, "dt": dt, "n_steps": n_steps,
            "t_target": t_target,
        },
        "metrics": {
            "mass_max_drift": float(mass_max_drift),
            "min_psi_global": float(min_psi_global),
            "diff_engine_vs_L_drift_euler": diff_euler,
            "diff_engine_vs_semidiscret": diff_exact,
            "com_distance_initial": com_distance_initial,
            "com_distance_final": com_distance_final,
            "com_convergence_ratio": com_convergence_ratio,
            "var_total_initial": float(var_total_arr[0]),
            "var_total_final": float(var_total_arr[-1]),
            "var_ratio": var_ratio,
            "continuum_var_gap_pct": float(continuum_var_gap),
            "spectral_radius_L_drift": props["spectral_radius"],
        },
        "L_drift_properties": props,
    }


def run_step_2_2() -> dict:
    """Exécute le micro-test de signe puis les 4 cas."""
    # Micro-test de signe (bloquant)
    signe_result = micro_test_signe(k=0.1)
    if signe_result["verdict"] != "PASS":
        print("\n\n!!! ARRÊT : micro-test de signe en FAIL, tests matriciels annulés.")
        return {
            "global_verdict": "FAIL",
            "micro_test_signe": signe_result,
            "cases": {},
        }

    # 4 cas
    cases_config = [
        (0.1, (2.0, 2.0, 2.0), "1_k0.1_centre"),
        (0.5, (2.0, 2.0, 2.0), "2_k0.5_centre"),
        (0.1, (1.0, 2.0, 3.0), "3_k0.1_decentre"),
        (0.5, (1.0, 2.0, 3.0), "4_k0.5_decentre"),
    ]

    results = {}
    for k, center, label in cases_config:
        result = test_drift_one_case(k=k, center_Phi=center, label=label)
        results[label] = result
        if result["verdict"] == "FAIL":
            print(f"\n!!! ARRÊT : cas {label} en FAIL, suivants annulés.")
            break

    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*60}")
    print(f"VERDICT GLOBAL §2.2 : {global_verdict}")
    print(f"{'='*60}")
    print(f"  Micro-test signe : {signe_result['verdict']}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    return {
        "global_verdict": global_verdict,
        "micro_test_signe": signe_result,
        "cases": results,
    }


if __name__ == "__main__":
    summary = run_step_2_2()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_drift_pure.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
