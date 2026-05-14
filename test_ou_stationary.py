"""
Test 6d-α §2.3 — Ornstein-Uhlenbeck stationnaire.

Plan validé Alex :
- Combinaison diffusion + drift en une seule étape Euler (matching
  référence matricielle L_total = D·L_diffusion + L_drift)
- h uniforme = 1 strict d'abord, h variable seulement après PASS
- Référence matricielle L_total explicite
- Distribution stationnaire engine vs vecteur propre droit de L_total
  associé à valeur propre 0
- Diagnostic non-bloquant : Var_∞ = D/k continuum

Cas testés :
1. k=0.1, D=0.1, Φ centré (2,2,2)
2. k=0.5, D=0.1, Φ centré (2,2,2)
3. k=0.1, D=0.05, Φ décentré (1,2,3)
4. k=0.5, D=0.2, Φ décentré (1,2,3)

Critères PASS :
- Σψ = 1 ± 1e-10
- min ψ ≥ -1e-12
- engine_combined vs L_total Euler à machine precision (1e-12)
- engine_combined vs semi-discret à erreur Euler théorique
- Convergence vers distribution stationnaire :
  ‖ψ_final - ψ_stat_discrete‖ < tolérance souple (10% de ψ_stat)
  pour t_target = 10/min(k, D)·dx²·...) au moins quelques τ_relaxation
- Diagnostic non-bloquant : ψ_stat_discrete vs Boltzmann ψ ∝ exp(-Φ/D)
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
    cfl_dt_max,
)
from mcq_v4.factorial_6d.engine import simulate_combined  # noqa: E402
from mcq_v4.factorial_6d.drift import (  # noqa: E402
    make_quadratic_potential,
    grad_Phi_max,
    cfl_dt_drift,
)
from mcq_v4.factorial_6d.reference_combined_3d import (  # noqa: E402
    build_L_total,
    euler_combined_3d,
    semi_discrete_combined_3d,
    stationary_distribution,
    boltzmann_distribution,
    verify_L_total_properties,
)


def test_ou_one_case(
    k: float, D: float, center_Phi: tuple[float, float, float], label: str
) -> dict:
    """Test OU pour une combinaison (k, D, center_Phi)."""
    print(f"\n{'='*60}")
    print(f"Cas : {label}")
    print(f"k = {k}, D = {D}, center_Phi = {center_Phi}")
    print(f"{'='*60}")

    sigma_0 = 1.5 * DX
    h = np.ones((N_AXIS, N_AXIS, N_AXIS))

    Phi = make_quadratic_potential(k=k, center=center_Phi)
    gp_max = grad_Phi_max(Phi)
    h_max = float(h.max())

    # CFL combinée stricte (diff + drift cumulés contribuent à diag(L_total))
    abs_diag_diff = 2.0 * D * h_max * 3 / (DX * DX)  # d=3
    abs_diag_drift = 3 * h_max * gp_max / DX
    abs_diag_total = abs_diag_diff + abs_diag_drift
    dt_cfl_combined = 0.5 / abs_diag_total
    dt = 0.9 * dt_cfl_combined

    # Temps cible : plusieurs constantes de relaxation
    # Pour OU, taux de relaxation = 2k (gaussienne stationnaire dans Φ)
    # τ_relax = 1/(2k). On vise t_target = 10 τ_relax.
    t_target = 10.0 / (2 * k)
    n_steps = int(np.ceil(t_target / dt))

    print(f"Paramètres : sigma_0={sigma_0}")
    print(f"  |diag(L_diff)|_max = {abs_diag_diff:.4f}, |diag(L_drift)|_max = {abs_diag_drift:.4f}")
    print(f"  CFL combinée = {dt_cfl_combined:.4f}")
    print(f"  dt = {dt:.4f}, n_steps = {n_steps}, t_target = {t_target:.4f}")

    state_template = make_gaussian_state(
        sigma_0=sigma_0, center=(2.0, 2.0, 2.0), h_uniform=1.0
    )
    state_init = State6dMinimal(
        psi=state_template.psi.copy(), h=h.copy(), h0=1.0, h_min=0.1
    )

    var_x_0, var_y_0, var_z_0 = state_init.variance_per_axis()
    com_x_0, com_y_0, com_z_0 = state_init.center_of_mass()
    print(f"\nInitialisation:")
    print(f"  Σψ = {state_init.total_mass():.10f}")
    print(f"  Var(0) = ({var_x_0:.4f}, {var_y_0:.4f}, {var_z_0:.4f})")
    print(f"  COM(0) = ({com_x_0:.4f}, {com_y_0:.4f}, {com_z_0:.4f})")

    L_total = build_L_total(Phi, h, D)
    props = verify_L_total_properties(L_total)
    print(f"\nPropriétés L_total :")
    print(f"  asymétrique : {not props['is_symmetric']} (asym={props['max_asymmetry']:.2e})")
    print(f"  conservation col_sums : {props['cols_sum_to_zero']} (max={props['max_col_sum']:.2e})")
    print(f"  M-matrix off-diag ≥ 0 : {props['off_diag_nonneg']}")
    print(f"  spectral radius : {props['spectral_radius']:.4f}")
    print(f"  vp max (devrait être 0) : {props['real_max_eigenvalue']:.4e}")

    # Distribution stationnaire discrète
    psi_stat, eig_res = stationary_distribution(Phi, h, D)
    psi_boltz = boltzmann_distribution(Phi, D)
    diff_stat_boltz = float(np.max(np.abs(psi_stat - psi_boltz)))
    print(f"\nDistribution stationnaire :")
    print(f"  residual vp 0 : {eig_res:.4e}")
    print(f"  diff stat_discrete vs Boltzmann continuum : {diff_stat_boltz:.4e}")

    # Variance de la distribution stationnaire discrète
    def var_per_axis_arr(psi_3d):
        coords = np.arange(N_AXIS) * DX
        psi_x = psi_3d.sum(axis=(1, 2))
        psi_y = psi_3d.sum(axis=(0, 2))
        psi_z = psi_3d.sum(axis=(0, 1))
        def v1d(p):
            m = (coords * p).sum() / p.sum()
            return float(((coords - m) ** 2 * p).sum() / p.sum())
        return v1d(psi_x), v1d(psi_y), v1d(psi_z)

    vx_stat, vy_stat, vz_stat = var_per_axis_arr(psi_stat)
    var_continuum_axe = D / k  # Var_∞ = D/k par axe continuum infini
    print(f"  Var par axe stationnaire discrète : ({vx_stat:.4f}, {vy_stat:.4f}, {vz_stat:.4f})")
    print(f"  Var par axe continuum infini D/k = {var_continuum_axe:.4f}")

    # Simulation
    state_final, logs = simulate_combined(
        state_init=state_init, Phi=Phi, D_eff=D,
        n_steps=n_steps, dt=dt,
        log_every=max(1, n_steps // 50),
    )

    # Référence matricielle
    psi_ref_euler = euler_combined_3d(state_init.psi, Phi, h, D, dt, n_steps)
    psi_ref_exact = semi_discrete_combined_3d(state_init.psi, Phi, h, D, n_steps * dt)

    diff_euler = float(np.max(np.abs(state_final.psi - psi_ref_euler)))
    diff_exact = float(np.max(np.abs(state_final.psi - psi_ref_exact)))

    print(f"\n=== Comparaison engine vs références ===")
    print(f"  diff max engine vs L_total Euler  : {diff_euler:.4e}")
    print(f"  diff max engine vs semi-discret   : {diff_exact:.4e}")

    # Distance à la stationnaire discrète
    diff_to_stat = float(np.max(np.abs(state_final.psi - psi_stat)))
    print(f"  diff max engine_final vs ψ_stat_discrete : {diff_to_stat:.4e}")

    # Invariants
    mass_max_drift = max(abs(m - 1.0) for m in logs["total_mass"])
    min_psi_global = min(logs["min_psi"])
    com_x_final = float(logs["com_x"][-1])
    com_y_final = float(logs["com_y"][-1])
    com_z_final = float(logs["com_z"][-1])
    com_distance_final = float(np.sqrt(
        (com_x_final - center_Phi[0])**2
        + (com_y_final - center_Phi[1])**2
        + (com_z_final - center_Phi[2])**2
    ))

    var_x_arr = np.array(logs["var_x"])
    var_y_arr = np.array(logs["var_y"])
    var_z_arr = np.array(logs["var_z"])
    var_x_final = float(var_x_arr[-1])
    var_y_final = float(var_y_arr[-1])
    var_z_final = float(var_z_arr[-1])

    print(f"\n=== Invariants ===")
    print(f"  Conservation masse drift : {mass_max_drift:.4e}")
    print(f"  Positivité min ψ        : {min_psi_global:.4e}")
    print(f"  Distance COM final - centre_Φ : {com_distance_final:.4f}")
    print(f"  Var final par axe : ({var_x_final:.4f}, {var_y_final:.4f}, {var_z_final:.4f})")
    print(f"  Var stationnaire discrète : ({vx_stat:.4f}, {vy_stat:.4f}, {vz_stat:.4f})")

    # Diagnostic continuum
    var_engine_axe_moy = (var_x_final + var_y_final + var_z_final) / 3
    var_stat_axe_moy = (vx_stat + vy_stat + vz_stat) / 3
    continuum_gap_pct = abs(var_engine_axe_moy - var_continuum_axe) / var_continuum_axe * 100
    discrete_var_gap_pct = abs(var_engine_axe_moy - var_stat_axe_moy) / var_stat_axe_moy * 100
    print(f"\n=== Diagnostic ===")
    print(f"  Var moyenne axe engine final     : {var_engine_axe_moy:.4f}")
    print(f"  Var moyenne axe stationnaire     : {var_stat_axe_moy:.4f}")
    print(f"  Var continuum D/k                : {var_continuum_axe:.4f}")
    print(f"  Écart engine vs stat discrète    : {discrete_var_gap_pct:.2f}%")
    print(f"  Écart engine vs continuum D/k    : {continuum_gap_pct:.2f}% (non-bloquant)")

    # VERDICT
    print(f"\n=== VERDICT ===")
    verdict = "PASS"
    reasons = []

    if mass_max_drift > 1e-10:
        verdict = "FAIL"
        reasons.append(f"Conservation: drift={mass_max_drift:.2e}")

    if min_psi_global < -1e-12:
        verdict = "FAIL"
        reasons.append(f"Positivité: min ψ={min_psi_global:.4e}")

    if diff_euler > 1e-12:
        verdict = "FAIL" if diff_euler > 1e-9 else "REVISION"
        reasons.append(f"Engine vs L_total Euler: {diff_euler:.4e}")

    erreur_euler_attendue = dt * t_target * 0.5
    seuil_semidiscret = max(erreur_euler_attendue * 10, 1e-2)
    if diff_exact > seuil_semidiscret:
        verdict = "REVISION"
        reasons.append(f"Engine vs semi-discret: {diff_exact:.4e} > {seuil_semidiscret:.4e}")

    # Convergence vers stationnaire à temps t_target = 10·τ_relax
    # Distance résiduelle attendue de l'ordre de exp(-10) ~ 1e-5 fois la
    # distance initiale. On accepte 10% comme seuil souple parce que
    # la convergence est asymptotique.
    diff_to_stat_normalised = diff_to_stat / float(psi_stat.max())
    if diff_to_stat_normalised > 0.10:
        verdict = "FAIL" if diff_to_stat_normalised > 0.5 else "REVISION"
        reasons.append(
            f"Convergence vers ψ_stat insuffisante : "
            f"diff/max_stat = {diff_to_stat_normalised:.4f} > 0.10"
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
            "k": k, "D": D, "center_Phi": list(center_Phi),
            "sigma_0": sigma_0, "dt": dt, "n_steps": n_steps,
            "t_target": t_target,
        },
        "metrics": {
            "mass_max_drift": float(mass_max_drift),
            "min_psi_global": float(min_psi_global),
            "diff_engine_vs_L_total_euler": diff_euler,
            "diff_engine_vs_semidiscret": diff_exact,
            "diff_engine_vs_stationary_discrete": diff_to_stat,
            "diff_stat_discrete_vs_boltzmann": diff_stat_boltz,
            "com_distance_final": com_distance_final,
            "var_x_final": var_x_final,
            "var_y_final": var_y_final,
            "var_z_final": var_z_final,
            "var_stationary_x": vx_stat,
            "var_stationary_y": vy_stat,
            "var_stationary_z": vz_stat,
            "var_continuum_per_axis_Doverk": var_continuum_axe,
            "continuum_var_gap_pct": float(continuum_gap_pct),
            "discrete_var_gap_pct": float(discrete_var_gap_pct),
            "eigenvalue_zero_residual": eig_res,
            "spectral_radius_L_total": props["spectral_radius"],
        },
        "L_total_properties": props,
    }


def run_step_2_3() -> dict:
    """Exécute les 4 cas OU."""
    cases_config = [
        (0.1, 0.1, (2.0, 2.0, 2.0), "1_k0.1_D0.1_centre"),
        (0.5, 0.1, (2.0, 2.0, 2.0), "2_k0.5_D0.1_centre"),
        (0.1, 0.05, (1.0, 2.0, 3.0), "3_k0.1_D0.05_decentre"),
        (0.5, 0.2, (1.0, 2.0, 3.0), "4_k0.5_D0.2_decentre"),
    ]

    results = {}
    for k, D, center, label in cases_config:
        result = test_ou_one_case(k=k, D=D, center_Phi=center, label=label)
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
    print(f"VERDICT GLOBAL §2.3 : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    return {"global_verdict": global_verdict, "cases": results}


if __name__ == "__main__":
    summary = run_step_2_3()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_ou_stationary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
