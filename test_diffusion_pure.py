"""
Test diffusion pure isotrope homogène — numerics-doc §2.1 AMENDÉ.

Conformément à la validation Alex pour étape 1 minimale de 6d-α :
- état minimal ψ[5,5,5] + h[5,5,5]
- pas de g_Ω, pas de bruit, pas de coupling, pas de drift
- h = h₀ uniforme
- diffusion pure conformal-conservative

HISTORIQUE DES VERSIONS DU TEST :
- v1 : comparaison à Var(t) = σ_0² + 2·D·t continuum infini → FAIL
  (référentiel inapproprié : grille 5×5×5 + Neumann ≠ espace infini)
- v2 (révision Alex) : référentiel = générateur Neumann discret 1D exact

POURQUOI v2 EST CORRECT, PAS UNE TRICHE :
Le critère continuum dVar/dt = 2D est valide pour un espace infini sans
bord. Sur 5×5×5 + Neumann :
- variance bornée par Var_max = 2.0 (uniforme sur 5 points centrés)
- spectre discret du Laplacien produit décroissance non-linéaire de dVar/dt
- les bords réfléchissants ralentissent la diffusion

Le bon critère discret est : le moteur 3D doit coïncider avec le
générateur Neumann discret 1D appliqué à la marginale, à machine
precision près (Euler) ou à l'erreur d'Euler théorique près (semi-discret).

Statut continuum dVar/dt=2D : diagnostic seulement, pas critère PASS/FAIL.

CRITÈRES PASS :
1. Conservation masse à 1e-10
2. Positivité à -1e-12
3. Isotropie axes à 5%
4. Centre de masse stable à 1e-10
5. Engine 3D marginale = (I+dt·D·L)^n·p_0 à 1e-12 (Euler discret exact)
6. Engine 3D marginale vs exp(D·L·t)·p_0 ≤ erreur Euler attendue *10

DIAGNOSTIC NON BLOQUANT :
- continuum_gap = |dVar_premier_step - 2D| / 2D
- convergence asymptotique vers uniforme Neumann
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
    simulate,
    cfl_dt_max,
    N_AXIS,
    DX,
)
from mcq_v4.factorial_6d.reference_neumann_1d import (  # noqa: E402
    euler_discrete_reference,
    semi_discrete_reference,
    variance_1d,
    marginal_x,
)


def test_pure_diffusion_v2() -> dict:
    """Test §2.1 v2 amendé : engine 3D vs référence Neumann discrète 1D."""
    D = 0.01
    sigma_0 = 1.5 * DX
    h0 = 1.0
    center = (2.0, 2.0, 2.0)
    dt = 0.5
    t_target = 10.0
    n_steps = int(np.ceil(t_target / dt))

    state_init = make_gaussian_state(
        sigma_0=sigma_0, center=center, h_uniform=h0
    )

    var_x_0, var_y_0, var_z_0 = state_init.variance_per_axis()
    print(f"=== Test 6d-α §2.1 v2 amendé ===")
    print(f"Paramètres : D={D}, σ_0={sigma_0}, dt={dt}, n_steps={n_steps}")
    print(f"\nInitialisation:")
    print(f"  Σψ = {state_init.total_mass():.10f}")
    print(f"  Var(0) = ({var_x_0:.6f}, {var_y_0:.6f}, {var_z_0:.6f})")
    print(f"  COM = {state_init.center_of_mass()}")

    dt_cfl = cfl_dt_max(h0, D)
    print(f"  CFL : dt={dt} < dt_max={dt_cfl:.4e}")

    # Simulation 3D
    state_final, logs = simulate(
        state_init=state_init, D_eff=D, n_steps=n_steps, dt=dt, log_every=1
    )

    # Référence Euler 1D
    psi_init_marg = marginal_x(state_init.psi)
    psi_ref_euler = euler_discrete_reference(psi_init_marg, D, dt, n_steps)

    # Référence semi-discret 1D
    psi_ref_exact = semi_discrete_reference(psi_init_marg, D, t_target)

    psi_engine_marg = marginal_x(state_final.psi)

    diff_euler = float(np.max(np.abs(psi_engine_marg - psi_ref_euler)))
    diff_exact = float(np.max(np.abs(psi_engine_marg - psi_ref_exact)))

    _, var_engine = variance_1d(psi_engine_marg)
    _, var_euler = variance_1d(psi_ref_euler)
    _, var_exact = variance_1d(psi_ref_exact)

    print(f"\n=== Comparaison engine 3D vs références Neumann 1D ===")
    print(f"  Var_x(t={t_target}) engine 3D        = {var_engine:.12f}")
    print(f"  Var_x(t={t_target}) ref Euler 1D     = {var_euler:.12f}")
    print(f"  Var_x(t={t_target}) ref semi-discret = {var_exact:.12f}")
    print(f"  diff max ψ (engine vs Euler 1D)    = {diff_euler:.4e}")
    print(f"  diff max ψ (engine vs semi-discret) = {diff_exact:.4e}")

    # Invariants
    mass_max_drift = max(abs(m - 1.0) for m in logs["total_mass"])
    min_psi_global = min(logs["min_psi"])

    var_x_arr = np.array(logs["var_x"])
    var_y_arr = np.array(logs["var_y"])
    var_z_arr = np.array(logs["var_z"])
    aniso_y_pct = np.abs(var_y_arr - var_x_arr) / np.maximum(var_x_arr, 1e-12) * 100
    aniso_z_pct = np.abs(var_z_arr - var_x_arr) / np.maximum(var_x_arr, 1e-12) * 100
    max_aniso = max(float(aniso_y_pct.max()), float(aniso_z_pct.max()))

    com_x_arr = np.array(logs["com_x"])
    com_y_arr = np.array(logs["com_y"])
    com_z_arr = np.array(logs["com_z"])
    com_drift = max(
        float(np.max(np.abs(com_x_arr - com_x_arr[0]))),
        float(np.max(np.abs(com_y_arr - com_y_arr[0]))),
        float(np.max(np.abs(com_z_arr - com_z_arr[0]))),
    )

    print(f"\n=== Invariants ===")
    print(f"  Conservation masse drift max : {mass_max_drift:.4e}  (seuil 1e-10)")
    print(f"  Positivité min ψ            : {min_psi_global:.4e}  (seuil -1e-12)")
    print(f"  Isotropie max axes          : {max_aniso:.6f}%  (seuil 5%)")
    print(f"  Centre de masse drift       : {com_drift:.4e}  (seuil 1e-10)")

    # Diagnostic continuum (non bloquant)
    dvar_dt_premier_step = (var_x_arr[1] - var_x_arr[0]) / dt
    dvar_dt_continuum = 2 * D
    continuum_gap_pct = (
        abs(dvar_dt_premier_step - dvar_dt_continuum) / dvar_dt_continuum * 100
    )
    print(f"\n=== Diagnostic continuum (NON BLOQUANT) ===")
    print(f"  dVar/dt premier step  = {dvar_dt_premier_step:.6f}")
    print(f"  2·D continuum infini  = {dvar_dt_continuum}")
    print(f"  continuum_gap         = {continuum_gap_pct:.2f}%")
    print(f"  (lecture §0ter : sur 5×5×5 Neumann, écart attendu et normal)")

    psi_uniform_target = 1.0 / (N_AXIS ** 3)
    deviation_from_uniform = float(np.max(np.abs(state_final.psi - psi_uniform_target)))
    print(f"\n  Déviation finale vs uniforme = {deviation_from_uniform:.4e}")
    print(f"  (à t={t_target}, encore loin de l'asymptote)")

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

    if max_aniso > 5.0:
        verdict = "FAIL" if max_aniso > 10.0 else "REVISION"
        reasons.append(f"Anisotropie: {max_aniso:.4f}%")

    if com_drift > 1e-10:
        verdict = "FAIL" if com_drift > 1e-6 else "REVISION"
        reasons.append(f"Centre de masse drift: {com_drift:.4e}")

    if diff_euler > 1e-12:
        verdict = "FAIL" if diff_euler > 1e-9 else "REVISION"
        reasons.append(
            f"Engine vs Euler discret 1D: diff={diff_euler:.4e} > 1e-12 "
            f"(devrait être machine precision)"
        )

    erreur_euler_attendue = dt * D * t_target
    seuil_semidiscret = erreur_euler_attendue * 10
    if diff_exact > seuil_semidiscret:
        verdict = "REVISION"
        reasons.append(
            f"Engine vs semi-discret: diff={diff_exact:.4e} > {seuil_semidiscret:.4e}"
        )

    if verdict == "PASS":
        print(f"  Tous les critères PASS ✓")
        print(f"  Moteur 6d-α étape 1 (diffusion pure conformal-conservative h=h₀)")
        print(f"  validé contre référence Neumann discrète.")
    else:
        print(f"  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "D": D,
            "sigma_0": sigma_0,
            "h0": h0,
            "dt": dt,
            "n_steps": n_steps,
            "t_target": t_target,
        },
        "metrics": {
            "mass_max_drift": float(mass_max_drift),
            "min_psi_global": float(min_psi_global),
            "max_aniso_pct": max_aniso,
            "com_drift": com_drift,
            "diff_engine_vs_euler1d": diff_euler,
            "diff_engine_vs_semidiscret": diff_exact,
            "var_engine_final": float(var_engine),
            "var_euler1d_final": float(var_euler),
            "var_semidiscret_final": float(var_exact),
            "continuum_gap_pct": float(continuum_gap_pct),
            "deviation_from_uniform": deviation_from_uniform,
            "dvar_dt_first_step": float(dvar_dt_premier_step),
            "dvar_dt_continuum": float(dvar_dt_continuum),
        },
    }


if __name__ == "__main__":
    result = test_pure_diffusion_v2()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_diffusion_pure_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if result["verdict"] == "PASS" else 1)
