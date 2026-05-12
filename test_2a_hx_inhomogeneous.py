"""
Test 6d-α étape 2a — diffusion avec h(x,y,z) = h_x(x) inhomogène.

Plan validé par Alex :
- 2a : h non-uniforme séparable en x seulement (h_x(x), constant en y,z)
- 2b : h(x,y,z) non séparable (seulement après 2a PASS)

Référence = générateur matriciel 3D L_3D(h) explicite.
Critère central : engine 3D vs (I + dt·D·L_3D)^n · ψ_0 à machine precision.

Profils testés (ordre croissant de raideur) :
1. h_x = [1, 1, 1, 1, 1] : contrôle (équivalent à étape 1)
2. h_x = [1.0, 0.8, 0.6, 0.8, 1.0] : contraction centrale douce
3. h_x = [1.0, 0.6, 0.3, 0.6, 1.0] : contraction centrale plus raide
   (seulement si 2 passe)

Critères PASS pour chaque profil :
- conservation masse à 1e-10
- positivité à -1e-12
- engine 3D vs Euler 3D matriciel à machine precision (1e-12)
- engine 3D vs semi-discret 3D à erreur Euler théorique (souple 1e-3)
- Var_y ≈ Var_z (puisque h est constant en y et z)
- Centre de masse stable si h_x symétrique (ce qui est le cas
  pour tous nos profils)
- pas de clipping h (h est fixe dans toute cette étape)

Anisotropie attendue : Var_x ≠ Var_y/z (parce que h_x crée une
résistance différente le long de x).
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
from mcq_v4.factorial_6d.reference_neumann_3d import (  # noqa: E402
    build_L3d_neumann,
    euler_discrete_3d,
    semi_discrete_3d,
    verify_L3d_properties,
)


def make_h_profile(profile_x: list[float]) -> np.ndarray:
    """
    Construit h[i,j,k] = h_x[i] (constant en j, k).

    profile_x : liste de N valeurs.
    """
    N = N_AXIS
    assert len(profile_x) == N, f"profile_x doit avoir {N} valeurs"
    h = np.zeros((N, N, N), dtype=float)
    for i in range(N):
        h[i, :, :] = profile_x[i]
    return h


def test_2a_one_profile(profile_x: list[float], label: str) -> dict:
    """
    Test diffusion pure pour un profil h_x donné.
    """
    print(f"\n{'='*60}")
    print(f"Profil : {label}")
    print(f"h_x = {profile_x}")
    print(f"{'='*60}")

    # Paramètres
    D = 0.01
    sigma_0 = 1.5 * DX
    h_max = max(profile_x)
    dt = 0.5
    t_target = 10.0
    n_steps = int(np.ceil(t_target / dt))

    # CFL
    dt_cfl = cfl_dt_max(h_max, D)
    print(f"Paramètres : D={D}, σ_0={sigma_0}, dt={dt}, n_steps={n_steps}")
    print(f"CFL : dt={dt} < dt_max={dt_cfl:.4e} (h_max={h_max})")
    if dt > dt_cfl:
        raise RuntimeError(f"CFL violation : dt={dt} > dt_max={dt_cfl}")

    # Construction h(θ) et état initial
    h = make_h_profile(profile_x)
    state_init = make_gaussian_state(
        sigma_0=sigma_0, center=(2.0, 2.0, 2.0), h_uniform=1.0
    )
    # Remplacer h par notre profil
    state_init = State6dMinimal(
        psi=state_init.psi.copy(),
        h=h.copy(),
        h0=h_max,
        h_min=min(profile_x),
    )

    print(f"\nInitialisation:")
    print(f"  Σψ = {state_init.total_mass():.10f}")
    var_x_0, var_y_0, var_z_0 = state_init.variance_per_axis()
    print(f"  Var(0) = ({var_x_0:.6f}, {var_y_0:.6f}, {var_z_0:.6f})")
    print(f"  COM = {state_init.center_of_mass()}")
    print(f"  h range : [{h.min()}, {h.max()}]")

    # Vérification propriétés de L_3D
    L_3D = build_L3d_neumann(h)
    props = verify_L3d_properties(L_3D)
    print(f"\nPropriétés L_3D :")
    print(f"  symétrique : {props['is_symmetric']} (asym={props['max_asymmetry']:.2e})")
    print(f"  conservation : {props['rows_sum_to_zero']} (max row sum={props['max_row_sum']:.2e})")
    print(f"  M-matrix off-diag ≥ 0 : {props['off_diag_nonneg']}")
    print(f"  spectral radius : {props['spectral_radius']:.4f}")

    # CFL spectral (vrai)
    dt_cfl_spectral = 2.0 / props["spectral_radius"] / D  # implicite : (I + dt·D·L) stable si dt·D·||L|| < 2
    print(f"  dt CFL spectral (2/(ρ·D)) = {dt_cfl_spectral:.4f}")
    if dt > dt_cfl_spectral:
        print(f"  ATTENTION : dt={dt} > CFL spectral={dt_cfl_spectral:.4f}")

    # Simulation
    state_final, logs = simulate(
        state_init=state_init, D_eff=D, n_steps=n_steps, dt=dt, log_every=1
    )

    # Référence Euler 3D matricielle
    psi_ref_euler = euler_discrete_3d(state_init.psi, h, D, dt, n_steps)

    # Référence semi-discrète
    psi_ref_exact = semi_discrete_3d(state_init.psi, h, D, t_target)

    # Comparaisons
    diff_euler = float(np.max(np.abs(state_final.psi - psi_ref_euler)))
    diff_exact = float(np.max(np.abs(state_final.psi - psi_ref_exact)))

    print(f"\n=== Comparaison engine vs références ===")
    print(f"  diff max engine vs Euler 3D matriciel : {diff_euler:.4e}")
    print(f"  diff max engine vs semi-discret 3D    : {diff_exact:.4e}")
    erreur_euler_attendue = dt * D * t_target
    print(f"  (erreur Euler théorique attendue ~ {erreur_euler_attendue:.4e})")

    # Invariants
    mass_max_drift = max(abs(m - 1.0) for m in logs["total_mass"])
    min_psi_global = min(logs["min_psi"])

    var_x_arr = np.array(logs["var_x"])
    var_y_arr = np.array(logs["var_y"])
    var_z_arr = np.array(logs["var_z"])

    # Anisotropie y vs z (devraient être identiques car h est constant en y,z)
    aniso_yz_pct = np.abs(var_y_arr - var_z_arr) / np.maximum(var_y_arr, 1e-12) * 100
    max_aniso_yz = float(aniso_yz_pct.max())

    # Var_x vs Var_y (attendu : différent puisque h_x non uniforme)
    diff_xy_final = float(abs(var_x_arr[-1] - var_y_arr[-1]))
    relative_xy_final = diff_xy_final / max(var_y_arr[-1], 1e-12) * 100

    # Centre de masse
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
    print(f"  Anisotropie y/z (devrait être 0) : {max_aniso_yz:.4f}%  (seuil 5%)")
    print(f"  Centre de masse drift       : {com_drift:.4e}  (seuil 1e-10)")
    print(f"  Anisotropie x vs y au temps final : {relative_xy_final:.4f}%")
    print(f"    (attendue non nulle pour profil non-uniforme,")
    print(f"     attendue nulle pour profil uniforme)")

    # Variance traces
    print(f"\n=== Variance traces ===")
    print(f"  {'t':>8} {'Var_x':>10} {'Var_y':>10} {'Var_z':>10}")
    indices_print = [0, 1, len(logs["t"]) // 4, len(logs["t"]) // 2, -1]
    for idx in indices_print:
        if idx < 0:
            idx = len(logs["t"]) + idx
        t = logs["t"][idx]
        vx = logs["var_x"][idx]
        vy = logs["var_y"][idx]
        vz = logs["var_z"][idx]
        print(f"  {t:>8.4f} {vx:>10.6f} {vy:>10.6f} {vz:>10.6f}")

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

    if max_aniso_yz > 5.0:
        verdict = "FAIL" if max_aniso_yz > 10.0 else "REVISION"
        reasons.append(f"Anisotropie y/z (devrait être 0): {max_aniso_yz:.4f}%")

    if com_drift > 1e-10:
        verdict = "FAIL" if com_drift > 1e-6 else "REVISION"
        reasons.append(f"Centre de masse drift: {com_drift:.4e}")

    if diff_euler > 1e-12:
        verdict = "FAIL" if diff_euler > 1e-9 else "REVISION"
        reasons.append(
            f"Engine vs Euler 3D matriciel: diff={diff_euler:.4e} > 1e-12 "
            f"(devrait être machine precision)"
        )

    seuil_semidiscret = erreur_euler_attendue * 10
    if diff_exact > seuil_semidiscret:
        verdict = "REVISION"
        reasons.append(
            f"Engine vs semi-discret: diff={diff_exact:.4e} > {seuil_semidiscret:.4e}"
        )

    if verdict == "PASS":
        print(f"  PASS ✓ pour profil {label}")
    else:
        print(f"  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "label": label,
        "profile_x": profile_x,
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "D": D,
            "sigma_0": sigma_0,
            "dt": dt,
            "n_steps": n_steps,
            "t_target": t_target,
            "h_min": min(profile_x),
            "h_max": max(profile_x),
        },
        "metrics": {
            "mass_max_drift": float(mass_max_drift),
            "min_psi_global": float(min_psi_global),
            "max_aniso_yz_pct": max_aniso_yz,
            "relative_xy_final_pct": relative_xy_final,
            "com_drift": com_drift,
            "diff_engine_vs_euler_3d": diff_euler,
            "diff_engine_vs_semidiscret_3d": diff_exact,
            "spectral_radius_L3d": props["spectral_radius"],
            "var_x_final": float(var_x_arr[-1]),
            "var_y_final": float(var_y_arr[-1]),
            "var_z_final": float(var_z_arr[-1]),
        },
        "L3d_properties": props,
    }


def run_step_2a() -> dict:
    """Exécute les trois profils de l'étape 2a."""
    profiles = [
        ([1.0, 1.0, 1.0, 1.0, 1.0], "1_uniforme_controle"),
        ([1.0, 0.8, 0.6, 0.8, 1.0], "2_contraction_douce"),
        ([1.0, 0.6, 0.3, 0.6, 1.0], "3_contraction_raide"),
    ]

    results = {}
    for profile, label in profiles:
        result = test_2a_one_profile(profile, label)
        results[label] = result
        # Stop si fail strict
        if result["verdict"] == "FAIL":
            print(f"\n!!! ARRÊT : profil {label} en FAIL, profils suivants annulés.")
            break

    # Verdict global
    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*60}")
    print(f"VERDICT GLOBAL ÉTAPE 2a : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    return {
        "global_verdict": global_verdict,
        "profiles": results,
    }


if __name__ == "__main__":
    summary = run_step_2a()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_2a_hx_inhomogeneous.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
