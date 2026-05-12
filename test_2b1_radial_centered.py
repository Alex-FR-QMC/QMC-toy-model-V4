"""
Test 6d-α étape 2b-1 — diffusion avec h(θ) radial centré non séparable.

Plan validé par Alex :
- 2b-1 : h radial 3D, h(i,j,k) = 1 - a·exp(-r²/2σ²) où r = ‖(i,j,k) - centre‖
- Objectif : tester h(θ) RÉELLEMENT 3D non factorisable
- Critère central : isotropie x/y/z préservée (symétrie sous permutation)
- engine 3D vs L_3D Euler à machine precision

Différence avec 2a (h_x seul) :
- 2a : h(i,j,k) = h_x(i) → constant en j,k → symétrie partielle (y=z)
- 2b-1 : h(i,j,k) dépend de la distance euclidienne au centre →
         symétrie sous permutation x↔y↔z (mais perte de toute factorisation
         h ≠ h_x(x)·h_y(y)·h_z(z))

Profils testés :
1. a=0.0 (témoin, équivalent à 2a profil 1) — pour vérifier code de
   construction radiale
2. a=0.4, σ=1.5 : creux radial modéré (h_min au centre ≈ 0.6)
3. a=0.7, σ=1.0 : creux radial plus marqué (h_min au centre ≈ 0.3)

Critères PASS :
- conservation masse à 1e-10
- positivité à -1e-12
- engine 3D vs Euler 3D matriciel à machine precision (1e-12)
- engine 3D vs semi-discret 3D à erreur Euler théorique
- isotropie x/y/z à 1e-12 (machine precision, parce que symétrie par
  permutation de h et de ψ initial centré symétrique)
- centre de masse stable à 1e-10
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


def make_h_radial(amplitude: float, sigma_r: float, h_max: float = 1.0) -> np.ndarray:
    """
    Construit h(i,j,k) = h_max·(1 - a·exp(-r²/2σ²)) où r = distance
    euclidienne entre (i,j,k) et le centre géométrique (2,2,2).

    - amplitude = a ∈ [0, 1) : profondeur du creux central
    - sigma_r : largeur du creux
    - h_max : valeur asymptotique loin du centre (≤ h_max)

    Si a > 0 : h_min au centre, h_max aux coins.
    Si a = 0 : h_max uniforme.

    Garantit h > 0 partout.
    Symétrique sous toute permutation des axes (x↔y↔z).
    """
    N = N_AXIS
    center = np.array([2.0, 2.0, 2.0])
    h = np.zeros((N, N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                r2 = (i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2
                h[i, j, k] = h_max * (1.0 - amplitude * np.exp(-r2 / (2 * sigma_r ** 2)))
    return h


def test_2b1_one_profile(
    amplitude: float, sigma_r: float, label: str
) -> dict:
    """Test diffusion pure pour un profil h radial donné."""
    print(f"\n{'='*60}")
    print(f"Profil : {label}")
    print(f"h(r) = 1 - {amplitude}·exp(-r²/{2*sigma_r**2:.1f})")
    print(f"{'='*60}")

    # Paramètres
    D = 0.01
    sigma_0 = 1.5 * DX
    dt = 0.5
    t_target = 10.0
    n_steps = int(np.ceil(t_target / dt))

    # h(θ) radial
    h = make_h_radial(amplitude=amplitude, sigma_r=sigma_r)

    print(f"Paramètres : D={D}, σ_0={sigma_0}, dt={dt}, n_steps={n_steps}")
    print(f"\nProfil h(θ) :")
    print(f"  min(h) = {h.min():.4f} (au centre approximativement)")
    print(f"  max(h) = {h.max():.4f} (aux coins)")
    print(f"  h(2,2,2) = {h[2,2,2]:.4f} (centre exact)")
    print(f"  h(0,0,0) = {h[0,0,0]:.4f}, h(0,4,0) = {h[0,4,0]:.4f}, h(4,4,4) = {h[4,4,4]:.4f}")

    # Vérification de la symétrie de h sous permutations
    # h doit être invariant sous (i,j,k) → toute permutation
    sym_test_xy = np.max(np.abs(h - h.transpose(1, 0, 2)))
    sym_test_xz = np.max(np.abs(h - h.transpose(2, 1, 0)))
    sym_test_yz = np.max(np.abs(h - h.transpose(0, 2, 1)))
    print(f"\nSymétrie h sous permutations :")
    print(f"  max |h - h^T_xy| = {sym_test_xy:.4e}")
    print(f"  max |h - h^T_xz| = {sym_test_xz:.4e}")
    print(f"  max |h - h^T_yz| = {sym_test_yz:.4e}")

    # CFL
    h_max = h.max()
    dt_cfl_local = cfl_dt_max(h_max, D)
    if dt > dt_cfl_local:
        raise RuntimeError(f"CFL : dt={dt} > dt_max={dt_cfl_local}")
    print(f"CFL local : dt={dt} < dt_max={dt_cfl_local:.4e}")

    # État initial : gaussienne centrée (symétrique)
    state_template = make_gaussian_state(
        sigma_0=sigma_0, center=(2.0, 2.0, 2.0), h_uniform=1.0
    )
    state_init = State6dMinimal(
        psi=state_template.psi.copy(),
        h=h.copy(),
        h0=h_max,
        h_min=float(h.min()),
    )

    var_x_0, var_y_0, var_z_0 = state_init.variance_per_axis()
    print(f"\nInitialisation:")
    print(f"  Σψ = {state_init.total_mass():.10f}")
    print(f"  Var(0) = ({var_x_0:.6f}, {var_y_0:.6f}, {var_z_0:.6f})")
    print(f"  COM = {state_init.center_of_mass()}")

    # Vérification propriétés de L_3D
    L_3D = build_L3d_neumann(h)
    props = verify_L3d_properties(L_3D)
    print(f"\nPropriétés L_3D :")
    print(f"  symétrique : {props['is_symmetric']} (asym={props['max_asymmetry']:.2e})")
    print(f"  conservation : {props['rows_sum_to_zero']} (max row sum={props['max_row_sum']:.2e})")
    print(f"  M-matrix off-diag ≥ 0 : {props['off_diag_nonneg']}")
    print(f"  spectral radius : {props['spectral_radius']:.4f}")

    # Simulation
    state_final, logs = simulate(
        state_init=state_init, D_eff=D, n_steps=n_steps, dt=dt, log_every=1
    )

    # Référence Euler 3D matricielle
    psi_ref_euler = euler_discrete_3d(state_init.psi, h, D, dt, n_steps)
    # Référence semi-discrète
    psi_ref_exact = semi_discrete_3d(state_init.psi, h, D, t_target)

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

    # Isotropie x/y/z attendue parfaite (symétrie sous permutation
    # de h ET symétrie de ψ initial centré symétrique)
    aniso_xy = float(np.max(np.abs(var_x_arr - var_y_arr)))
    aniso_xz = float(np.max(np.abs(var_x_arr - var_z_arr)))
    aniso_yz = float(np.max(np.abs(var_y_arr - var_z_arr)))
    max_aniso_abs = max(aniso_xy, aniso_xz, aniso_yz)

    # Aussi en pourcentage pour lisibilité
    aniso_xy_pct = aniso_xy / max(var_x_arr[-1], 1e-12) * 100
    aniso_xz_pct = aniso_xz / max(var_x_arr[-1], 1e-12) * 100
    aniso_yz_pct = aniso_yz / max(var_y_arr[-1], 1e-12) * 100
    max_aniso_pct = max(aniso_xy_pct, aniso_xz_pct, aniso_yz_pct)

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
    print(f"  Isotropie x/y/z absolue     : {max_aniso_abs:.4e}  (attendu machine precision)")
    print(f"  Isotropie x/y/z relative    : {max_aniso_pct:.4e}%")
    print(f"  Centre de masse drift       : {com_drift:.4e}  (seuil 1e-10)")

    # Variance traces
    print(f"\n=== Variance traces ===")
    print(f"  {'t':>8} {'Var_x':>10} {'Var_y':>10} {'Var_z':>10} {'aniso_abs':>11}")
    indices_print = [0, 1, len(logs["t"]) // 4, len(logs["t"]) // 2, -1]
    for idx in indices_print:
        if idx < 0:
            idx = len(logs["t"]) + idx
        t = logs["t"][idx]
        vx = logs["var_x"][idx]
        vy = logs["var_y"][idx]
        vz = logs["var_z"][idx]
        amax = max(abs(vy - vx), abs(vz - vx), abs(vy - vz))
        print(f"  {t:>8.4f} {vx:>10.6f} {vy:>10.6f} {vz:>10.6f} {amax:>10.4e}")

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

    # Isotropie : attendue à machine precision parce que h ET ψ
    # initial sont tous deux symétriques sous permutation des axes
    if max_aniso_abs > 1e-12:
        verdict = "FAIL" if max_aniso_abs > 1e-9 else "REVISION"
        reasons.append(
            f"Isotropie x/y/z violée: max diff={max_aniso_abs:.4e} > 1e-12 "
            f"(attendue machine precision par symétrie)"
        )

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
        "params_h": {"amplitude": amplitude, "sigma_r": sigma_r},
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "D": D,
            "sigma_0": sigma_0,
            "dt": dt,
            "n_steps": n_steps,
            "t_target": t_target,
            "h_min": float(h.min()),
            "h_max": float(h.max()),
        },
        "metrics": {
            "mass_max_drift": float(mass_max_drift),
            "min_psi_global": float(min_psi_global),
            "max_aniso_absolute": float(max_aniso_abs),
            "max_aniso_relative_pct": float(max_aniso_pct),
            "com_drift": com_drift,
            "diff_engine_vs_euler_3d": diff_euler,
            "diff_engine_vs_semidiscret_3d": diff_exact,
            "spectral_radius_L3d": props["spectral_radius"],
            "var_x_final": float(var_x_arr[-1]),
            "var_y_final": float(var_y_arr[-1]),
            "var_z_final": float(var_z_arr[-1]),
            "h_symmetry_check": {
                "max_xy_xz_yz_asymmetry": float(max(sym_test_xy, sym_test_xz, sym_test_yz)),
            },
        },
        "L3d_properties": props,
    }


def run_step_2b1() -> dict:
    """Exécute les trois profils de l'étape 2b-1 (h radial centré)."""
    profiles = [
        (0.0, 1.5, "1_radial_uniforme_temoin"),
        (0.4, 1.5, "2_radial_creux_modere"),
        (0.7, 1.0, "3_radial_creux_marque"),
    ]

    results = {}
    for amplitude, sigma_r, label in profiles:
        result = test_2b1_one_profile(amplitude, sigma_r, label)
        results[label] = result
        if result["verdict"] == "FAIL":
            print(f"\n!!! ARRÊT : profil {label} en FAIL, suivants annulés.")
            break

    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*60}")
    print(f"VERDICT GLOBAL ÉTAPE 2b-1 : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    return {
        "global_verdict": global_verdict,
        "profiles": results,
    }


if __name__ == "__main__":
    summary = run_step_2b1()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_2b1_radial_centered.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
