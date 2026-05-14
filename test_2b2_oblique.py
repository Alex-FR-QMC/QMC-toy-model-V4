"""
Test 6d-α étape 2b-2 — diffusion avec h(θ) en motif oblique non séparable.

Plan validé par Alex :
- 2b-2A : vallée plane i≈j  → h(i,j,k) = 1 - a·exp(-(i-j)²/2σ²)
         engage x et y, laisse z comme témoin
- 2b-2B : diagonale principale i≈j≈k → h(i,j,k) = 1 - a·exp(-((i-j)² + (i-k)²)/2σ²)
         engage les trois axes, préserve symétrie par permutation
- 2b-2C : plan oblique asymétrique → différé à plus tard

OBJECTIFS :
- Tester une perte de séparabilité explicite (h pas factorisable)
- Tester une anisotropie structurée différente de 2a (h_x seul)
- engine 3D vs L_3D matriciel à machine precision sur motifs obliques
- COM stable si motif symétrique
- Anisotropie attendue : Var_x ≈ Var_y ≠ Var_z (Option A)
                         Var_x ≈ Var_y ≈ Var_z (Option B, par permutation)

PRUDENCE TERMINOLOGIQUE (rappel Alex) :
On parle de "perte de séparabilité" et "couplage conforme non trivial",
PAS de "cross-talk métrique fort". Le cross-talk au sens fort
(propriété 5 numerics-doc §1.8) demande comparaison h plein vs h
projeté sur marginales, ce qui n'est pas le test ici.
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


def make_h_diagonal_xy(
    amplitude: float, sigma_diag: float, h_max: float = 1.0
) -> np.ndarray:
    """
    Option A : vallée plane le long de i=j, indépendante de k.

    h(i,j,k) = h_max · (1 - a·exp(-(i-j)²/2σ²))

    Propriétés :
    - h_min sur le plan i=j (vallée)
    - constant en k (z est témoin)
    - invariant sous échange (i ↔ j) : symétrie xy
    - NON invariant sous (i ↔ k) ou (j ↔ k) : pas de symétrie xz/yz
    - NON factorisable : h(0,0,k)·h(1,1,k) = h_max²·1·1 mais h(0,1,k) = h_max·exp(-1/(2σ²)) <- pas de factorisation simple
    """
    N = N_AXIS
    h = np.zeros((N, N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                d2 = (i - j) ** 2
                h[i, j, k] = h_max * (1.0 - amplitude * np.exp(-d2 / (2 * sigma_diag ** 2)))
    return h


def make_h_diagonal_xyz(
    amplitude: float, sigma_diag: float, h_max: float = 1.0
) -> np.ndarray:
    """
    Option B : vallée le long de i=j=k (diagonale principale).

    h(i,j,k) = h_max · (1 - a·exp(-((i-j)² + (i-k)²)/2σ²))

    Propriétés :
    - h_min sur la diagonale i=j=k
    - invariant sous permutation cyclique (i,j,k) → (j,k,i) ?
      Vérifions : (j-k)² + (j-i)² = (k-j)² + (i-j)². OK invariant.
    - en fait, invariant sous toute permutation (i,j,k) par symétrie
      de la fonction (i-j)² + (i-k)² + (j-k)² (somme symétrique).
      Note : (i-j)² + (i-k)² ne contient pas (j-k)², donc pas
      strictement invariant sous tous échanges (i↔j) seul. Vérifions.

    Pour Option B strictement symétrique sous toute permutation,
    on devrait écrire : -((i-j)² + (j-k)² + (i-k)²)/(2σ²·3)
    qui est symétrique par construction.
    """
    N = N_AXIS
    h = np.zeros((N, N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Forme symétrique sous permutation des 3 axes
                d2 = (i - j) ** 2 + (j - k) ** 2 + (i - k) ** 2
                h[i, j, k] = h_max * (1.0 - amplitude * np.exp(-d2 / (2 * sigma_diag ** 2)))
    return h


def check_h_symmetries(h: np.ndarray) -> dict:
    """Vérifie les symétries de h sous les 6 permutations possibles."""
    perms = {
        "identity": (0, 1, 2),
        "swap_xy": (1, 0, 2),
        "swap_xz": (2, 1, 0),
        "swap_yz": (0, 2, 1),
        "cycle_xyz": (1, 2, 0),
        "cycle_xzy": (2, 0, 1),
    }
    return {
        name: float(np.max(np.abs(h - h.transpose(p))))
        for name, p in perms.items()
    }


def test_2b2_one_profile(
    h_builder, amplitude: float, sigma_diag: float, label: str,
    expected_symmetries: dict
) -> dict:
    """Test diffusion pour un profil h en motif oblique."""
    print(f"\n{'='*60}")
    print(f"Profil : {label}")
    print(f"Paramètres h : amplitude={amplitude}, sigma_diag={sigma_diag}")
    print(f"{'='*60}")

    # Paramètres
    D = 0.01
    sigma_0 = 1.5 * DX
    dt = 0.5
    t_target = 10.0
    n_steps = int(np.ceil(t_target / dt))

    # h(θ)
    h = h_builder(amplitude=amplitude, sigma_diag=sigma_diag)
    h_max = float(h.max())
    h_min = float(h.min())

    print(f"\nProfil h(θ) :")
    print(f"  range : [{h_min:.4f}, {h_max:.4f}]")
    print(f"  h(0,0,0)={h[0,0,0]:.4f}, h(2,2,2)={h[2,2,2]:.4f}, h(4,4,4)={h[4,4,4]:.4f}")
    print(f"  h(0,2,4)={h[0,2,4]:.4f}, h(0,4,0)={h[0,4,0]:.4f}, h(4,0,4)={h[4,0,4]:.4f}")

    # Symétries de h
    syms = check_h_symmetries(h)
    print(f"\nSymétries de h (0 = invariant sous la permutation) :")
    for name, val in syms.items():
        marker = "✓" if val < 1e-12 else "✗"
        expected = expected_symmetries.get(name, "any")
        check_expected = ""
        if expected == "invariant":
            check_expected = " (attendu invariant)" if val < 1e-12 else " (ATTENDU INVARIANT mais ne l'est pas)"
        elif expected == "non_invariant":
            check_expected = " (attendu non invariant)" if val > 1e-12 else " (attendu non invariant mais l'est)"
        print(f"  {marker} {name:>12} : {val:.4e}{check_expected}")

    # CFL
    dt_cfl_local = cfl_dt_max(h_max, D)
    if dt > dt_cfl_local:
        raise RuntimeError(f"CFL : dt={dt} > dt_max={dt_cfl_local}")

    # État initial : gaussienne centrée (symétrique sous permutation par construction)
    state_template = make_gaussian_state(
        sigma_0=sigma_0, center=(2.0, 2.0, 2.0), h_uniform=1.0
    )
    state_init = State6dMinimal(
        psi=state_template.psi.copy(),
        h=h.copy(),
        h0=h_max,
        h_min=h_min,
    )

    var_x_0, var_y_0, var_z_0 = state_init.variance_per_axis()
    print(f"\nInitialisation:")
    print(f"  Σψ = {state_init.total_mass():.10f}")
    print(f"  Var(0) = ({var_x_0:.6f}, {var_y_0:.6f}, {var_z_0:.6f})")
    print(f"  COM = {state_init.center_of_mass()}")

    # L_3D
    L_3D = build_L3d_neumann(h)
    props = verify_L3d_properties(L_3D)
    print(f"\nPropriétés L_3D :")
    print(f"  symétrique : {props['is_symmetric']} (asym={props['max_asymmetry']:.2e})")
    print(f"  conservation : {props['rows_sum_to_zero']} (max row sum={props['max_row_sum']:.2e})")
    print(f"  M-matrix : {props['off_diag_nonneg']}")
    print(f"  spectral radius : {props['spectral_radius']:.4f}")

    # Simulation
    state_final, logs = simulate(
        state_init=state_init, D_eff=D, n_steps=n_steps, dt=dt, log_every=1
    )

    # Références
    psi_ref_euler = euler_discrete_3d(state_init.psi, h, D, dt, n_steps)
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

    aniso_xy = float(np.max(np.abs(var_x_arr - var_y_arr)))
    aniso_xz = float(np.max(np.abs(var_x_arr - var_z_arr)))
    aniso_yz = float(np.max(np.abs(var_y_arr - var_z_arr)))

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
    print(f"  |Var_x - Var_y| max         : {aniso_xy:.4e}")
    print(f"  |Var_x - Var_z| max         : {aniso_xz:.4e}")
    print(f"  |Var_y - Var_z| max         : {aniso_yz:.4e}")
    print(f"  Centre de masse drift       : {com_drift:.4e}  (seuil 1e-10)")

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

    if com_drift > 1e-10:
        verdict = "FAIL" if com_drift > 1e-6 else "REVISION"
        reasons.append(f"Centre de masse drift: {com_drift:.4e}")

    if diff_euler > 1e-12:
        verdict = "FAIL" if diff_euler > 1e-9 else "REVISION"
        reasons.append(
            f"Engine vs Euler 3D matriciel: diff={diff_euler:.4e} > 1e-12"
        )

    seuil_semidiscret = erreur_euler_attendue * 10
    if diff_exact > seuil_semidiscret:
        verdict = "REVISION"
        reasons.append(
            f"Engine vs semi-discret: diff={diff_exact:.4e} > {seuil_semidiscret:.4e}"
        )

    # Vérification des anisotropies attendues
    # Pour Option A : Var_x ≈ Var_y (symétrie xy de h), Var_z différent
    # Pour Option B : Var_x ≈ Var_y ≈ Var_z (symétrie par permutation)
    aniso_expected = expected_symmetries.get("aniso_attendu", None)
    if aniso_expected == "xy_equal_z_different":
        # Var_x doit être ≈ Var_y (symétrie xy) à machine precision
        if aniso_xy > 1e-12:
            verdict = "FAIL" if aniso_xy > 1e-9 else "REVISION"
            reasons.append(
                f"Anisotropie xy attendue nulle (symétrie i↔j), mesurée {aniso_xy:.4e}"
            )
        # Var_z attendu différent de Var_x si profil non trivial
        # (on ne fait pas de test FAIL pour la différence z, on la rapporte)
    elif aniso_expected == "all_equal":
        # Var_x ≈ Var_y ≈ Var_z à machine precision
        if max(aniso_xy, aniso_xz, aniso_yz) > 1e-12:
            verdict = "FAIL" if max(aniso_xy, aniso_xz, aniso_yz) > 1e-9 else "REVISION"
            reasons.append(
                f"Anisotropie xyz attendue nulle (symétrie permutation), "
                f"max={max(aniso_xy, aniso_xz, aniso_yz):.4e}"
            )

    if verdict == "PASS":
        print(f"  PASS ✓ pour profil {label}")
        if aniso_expected == "xy_equal_z_different":
            print(f"    → Var_x = Var_y (symétrie i↔j préservée à {aniso_xy:.2e})")
            print(f"    → Var_z {'≠' if aniso_xz > 1e-10 else '='} Var_x (effet du motif sur z)")
    else:
        print(f"  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "label": label,
        "params_h": {"amplitude": amplitude, "sigma_diag": sigma_diag},
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "D": D, "sigma_0": sigma_0, "dt": dt, "n_steps": n_steps,
            "t_target": t_target, "h_min": h_min, "h_max": h_max,
        },
        "h_symmetries": syms,
        "metrics": {
            "mass_max_drift": float(mass_max_drift),
            "min_psi_global": float(min_psi_global),
            "aniso_xy_absolute": aniso_xy,
            "aniso_xz_absolute": aniso_xz,
            "aniso_yz_absolute": aniso_yz,
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


def run_step_2b2() -> dict:
    """Exécute Option A puis Option B."""

    # Option A : vallée plane i≈j, indépendante de k
    expected_a = {
        "identity": "invariant",
        "swap_xy": "invariant",  # i↔j → (i-j)² invariant, k inchangé : invariant
        "swap_xz": "non_invariant",  # i↔k change la fonction
        "swap_yz": "non_invariant",
        "cycle_xyz": "non_invariant",
        "cycle_xzy": "non_invariant",
        "aniso_attendu": "xy_equal_z_different",
    }

    # Option B : symétrique sous toute permutation
    expected_b = {
        "identity": "invariant",
        "swap_xy": "invariant",
        "swap_xz": "invariant",
        "swap_yz": "invariant",
        "cycle_xyz": "invariant",
        "cycle_xzy": "invariant",
        "aniso_attendu": "all_equal",
    }

    profiles = [
        (make_h_diagonal_xy, 0.0, 1.0, "A1_temoin_uniforme", expected_a),
        (make_h_diagonal_xy, 0.4, 1.0, "A2_vallee_ij_moderee", expected_a),
        (make_h_diagonal_xy, 0.7, 1.0, "A3_vallee_ij_marquee", expected_a),
        (make_h_diagonal_xyz, 0.4, 1.0, "B1_diag_xyz_moderee", expected_b),
        (make_h_diagonal_xyz, 0.7, 1.0, "B2_diag_xyz_marquee", expected_b),
    ]

    results = {}
    for builder, amplitude, sigma_diag, label, expected in profiles:
        result = test_2b2_one_profile(builder, amplitude, sigma_diag, label, expected)
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
    print(f"VERDICT GLOBAL ÉTAPE 2b-2 : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    return {
        "global_verdict": global_verdict,
        "profiles": results,
    }


if __name__ == "__main__":
    summary = run_step_2b2()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_2b2_oblique.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
