"""
6d-β cycle 2 — session 1 : perturbation locale minimale.

Trace de bifurcation : voir 6d-beta-bifurcation-cycle-2.md

Question (formulée par Alex) :
«Que fait le système quand quelque chose localement cesse d'être
exactement comme avant ?»

PAS :
- mesurer "asymétrie"
- mesurer "viabilité"
- mesurer "dépopulation directionnelle"
- comparer plusieurs directions
- comparer plusieurs amplitudes
- comparer plusieurs instants

Une seule perturbation. Un seul lieu. Un seul instant.

Cas : ÉLEVÉ (β=60, A↔B2)
État perturbé : à t_perturb = 100 (après le pic ~25, durant le plateau)
Lieu : cellule centrale (2,2,2)
Amplitude : 1e-6 relatif
Observable : Dh(t) après perturbation, comparée à trajectoire non perturbée

Pas de circulation §11.4. Pas d'amendement spec. Juste un contact.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mcq_v4.factorial_6d import N_AXIS, DX, cfl_dt_max  # noqa: E402
from mcq_v4.factorial_6d.engine import (  # noqa: E402
    compute_diffusion_flux, compute_divergence,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero  # noqa: E402


def rhs_coupled(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)
    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi_dt, dh_dt


def step_engine_euler(psi, h, D, beta, gamma, h0, dt):
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi_dt, h + dt * dh_dt


def make_psi_A(sigma_0=1.8):
    coords = np.arange(N_AXIS) * DX
    center = (N_AXIS - 1) * DX / 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - center) ** 2
                      + (coords[j] - center) ** 2
                      + (coords[k] - center) ** 2)
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma_0 ** 2)
    return psi / psi.sum()


def make_psi_B2(sigma_0=1.0):
    coords = np.arange(N_AXIS) * DX
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for c in [(1.5, 2.0, 2.0), (2.5, 2.0, 2.0)]:
        cx, cy, cz = c
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r2 = ((coords[i] - cx) ** 2
                          + (coords[j] - cy) ** 2
                          + (coords[k] - cz) ** 2)
                    psi[i, j, k] += np.exp(-0.5 * r2 / sigma_0 ** 2)
    return psi / psi.sum()


def apply_perturbation(psi, i, j, k, epsilon_relative):
    """Acte expérimental exogène (§6.6) :
    perturbation locale d'une cellule de psi, suivie de renormalisation.

    Cette fonction n'est PAS dans le moteur. Elle est appliquée
    de l'extérieur, entre deux pas du moteur.
    """
    psi_pert = psi.copy()
    psi_pert[i, j, k] += epsilon_relative * psi_pert[i, j, k]
    # Renormalisation pour préserver la normalisation de psi
    psi_pert /= psi_pert.sum()
    return psi_pert


def simulate_two_branches(
    psi_A_init, psi_B_init, h_init, D, beta, gamma, h0,
    t_sim, dt, t_perturb, perturb_indices, perturb_epsilon,
    n_snapshots=120,
):
    """Simule deux branches :
    - branche 'control' : trajectoire AB sans perturbation
    - branche 'perturbed' : trajectoire AB identique jusqu'à t_perturb,
      puis B perturbé une fois, puis évolution continue

    Note : la perturbation est appliquée à B (le partenaire), pour
    rester cohérent avec le format des cycles 1 (A vs B).
    On aurait pu choisir A : ce choix est arbitraire et noté ici comme tel.
    """
    n_steps = int(np.ceil(t_sim / dt))
    step_perturb = int(np.ceil(t_perturb / dt))

    snapshot_indices = sorted(set(
        list(range(0, min(20, n_steps + 1)))
        + list(np.linspace(0, n_steps, n_snapshots, dtype=int))
        # Aussi : snapshots dense autour de la perturbation
        + list(range(max(0, step_perturb - 5),
                     min(n_steps + 1, step_perturb + 20)))
    ))
    snapshot_set = set(snapshot_indices)

    # Branche control
    psi_A_c = psi_A_init.copy()
    h_A_c = h_init.copy()
    psi_B_c = psi_B_init.copy()
    h_B_c = h_init.copy()

    # Branche perturbée
    psi_A_p = psi_A_init.copy()
    h_A_p = h_init.copy()
    psi_B_p = psi_B_init.copy()
    h_B_p = h_init.copy()

    t_list = [0.0]
    Dh_control = []
    Dh_perturbed = []
    # Initial
    norm_A = max(np.linalg.norm(h_A_c), 1e-30)
    Dh_control.append(float(np.linalg.norm(h_B_c - h_A_c) / norm_A))
    Dh_perturbed.append(float(np.linalg.norm(h_B_p - h_A_p) / norm_A))

    i_pert, j_pert, k_pert = perturb_indices

    for step in range(1, n_steps + 1):
        # Control
        psi_A_c, h_A_c = step_engine_euler(
            psi_A_c, h_A_c, D, beta, gamma, h0, dt)
        psi_B_c, h_B_c = step_engine_euler(
            psi_B_c, h_B_c, D, beta, gamma, h0, dt)

        # Perturbée : identique au control sauf au step_perturb
        psi_A_p, h_A_p = step_engine_euler(
            psi_A_p, h_A_p, D, beta, gamma, h0, dt)
        psi_B_p, h_B_p = step_engine_euler(
            psi_B_p, h_B_p, D, beta, gamma, h0, dt)
        if step == step_perturb:
            psi_B_p = apply_perturbation(
                psi_B_p, i_pert, j_pert, k_pert, perturb_epsilon)

        if step in snapshot_set:
            t = step * dt
            t_list.append(t)
            norm_A_c = max(np.linalg.norm(h_A_c), 1e-30)
            norm_A_p = max(np.linalg.norm(h_A_p), 1e-30)
            Dh_control.append(
                float(np.linalg.norm(h_B_c - h_A_c) / norm_A_c))
            Dh_perturbed.append(
                float(np.linalg.norm(h_B_p - h_A_p) / norm_A_p))

    return (np.array(t_list), np.array(Dh_control),
            np.array(Dh_perturbed), step_perturb * dt)


def run_cycle_2_session_1():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0

    t_sim = 500.0
    t_perturb = 100.0
    perturb_indices = (2, 2, 2)  # cellule centrale
    perturb_epsilon = 1e-6        # amplitude relative

    print(f"{'='*78}")
    print(f"6d-β CYCLE 2 — SESSION 1")
    print(f"  Perturbation locale minimale")
    print(f"  Cas : ÉLEVÉ (β={beta}, A↔B2)")
    print(f"  Perturbation : ψ_B[{perturb_indices}] += {perturb_epsilon} relatif")
    print(f"  Instant : t = {t_perturb}")
    print(f"  Question : que fait le système ?")
    print(f"{'='*78}")

    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_A = make_psi_A(sigma_0=1.8)
    psi_B = make_psi_B2(sigma_0=1.0)
    psi_max = max(float(psi_A.max()), float(psi_B.max()))
    rate_h = beta * psi_max + gamma
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / rate_h)
    print(f"  dt fixé : {dt:.6f}\n")

    times, Dh_control, Dh_perturbed, t_actual_perturb = simulate_two_branches(
        psi_A, psi_B, h_init, D, beta, gamma, h0,
        t_sim, dt, t_perturb, perturb_indices, perturb_epsilon,
        n_snapshots=120,
    )

    # Mesure minimale : différence entre les deux trajectoires
    diff = Dh_perturbed - Dh_control
    abs_diff = np.abs(diff)

    # Indice du moment effectif de perturbation dans les snapshots
    idx_pert = int(np.searchsorted(times, t_actual_perturb))

    print(f"  Snapshots avant perturbation effective : {idx_pert}")
    print(f"  Snapshots après perturbation effective : "
          f"{len(times) - idx_pert}")

    # Vérification : avant perturbation, les deux trajectoires doivent être identiques
    max_diff_before = float(np.max(abs_diff[:idx_pert])) if idx_pert > 0 else 0.0
    print(f"\n  Vérification (max |diff| avant perturbation) : "
          f"{max_diff_before:.2e}")
    print(f"  → doit être < 1e-15 si tout va bien")

    # Observation après perturbation
    if idx_pert < len(times):
        diff_post = abs_diff[idx_pert:]
        print(f"\n  Après perturbation :")
        print(f"    max |Dh_perturbed - Dh_control|  = {diff_post.max():.4e}")
        print(f"    |Dh_perturbed - Dh_control| final = {abs_diff[-1]:.4e}")
        print(f"    valeur de Dh_control juste après = "
              f"{Dh_control[idx_pert]:.4e}")
        print(f"    valeur de Dh_perturbed juste après = "
              f"{Dh_perturbed[idx_pert]:.4e}")

        # Affichage de quelques échantillons après perturbation
        print(f"\n  Échantillons après perturbation (jusqu'à t=200) :")
        print(f"    {'t':>10} {'Dh_control':>14} {'Dh_perturbed':>14} "
              f"{'|diff|':>12} {'diff/Dh':>12}")
        for i in range(idx_pert, min(idx_pert + 15, len(times))):
            if times[i] - t_actual_perturb < 100:
                ratio = (abs_diff[i] / max(Dh_control[i], 1e-30)
                         if Dh_control[i] > 0 else 0)
                print(f"    {times[i]:>10.2f} {Dh_control[i]:>14.4e} "
                      f"{Dh_perturbed[i]:>14.4e} {abs_diff[i]:>12.4e} "
                      f"{ratio:>12.4e}")

        # Comportement asymptotique
        print(f"\n  Comportement aux temps longs (t > 400) :")
        late_indices = [i for i, t in enumerate(times) if t > 400]
        if late_indices:
            late_diff = abs_diff[late_indices]
            print(f"    max |diff| sur t > 400 : {late_diff.max():.4e}")
            print(f"    moyenne |diff| sur t > 400 : {late_diff.mean():.4e}")
            print(f"    Dh_control moyen sur t > 400 : "
                  f"{Dh_control[late_indices].mean():.4e}")

    # Pas de A/B/C ici. Pas d'interprétation.
    # Juste les chiffres.

    results = {
        "params": {
            "beta": beta, "t_sim": t_sim, "dt": dt,
            "t_perturb": t_perturb, "t_actual_perturb": t_actual_perturb,
            "perturb_indices": list(perturb_indices),
            "perturb_epsilon": perturb_epsilon,
        },
        "times": [float(t) for t in times],
        "Dh_control": [float(x) for x in Dh_control],
        "Dh_perturbed": [float(x) for x in Dh_perturbed],
        "abs_diff": [float(x) for x in abs_diff],
        "idx_perturb_in_snapshots": idx_pert,
        "max_diff_before_perturbation": max_diff_before,
        "max_diff_after_perturbation":
            float(abs_diff[idx_pert:].max()) if idx_pert < len(times) else 0.0,
    }

    return results


if __name__ == "__main__":
    summary = run_cycle_2_session_1()
    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_1_minimal_perturbation.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats : {output_path}")
