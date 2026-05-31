"""
6d-β cycle 2 — disjonction timing vs localisation.

Question :
Le facteur ~2.6 (perturbée vs soeurs) provient-il principalement
du timing (perturbation post-stabilisation) ou principalement de
la localisation spatiale (perturbation concentrée sur 1 cellule) ?

Test miroir :
- Perturbation IDENTIQUE : psi[2,2,2] *= 1.01, renormalisée
- Mais injectée à t=0 au lieu de t=200
- Suivie ensuite de l'évolution complète

Comparaison avec :
- Référence : init standard, évoluée
- Soeurs : init micro-variée 1% distribuée

Si le facteur ~2.6 vient du timing :
  perturbée-init devrait se rapprocher des soeurs (perte du facteur)
Si le facteur vient de la localisation :
  perturbée-init devrait conserver le facteur ~2.6

Une seule disjonction. Pas de matrice.
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


def rhs(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi = compute_divergence(Jx, Jy, Jz)
    dh = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi, dh


def step(psi, h, D, beta, gamma, h0, dt):
    dpsi, dh = rhs(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi, h + dt * dh


def make_psi_centered(sigma=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - c)**2 + (coords[j] - c)**2 +
                      (coords[k] - c)**2)
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()


def evolve(psi, h, D, beta, gamma, h0, dt, n_steps, snapshot_every):
    psi_list = [psi.copy()]
    h_list = [h.copy()]
    t_list = [0.0]
    for k in range(1, n_steps + 1):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        if k % snapshot_every == 0:
            psi_list.append(psi.copy())
            h_list.append(h.copy())
            t_list.append(k * dt)
    return np.array(t_list), psi_list, h_list


def run_test():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0
    t_total = 500.0  # stab + obs cumulés

    psi_ref_init = make_psi_centered(sigma=1.5)
    h_ref_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi_ref_init.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_total = int(t_total / dt)
    snapshot_every = max(1, n_total // 200)

    print(f"{'='*70}")
    print(f"Disjonction timing vs localisation")
    print(f"  β={beta}, dt={dt:.5f}, t_total={t_total}")
    print(f"{'='*70}\n")

    # === Référence : init standard, évolution complète ===
    print("Trajectoire de référence...")
    times, psi_ref_traj, h_ref_traj = evolve(
        psi_ref_init.copy(), h_ref_init.copy(),
        D, beta, gamma, h0, dt, n_total, snapshot_every,
    )

    # === Perturbée-INIT : perturbation injectée à t=0 ===
    print("Trajectoire perturbée-à-l'initialisation...")
    psi_pert_init = psi_ref_init.copy()
    psi_pert_init[2, 2, 2] *= 1.01
    psi_pert_init /= psi_pert_init.sum()
    delta_psi_init = float(np.linalg.norm(psi_pert_init - psi_ref_init))
    print(f"  ‖Δψ‖ à l'init = {delta_psi_init:.4e}")

    _, psi_pi_traj, h_pi_traj = evolve(
        psi_pert_init, h_ref_init.copy(),
        D, beta, gamma, h0, dt, n_total, snapshot_every,
    )

    # Distances perturbée-init vs ref
    delta_psi_pi = []
    delta_h_pi = []
    for k in range(len(times)):
        delta_psi_pi.append(
            float(np.linalg.norm(psi_pi_traj[k] - psi_ref_traj[k]))
        )
        delta_h_pi.append(
            float(np.linalg.norm(h_pi_traj[k] - h_ref_traj[k]))
        )
    delta_psi_pi = np.array(delta_psi_pi)
    delta_h_pi = np.array(delta_h_pi)

    # === Affichage : trajectoire complète des écarts ===
    print(f"\nÉvolution des écarts perturbée-init vs ref :")
    print(f"  {'t':>8} {'‖Δψ‖':>12} {'‖Δh‖':>12} {'ratio Δh/Δψ':>12}")
    sample_t = [0.0, 5.0, 20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 499.0]
    for t_target in sample_t:
        idx = int(np.searchsorted(times, t_target))
        if idx >= len(times):
            continue
        ratio = (delta_h_pi[idx] / max(delta_psi_pi[idx], 1e-30)
                 if delta_psi_pi[idx] > 0 else 0)
        print(f"  {times[idx]:>8.2f} {delta_psi_pi[idx]:>12.4e} "
              f"{delta_h_pi[idx]:>12.4e} {ratio:>12.4e}")

    # === Comparaison avec les valeurs perturbée-POST des tests précédents ===
    # Du test précédent (geometric_reformulation) :
    #   perturbée-POST : ‖Δψ‖ = 2.0007e-4, ‖Δh‖ = 1.0737e-3 (saturé)
    #   soeurs (mean)  : ‖Δψ‖ = 7.6e-5,   ‖Δh‖ = 4.2e-4 (saturé)
    #   ratio pert/soeurs ≈ 2.62

    print(f"\n{'─'*70}")
    print(f"Comparaison aux résultats antérieurs (à temps long, t > 400) :")
    print(f"{'─'*70}")
    late_idx = [i for i, t in enumerate(times) if t > 400]
    if late_idx:
        pi_psi_late = float(delta_psi_pi[late_idx].mean())
        pi_h_late = float(delta_h_pi[late_idx].mean())
        print(f"\n  Perturbée-INIT (ce test) à t>400 :")
        print(f"    ‖Δψ‖ moyen = {pi_psi_late:.4e}")
        print(f"    ‖Δh‖ moyen = {pi_h_late:.4e}")
        print(f"\n  Référence externe (tests précédents) :")
        print(f"    Perturbée-POST : ‖Δψ‖ = 2.00e-04, ‖Δh‖ = 1.07e-03")
        print(f"    Soeurs (mean)  : ‖Δψ‖ = 7.64e-05, ‖Δh‖ = 4.25e-04")

        # Le test de disjonction
        ratio_pi_vs_post_psi = pi_psi_late / 2.0007e-4
        ratio_pi_vs_post_h = pi_h_late / 1.0737e-3
        ratio_pi_vs_sisters_psi = pi_psi_late / 7.6358e-5
        ratio_pi_vs_sisters_h = pi_h_late / 4.2453e-4

        print(f"\n  Position de perturbée-INIT :")
        print(f"    relative à perturbée-POST :")
        print(f"      ratio Δψ : {ratio_pi_vs_post_psi:.3f}")
        print(f"      ratio Δh : {ratio_pi_vs_post_h:.3f}")
        print(f"    relative aux soeurs (mean) :")
        print(f"      ratio Δψ : {ratio_pi_vs_sisters_psi:.3f}")
        print(f"      ratio Δh : {ratio_pi_vs_sisters_h:.3f}")

        # Verdict provisoire
        print(f"\n  Lecture de la disjonction :")
        if (abs(ratio_pi_vs_post_psi - 1.0) < 0.2 and
                abs(ratio_pi_vs_post_h - 1.0) < 0.2):
            print(f"    Perturbée-INIT ≈ Perturbée-POST")
            print(f"    → facteur ~2.6 vient principalement de la "
                  f"LOCALISATION (concentration spatiale)")
            print(f"    → le timing post-stabilisation n'ajoute pas d'effet")
        elif (abs(ratio_pi_vs_sisters_psi - 1.0) < 0.5 and
                abs(ratio_pi_vs_sisters_h - 1.0) < 0.5):
            print(f"    Perturbée-INIT ≈ Soeurs")
            print(f"    → facteur ~2.6 vient principalement du TIMING")
            print(f"    → perturber post-stabilisation est qualitativement"
                  f" différent")
        else:
            print(f"    Perturbée-INIT entre les deux références")
            print(f"    → effet mixte timing + localisation")
            print(f"    → la disjonction ne tranche pas nettement")

    return {
        "params": {
            "beta": beta, "D": D, "gamma": gamma, "h0": h0,
            "dt": dt, "t_total": t_total,
        },
        "times": [float(t) for t in times],
        "delta_psi_pi": [float(x) for x in delta_psi_pi],
        "delta_h_pi": [float(x) for x in delta_h_pi],
        "reference_perturbed_post_psi": 2.0007e-4,
        "reference_perturbed_post_h": 1.0737e-3,
        "reference_sisters_mean_psi": 7.6358e-5,
        "reference_sisters_mean_h": 4.2453e-4,
    }


if __name__ == "__main__":
    out = run_test()
    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_disjonction_timing_vs_localisation.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
