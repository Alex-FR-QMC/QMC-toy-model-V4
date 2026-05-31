"""
6d-β cycle 2 — survie du ratio Δh/Δψ sous changement de régime.

Question :
Le ratio ~5.37 (Δh/Δψ) observé sous β=60 survit-il à un changement
minimal de régime (β=30) ?

Si oui : contrainte structurelle possible du couplage ψ↔h
Si non : effet local de régime

Protocole IDENTIQUE au test de disjonction, sauf β=30.
- Perturbation : psi[2,2,2] *= 1.01, renormalisée, à t=0
- Comparaison à trajectoire de référence
- Mesure ||Δψ|| et ||Δh|| sur 500 unités de temps
- Lecture du ratio à temps long

Une seule variation. Pas de matrice.
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


def run_test(beta):
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_total = 500.0

    psi_ref_init = make_psi_centered(sigma=1.5)
    h_ref_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi_ref_init.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_total = int(t_total / dt)
    snapshot_every = max(1, n_total // 200)

    print(f"\n{'='*70}")
    print(f"β = {beta}")
    print(f"  dt = {dt:.5f}")
    print(f"{'='*70}")

    # Référence
    times, psi_ref_traj, h_ref_traj = evolve(
        psi_ref_init.copy(), h_ref_init.copy(),
        D, beta, gamma, h0, dt, n_total, snapshot_every,
    )

    # Perturbée à l'init
    psi_pert = psi_ref_init.copy()
    psi_pert[2, 2, 2] *= 1.01
    psi_pert /= psi_pert.sum()

    _, psi_pi_traj, h_pi_traj = evolve(
        psi_pert, h_ref_init.copy(),
        D, beta, gamma, h0, dt, n_total, snapshot_every,
    )

    # Calcul des écarts
    delta_psi = np.array([
        float(np.linalg.norm(psi_pi_traj[k] - psi_ref_traj[k]))
        for k in range(len(times))
    ])
    delta_h = np.array([
        float(np.linalg.norm(h_pi_traj[k] - h_ref_traj[k]))
        for k in range(len(times))
    ])

    # État stationnaire pour référence : valeurs finales de psi et h
    psi_final = psi_ref_traj[-1]
    h_final = h_ref_traj[-1]
    print(f"\nÉtat stationnaire ref (t={times[-1]:.1f}) :")
    print(f"  ψ: min={psi_final.min():.4e}, max={psi_final.max():.4e}")
    print(f"  h: min={h_final.min():.4e}, max={h_final.max():.4e}, "
          f"mean={h_final.mean():.4e}")

    # Évolution des écarts
    print(f"\nÉvolution des écarts :")
    print(f"  {'t':>8} {'‖Δψ‖':>12} {'‖Δh‖':>12} {'ratio':>10}")
    sample_t = [0.0, 5.0, 20.0, 100.0, 200.0, 400.0, 499.0]
    for t_target in sample_t:
        idx = int(np.searchsorted(times, t_target))
        if idx >= len(times):
            continue
        ratio = (delta_h[idx] / max(delta_psi[idx], 1e-30)
                 if delta_psi[idx] > 0 else 0)
        print(f"  {times[idx]:>8.2f} {delta_psi[idx]:>12.4e} "
              f"{delta_h[idx]:>12.4e} {ratio:>10.4f}")

    # Statistiques à temps long
    late_idx = [i for i, t in enumerate(times) if t > 400]
    if late_idx:
        psi_late = float(delta_psi[late_idx].mean())
        h_late = float(delta_h[late_idx].mean())
        ratio_late = h_late / max(psi_late, 1e-30)
        print(f"\nMoyennes à t > 400 :")
        print(f"  ‖Δψ‖ = {psi_late:.4e}")
        print(f"  ‖Δh‖ = {h_late:.4e}")
        print(f"  ratio Δh/Δψ = {ratio_late:.4f}")
    else:
        psi_late = h_late = ratio_late = None

    return {
        "beta": beta,
        "dt": dt,
        "psi_stationary_max": float(psi_final.max()),
        "h_stationary_min": float(h_final.min()),
        "h_stationary_mean": float(h_final.mean()),
        "psi_late": psi_late,
        "h_late": h_late,
        "ratio_late": ratio_late,
        "times": [float(t) for t in times],
        "delta_psi": [float(x) for x in delta_psi],
        "delta_h": [float(x) for x in delta_h],
    }


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"Survie du ratio Δh/Δψ sous changement de régime")
    print(f"  Protocole : perturbation init psi[2,2,2] *= 1.01 renormalisée")
    print(f"  Référence : β=60 a donné ratio ≈ 5.37")
    print(f"{'='*70}")

    out_60 = run_test(beta=60.0)
    out_30 = run_test(beta=30.0)

    print(f"\n\n{'='*70}")
    print(f"COMPARAISON FINALE")
    print(f"{'='*70}")
    print(f"\n  β = 60 (référence)")
    print(f"    ‖Δψ‖   = {out_60['psi_late']:.4e}")
    print(f"    ‖Δh‖   = {out_60['h_late']:.4e}")
    print(f"    ratio  = {out_60['ratio_late']:.4f}")
    print(f"\n  β = 30 (test)")
    print(f"    ‖Δψ‖   = {out_30['psi_late']:.4e}")
    print(f"    ‖Δh‖   = {out_30['h_late']:.4e}")
    print(f"    ratio  = {out_30['ratio_late']:.4f}")

    if out_60['ratio_late'] and out_30['ratio_late']:
        relative_change = (
            abs(out_30['ratio_late'] - out_60['ratio_late'])
            / out_60['ratio_late']
        )
        print(f"\n  Changement relatif du ratio : "
              f"{relative_change*100:.2f}%")

        print(f"\n  Lecture :")
        if relative_change < 0.05:
            print(f"    Ratio quasi-identique (<5%)")
            print(f"    → suggère une contrainte structurelle du couplage ψ↔h")
            print(f"    → indépendant du régime testé")
        elif relative_change < 0.2:
            print(f"    Ratio modérément stable ({relative_change*100:.0f}%)")
            print(f"    → contrainte structurelle partielle")
            print(f"    → modulation du régime présente mais bornée")
        else:
            print(f"    Ratio significativement différent (>20%)")
            print(f"    → effet de régime dominant")
            print(f"    → le couplage n'impose pas un ratio universel")

    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_ratio_survival_beta30_vs_beta60.json"
    with open(output_path, "w") as f:
        json.dump({"beta_60": out_60, "beta_30": out_30}, f, indent=2)
    print(f"\nRésultats : {output_path}")
