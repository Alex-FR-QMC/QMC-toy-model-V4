"""
6d-β cycle 2 — transition qualitative entre β=30 et β=60.

Question :
Entre β=30 (attracteur homogène dissipatif) et β=60 (attracteur
structuré conservatif), existe-t-il une transition qualitative ?

Test minimal : deux β intermédiaires (40, 50).
Pas une grille. Pas une cartographie. Deux points.

Mesures (identiques aux tests précédents) :
- Structure de l'attracteur stationnaire (min, max, mean de ψ et h)
- Comportement de la perturbation (Δψ et Δh à temps long)
- Ratio Δh/Δψ

Trois sous-intervalles : [30,40], [40,50], [50,60].
On verra où se situe la transition, si elle existe.
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


def evolve(psi, h, D, beta, gamma, h0, dt, n_steps):
    """Évolution sans snapshots — on ne garde que l'état final
    et quelques snapshots clés."""
    snapshots = {0: (psi.copy(), h.copy())}
    sample_steps = {n_steps // 4, n_steps // 2,
                    3 * n_steps // 4, n_steps}
    for k in range(1, n_steps + 1):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        if k in sample_steps:
            snapshots[k] = (psi.copy(), h.copy())
    return snapshots


def run_one_beta(beta):
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_total = 500.0

    psi_init = make_psi_centered(sigma=1.5)
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max_init = float(psi_init.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max_init + gamma))
    n_total = int(t_total / dt)

    # Référence
    snaps_ref = evolve(psi_init.copy(), h_init.copy(),
                       D, beta, gamma, h0, dt, n_total)
    psi_ref_final, h_ref_final = snaps_ref[n_total]

    # Perturbée à l'init
    psi_pert = psi_init.copy()
    psi_pert[2, 2, 2] *= 1.01
    psi_pert /= psi_pert.sum()
    snaps_pert = evolve(psi_pert, h_init.copy(),
                        D, beta, gamma, h0, dt, n_total)
    psi_pert_final, h_pert_final = snaps_pert[n_total]

    # Mesures sur attracteur (référence)
    psi_min, psi_max = float(psi_ref_final.min()), float(psi_ref_final.max())
    h_min, h_max = float(h_ref_final.min()), float(h_ref_final.max())
    h_mean = float(h_ref_final.mean())
    # Indicateur de structure : ratio max/min
    psi_inhomogeneity = psi_max / max(psi_min, 1e-30)
    h_inhomogeneity = h_max / max(h_min, 1e-30)

    # Mesures sur la perturbation
    delta_psi = float(np.linalg.norm(psi_pert_final - psi_ref_final))
    delta_h = float(np.linalg.norm(h_pert_final - h_ref_final))
    ratio = delta_h / max(delta_psi, 1e-30)

    return {
        "beta": beta,
        "dt": dt,
        "psi_attractor": {
            "min": psi_min, "max": psi_max,
            "inhomogeneity_ratio": psi_inhomogeneity,
        },
        "h_attractor": {
            "min": h_min, "max": h_max, "mean": h_mean,
            "inhomogeneity_ratio": h_inhomogeneity,
        },
        "perturbation_final": {
            "delta_psi": delta_psi,
            "delta_h": delta_h,
            "ratio_h_over_psi": ratio,
        },
    }


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"Transition entre β=30 et β=60")
    print(f"  Points : 30, 40, 50, 60")
    print(f"  Mesures à t=500 (référence + perturbée à l'init)")
    print(f"{'='*70}")

    betas = [30.0, 40.0, 50.0, 60.0]
    results = []
    for b in betas:
        print(f"\nβ = {b}...")
        r = run_one_beta(b)
        results.append(r)

    # Tableau de synthèse
    print(f"\n\n{'='*70}")
    print(f"SYNTHÈSE — Attracteur stationnaire (référence)")
    print(f"{'='*70}")
    print(f"\n  {'β':>6} {'ψ_min':>12} {'ψ_max':>12} "
          f"{'ψ_inhomo':>10} {'h_min':>14} {'h_max':>10} "
          f"{'h_inhomo':>12}")
    for r in results:
        ph = r['psi_attractor']
        hh = r['h_attractor']
        print(f"  {r['beta']:>6.1f} {ph['min']:>12.4e} {ph['max']:>12.4e} "
              f"{ph['inhomogeneity_ratio']:>10.2f} "
              f"{hh['min']:>14.4e} {hh['max']:>10.4f} "
              f"{hh['inhomogeneity_ratio']:>12.4e}")

    print(f"\n\n{'='*70}")
    print(f"SYNTHÈSE — Perturbation (à temps long)")
    print(f"{'='*70}")
    print(f"\n  {'β':>6} {'‖Δψ‖':>14} {'‖Δh‖':>14} {'ratio':>10}")
    for r in results:
        pf = r['perturbation_final']
        print(f"  {r['beta']:>6.1f} {pf['delta_psi']:>14.4e} "
              f"{pf['delta_h']:>14.4e} {pf['ratio_h_over_psi']:>10.4f}")

    # Lecture
    print(f"\n\n{'='*70}")
    print(f"LECTURE")
    print(f"{'='*70}")
    # Δψ aux quatre β :
    psis = [r['perturbation_final']['delta_psi'] for r in results]
    print(f"\n  Δψ (perturbation au temps long) :")
    for b, p in zip(betas, psis):
        order = int(np.floor(np.log10(max(p, 1e-300))))
        print(f"    β={b:.0f} : {p:.4e}  (ordre 10^{order})")

    # Identifier où se produit la transition (si elle existe)
    log_psis = [np.log10(max(p, 1e-300)) for p in psis]
    diffs = [log_psis[i+1] - log_psis[i] for i in range(3)]
    print(f"\n  Sauts en log10(Δψ) entre β successifs :")
    intervals = [(30, 40), (40, 50), (50, 60)]
    for (b1, b2), d in zip(intervals, diffs):
        print(f"    [{b1}, {b2}] : Δlog10 = {d:+.2f}")
    print(f"\n  Plus grand saut : interval "
          f"[{intervals[np.argmax(np.abs(diffs))][0]}, "
          f"{intervals[np.argmax(np.abs(diffs))][1]}]")

    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_beta_transition_30_to_60.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats : {output_path}")
