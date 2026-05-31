"""
6d-β cycle 2 — universalité vs héritage de l'attracteur β=60.

Question (formulée par Alex) :
Le régime structuré β=60 est-il universel ou géométriquement hérité ?

Test minimal :
- β = 60 fixé
- Masse totale ψ = 1 conservée pour toutes les inits
- 3 géométries initiales différentes :
  (G1) gaussienne large    : sigma=2.5
  (G2) gaussienne étroite  : sigma=0.8
  (G3) double pic          : deux gaussiennes sigma=0.7 décentrées
- Comparaison aux résultats antérieurs : gaussienne sigma=1.5 (la "référence")

Mesures :
- Attracteur final : ψ_min/max, h_min/max, structure spatiale
- Position de h_argmin
- Si on perturbe à l'init : Δψ et Δh à temps long, ratio Δh/Δψ

Cible :
- si tous convergent vers le même attracteur structuré universel
  → régime structuré est intrinsèque au système
- si chaque init produit un attracteur différent
  → régime structuré est conditionné par la géométrie initiale
  → "Γ_meta", héritage morphologique
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


def make_psi_gaussian(sigma, center=None):
    coords = np.arange(N_AXIS) * DX
    if center is None:
        center = ((N_AXIS - 1) * DX / 2.0,) * 3
    cx, cy, cz = center
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - cx)**2 + (coords[j] - cy)**2 +
                      (coords[k] - cz)**2)
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()


def make_psi_double_peak(sigma=0.7):
    coords = np.arange(N_AXIS) * DX
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    # Deux pics symétriques le long de l'axe x
    centers = [(1.0, 2.0, 2.0), (3.0, 2.0, 2.0)]
    for cx, cy, cz in centers:
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r2 = ((coords[i] - cx)**2 + (coords[j] - cy)**2 +
                          (coords[k] - cz)**2)
                    psi[i, j, k] += np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()


def evolve(psi, h, D, beta, gamma, h0, dt, n_steps):
    for k in range(1, n_steps + 1):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return psi, h


def measure_attractor(psi, h):
    """Caractéristiques d'un état stationnaire."""
    psi_min = float(psi.min())
    psi_max = float(psi.max())
    h_min = float(h.min())
    h_max = float(h.max())
    h_mean = float(h.mean())
    # Position de h_argmin
    argmin_h_flat = int(np.argmin(h))
    argmin_h = tuple(int(x) for x in np.unravel_index(argmin_h_flat, h.shape))
    # Position de ψ_argmax
    argmax_psi_flat = int(np.argmax(psi))
    argmax_psi = tuple(int(x) for x in
                       np.unravel_index(argmax_psi_flat, psi.shape))
    return {
        "psi_min": psi_min, "psi_max": psi_max,
        "psi_inhomogeneity": psi_max / max(psi_min, 1e-30),
        "h_min": h_min, "h_max": h_max, "h_mean": h_mean,
        "h_inhomogeneity": h_max / max(h_min, 1e-30),
        "h_argmin": argmin_h,
        "psi_argmax": argmax_psi,
    }


def run_one_geometry(label, psi_init_fn, beta):
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_total = 500.0

    psi_init = psi_init_fn()
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max_init = float(psi_init.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max_init + gamma))
    n_total = int(t_total / dt)

    print(f"\n  Géométrie {label}")
    print(f"    psi initial: min={psi_init.min():.4e}, "
          f"max={psi_init.max():.4e}, sum={psi_init.sum():.6f}")
    print(f"    psi_argmax initial : "
          f"{np.unravel_index(int(np.argmax(psi_init)), psi_init.shape)}")
    print(f"    dt={dt:.5f}, n_steps={n_total}")

    # Évolution référence
    psi_final, h_final = evolve(
        psi_init.copy(), h_init.copy(),
        D, beta, gamma, h0, dt, n_total,
    )
    attractor = measure_attractor(psi_final, h_final)

    # Perturbée à l'init (même protocole que tests précédents)
    psi_pert = psi_init.copy()
    psi_pert[2, 2, 2] *= 1.01
    psi_pert /= psi_pert.sum()
    psi_pert_final, h_pert_final = evolve(
        psi_pert, h_init.copy(),
        D, beta, gamma, h0, dt, n_total,
    )
    delta_psi = float(np.linalg.norm(psi_pert_final - psi_final))
    delta_h = float(np.linalg.norm(h_pert_final - h_final))
    ratio = delta_h / max(delta_psi, 1e-30)

    print(f"    Attracteur :")
    print(f"      ψ_inhomo = {attractor['psi_inhomogeneity']:.3f}")
    print(f"      h_min = {attractor['h_min']:.4e}")
    print(f"      h_max = {attractor['h_max']:.4f}")
    print(f"      h_argmin = {attractor['h_argmin']}")
    print(f"      psi_argmax = {attractor['psi_argmax']}")
    print(f"    Perturbation finale :")
    print(f"      ‖Δψ‖ = {delta_psi:.4e}")
    print(f"      ‖Δh‖ = {delta_h:.4e}")
    print(f"      ratio Δh/Δψ = {ratio:.4f}")

    return {
        "label": label,
        "psi_init_max": float(psi_init.max()),
        "psi_init_argmax": tuple(int(x) for x in
                                  np.unravel_index(int(np.argmax(psi_init)),
                                                   psi_init.shape)),
        "dt": dt,
        "attractor": attractor,
        "delta_psi": delta_psi,
        "delta_h": delta_h,
        "ratio_h_over_psi": ratio,
    }


if __name__ == "__main__":
    beta = 60.0

    print(f"{'='*70}")
    print(f"Universalité vs héritage de l'attracteur β=60")
    print(f"{'='*70}")

    # Tests
    results = []

    # G_ref : la gaussienne sigma=1.5 (référence des tests précédents)
    results.append(run_one_geometry(
        "G_ref (gaussienne sigma=1.5)",
        lambda: make_psi_gaussian(sigma=1.5),
        beta,
    ))

    # G1 : gaussienne plus large
    results.append(run_one_geometry(
        "G1 (gaussienne sigma=2.5, plus large)",
        lambda: make_psi_gaussian(sigma=2.5),
        beta,
    ))

    # G2 : gaussienne plus étroite
    results.append(run_one_geometry(
        "G2 (gaussienne sigma=0.8, plus étroite)",
        lambda: make_psi_gaussian(sigma=0.8),
        beta,
    ))

    # G3 : double pic
    results.append(run_one_geometry(
        "G3 (double pic sigma=0.7)",
        make_psi_double_peak,
        beta,
    ))

    # Synthèse
    print(f"\n\n{'='*70}")
    print(f"SYNTHÈSE")
    print(f"{'='*70}\n")
    print(f"  {'géo':<35} {'ψ_inhomo':>10} {'h_min':>14} "
          f"{'h_argmin':>14} {'ratio':>10}")
    for r in results:
        att = r["attractor"]
        print(f"  {r['label']:<35} "
              f"{att['psi_inhomogeneity']:>10.3f} "
              f"{att['h_min']:>14.4e} "
              f"{str(att['h_argmin']):>14} "
              f"{r['ratio_h_over_psi']:>10.4f}")

    # Question : tous les attracteurs sont-ils "identiques" ?
    # Critère qualitatif : structure spatiale (h_argmin), niveau h_min,
    # ratio Δh/Δψ
    print(f"\n  Lecture :")
    h_mins = [r["attractor"]["h_min"] for r in results]
    psi_inhomos = [r["attractor"]["psi_inhomogeneity"] for r in results]
    ratios = [r["ratio_h_over_psi"] for r in results]
    argmins = [r["attractor"]["h_argmin"] for r in results]

    # Classification grossière des régimes
    print(f"\n    Régimes apparents :")
    for r in results:
        att = r["attractor"]
        if att["psi_inhomogeneity"] < 1.1 and att["h_min"] > 0.1:
            regime = "homogène"
        elif att["h_min"] < 1e-30:
            regime = "structuré quasi-singulier"
        else:
            regime = "structuré (intermédiaire)"
        print(f"      {r['label']}: {regime}")

    # Universalité ou héritage ?
    distinct_regimes = set()
    for r in results:
        att = r["attractor"]
        if att["psi_inhomogeneity"] < 1.1 and att["h_min"] > 0.1:
            distinct_regimes.add("homogène")
        elif att["h_min"] < 1e-30:
            distinct_regimes.add("structuré quasi-singulier")
        else:
            distinct_regimes.add("structuré intermédiaire")

    if len(distinct_regimes) == 1:
        print(f"\n    → tous les inits convergent vers le même régime")
        print(f"    → l'attracteur est universel à β=60")
    else:
        print(f"\n    → {len(distinct_regimes)} régimes distincts observés "
              f"selon l'init")
        print(f"    → l'attracteur est conditionné par la géométrie initiale")
        print(f"    → héritage morphologique présent")

    # Ratio 5.3666 : survit-il aux changements d'init ?
    print(f"\n    Ratio Δh/Δψ par init :")
    for r in results:
        print(f"      {r['label']}: {r['ratio_h_over_psi']:.4f}")

    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_geometric_inheritance.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats : {output_path}")
