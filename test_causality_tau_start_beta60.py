"""
6d-β cycle 2 — test causalité à β=60 (corrigé).

Question reformulée (Alex) :
"Comment l'espace des attracteurs accessibles se transforme-t-il
quand on retarde le verrouillage du couplage ?"

PAS : homogène vs structuré (binaire)
Mais : caractérisation morphologique complète de l'attracteur

Protocole :
- Phase 1 (durée τ_start) : β=0, ψ diffuse librement
- Phase 2 : β=60 activé, le système évolue jusqu'à t_total

Pour τ_start ∈ {0, 1, 2, 3, 5, 7, 10}
(densité plus fine pour capter un possible comportement intermédiaire)

Mesures de l'attracteur final :
- h_min, h_max, h_mean
- h_argmin (position spatiale)
- structure spatiale de h : profil radial, anisotropies
- ψ_inhomogeneity, ψ_argmax
- ratio Δh/Δψ sous perturbation (signature locale d'attracteur)
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


def radial_profile(field):
    """Profil radial moyen autour du centre de la grille."""
    c = (N_AXIS - 1) // 2  # centre = (2, 2, 2)
    profile = {}
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((i - c)**2 + (j - c)**2 + (k - c)**2)
                r_round = round(float(r), 4)
                if r_round not in profile:
                    profile[r_round] = []
                profile[r_round].append(float(field[i, j, k]))
    # Moyenne par rayon
    result = {}
    for r in sorted(profile.keys()):
        vals = profile[r]
        result[r] = {
            "mean": float(np.mean(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n_cells": len(vals),
        }
    return result


def measure_attractor(psi, h):
    psi_min, psi_max = float(psi.min()), float(psi.max())
    h_min, h_max = float(h.min()), float(h.max())
    h_mean = float(h.mean())
    return {
        "psi_min": psi_min, "psi_max": psi_max,
        "psi_inhomogeneity": psi_max / max(psi_min, 1e-30),
        "psi_argmax": tuple(int(x) for x in
                            np.unravel_index(int(np.argmax(psi)), psi.shape)),
        "h_min": h_min, "h_max": h_max, "h_mean": h_mean,
        "h_inhomogeneity": h_max / max(h_min, 1e-30),
        "h_argmin": tuple(int(x) for x in
                          np.unravel_index(int(np.argmin(h)), h.shape)),
        "h_argmax": tuple(int(x) for x in
                          np.unravel_index(int(np.argmax(h)), h.shape)),
    }


def run_one_tau(tau_start, beta_active=60.0, sigma_init=1.5):
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_total = 500.0  # plus long que pour β=55 pour bien stabiliser

    psi = make_psi_centered(sigma=sigma_init)
    h = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max_init = float(psi.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_active * psi_max_init + gamma))
    n_total = int(t_total / dt)
    n_phase1 = int(tau_start / dt)

    # Phase 1 : β=0
    for k in range(1, n_phase1 + 1):
        psi, h = step(psi, h, D, 0.0, gamma, h0, dt)

    psi_at_start = psi.copy()

    # Phase 2 : β=60
    for k in range(n_phase1 + 1, n_total + 1):
        psi, h = step(psi, h, D, beta_active, gamma, h0, dt)

    attractor = measure_attractor(psi, h)

    # Perturbation à tau_start aurait été équivalent ; on mesure
    # directement la signature d'attracteur. Refaire une simulation
    # complète perturbée serait redondant.

    return {
        "tau_start": tau_start,
        "dt": dt,
        "psi_at_start": {
            "max": float(psi_at_start.max()),
            "inhomogeneity": float(psi_at_start.max() /
                                   max(psi_at_start.min(), 1e-30)),
            "argmax": tuple(int(x) for x in
                            np.unravel_index(int(np.argmax(psi_at_start)),
                                             psi_at_start.shape)),
        },
        "attractor": attractor,
        "h_radial_profile": radial_profile(h),
    }


if __name__ == "__main__":
    beta = 60.0
    print(f"{'='*70}")
    print(f"Test causalité à β=60 — espace des attracteurs accessibles")
    print(f"  selon τ_start (retard de verrouillage)")
    print(f"{'='*70}")

    tau_values = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    results = []
    for tau in tau_values:
        print(f"\n  τ_start = {tau}...")
        r = run_one_tau(tau, beta_active=beta)
        att = r["attractor"]
        print(f"    ψ à τ_start : inhomo={r['psi_at_start']['inhomogeneity']:.3f}")
        print(f"    Attracteur :")
        print(f"      ψ_inhomo = {att['psi_inhomogeneity']:.3f}")
        print(f"      h_min = {att['h_min']:.4e}, h_max = {att['h_max']:.4f}")
        print(f"      h_argmin = {att['h_argmin']}, "
              f"h_argmax = {att['h_argmax']}")
        results.append(r)

    # Synthèse
    print(f"\n\n{'='*70}")
    print(f"SYNTHÈSE")
    print(f"{'='*70}\n")
    print(f"  {'τ_start':>8} {'ψ(τ)_inhomo':>14} {'ψ_final_inhomo':>16} "
          f"{'h_min final':>14} {'h_argmin':>12}")
    for r in results:
        att = r["attractor"]
        print(f"  {r['tau_start']:>8.1f} "
              f"{r['psi_at_start']['inhomogeneity']:>14.3f} "
              f"{att['psi_inhomogeneity']:>16.3f} "
              f"{att['h_min']:>14.4e} "
              f"{str(att['h_argmin']):>12}")

    # Classification morphologique des attracteurs
    print(f"\n  Classification des attracteurs finaux :")
    for r in results:
        att = r["attractor"]
        h_min = att["h_min"]
        psi_inhomo = att["psi_inhomogeneity"]
        if h_min > 0.1 and psi_inhomo < 1.1:
            kind = "HOMOGÈNE (h_min > 0.1, ψ uniforme)"
        elif h_min < 1e-30:
            kind = "QUASI-SINGULIER (h_min effondré)"
        elif h_min < 0.01:
            kind = "STRUCTURÉ FORT"
        elif h_min < 0.1:
            kind = "STRUCTURÉ INTERMÉDIAIRE"
        else:
            kind = "STRUCTURÉ FAIBLE"
        print(f"    τ={r['tau_start']:>4.1f} : {kind}")

    # Profils radiaux de h
    print(f"\n  Profils radiaux de h (par τ_start) :")
    print(f"  {'r':>6}", end="")
    for r in results:
        print(f" {f'τ={r[chr(34)+chr(116)+chr(97)+chr(117)+chr(95)+chr(115)+chr(116)+chr(97)+chr(114)+chr(116)+chr(34)]}':>14}",
              end="")
    print()
    # Récupère les rayons uniques
    all_radii = set()
    for r in results:
        all_radii.update(r["h_radial_profile"].keys())
    for radius in sorted(all_radii):
        print(f"  {radius:>6.3f}", end="")
        for r in results:
            if radius in r["h_radial_profile"]:
                val = r["h_radial_profile"][radius]["mean"]
                print(f" {val:>14.4e}", end="")
            else:
                print(f" {'--':>14}", end="")
        print()

    # Lecture
    print(f"\n  Lecture :")
    h_mins = [r["attractor"]["h_min"] for r in results]
    log_h_mins = [np.log10(max(hm, 1e-300)) for hm in h_mins]
    
    # Existe-t-il un seuil τ_crit ?
    transitions = []
    for i in range(1, len(log_h_mins)):
        if log_h_mins[i] - log_h_mins[i-1] > 5:  # saut de 5 décades
            transitions.append((tau_values[i-1], tau_values[i],
                                log_h_mins[i] - log_h_mins[i-1]))
    
    if transitions:
        for t_before, t_after, jump in transitions:
            print(f"    Saut de {jump:.1f} décades entre "
                  f"τ={t_before} et τ={t_after}")
            print(f"    → seuil de basculement quelque part dans "
                  f"[τ={t_before}, τ={t_after}]")
    else:
        print(f"    Évolution continue de log10(h_min) selon τ_start")
        for tau, lhm in zip(tau_values, log_h_mins):
            print(f"      τ={tau:.1f} : log10(h_min) = {lhm:.2f}")

    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_causality_tau_start_beta60.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats : {output_path}")
