"""
6d-β cycle 2 — Propriété 4 : opacité différentielle.

Question reformulée (Alex) :
"Quelles composantes de l'histoire survivent encore après
verrouillage morphologique ?"

PAS : reconstructible ou non (booléen)
Mais : opacité différentielle (par dimension morphodynamique)

Protocole :
- Même init (ψ gaussienne sigma=1.5)
- Même perturbation (psi *= 1.01 sur même cellule)
- Même β=60 (régime structuré)
- Deux instants de perturbation différents : t1=5 et t2=50

t1 = 5 : pendant la transition de couplage (cf. propriété 2)
t2 = 50 : après la transition, dans le régime verrouillé

Évolution complète jusqu'à t_total=500.
Mesures sur l'état final ET sur la trajectoire :
1. ||Δψ_final||, ||Δh_final||  (écart global final entre les deux histoires)
2. h_argmin, ψ_argmax  (structure spatiale)
3. h_inhomogeneity, h_min, h_max  (caractéristiques de l'attracteur)
4. Distribution radiale de h
5. Évolution temporelle de |Δψ|(t), |Δh|(t)

Lectures possibles :
- Différence nulle ou ε_machine sur toutes les dimensions
  → opacité totale (instant de perturbation effacé)
- Différence significative sur ||Δψ|| seulement
  → opacité différentielle légère (énergie de perturbation visible, instant non identifiable)
- Différence sur multiples dimensions mais pas reconstructible
  → opacité différentielle riche
- Différence claire et structurée
  → quasi-transparence
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


def apply_perturbation_psi(psi, indices=(2, 2, 2), eps_rel=0.01):
    psi_pert = psi.copy()
    i, j, k = indices
    psi_pert[i, j, k] *= (1.0 + eps_rel)
    psi_pert /= psi_pert.sum()
    return psi_pert


def radial_profile(field, center=(2, 2, 2)):
    cx, cy, cz = center
    profile = {}
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((i - cx)**2 + (j - cy)**2 + (k - cz)**2)
                r_round = round(float(r), 4)
                if r_round not in profile:
                    profile[r_round] = []
                profile[r_round].append(float(field[i, j, k]))
    result = {}
    for r in sorted(profile.keys()):
        result[r] = float(np.mean(profile[r]))
    return result


def run_one_history(t_perturb, beta=60.0):
    """Évolue le système avec perturbation à l'instant t_perturb."""
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_total = 500.0

    psi = make_psi_centered(sigma=1.5)
    h = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max_init = float(psi.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max_init + gamma))
    n_total = int(t_total / dt)
    n_perturb = int(t_perturb / dt)
    snapshot_every = max(1, n_total // 200)

    times = [0.0]
    psi_snapshots = [psi.copy()]
    h_snapshots = [h.copy()]

    for k in range(1, n_total + 1):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        if k == n_perturb:
            psi = apply_perturbation_psi(psi)
        if k % snapshot_every == 0 or k == n_total:
            times.append(k * dt)
            psi_snapshots.append(psi.copy())
            h_snapshots.append(h.copy())

    return {
        "times": np.array(times),
        "psi_final": psi.copy(),
        "h_final": h.copy(),
        "psi_snapshots": psi_snapshots,
        "h_snapshots": h_snapshots,
        "dt": dt,
        "t_perturb": t_perturb,
    }


def compare_histories(H1, H2):
    """Compare deux histoires sur plusieurs dimensions morphodynamiques."""
    psi1 = H1["psi_final"]
    h1 = H1["h_final"]
    psi2 = H2["psi_final"]
    h2 = H2["h_final"]

    results = {}

    # 1. Écart global final
    results["delta_psi_final"] = float(np.linalg.norm(psi2 - psi1))
    results["delta_h_final"] = float(np.linalg.norm(h2 - h1))

    # 2. Caractéristiques de chaque attracteur final
    for label, psi, h in [("H1", psi1, h1), ("H2", psi2, h2)]:
        results[f"{label}_psi_min"] = float(psi.min())
        results[f"{label}_psi_max"] = float(psi.max())
        results[f"{label}_psi_inhomo"] = float(psi.max() / max(psi.min(), 1e-30))
        results[f"{label}_psi_argmax"] = tuple(int(x) for x in
                                                np.unravel_index(int(np.argmax(psi)),
                                                                 psi.shape))
        results[f"{label}_h_min"] = float(h.min())
        results[f"{label}_h_max"] = float(h.max())
        results[f"{label}_h_mean"] = float(h.mean())
        results[f"{label}_h_argmin"] = tuple(int(x) for x in
                                              np.unravel_index(int(np.argmin(h)),
                                                               h.shape))

    # 3. Différences point par point (signature spatiale)
    diff_psi = psi2 - psi1
    diff_h = h2 - h1
    results["diff_psi_argmax_abs"] = tuple(int(x) for x in
                                            np.unravel_index(int(np.argmax(np.abs(diff_psi))),
                                                             diff_psi.shape))
    results["diff_h_argmax_abs"] = tuple(int(x) for x in
                                          np.unravel_index(int(np.argmax(np.abs(diff_h))),
                                                           diff_h.shape))
    results["diff_psi_max_abs"] = float(np.max(np.abs(diff_psi)))
    results["diff_h_max_abs"] = float(np.max(np.abs(diff_h)))

    # 4. Distribution radiale comparée
    radial_h1 = radial_profile(h1)
    radial_h2 = radial_profile(h2)
    results["radial_h1"] = radial_h1
    results["radial_h2"] = radial_h2

    # 5. Ratio Δh/Δψ final (signature locale d'attracteur)
    if results["delta_psi_final"] > 1e-30:
        results["ratio_dh_over_dpsi"] = results["delta_h_final"] / results["delta_psi_final"]
    else:
        results["ratio_dh_over_dpsi"] = None

    return results


def run_test():
    print(f"{'='*70}")
    print(f"Propriété 4 — Opacité différentielle")
    print(f"  Question : quelles composantes de l'histoire survivent")
    print(f"  après verrouillage morphologique ?")
    print(f"{'='*70}\n")

    t1 = 5.0  # pendant la transition de couplage
    t2 = 50.0  # après transition, régime verrouillé

    print(f"  H1 : perturbation à t={t1} (pendant transition de couplage)")
    print(f"  H2 : perturbation à t={t2} (après transition, verrouillé)")
    print(f"  Toutes choses égales par ailleurs.\n")

    print(f"  Évolution H1...")
    H1 = run_one_history(t_perturb=t1)
    print(f"  Évolution H2...")
    H2 = run_one_history(t_perturb=t2)

    comp = compare_histories(H1, H2)

    # Affichage
    print(f"\n  ÉCARTS FINAUX ENTRE LES DEUX HISTOIRES :")
    print(f"    ||Δψ_final|| = {comp['delta_psi_final']:.4e}")
    print(f"    ||Δh_final|| = {comp['delta_h_final']:.4e}")
    if comp["ratio_dh_over_dpsi"]:
        print(f"    ratio Δh/Δψ = {comp['ratio_dh_over_dpsi']:.4f}")

    print(f"\n  ATTRACTEURS FINAUX :")
    print(f"    {'':>15} {'H1 (t_pert=5)':>20} {'H2 (t_pert=50)':>20}")
    print(f"    {'ψ_inhomo':>15} {comp['H1_psi_inhomo']:>20.4f} "
          f"{comp['H2_psi_inhomo']:>20.4f}")
    print(f"    {'h_min':>15} {comp['H1_h_min']:>20.4e} "
          f"{comp['H2_h_min']:>20.4e}")
    print(f"    {'h_max':>15} {comp['H1_h_max']:>20.4f} "
          f"{comp['H2_h_max']:>20.4f}")
    print(f"    {'h_argmin':>15} {str(comp['H1_h_argmin']):>20} "
          f"{str(comp['H2_h_argmin']):>20}")
    print(f"    {'ψ_argmax':>15} {str(comp['H1_psi_argmax']):>20} "
          f"{str(comp['H2_psi_argmax']):>20}")

    print(f"\n  SIGNATURE SPATIALE DE LA DIFFÉRENCE :")
    print(f"    argmax|Δψ| = {comp['diff_psi_argmax_abs']}, "
          f"valeur = {comp['diff_psi_max_abs']:.4e}")
    print(f"    argmax|Δh| = {comp['diff_h_argmax_abs']}, "
          f"valeur = {comp['diff_h_max_abs']:.4e}")

    print(f"\n  PROFIL RADIAL DE h (autour du centre 2,2,2) :")
    print(f"    {'r':>8} {'h1(r)':>14} {'h2(r)':>14} {'|Δh|(r)':>14}")
    for r in sorted(comp["radial_h1"].keys()):
        h1_r = comp["radial_h1"][r]
        h2_r = comp["radial_h2"][r]
        delta = abs(h2_r - h1_r)
        print(f"    {r:>8.3f} {h1_r:>14.4e} {h2_r:>14.4e} {delta:>14.4e}")

    # Lecture par dimension
    print(f"\n  LECTURE PAR DIMENSION (opacité différentielle) :")

    # (a) Énergie de perturbation
    energy_distinguishable = comp["delta_psi_final"] > 1e-15
    print(f"    (a) Énergie de perturbation (Δψ_final > seuil)  : "
          f"{'DISTINGUABLE' if energy_distinguishable else 'OPAQUE'}")

    # (b) Lieu de perturbation
    same_argmax_psi = comp["H1_psi_argmax"] == comp["H2_psi_argmax"]
    same_argmin_h = comp["H1_h_argmin"] == comp["H2_h_argmin"]
    print(f"    (b) Lieu de la perturbation préservé             : "
          f"{'OUI' if (same_argmax_psi and same_argmin_h) else 'NON'}")
    print(f"        (ψ_argmax: {'identique' if same_argmax_psi else 'différent'}, "
          f"h_argmin: {'identique' if same_argmin_h else 'différent'})")

    # (c) Régime d'attracteur (structuré/homogène)
    same_regime = (
        (comp["H1_h_min"] < 1e-30 and comp["H2_h_min"] < 1e-30) or
        (comp["H1_h_min"] > 0.1 and comp["H2_h_min"] > 0.1)
    )
    print(f"    (c) Régime d'attracteur identique               : "
          f"{'OUI' if same_regime else 'NON'}")

    # (d) Instant de perturbation reconstructible ?
    # Si Δψ_final ≠ 0 mais aucune signature ne dépend explicitement de t1/t2
    # → opacité sur l'instant
    instant_signature_in_diff_position = (
        comp["diff_psi_argmax_abs"] != (2, 2, 2)
    )
    print(f"    (d) Position max(|Δψ|) hors cellule perturbée   : "
          f"{'OUI' if instant_signature_in_diff_position else 'NON'}")
    if instant_signature_in_diff_position:
        print(f"        → la trace s'est délocalisée, "
              f"pas d'inférence directe sur lieu original")

    return comp, H1, H2


if __name__ == "__main__":
    comp, H1, H2 = run_test()
    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_property_4_self_opacity.json"
    # Préparer pour JSON (radial dict avec clés float → str)
    comp_serializable = {}
    for k, v in comp.items():
        if k.startswith("radial_"):
            comp_serializable[k] = {str(rk): rv for rk, rv in v.items()}
        elif isinstance(v, tuple):
            comp_serializable[k] = list(v)
        else:
            comp_serializable[k] = v
    with open(output_path, "w") as f:
        json.dump(comp_serializable, f, indent=2)
    print(f"\nRésultats : {output_path}")
