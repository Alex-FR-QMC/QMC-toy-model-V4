"""
6d-γ contact 11 — activité du couplage résiduel.

Contact 9 : hiérarchie réorganisée par rapport au contact 8, mais
garde-fou d'amplitude violé pour P2' (ratio 3.8).
Contact 10 : calibration de l'entrée absorbe ~90% de l'écart de
réponse, laisse 10% de couplage résiduel.

Question (Alex) : ce résiduel de 10% est-il morphodynamiquement actif ?

Reformulation : pas "refaire contact 9 proprement", mais tester si
un faible couplage géométrique suffit à réorganiser la hiérarchie.

Lectures préinscrites :
- Si hiérarchie réorganisée du contact 9 DISPARAÎT après calibration
  → l'effet du contact 9 était essentiellement porté par l'amplitude
  → le 10% résiduel n'est pas suffisant pour réorganiser
- Si hiérarchie réorganisée PERSISTE malgré la calibration
  → un faible couplage géométrique (10%) suffit à réorganiser
  → effet faible mais structurellement actif

Protocole : identique au contact 9, sauf P2' (strength 0.05) → P2''
(strength 0.0128 calibré à entrée de P2 originale, valeur trouvée
au contact 10). P1' et P3' inchangés.

Ordre attendu (préinscrit) :
- Contact 8 (originales)   : P1-P2 < P2-P3 < P1-P3
- Contact 9 (variantes brutes) : P2'-P3' < P1'-P2' < P1'-P3'
  → réorganisation, mais amplitude violée
- Contact 11 (variantes calibrées) :
    si retour à P1''-P2'' < P2''-P3'' < P1''-P3'' (= ordre 8)
      → réorganisation 9 était amplitude
    si maintien de P2''-P3'' < P1''-P2'' < P1''-P3'' (= ordre 9)
      → réorganisation persiste à amplitude calibrée → 10% suffit
    si autre ordre encore → résultat à examiner
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


def evolve(psi, h, D, beta, gamma, h0, dt, n_steps):
    for _ in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return psi, h


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


def P1prime_plateau_central(psi, strength=0.05, radius=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i] - c)**2 + (coords[j] - c)**2 +
                            (coords[k] - c)**2)
                if r <= radius:
                    factor[i, j, k] += strength
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P2prime_corona_peripheral(psi, strength, radius_inner=2.0):
    """Couronne périphérique. Strength varie pour calibration."""
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i] - c)**2 + (coords[j] - c)**2 +
                            (coords[k] - c)**2)
                if r >= radius_inner:
                    factor[i, j, k] += strength
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P3prime_anisotropic_z(psi, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for k in range(N_AXIS):
        dist_z = abs(k - c)
        factor[:, :, k] *= 1.0 + strength * (2.0 * dist_z / c - 1.0)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def distance(state_a, state_b):
    psi_a, h_a = state_a
    psi_b, h_b = state_b
    return float(np.sqrt(np.linalg.norm(psi_a - psi_b)**2 +
                          np.linalg.norm(h_a - h_b)**2))


def run():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0
    strength_P2pp_calibrated = 0.012789  # depuis contact 10

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))

    print(f"{'='*70}")
    print(f"6d-γ contact 11 — activité du couplage résiduel")
    print(f"  β=60, dt={dt:.5f}")
    print(f"  P2'' calibrée : strength={strength_P2pp_calibrated} "
          f"(au lieu de 0.05)")
    print(f"  P1' et P3' inchangés")
    print(f"{'='*70}\n")

    n_stab = int(50.0 / dt)
    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)
    print(f"  base : ψ_inhomo="
          f"{psi_base.max()/max(psi_base.min(),1e-30):.3f}, "
          f"h_min={h_base.min():.3e}")

    n_short = int(10.0 / dt)

    perturbations = {
        "P1''": lambda p: P1prime_plateau_central(p, strength=0.05),
        "P2''": lambda p: P2prime_corona_peripheral(
            p, strength=strength_P2pp_calibrated),
        "P3''": lambda p: P3prime_anisotropic_z(p, strength=0.05),
    }

    # Vérification amplitudes d'entrée
    print(f"\n  Amplitudes d'entrée des perturbations :")
    for name, P in perturbations.items():
        amp = float(np.linalg.norm(P(psi_base) - psi_base))
        print(f"    {name} : ||ΔP|| = {amp:.4e}")

    # Application + formation des Rᵢ
    R_states = {}
    for name, P in perturbations.items():
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        R_states[name] = [psi_R, h_R]

    # Relaxation aux mêmes checkpoints
    checkpoints_t = [5.0, 20.0, 50.0, 100.0, 200.0]
    pairs = [("P1''", "P2''"), ("P1''", "P3''"), ("P2''", "P3''")]
    distances_at_t = {}
    t_current = 0.0
    for t_target in checkpoints_t:
        dt_seg = t_target - t_current
        n_seg = int(dt_seg / dt)
        for name in R_states:
            psi_r, h_r = R_states[name]
            psi_r, h_r = evolve(psi_r, h_r, D, beta, gamma, h0, dt, n_seg)
            R_states[name] = [psi_r, h_r]
        d = {}
        for pair in pairs:
            d[f"{pair[0]}-{pair[1]}"] = distance(
                R_states[pair[0]], R_states[pair[1]])
        distances_at_t[t_target] = d
        t_current = t_target

    print(f"\n  Distances entre Rᵢ''(t) :")
    header = "    {:>8} {:>14} {:>14} {:>14}".format(
        "t", "P1''-P2''", "P1''-P3''", "P2''-P3''")
    print(header)
    for t in checkpoints_t:
        d = distances_at_t[t]
        row = "    {:>8.0f} {:>14.4e} {:>14.4e} {:>14.4e}".format(
            t, d["P1''-P2''"], d["P1''-P3''"], d["P2''-P3''"])
        print(row)

    print(f"\n  Ordre des paires :")
    orders = []
    for t in checkpoints_t:
        d = distances_at_t[t]
        sorted_pairs = sorted(d.items(), key=lambda x: x[1])
        order = tuple(p for p, _ in sorted_pairs)
        orders.append(order)
        print(f"    t={t:>5.0f} : {' < '.join(order)}")

    all_same = all(o == orders[0] for o in orders)

    # Comparaison avec contacts 8 et 9
    print(f"\n  Comparaison ordres finaux :")
    print(f"    Contact 8 (originales)        : P1-P2 < P2-P3 < P1-P3")
    print(f"    Contact 9 (variantes brutes)  : P2'-P3' < P1'-P2' < P1'-P3'")
    print(f"    Contact 11 (calibrées)        : "
          f"{' < '.join(orders[-1])}")

    # Verdict préinscrit
    order_8_pattern = ("P1''-P2''", "P2''-P3''", "P1''-P3''")
    order_9_pattern = ("P2''-P3''", "P1''-P2''", "P1''-P3''")
    print(f"\n  Verdict (préinscrit) :")
    if orders[-1] == order_8_pattern:
        verdict = ("RETOUR À L'ORDRE 8 : la réorganisation du contact 9 "
                   "était essentiellement portée par l'AMPLITUDE. Le 10% "
                   "résiduel ne suffit PAS à réorganiser la hiérarchie.")
    elif orders[-1] == order_9_pattern:
        verdict = ("MAINTIEN DE L'ORDRE 9 : la réorganisation persiste "
                   "à amplitude calibrée. Un faible couplage géométrique "
                   "(10%) SUFFIT à réorganiser. Effet faible mais "
                   "structurellement actif.")
    else:
        verdict = (f"AUTRE ORDRE : {' < '.join(orders[-1])}. À examiner "
                   f"sans présupposer.")
    print(f"    {verdict}")
    if not all_same:
        print(f"    (note : ordre non stable en temps — verdict basé sur t=200)")

    output = {
        "beta": beta, "dt": dt,
        "strength_P2pp": strength_P2pp_calibrated,
        "base": {"psi_inhomo": float(psi_base.max() /
                                      max(psi_base.min(), 1e-30)),
                 "h_min": float(h_base.min())},
        "distances_at_t": {str(t): distances_at_t[t]
                           for t in checkpoints_t},
        "orders": [list(o) for o in orders],
        "all_orders_same": all_same,
        "verdict": verdict,
    }
    out_path = REPO_ROOT / "results" / "phase6d_gamma" / \
        "contact_11_residual_coupling_activity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats : {out_path}")


if __name__ == "__main__":
    run()
