"""
6d-γ contact 8 — dynamique d'émergence de la hiérarchie sous S.

Question (Alex) :
Avant de tester si la hiérarchie est robuste à une autre famille S',
comprendre comment elle émerge dans S elle-même.
Quelle paire perd sa distinction EN PREMIER ?

Le contact 7 a donné une hiérarchie finale à t=200. Cette mesure est
un instantané. Sa dynamique d'émergence pourrait être :
- ordre STABLE : à tout t, même hiérarchie (P2-P3 < P1-P2 < P1-P3)
- ordre TEMPORELLEMENT STRATIFIÉ : P2-P3 d'abord, P1-P3 ensuite,
  P1-P2 en dernier ; échelles temporelles séparées
- ordre INSTABLE : la hiérarchie change selon t, la hiérarchie finale
  est un instantané d'un processus non-monotone

Préinscription : les trois lectures sont fixées avant mesure pour
éviter de réinterpréter le résultat après coup.

Protocole :
- mêmes P1, P2, P3 que contact 7 (centrale, périphérique, anisotrope)
- même état de base β=60 verrouillé
- application de chaque Pᵢ → Rᵢ après relax court n_short
- relaxation continue, mais on enregistre les distances entre Rᵢ(t)
  aux instants t = 5, 20, 50, 100, 200
- traçabilité temporelle des trois distances P1-P2, P1-P3, P2-P3
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


def P1_central(psi, strength=0.05):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - c)**2 + (coords[j] - c)**2 +
                      (coords[k] - c)**2)
                factor[i, j, k] += strength * np.exp(-0.5 * r2 / 0.8**2)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P2_peripheral(psi, strength=0.05):
    coords = np.arange(N_AXIS) * DX
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i]**2 + coords[j]**2 + coords[k]**2)
                factor[i, j, k] += strength * np.exp(-0.5 * r2 / 0.8**2)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P3_anisotropic(psi, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        dist_x = abs(i - c)
        factor[i, :, :] *= 1.0 + strength * (2.0 * dist_x / c - 1.0)
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

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))

    print(f"{'='*70}")
    print(f"6d-γ contact 8 — dynamique d'émergence de la hiérarchie sous S")
    print(f"  β=60, dt={dt:.5f}")
    print(f"  Mesure des distances Rᵢ(t) à t = 5, 20, 50, 100, 200")
    print(f"{'='*70}\n")

    # État de base
    n_stab = int(50.0 / dt)
    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)
    print(f"  base : ψ_inhomo="
          f"{psi_base.max()/max(psi_base.min(),1e-30):.3f}, "
          f"h_min={h_base.min():.3e}")

    # Appliquer les perturbations et faire un court relax (formation Rᵢ)
    n_short = int(10.0 / dt)
    perturbations = {"P1": P1_central, "P2": P2_peripheral,
                     "P3": P3_anisotropic}
    R_states = {}
    for name, P in perturbations.items():
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        R_states[name] = [psi_R, h_R]
        print(f"  R[{name}] formé à t=10 post-perturbation")

    # Relaxation progressive : enregistrer distances à temps intermédiaires
    checkpoints_t = [5.0, 20.0, 50.0, 100.0, 200.0]
    # On part de R_states (t=0 pour S), et on relaxe progressivement
    # en accumulant les pas pour atteindre chaque checkpoint
    distances_at_t = {}
    t_current = 0.0
    for t_target in checkpoints_t:
        dt_seg = t_target - t_current
        n_seg = int(dt_seg / dt)
        for name in R_states:
            psi_r, h_r = R_states[name]
            psi_r, h_r = evolve(psi_r, h_r, D, beta, gamma, h0, dt, n_seg)
            R_states[name] = [psi_r, h_r]
        # Mesurer toutes les paires
        d = {}
        for pair in [("P1", "P2"), ("P1", "P3"), ("P2", "P3")]:
            d[f"{pair[0]}-{pair[1]}"] = distance(
                R_states[pair[0]], R_states[pair[1]])
        distances_at_t[t_target] = d
        t_current = t_target

    # Affichage temporel
    print(f"\n  Distances entre Rᵢ(t) au cours de la relaxation :")
    print(f"    {'t':>8} {'P1-P2':>14} {'P1-P3':>14} {'P2-P3':>14}")
    for t in checkpoints_t:
        d = distances_at_t[t]
        print(f"    {t:>8.0f} {d['P1-P2']:>14.4e} "
              f"{d['P1-P3']:>14.4e} {d['P2-P3']:>14.4e}")

    # Pour chaque temps, donner l'ordre (de la plus proche à la plus distante)
    print(f"\n  Ordre des paires (la plus proche → la plus distante) :")
    for t in checkpoints_t:
        d = distances_at_t[t]
        sorted_pairs = sorted(d.items(), key=lambda x: x[1])
        order = " < ".join(p for p, _ in sorted_pairs)
        print(f"    t={t:>5.0f} : {order}")

    # Verdict — comparer les ordres
    orders = []
    for t in checkpoints_t:
        d = distances_at_t[t]
        sorted_pairs = sorted(d.items(), key=lambda x: x[1])
        orders.append(tuple(p for p, _ in sorted_pairs))

    all_same = all(o == orders[0] for o in orders)
    print(f"\n  Verdict (préinscrit avant mesure) :")
    if all_same:
        print(f"    ORDRE STABLE : la même hiérarchie à tout t")
        print(f"    → la hiérarchie n'est pas un instantané, elle est")
        print(f"      intrinsèque à la famille S, juste plus prononcée à long t.")
    else:
        # Est-ce une stratification monotone (P2-P3 d'abord, etc.) ou
        # une instabilité ?
        # Compter les changements d'ordre
        n_changes = sum(1 for i in range(1, len(orders))
                        if orders[i] != orders[i-1])
        print(f"    ORDRE NON STABLE : {n_changes} changement(s) au cours du temps")
        # Identifier la première paire à devenir indiscernable
        # (i.e. celle dont la distance décroît le plus vite relativement)
        decays = {}
        for pair in ["P1-P2", "P1-P3", "P2-P3"]:
            d_init = distances_at_t[checkpoints_t[0]][pair]
            d_final = distances_at_t[checkpoints_t[-1]][pair]
            decays[pair] = d_final / max(d_init, 1e-30)
        slowest_decay = max(decays.items(), key=lambda x: x[1])
        fastest_decay = min(decays.items(), key=lambda x: x[1])
        print(f"    décroissance la plus rapide : {fastest_decay[0]} "
              f"(ratio {fastest_decay[1]:.4f})")
        print(f"    décroissance la plus lente : {slowest_decay[0]} "
              f"(ratio {slowest_decay[1]:.4f})")
        if abs(decays["P1-P2"] - decays["P1-P3"]) < 0.1 and \
           decays["P2-P3"] < min(decays["P1-P2"], decays["P1-P3"]) / 2:
            print(f"    → stratification temporelle plausible (P2-P3 disparaît")
            print(f"      nettement avant les autres)")
        else:
            print(f"    → pattern à examiner ; ne pas conclure immédiatement")

    output = {
        "beta": beta, "dt": dt,
        "base": {"psi_inhomo": float(psi_base.max() /
                                      max(psi_base.min(), 1e-30)),
                 "h_min": float(h_base.min())},
        "distances_at_t": {str(t): distances_at_t[t]
                           for t in checkpoints_t},
        "orders": [list(o) for o in orders],
        "all_orders_same": all_same,
    }
    out_path = REPO_ROOT / "results" / "phase6d_gamma" / \
        "contact_8_hierarchy_emergence.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats : {out_path}")


if __name__ == "__main__":
    run()
