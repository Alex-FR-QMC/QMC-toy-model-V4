"""
6d-γ contact 9 — robustesse de la hiérarchie aux variantes minimales
                   de perturbations primaires.

Contact 8 a montré : ordre stable P1-P2 < P2-P3 < P1-P3 à tout t,
sous S = relaxation prolongée à β=60. La relaxation agit comme
compression anisotrope préservant l'ordre.

Question (Alex) : la hiérarchie dépend-elle de la réalisation
géométrique précise de P1/P2/P3, ou du rôle fonctionnel
(centre/périphérie/anisotropie) ?

Protocole : strictement identique au contact 8, sauf P1/P2/P3
remplacés par variantes minimales conservant le rôle fonctionnel :

- P1' : plateau central uniforme  (au lieu de gaussienne centrale)
- P2' : couronne périphérique isotrope (au lieu de coin (0,0,0))
- P3' : anisotropie axe z         (au lieu d'axe x)

Garde-fou (Alex) : vérifier que les amplitudes ||P'(ψ) - ψ||
restent du même ordre que ||P(ψ) - ψ|| originales. Sinon une
réorganisation de hiérarchie pourrait venir d'une différence
d'intensité, pas de géométrie.

Verdict préinscrit :
- si même ordre P1'-P2' < P2'-P3' < P1'-P3' → la hiérarchie
  ne dépend pas de la réalisation, candidate pour propriété du système
- si ordre réorganisé → la géométrie fine compte plus que ce que
  suggérait le contact 8
- si amplitudes hors du garde-fou → résultat non interprétable
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


# === Perturbations ORIGINALES (contact 7-8) pour le garde-fou ===

def P1_central_original(psi, strength=0.05):
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


def P2_peripheral_original(psi, strength=0.05):
    coords = np.arange(N_AXIS) * DX
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i]**2 + coords[j]**2 + coords[k]**2)
                factor[i, j, k] += strength * np.exp(-0.5 * r2 / 0.8**2)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P3_anisotropic_original(psi, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        dist_x = abs(i - c)
        factor[i, :, :] *= 1.0 + strength * (2.0 * dist_x / c - 1.0)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


# === Variantes minimales (contact 9) ===

def P1prime_plateau_central(psi, strength=0.05, radius=1.5):
    """Plateau central uniforme : boost constant dans une boule
    autour du centre (au lieu de gaussienne)."""
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


def P2prime_corona_peripheral(psi, strength=0.05, radius_inner=2.0):
    """Couronne périphérique isotrope : boost à toutes les cellules
    à distance > radius_inner du centre (au lieu de coin (0,0,0))."""
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
    """Anisotropie le long de l'axe z (au lieu de x)."""
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

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))

    print(f"{'='*70}")
    print(f"6d-γ contact 9 — robustesse de la hiérarchie aux variantes")
    print(f"  β=60, dt={dt:.5f}")
    print(f"  Protocole identique au contact 8, P1/P2/P3 → P1'/P2'/P3'")
    print(f"{'='*70}\n")

    # État de base
    n_stab = int(50.0 / dt)
    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)
    print(f"  base : ψ_inhomo="
          f"{psi_base.max()/max(psi_base.min(),1e-30):.3f}, "
          f"h_min={h_base.min():.3e}")

    # === Garde-fou d'amplitude ===
    print(f"\n  Garde-fou d'amplitude (||P(ψ) - ψ|| originales vs variantes) :")
    originals = {"P1": P1_central_original, "P2": P2_peripheral_original,
                 "P3": P3_anisotropic_original}
    variants = {"P1'": P1prime_plateau_central,
                "P2'": P2prime_corona_peripheral,
                "P3'": P3prime_anisotropic_z}
    amplitudes_orig = {}
    amplitudes_var = {}
    for (n_orig, P_orig), (n_var, P_var) in zip(originals.items(),
                                                 variants.items()):
        amp_orig = float(np.linalg.norm(P_orig(psi_base) - psi_base))
        amp_var = float(np.linalg.norm(P_var(psi_base) - psi_base))
        amplitudes_orig[n_orig] = amp_orig
        amplitudes_var[n_var] = amp_var
        ratio = amp_var / max(amp_orig, 1e-30)
        same_order = 0.3 < ratio < 3.0
        flag = "OK" if same_order else "ATTENTION"
        print(f"    {n_orig}={amp_orig:.4e}  →  {n_var}={amp_var:.4e}"
              f"  ratio={ratio:.3f}  {flag}")

    amplitudes_ok = all(
        0.3 < amplitudes_var[v] / max(amplitudes_orig[o], 1e-30) < 3.0
        for o, v in zip(originals.keys(), variants.keys())
    )
    if not amplitudes_ok:
        print(f"\n  ATTENTION : au moins une variante a une amplitude hors du")
        print(f"  facteur [0.3, 3.0] de l'originale. Résultat à interpréter")
        print(f"  avec prudence — une réorganisation de hiérarchie pourrait")
        print(f"  venir de l'intensité, pas de la géométrie.")
    else:
        print(f"\n  Amplitudes comparables (toutes dans facteur [0.3, 3.0]).")

    # === Application des variantes, formation des Rᵢ', mesures temporelles ===
    n_short = int(10.0 / dt)
    R_states = {}
    for name, P in variants.items():
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        R_states[name] = [psi_R, h_R]
        print(f"  R[{name}] formé à t=10 post-perturbation")

    # Mesures aux mêmes checkpoints que contact 8
    checkpoints_t = [5.0, 20.0, 50.0, 100.0, 200.0]
    pairs = [("P1'", "P2'"), ("P1'", "P3'"), ("P2'", "P3'")]
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

    # Affichage
    print(f"\n  Distances entre Rᵢ'(t) au cours de la relaxation :")
    header = "    {:>8} {:>14} {:>14} {:>14}".format(
        "t", "P1'-P2'", "P1'-P3'", "P2'-P3'")
    print(header)
    for t in checkpoints_t:
        d = distances_at_t[t]
        row = "    {:>8.0f} {:>14.4e} {:>14.4e} {:>14.4e}".format(
            t, d["P1'-P2'"], d["P1'-P3'"], d["P2'-P3'"])
        print(row)

    # Ordres
    print(f"\n  Ordre des paires (la plus proche → la plus distante) :")
    orders = []
    for t in checkpoints_t:
        d = distances_at_t[t]
        sorted_pairs = sorted(d.items(), key=lambda x: x[1])
        order = tuple(p for p, _ in sorted_pairs)
        orders.append(order)
        print(f"    t={t:>5.0f} : {' < '.join(order)}")

    all_same = all(o == orders[0] for o in orders)

    # Comparaison avec contact 8
    print(f"\n  Verdict (préinscrit) :")
    print(f"    Contact 8 (originales) : P1-P2 < P2-P3 < P1-P3")
    print(f"    Contact 9 (variantes)  : "
          f"{' < '.join(orders[-1])}")

    # Mappage : P1'-P2' joue le rôle de P1-P2, etc.
    expected_order = ("P1'-P2'", "P2'-P3'", "P1'-P3'")
    if orders[-1] == expected_order and all_same:
        verdict = ("HIÉRARCHIE STABLE : même ordre que contact 8, stable à"
                   " tout t. La hiérarchie semble dépendre du RÔLE fonctionnel,"
                   " pas de la réalisation. Candidat pour propriété du système.")
    elif orders[-1] == expected_order and not all_same:
        verdict = ("HIÉRARCHIE FINALE STABLE mais ordre instable en temps."
                   " À examiner.")
    elif all_same and orders[-1] != expected_order:
        verdict = (f"HIÉRARCHIE RÉORGANISÉE : ordre stable mais différent de "
                   f"contact 8. La géométrie fine compte plus que ce que"
                   f" suggérait le contact 8.")
    else:
        verdict = "PATTERN COMPLEXE : ordre non stable ET différent."
    print(f"\n    {verdict}")

    if not amplitudes_ok:
        print(f"\n    Rappel : amplitudes hors garde-fou. Le verdict est à")
        print(f"    interpréter avec prudence.")

    output = {
        "beta": beta, "dt": dt,
        "amplitudes_original": amplitudes_orig,
        "amplitudes_variant": amplitudes_var,
        "amplitudes_ok": amplitudes_ok,
        "base": {"psi_inhomo": float(psi_base.max() /
                                      max(psi_base.min(), 1e-30)),
                 "h_min": float(h_base.min())},
        "distances_at_t": {str(t): distances_at_t[t]
                           for t in checkpoints_t},
        "orders": [list(o) for o in orders],
        "all_orders_same": all_same,
        "expected_order": list(expected_order),
        "matches_contact_8": orders[-1] == expected_order,
        "verdict": verdict,
    }
    out_path = REPO_ROOT / "results" / "phase6d_gamma" / \
        "contact_9_hierarchy_robustness.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats : {out_path}")


if __name__ == "__main__":
    run()
