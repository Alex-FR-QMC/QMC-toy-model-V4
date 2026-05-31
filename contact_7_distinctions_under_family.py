"""
6d-γ contact 7 — distinctions sous une famille de transformations.

Question (Alex) :
"Quelles perturbations cessent de produire des distinctions nouvelles,
et lesquelles continuent à en produire ?"

Reformulation finale (Alex) :
S n'est PAS une projection de la générativité. S est une famille de
transformations sous laquelle certaines distinctions survivent ou
s'effondrent. Aucune charge ontologique sur S ni sur 𝒢.

Cadre négatif (préinscrit, voir trace §5bis) :
- si distinction disparaît sous S → disparition SOUS CETTE FAMILLE
- si distinction survit sous S → survie SOUS CETTE FAMILLE
- ne conclut pas directement sur générativité en général

Protocole (validé Alex) :
- 3 perturbations primaires qualitativement diverses :
  P1 = compression centrale (vers (2,2,2))
  P2 = compression périphérique (vers un coin (0,0,0))
  P3 = déformation anisotrope (étirement le long de l'axe x)
- 2 régimes : β=45 (non verrouillé) et β=60 (verrouillé)
- S = relaxation prolongée
- pour chaque régime :
    appliquer chaque Pᵢ → obtenir Rᵢ
    appliquer S à chaque Rᵢ → obtenir S(Rᵢ)
    examiner la matrice de distance entre les S(Rᵢ)

Point d'attention (Alex) :
Ne pas seulement regarder lesquelles distinctions survivent.
Regarder surtout SI CERTAINES DISTINCTIONS DISPARAISSENT
ALORS QUE D'AUTRES SURVIVENT sous la même famille.
C'est là qu'une structuration émergente pourrait apparaître.
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


# === Trois perturbations primaires qualitativement distinctes ===

def P1_central(psi, strength=0.05):
    """Compression vers le centre (2,2,2). Concentre ψ au centre."""
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
    """Compression vers un coin (0,0,0). Concentre ψ en périphérie."""
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
    """Étirement anisotrope le long de l'axe x.
    Augmente ψ aux extrémités x=0 et x=4, diminue au centre."""
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        dist_x = abs(i - c)
        # +strength aux extrémités, -strength au centre
        factor[i, :, :] *= 1.0 + strength * (2.0 * dist_x / c - 1.0)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def distance(state_a, state_b):
    """Distance L2 entre deux états (psi, h)."""
    psi_a, h_a = state_a
    psi_b, h_b = state_b
    return float(np.sqrt(np.linalg.norm(psi_a - psi_b)**2 +
                          np.linalg.norm(h_a - h_b)**2))


def run_regime(beta, label, perturbations, t_relax_S):
    gamma = 1.0
    D = 0.1
    h0 = 1.0

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_S = int(t_relax_S / dt)

    print(f"\n  {'─'*60}")
    print(f"  RÉGIME {label} (β={beta})")
    print(f"  {'─'*60}")

    # État de base : laisser le système se stabiliser
    n_stab = int(50.0 / dt)
    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)
    print(f"    base : ψ_inhomo="
          f"{psi_base.max()/max(psi_base.min(),1e-30):.3f}, "
          f"h_min={h_base.min():.3e}")

    # Appliquer chaque perturbation primaire, obtenir Rᵢ (après court relax)
    # puis appliquer S, obtenir S(Rᵢ)
    n_short = int(10.0 / dt)  # court relax post-perturbation (formation Rᵢ)
    R_states = {}
    SR_states = {}
    for name, P in perturbations.items():
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        R_states[name] = (psi_R, h_R)
        # S = relaxation prolongée
        psi_SR, h_SR = evolve(psi_R.copy(), h_R.copy(),
                               D, beta, gamma, h0, dt, n_S)
        SR_states[name] = (psi_SR, h_SR)
        print(f"    {name}: R produit, S(R) calculé")

    # Matrices de distance
    names = list(perturbations.keys())
    print(f"\n    Matrice de distance entre Rᵢ (avant S) :")
    print(f"    {'':>4} " + " ".join(f"{n:>12}" for n in names))
    d_R = {}
    for i in names:
        row = []
        for j in names:
            d = distance(R_states[i], R_states[j])
            d_R[(i, j)] = d
            row.append(f"{d:>12.4e}")
        print(f"    {i:>4} " + " ".join(row))

    print(f"\n    Matrice de distance entre S(Rᵢ) (après S) :")
    print(f"    {'':>4} " + " ".join(f"{n:>12}" for n in names))
    d_SR = {}
    for i in names:
        row = []
        for j in names:
            d = distance(SR_states[i], SR_states[j])
            d_SR[(i, j)] = d
            row.append(f"{d:>12.4e}")
        print(f"    {i:>4} " + " ".join(row))

    # Pour chaque paire (i, j), comparer d_R et d_SR
    # Une distinction "s'effondre" si d_SR << d_R
    # Une distinction "survit" si d_SR comparable ou > d_R
    print(f"\n    Évolution des distinctions sous S "
          f"(ratio d_SR / d_R) :")
    print(f"    {'paire':>10} {'d_R':>14} {'d_S(R)':>14} {'ratio':>10}")
    ratios = {}
    for i_idx, i in enumerate(names):
        for j in names[i_idx+1:]:
            r = d_SR[(i, j)] / max(d_R[(i, j)], 1e-30)
            ratios[f"{i}-{j}"] = r
            verdict = ("EFFONDRE" if r < 0.1 else
                       "RÉDUIT" if r < 0.5 else
                       "MAINTENU" if r < 2.0 else
                       "AMPLIFIÉ")
            print(f"    {i+'-'+j:>10} {d_R[(i,j)]:>14.4e} "
                  f"{d_SR[(i,j)]:>14.4e} {r:>10.4f}  {verdict}")

    return {
        "beta": beta, "label": label, "dt": dt,
        "base_psi_inhomo": float(psi_base.max() / max(psi_base.min(), 1e-30)),
        "base_h_min": float(h_base.min()),
        "distances_R": {f"{i}-{j}": d_R[(i, j)]
                        for i in names for j in names if i < j},
        "distances_SR": {f"{i}-{j}": d_SR[(i, j)]
                         for i in names for j in names if i < j},
        "ratios_SR_over_R": ratios,
    }


if __name__ == "__main__":
    perturbations = {
        "P1": P1_central,
        "P2": P2_peripheral,
        "P3": P3_anisotropic,
    }

    print(f"{'='*70}")
    print(f"6d-γ contact 7 — distinctions sous une famille de transformations")
    print(f"  3 perturbations : centrale (P1), périphérique (P2), anisotrope (P3)")
    print(f"  S = relaxation prolongée (t=200)")
    print(f"  2 régimes : β=45 (non verrouillé), β=60 (verrouillé)")
    print(f"{'='*70}")

    r_unlocked = run_regime(45.0, "NON VERROUILLÉ", perturbations,
                            t_relax_S=200.0)
    r_locked = run_regime(60.0, "VERROUILLÉ", perturbations,
                          t_relax_S=200.0)

    # Lecture comparée
    print(f"\n{'='*70}")
    print(f"LECTURE COMPARÉE — structure différentielle de survie/effondrement")
    print(f"{'='*70}")
    print(f"\n  Ratios d_S(R) / d_R par paire et régime :")
    print(f"    {'paire':>10} {'β=45':>14} {'β=60':>14}")
    for pair in r_unlocked["ratios_SR_over_R"]:
        r45 = r_unlocked["ratios_SR_over_R"][pair]
        r60 = r_locked["ratios_SR_over_R"][pair]
        print(f"    {pair:>10} {r45:>14.4f} {r60:>14.4f}")

    # Point d'attention (Alex) : disparition pour certaines, survie pour d'autres,
    # sous la MÊME famille de transformations ?
    print(f"\n  Point d'attention : structure différentielle ?")
    for label, regime in [("β=45", r_unlocked), ("β=60", r_locked)]:
        rs = list(regime["ratios_SR_over_R"].values())
        spread = max(rs) - min(rs)
        all_collapsed = all(r < 0.1 for r in rs)
        all_maintained = all(0.5 < r < 2.0 for r in rs)
        differential = (max(rs) > 0.5 and min(rs) < 0.1)
        print(f"    {label} : spread={spread:.4f}, "
              f"all_collapsed={all_collapsed}, "
              f"all_maintained={all_maintained}, "
              f"DIFFÉRENTIEL={differential}")

    out = {"beta_45": r_unlocked, "beta_60": r_locked}
    output_dir = REPO_ROOT / "results" / "phase6d_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "contact_7_distinctions_under_family.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {out_path}")
