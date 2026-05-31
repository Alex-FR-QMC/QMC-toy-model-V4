"""
6d-γ — test de bifurcation structurelle (pas un gradient).

Question minimale (formulée avec Alex) :
"La structure relationnelle apparaît-elle seulement avec le
verrouillage, ou existe-t-elle déjà avant ?"

PAS un gradient continu en β. PAS une cartographie.
DEUX régimes seulement, mêmes observables :
- β=45 : NON verrouillé (attracteur homogène)
- β=60 : VERROUILLÉ (attracteur structuré)

Observables (identiques contact 2) :
- C3 : ||ψ_AB − ψ_BA|| (non-commutativité)
- proxy séparabilité : non-linéarité relative pour 3 paires
  (proches / orthogonal / éloignés)
  ET surtout : le PROFIL (proches < ortho < éloignés) survit-il ?

Lecture :
- Si la structure relationnelle (profil de non-linéarité + C3) est
  ABSENTE à β=45 et PRÉSENTE à β=60 → elle apparaît AVEC le verrouillage.
  Support pour : C3 et non-séparabilité co-émergent au verrouillage (L2).
- Si elle est DÉJÀ présente à β=45 → elle préexiste au verrouillage.
  Support pour : C3 / séparabilité sont (partiellement) indépendants
  du verrouillage (L1 ou L3).

Garde-fou : on NE mesure PAS une variation fine. On répond à une
question binaire de présence/absence structurelle.
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
    for _ in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return psi, h


def compression_region(psi, center, sigma_r=0.8, strength=0.05):
    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = center
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - cx)**2 + (coords[j] - cy)**2 +
                      (coords[k] - cz)**2)
                factor[i, j, k] += strength * np.exp(-0.5 * r2 / sigma_r**2)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def compression_along_axis(psi, axis, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for idx in range(N_AXIS):
        dist = abs(idx - c)
        local = 1.0 + strength * (1.0 - dist / c)
        if axis == 0:
            factor[idx, :, :] *= local
        elif axis == 1:
            factor[:, idx, :] *= local
        else:
            factor[:, :, idx] *= local
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def measure_nonlinearity(psi_base, h_base, op_A, op_B,
                         D, beta, gamma, h0, dt, n_relax):
    psi_a, h_a = op_A(psi_base.copy()), h_base.copy()
    psi_a, h_a = evolve(psi_a, h_a, D, beta, gamma, h0, dt, n_relax)
    resp_A = psi_a - psi_base
    psi_b, h_b = op_B(psi_base.copy()), h_base.copy()
    psi_b, h_b = evolve(psi_b, h_b, D, beta, gamma, h0, dt, n_relax)
    resp_B = psi_b - psi_base
    psi_ab, h_ab = op_B(op_A(psi_base.copy())), h_base.copy()
    psi_ab, h_ab = evolve(psi_ab, h_ab, D, beta, gamma, h0, dt, n_relax)
    resp_AB = psi_ab - psi_base
    nonlin = float(np.linalg.norm(resp_AB - (resp_A + resp_B)))
    norm_AB = float(np.linalg.norm(resp_AB))
    return nonlin / max(norm_AB, 1e-30)


def measure_C3(psi_base, h_base, D, beta, gamma, h0, dt, n_relax):
    opA = lambda p: compression_along_axis(p, 0, 0.05)
    opB = lambda p: compression_along_axis(p, 1, 0.05)
    # AB
    psi_ab, h_ab = opA(psi_base.copy()), h_base.copy()
    psi_ab, h_ab = evolve(psi_ab, h_ab, D, beta, gamma, h0, dt, n_relax)
    psi_ab = opB(psi_ab)
    psi_ab, h_ab = evolve(psi_ab, h_ab, D, beta, gamma, h0, dt, n_relax)
    # BA
    psi_ba, h_ba = opB(psi_base.copy()), h_base.copy()
    psi_ba, h_ba = evolve(psi_ba, h_ba, D, beta, gamma, h0, dt, n_relax)
    psi_ba = opA(psi_ba)
    psi_ba, h_ba = evolve(psi_ba, h_ba, D, beta, gamma, h0, dt, n_relax)
    return float(np.linalg.norm(psi_ab - psi_ba))


def run_regime(beta, label):
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_relax = 100.0

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_relax = int(t_relax / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, int(50.0 / dt))

    print(f"\n  {'─'*60}")
    print(f"  RÉGIME {label} (β={beta})")
    print(f"  {'─'*60}")
    print(f"    état de base : ψ_inhomo="
          f"{psi_base.max()/max(psi_base.min(),1e-30):.3f}, "
          f"h_min={h_base.min():.3e}")

    # C3
    c3 = measure_C3(psi_base, h_base, D, beta, gamma, h0, dt, n_relax)
    print(f"    C3 (||ψ_AB−ψ_BA||) = {c3:.4e}")

    # Profil de séparabilité
    nl_close = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_region(p, (1.5, 2.0, 2.0), strength=0.05),
        lambda p: compression_region(p, (2.5, 2.0, 2.0), strength=0.05),
        D, beta, gamma, h0, dt, 2 * n_relax)
    nl_ortho = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_along_axis(p, 0, 0.05),
        lambda p: compression_along_axis(p, 1, 0.05),
        D, beta, gamma, h0, dt, 2 * n_relax)
    nl_far = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_region(p, (0.0, 0.0, 0.0), strength=0.05),
        lambda p: compression_region(p, (4.0, 4.0, 4.0), strength=0.05),
        D, beta, gamma, h0, dt, 2 * n_relax)

    spread = max(nl_close, nl_ortho, nl_far) - min(nl_close, nl_ortho, nl_far)
    print(f"    séparabilité : proches={nl_close:.4f}, "
          f"ortho={nl_ortho:.4f}, éloignés={nl_far:.4f}")
    print(f"    profil monotone (proches<ortho<éloignés) ? "
          f"{nl_close < nl_ortho < nl_far}")
    print(f"    spread = {spread:.4f}")

    return {
        "beta": beta, "label": label, "dt": dt,
        "psi_inhomo_base": float(psi_base.max()/max(psi_base.min(),1e-30)),
        "h_min_base": float(h_base.min()),
        "C3": c3,
        "nl_close": nl_close, "nl_ortho": nl_ortho, "nl_far": nl_far,
        "spread": spread,
        "profile_monotone": bool(nl_close < nl_ortho < nl_far),
    }


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"6d-γ — bifurcation structurelle (2 régimes, pas de gradient)")
    print(f"  Question : la structure relationnelle apparaît-elle AVEC")
    print(f"  le verrouillage, ou existe-t-elle déjà avant ?")
    print(f"{'='*70}")

    r_unlocked = run_regime(45.0, "NON VERROUILLÉ")
    r_locked = run_regime(60.0, "VERROUILLÉ")

    print(f"\n{'='*70}")
    print(f"BIFURCATION STRUCTURELLE")
    print(f"{'='*70}")
    print(f"\n  {'observable':<22} {'β=45 (non verr.)':>18} "
          f"{'β=60 (verr.)':>16}")
    print(f"  {'C3':<22} {r_unlocked['C3']:>18.4e} "
          f"{r_locked['C3']:>16.4e}")
    print(f"  {'spread séparabilité':<22} {r_unlocked['spread']:>18.4f} "
          f"{r_locked['spread']:>16.4f}")
    print(f"  {'profil monotone':<22} {str(r_unlocked['profile_monotone']):>18} "
          f"{str(r_locked['profile_monotone']):>16}")

    print(f"\n  Lecture :")
    # C3 présent dans les deux ? ou seulement verrouillé ?
    c3_unlocked_significant = r_unlocked['C3'] > 1e-10
    c3_locked_significant = r_locked['C3'] > 1e-10
    struct_unlocked = r_unlocked['spread'] > 0.3
    struct_locked = r_locked['spread'] > 0.3

    print(f"    C3 significatif β=45 ? {c3_unlocked_significant} "
          f"(={r_unlocked['C3']:.2e})")
    print(f"    C3 significatif β=60 ? {c3_locked_significant} "
          f"(={r_locked['C3']:.2e})")
    print(f"    structure relationnelle β=45 ? {struct_unlocked} "
          f"(spread={r_unlocked['spread']:.3f})")
    print(f"    structure relationnelle β=60 ? {struct_locked} "
          f"(spread={r_locked['spread']:.3f})")

    print(f"\n  Verdict de bifurcation :")
    if not struct_unlocked and struct_locked:
        print(f"    La structure relationnelle APPARAÎT AVEC LE VERROUILLAGE.")
        print(f"    → absente à β=45, présente à β=60")
        print(f"    → support pour co-émergence C3/non-séparabilité (L2)")
    elif struct_unlocked and struct_locked:
        print(f"    La structure relationnelle PRÉEXISTE au verrouillage.")
        print(f"    → présente dès β=45")
        print(f"    → support pour indépendance partielle (L1 ou L3)")
    elif struct_unlocked and not struct_locked:
        print(f"    Structure présente AVANT mais pas APRÈS (inattendu).")
    else:
        print(f"    Structure absente dans les deux régimes.")
        print(f"    → ni C3 ni séparabilité ne sont structurellement actifs")
        print(f"      sous ces conditions")

    out = {"unlocked_beta45": r_unlocked, "locked_beta60": r_locked}
    output_dir = REPO_ROOT / "results" / "phase6d_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "contact_3_structural_bifurcation.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
