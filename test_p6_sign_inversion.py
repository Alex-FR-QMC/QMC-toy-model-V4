"""
6d-γ — extra(-P6) vs extra(P6) : test de la nature de la fonctionnelle.

Le test de scaling avait montré ||extra|| ∝ amplitude.
La symétrisation observée (P6 antisymétrique → extra pair) suggère
une fonctionnelle quadratique en la composante antisymétrique de P.

Test discriminant : comparer extra(P6) à extra(P6_neg), où P6_neg
est P6 avec signes inversés (boost à i=4, dépression à i=0).

- extra(P6_neg) = -extra(P6) → fonctionnelle LINÉAIRE en P
- extra(P6_neg) = +extra(P6) → fonctionnelle QUADRATIQUE en P (insensible
  au signe de la composante antisymétrique)
- intermédiaire → cas plus complexe

Si on observe le cas quadratique, alors extra n'est PAS une coordonnée
conservée linéaire indépendante. C'est une réponse secondaire de
l'opérateur, qui projette toute la composante antisymétrique sur un
même mode pair selon une norme.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import json
from scipy.optimize import brentq

from mcq_v4.factorial_6d import N_AXIS, DX, cfl_dt_max
from mcq_v4.factorial_6d.engine import compute_diffusion_flux, compute_divergence
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero


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
                r2 = ((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                psi[i,j,k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()


def P1prime(psi, strength=0.05, radius=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= radius: factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

def P2prime(psi, strength, radius_inner=2.0):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r >= radius_inner: factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

def P3prime(psi, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for k in range(N_AXIS):
        d = abs(k - c)
        factor[:,:,k] *= 1.0 + strength * (2.0 * d / c - 1.0)
    p = psi * factor
    return p / p.sum()

def P4(psi, strength, r_inner=1.0, r_outer=2.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= r_inner: factor[i,j,k] += strength
                elif r >= r_outer: factor[i,j,k] -= strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()

def P5_neighbors_only(psi, strength):
    factor = np.ones_like(psi)
    neighbors = [(1,2,2), (3,2,2), (2,1,2), (2,3,2), (2,2,1), (2,2,3)]
    for (i,j,k) in neighbors:
        factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

def P6_face_dipole(psi, strength):
    """+face i=0, -face i=4."""
    factor = np.ones_like(psi)
    factor[0, :, :] += strength
    factor[4, :, :] -= strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()

def P6_neg(psi, strength):
    """Inverse de P6 : -face i=0, +face i=4."""
    factor = np.ones_like(psi)
    factor[0, :, :] -= strength
    factor[4, :, :] += strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    def input_amp(P, s): return float(np.linalg.norm(P(psi_base, s) - psi_base))
    amp_target = input_amp(P1prime, 0.05)
    s_P5 = brentq(lambda s: input_amp(P5_neighbors_only, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)
    s_P6 = brentq(lambda s: input_amp(P6_face_dipole, s) - amp_target,
                  1e-4, 0.99, xtol=1e-6)

    perturbations_15 = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
        ("P5", lambda p: P5_neighbors_only(p, strength=s_P5)),
    ]
    r_15 = []
    for name, P in perturbations_15:
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        r_15.append(psi_R - psi_base)
    R_base = np.column_stack([r.flatten() for r in r_15])

    # Calculer r_6 et r_6_neg
    def compute_r(P, s):
        psi_p = P(psi_base.copy(), s)
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        return psi_R - psi_base

    r_6 = compute_r(P6_face_dipole, s_P6)
    r_6_neg = compute_r(P6_neg, s_P6)

    print(f"=== Test signe inversé : extra(P6) vs extra(-P6) ===\n")
    print(f"  s_P6 = {s_P6:.5f}")
    print(f"  Amplitude entrée P6 et P6_neg :")
    print(f"    ||P6(ψ) - ψ||     = {input_amp(P6_face_dipole, s_P6):.4e}")
    print(f"    ||P6_neg(ψ) - ψ|| = {input_amp(P6_neg, s_P6):.4e}")

    # Calculer extra pour les deux
    r_6_flat = r_6.flatten()
    r_6_neg_flat = r_6_neg.flatten()

    coefs_pos, _, _, _ = np.linalg.lstsq(R_base, r_6_flat, rcond=None)
    coefs_neg, _, _, _ = np.linalg.lstsq(R_base, r_6_neg_flat, rcond=None)

    extra_pos = r_6_flat - R_base @ coefs_pos
    extra_neg = r_6_neg_flat - R_base @ coefs_neg

    print(f"\n  ||r_6||      = {np.linalg.norm(r_6_flat):.4e}")
    print(f"  ||r_6_neg||  = {np.linalg.norm(r_6_neg_flat):.4e}")
    print(f"  ||extra_pos|| = {np.linalg.norm(extra_pos):.4e}")
    print(f"  ||extra_neg|| = {np.linalg.norm(extra_neg):.4e}")

    # Test linéaire : extra_neg = -extra_pos ?
    # Test quadratique : extra_neg = +extra_pos ?
    diff_linear = extra_pos + extra_neg  # devrait être 0 si linéaire (extra_neg = -extra_pos)
    diff_quad = extra_pos - extra_neg     # devrait être 0 si quadratique (extra_neg = +extra_pos)

    rel_linear = np.linalg.norm(diff_linear) / max(np.linalg.norm(extra_pos), 1e-30)
    rel_quad = np.linalg.norm(diff_quad) / max(np.linalg.norm(extra_pos), 1e-30)

    print(f"\n  ||extra_pos + extra_neg|| / ||extra_pos||  = {rel_linear:.4e}")
    print(f"    (devrait être 0 si LINÉAIRE en P)")
    print(f"  ||extra_pos - extra_neg|| / ||extra_pos||  = {rel_quad:.4e}")
    print(f"    (devrait être 0 si QUADRATIQUE en signe de P)")

    # Et le produit scalaire normalisé
    if np.linalg.norm(extra_pos) > 1e-30 and np.linalg.norm(extra_neg) > 1e-30:
        cos_pos_neg = float((extra_pos * extra_neg).sum() /
                            (np.linalg.norm(extra_pos) * np.linalg.norm(extra_neg)))
        print(f"\n  cos(angle(extra_pos, extra_neg)) = {cos_pos_neg:+.6f}")
        print(f"    +1 → extra_neg = +extra_pos (QUADRATIQUE)")
        print(f"    -1 → extra_neg = -extra_pos (LINÉAIRE)")
        print(f"     0 → orthogonal (autre)")
    else:
        cos_pos_neg = None

    # Verdict
    print(f"\n=== Verdict ===")
    if cos_pos_neg is not None:
        if cos_pos_neg > 0.99:
            verdict = ("QUADRATIQUE EN SIGNE : extra(-P6) ≈ +extra(P6). "
                       "La 'coordonnée' γ n'est pas une fonctionnelle linéaire "
                       "de P. C'est une réponse de l'opérateur qui détache le "
                       "signe de la composante antisymétrique.")
        elif cos_pos_neg < -0.99:
            verdict = ("LINÉAIRE : extra(-P6) ≈ -extra(P6). γ est compatible "
                       "avec une coordonnée linéaire conservée.")
        else:
            verdict = f"INTERMÉDIAIRE : cos = {cos_pos_neg:.4f}. Cas à examiner."
    else:
        verdict = "Données insuffisantes."
    print(f"  {verdict}")

    output = {
        "s_P6": s_P6,
        "amp_target": amp_target,
        "norm_r_6": float(np.linalg.norm(r_6_flat)),
        "norm_r_6_neg": float(np.linalg.norm(r_6_neg_flat)),
        "norm_extra_pos": float(np.linalg.norm(extra_pos)),
        "norm_extra_neg": float(np.linalg.norm(extra_neg)),
        "rel_linear": float(rel_linear),
        "rel_quad": float(rel_quad),
        "cos_pos_neg": cos_pos_neg,
        "verdict": verdict,
    }
    out_path = "/home/claude/mcq_v4/p6_sign_inversion.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
