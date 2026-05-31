"""
6d-γ — garde-fou négatif de la proxy de séparabilité.

Le contact 1 a montré une sous-additivité forte (non-linéarité
relative 0.75) entre directions A et B. Mais cela peut être :
- (saturation) une manifestation globale du verrouillage morphologique
  déjà observé en propriété 3 (toute perturbation sature) ;
- (relationnel) un couplage spécifique entre directions de
  transformation.

Question discriminante (formulée avec Alex) :
"La sous-additivité dépend-elle réellement de la relation entre
directions ?"

PAS "la sous-additivité existe-t-elle ?" (déjà répondu : oui).

Protocole : mesurer la non-linéarité resp(A+B) − [resp(A)+resp(B)]
pour trois paires (A,B) à degrés de relation géométrique différents :
- proches    : deux compressions sur axes adjacents / plans voisins
- orthogonal : axes x et y (cas du contact 1)
- éloignés   : régions spatiales disjointes (coins opposés)

Si non-linéarité ≈ constante sur les trois → SATURATION GLOBALE
  (le 0.75 ne survit pas comme signature relationnelle).
Si non-linéarité varie fortement → RELATIONNEL
  (la proxy capte quelque chose de la relation entre directions).

Un seul régime (β=60). Trois paires. Pas de gradient de β.
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


def compression_along_axis(psi, axis, strength=0.05):
    """Compression vers le plan central le long d'un axe (0=x,1=y,2=z)."""
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


def compression_region(psi, center, sigma_r=0.8, strength=0.05):
    """Compression localisée autour d'une région (coin par ex.)."""
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


def measure_nonlinearity(psi_base, h_base, op_A, op_B,
                         D, beta, gamma, h0, dt, n_relax):
    """resp(A+B) − [resp(A)+resp(B)] en norme."""
    # resp A
    psi_a, h_a = op_A(psi_base.copy()), h_base.copy()
    psi_a, h_a = evolve(psi_a, h_a, D, beta, gamma, h0, dt, n_relax)
    resp_A = psi_a - psi_base
    # resp B
    psi_b, h_b = op_B(psi_base.copy()), h_base.copy()
    psi_b, h_b = evolve(psi_b, h_b, D, beta, gamma, h0, dt, n_relax)
    resp_B = psi_b - psi_base
    # resp A+B
    psi_ab, h_ab = op_B(op_A(psi_base.copy())), h_base.copy()
    psi_ab, h_ab = evolve(psi_ab, h_ab, D, beta, gamma, h0, dt, n_relax)
    resp_AB = psi_ab - psi_base

    resp_sum = resp_A + resp_B
    nonlin = float(np.linalg.norm(resp_AB - resp_sum))
    norm_AB = float(np.linalg.norm(resp_AB))
    return {
        "resp_A_norm": float(np.linalg.norm(resp_A)),
        "resp_B_norm": float(np.linalg.norm(resp_B)),
        "resp_AB_norm": norm_AB,
        "nonlinearity": nonlin,
        "nonlinearity_relative": nonlin / max(norm_AB, 1e-30),
    }


def run():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0
    t_relax = 100.0

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_relax = int(t_relax / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, int(50.0 / dt))

    print(f"{'='*70}")
    print(f"6d-γ — garde-fou négatif de la proxy de séparabilité")
    print(f"  β={beta}, dt={dt:.5f}")
    print(f"  Question : la sous-additivité dépend-elle de la relation A↔B ?")
    print(f"{'='*70}\n")

    s = 0.05

    # Paire 1 : PROCHES (axes x et y mais compressions sur plans voisins)
    # On utilise deux régions proches
    pair_close = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_region(p, (1.5, 2.0, 2.0), strength=s),
        lambda p: compression_region(p, (2.5, 2.0, 2.0), strength=s),
        D, beta, gamma, h0, dt, 2 * n_relax)
    print(f"  Paire PROCHES (régions (1.5,2,2) et (2.5,2,2)) :")
    print(f"    non-linéarité relative = {pair_close['nonlinearity_relative']:.4f}")

    # Paire 2 : ORTHOGONAL (axes x et y — cas du contact 1)
    pair_ortho = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_along_axis(p, 0, strength=s),
        lambda p: compression_along_axis(p, 1, strength=s),
        D, beta, gamma, h0, dt, 2 * n_relax)
    print(f"  Paire ORTHOGONAL (axes x et y) :")
    print(f"    non-linéarité relative = {pair_ortho['nonlinearity_relative']:.4f}")

    # Paire 3 : ÉLOIGNÉS (coins opposés)
    pair_far = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_region(p, (0.0, 0.0, 0.0), strength=s),
        lambda p: compression_region(p, (4.0, 4.0, 4.0), strength=s),
        D, beta, gamma, h0, dt, 2 * n_relax)
    print(f"  Paire ÉLOIGNÉS (coins (0,0,0) et (4,4,4)) :")
    print(f"    non-linéarité relative = {pair_far['nonlinearity_relative']:.4f}")

    # Verdict
    print(f"\n  {'─'*60}")
    print(f"  VERDICT — la proxy survit-elle à son garde-fou ?")
    print(f"  {'─'*60}")
    nls = [pair_close['nonlinearity_relative'],
           pair_ortho['nonlinearity_relative'],
           pair_far['nonlinearity_relative']]
    spread = max(nls) - min(nls)
    print(f"    non-linéarités : proches={nls[0]:.4f}, "
          f"ortho={nls[1]:.4f}, éloignés={nls[2]:.4f}")
    print(f"    spread = {spread:.4f}")

    if spread < 0.1:
        verdict = ("SATURATION GLOBALE — la sous-additivité est ~constante, "
                   "indépendante de la relation A↔B. Le 0.75 ne survit PAS "
                   "comme signature relationnelle.")
    elif spread > 0.3:
        verdict = ("RELATIONNEL — la sous-additivité varie fortement selon "
                   "la relation A↔B. La proxy capte quelque chose de "
                   "relationnel. Survit au garde-fou.")
    else:
        verdict = ("INTERMÉDIAIRE — variation modérée. Ni saturation pure "
                   "ni relationnel pur. À ne pas trancher.")
    print(f"\n    {verdict}")

    return {
        "params": {"beta": beta, "dt": dt, "t_relax": t_relax},
        "pair_close": pair_close,
        "pair_ortho": pair_ortho,
        "pair_far": pair_far,
        "spread": spread,
        "verdict": verdict,
    }


if __name__ == "__main__":
    out = run()
    output_dir = REPO_ROOT / "results" / "phase6d_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "contact_2_separability_guardrail.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
