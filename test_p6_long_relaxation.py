"""
6d-γ — Le mode γ est-il un mode lent en cours de convergence,
ou un mode déjà stationnaire à t=200 ?

Distinction (Alex) : coordonnée fondamentale vs mode faiblement couplé.

Le coefficient de couplage observé pour γ est ~50× plus faible que celui
de β. Mais ce rapport pourrait masquer une dynamique plus lente :
γ pourrait être un mode propre du même opérateur, juste avec une valeur
propre plus petite (donc convergence plus lente).

Test : prolonger la relaxation de P6 jusqu'à t=500, t=1000, t=2000.
Suivre ||extra|| dans le temps.

- ||extra|| stable à t=200 → mode déjà à son régime permanent
- ||extra|| continue à croître → mode lent pas encore convergé
- ||extra|| décroît à long terme → mode amortit progressivement,
  pas vraiment conservé
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
    factor = np.ones_like(psi)
    factor[0, :, :] += strength
    factor[4, :, :] -= strength
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

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    def input_amp(P, s): return float(np.linalg.norm(P(psi_base, s) - psi_base))
    amp_target = input_amp(P1prime, 0.05)
    s_P5 = brentq(lambda s: input_amp(P5_neighbors_only, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)
    s_P6 = brentq(lambda s: input_amp(P6_face_dipole, s) - amp_target,
                  1e-4, 0.99, xtol=1e-6)

    # On a besoin de la base sur P1-P5 mesurée à différents temps de relaxation
    # pour comparer. On utilise relax à t=200 comme base (cohérent avec le cycle).
    perturbations_15 = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
        ("P5", lambda p: P5_neighbors_only(p, strength=s_P5)),
    ]
    n_long = int(200.0 / dt)
    r_15 = []
    for name, P in perturbations_15:
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        r_15.append(psi_R - psi_base)
    R_base = np.column_stack([r.flatten() for r in r_15])

    # Évolution de P6 avec mesures intermédiaires
    psi_p6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_p6 = h_base.copy()
    psi_R6, h_R6 = evolve(psi_p6, h_p6, D, beta_lock, gamma_v, h0, dt, n_short)

    # Continuer la relaxation et mesurer extra à différents instants
    times = [50, 100, 200, 400, 800, 1500, 3000]
    print(f"=== Évolution temporelle de ||r_6|| et ||extra|| ===\n")
    print(f"  Mesures à t (relaxation prolongée après formation R_6 à t=10) :")
    print(f"  {'t':>8} {'||r_6||':>14} {'||extra||':>14} {'ratio':>14}")

    t_current = 0.0
    psi_curr, h_curr = psi_R6.copy(), h_R6.copy()
    results = []
    for t_target in times:
        dt_seg = t_target - t_current
        n_seg = int(dt_seg / dt)
        if n_seg > 0:
            psi_curr, h_curr = evolve(psi_curr, h_curr, D, beta_lock, gamma_v, h0, dt, n_seg)
        t_current = t_target

        r_6 = psi_curr - psi_base
        r_6_flat = r_6.flatten()
        coefs, _, _, _ = np.linalg.lstsq(R_base, r_6_flat, rcond=None)
        extra = r_6_flat - R_base @ coefs

        r_norm = float(np.linalg.norm(r_6_flat))
        extra_norm = float(np.linalg.norm(extra))
        ratio = extra_norm / r_norm
        results.append({"t": t_target, "r_norm": r_norm, "extra_norm": extra_norm,
                       "ratio": ratio})
        print(f"  {t_target:>8.0f} {r_norm:>14.4e} {extra_norm:>14.4e} {ratio:>14.4f}")

    # Diagnostic
    print(f"\n=== Diagnostic ===")
    extras = np.array([r["extra_norm"] for r in results])
    norms = np.array([r["r_norm"] for r in results])
    ts = np.array([r["t"] for r in results])

    # extra entre t=200 et t=3000
    ratio_change_extra = extras[-1] / extras[2]  # t=3000 / t=200
    ratio_change_r = norms[-1] / norms[2]
    print(f"  ||extra(t=3000)|| / ||extra(t=200)|| = {ratio_change_extra:.4f}")
    print(f"  ||r_6(t=3000)|| / ||r_6(t=200)|| = {ratio_change_r:.4f}")

    if 0.98 < ratio_change_extra < 1.02:
        verdict = ("STATIONNAIRE : ||extra|| stable au-delà de t=200. Le mode γ "
                   "est conservé et a atteint son régime permanent. Sa faiblesse "
                   "par rapport à β n'est pas due à une convergence incomplète. "
                   "Lecture (B) appuyée : mode secondaire faiblement couplé.")
    elif ratio_change_extra > 1.5:
        verdict = ("CROISSANCE : ||extra|| continue à croître. γ est un mode lent "
                   "pas encore convergé à t=200. Sa faiblesse à t=200 sous-estime "
                   "son poids réel. Statut à revoir.")
    elif ratio_change_extra < 0.5:
        verdict = ("DÉCROISSANCE : ||extra|| diminue. γ n'est pas vraiment conservé, "
                   "il s'amortit. C'est un mode quasi-conservé mais en relaxation "
                   "lente.")
    else:
        verdict = f"INTERMÉDIAIRE : ratio = {ratio_change_extra:.4f}"
    print(f"\n  Verdict : {verdict}")

    output = {
        "times": [r["t"] for r in results],
        "r_norms": [r["r_norm"] for r in results],
        "extra_norms": [r["extra_norm"] for r in results],
        "ratios": [r["ratio"] for r in results],
        "ratio_change_extra_200_to_3000": float(ratio_change_extra),
        "ratio_change_r_200_to_3000": float(ratio_change_r),
        "verdict": verdict,
    }
    out_path = "/home/claude/mcq_v4/p6_long_relaxation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
