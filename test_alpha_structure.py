"""
6d-γ — α transporte-t-il une information sur P ?

Asymétrie de traitement identifiée par Alex :
- β a été examiné avec 4 proxies en compétition, test de décorrélation, test interne P1/P4
- η a été décomposé en floor + variation, signe étudié
- α n'a jamais été testé. Considéré "essentiellement uniforme" sur la base
  d'une remarque rapide (1.27e-3 à 1.42e-3, variation de 12%)

Mais 12% de variation peut très bien encoder une structure.

Trois questions :
1. α varie-t-il significativement ?
2. Si oui, sa variation est-elle structurée ?
3. Si oui, l'information dans α est-elle la même que dans β, ou indépendante ?
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

    def input_amp(P, s):
        return float(np.linalg.norm(P(psi_base, s) - psi_base))
    amp_target = input_amp(P1prime, 0.05)
    s_P5 = brentq(lambda s: input_amp(P5_neighbors_only, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)

    perturbations = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
        ("P5", lambda p: P5_neighbors_only(p, strength=s_P5)),
    ]

    # Calcul des résidus + de plusieurs caractéristiques de P
    r_psi_list = []
    delta_inputs = []
    for name, P in perturbations:
        delta = P(psi_base) - psi_base
        delta_inputs.append(delta)
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        r_psi_list.append(psi_R - psi_base)

    # u_ref signe-harmonisé
    r_1_n = r_psi_list[0] / np.linalg.norm(r_psi_list[0])
    aligned = []
    for r in r_psi_list:
        if (r * r_1_n).sum() < 0:
            aligned.append(-r / np.linalg.norm(r))
        else:
            aligned.append(r / np.linalg.norm(r))
    u_ref = np.mean(aligned, axis=0)
    u_ref = u_ref / np.linalg.norm(u_ref)

    # === Question 1 : α varie-t-il ? ===
    print(f"=== Question 1 : α_i varie-t-il significativement ? ===\n")
    alphas = []
    for r in r_psi_list:
        a = float((r * u_ref).sum())
        alphas.append(a)

    names = [p[0] for p in perturbations]
    print(f"  {'cas':<6} {'α_i':>14} {'||r_i||':>14} {'α_i/||r_i||':>14}")
    for i, name in enumerate(names):
        r_norm = np.linalg.norm(r_psi_list[i])
        print(f"  {name:<6} {alphas[i]:>14.4e} {r_norm:>14.4e} "
              f"{alphas[i]/r_norm:>14.4f}")
    alpha_mean = float(np.mean(alphas))
    alpha_std = float(np.std(alphas))
    print(f"\n  α moyen : {alpha_mean:.4e}")
    print(f"  α écart-type : {alpha_std:.4e}")
    print(f"  Coefficient de variation σ/μ : {alpha_std/abs(alpha_mean):.4f}")
    print(f"  → si σ/μ < 0.05 : α essentiellement constant")
    print(f"  → si σ/μ > 0.20 : α varie significativement")

    # === Question 2 : si α varie, est-ce structuré ? ===
    print(f"\n=== Question 2 : la variation de α est-elle structurée ? ===\n")
    # Plusieurs caractéristiques scalaires des perturbations à tester comme prédicteurs
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    cc = (N_AXIS - 1) / 2.0
    dist_center = np.sqrt((I-cc)**2 + (J-cc)**2 + (K-cc)**2)

    chars = {}
    chars["delta_center"] = np.array([float(d[2,2,2]) for d in delta_inputs])
    chars["L2_norm_input"] = np.array([float(np.linalg.norm(d)) for d in delta_inputs])
    chars["sum_signed"] = np.array([float(d.sum()) for d in delta_inputs])  # masse signée
    chars["L1_norm_input"] = np.array([float(np.abs(d).sum()) for d in delta_inputs])
    chars["mean_dist_center"] = np.array([
        float((np.abs(d) * dist_center).sum() / max(np.abs(d).sum(), 1e-30))
        for d in delta_inputs
    ])
    chars["delta_periph_mean"] = np.array([
        float(d[dist_center >= 2.5].mean()) for d in delta_inputs
    ])
    chars["delta_first_shell"] = np.array([
        float(d[np.isclose(dist_center, 1.0, atol=1e-4)].mean()) for d in delta_inputs
    ])

    print(f"  Caractéristiques scalaires de P par cas :")
    print(f"  {'char':<25}" + "".join(f"{n:>13}" for n in names))
    for k, v in chars.items():
        row = f"  {k:<25}"
        for x in v:
            row += f"{x:>+13.4e}"
        print(row)
    row = f"  {'α (à expliquer)':<25}"
    for x in alphas: row += f"{x:>+13.4e}"
    print(row)

    # Corrélations entre α et chaque caractéristique
    print(f"\n  Corrélations entre α et chaque caractéristique :")
    alpha_array = np.array(alphas)
    for k, v in chars.items():
        if np.std(v) > 1e-30 and np.std(alpha_array) > 1e-30:
            corr = float(np.corrcoef(alpha_array, v)[0,1])
            print(f"    {k:<25}: corr(α, {k}) = {corr:+.4f}")

    # === Question 3 : si α varie, est-ce indépendant de β ? ===
    print(f"\n=== Question 3 : α encode-t-il la même information que β ? ===\n")
    beta_array = chars["delta_center"]
    if np.std(alpha_array) > 1e-30 and np.std(beta_array) > 1e-30:
        corr_alpha_beta = float(np.corrcoef(alpha_array, beta_array)[0,1])
        print(f"  Corrélation α vs β (= Δψ_centre) : {corr_alpha_beta:+.4f}")
        if abs(corr_alpha_beta) < 0.3:
            print(f"  → α est essentiellement INDÉPENDANT de β.")
            print(f"  Si α varie de façon structurée, c'est une SECONDE")
            print(f"  variable conservée, différente de β.")
        elif abs(corr_alpha_beta) > 0.9:
            print(f"  → α et β co-varient très fortement.")
            print(f"  α n'apporte pas d'information indépendante de β.")
        else:
            print(f"  → corrélation intermédiaire. α porte peut-être de")
            print(f"    l'information partiellement indépendante.")

    # Régression de α sur les caractéristiques
    # Pour chaque caractéristique, faire un fit α ≈ a_c · char + b_c
    # et regarder la fraction de variance expliquée
    print(f"\n  Variance de α expliquée par chaque caractéristique seule :")
    var_alpha = np.var(alpha_array)
    if var_alpha > 1e-30:
        for k, v in chars.items():
            if np.std(v) > 1e-30:
                slope = np.cov(alpha_array, v)[0,1] / np.var(v)
                intercept = np.mean(alpha_array) - slope * np.mean(v)
                pred = slope * v + intercept
                residual = alpha_array - pred
                var_explained = 1.0 - np.var(residual) / var_alpha
                print(f"    {k:<25}: R² = {var_explained:.4f}")

    output = {
        "perturbations": names,
        "alphas": alphas,
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "alpha_cv": alpha_std / abs(alpha_mean),
        "characteristics": {k: v.tolist() for k, v in chars.items()},
        "correlations_alpha_chars": {
            k: float(np.corrcoef(alpha_array, v)[0,1])
            for k, v in chars.items() if np.std(v) > 1e-30
        },
        "r_squared_by_char": {
            k: float(1.0 - np.var(alpha_array - (np.cov(alpha_array, v)[0,1] / np.var(v)) * v - (np.mean(alpha_array) - (np.cov(alpha_array, v)[0,1] / np.var(v)) * np.mean(v))) / np.var(alpha_array))
            for k, v in chars.items() if np.std(v) > 1e-30
        },
    }
    out_path = "/home/claude/mcq_v4/alpha_structure.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
