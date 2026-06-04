"""
6d-γ — Statut de Δψ_centre : variable conservée réelle, ou meilleur
proxy actuellement connu d'une variable plus profonde ?

Distinction (Alex) :
- Lecture (1) : Δψ_centre EST la variable conservée. α et β en sont des
  projections redondantes. Coefficients linéaires exacts attendus.
- Lecture (2) : Δψ_centre est le meilleur proxy actuel. α et β sont
  des projections d'une variable plus profonde c(P), dont Δψ_centre
  est très corrélé avec mais pas identique.

Trois tests sur les données existantes :
(I) la relation α = κ·Δψ_centre est-elle linéaire EXACTE (R²=1.0) ou
    a-t-elle une déviation systématique (R² < 1.0) ?
(II) une combinaison de caractéristiques prédit-elle α mieux que
     Δψ_centre seul ?
(III) β défini comme "projection de ε sur v(centre)" est-il
      strictement identique à Δψ_centre, ou en diverge-t-il ?
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

    r_1_n = r_psi_list[0] / np.linalg.norm(r_psi_list[0])
    aligned = []
    for r in r_psi_list:
        if (r * r_1_n).sum() < 0:
            aligned.append(-r / np.linalg.norm(r))
        else:
            aligned.append(r / np.linalg.norm(r))
    u_ref = np.mean(aligned, axis=0)
    u_ref = u_ref / np.linalg.norm(u_ref)

    alphas = [float((r * u_ref).sum()) for r in r_psi_list]
    epsilons = [r_psi_list[i] - alphas[i] * u_ref for i in range(5)]
    dpsi_center = [float(d[2,2,2]) for d in delta_inputs]

    # Pour β défini comme "projection de ε sur la direction v"
    # v ajusté sur P1-P4 (P5 séparé)
    c_vec_4 = np.array(dpsi_center[:4])
    v_fit = sum(c_vec_4[i] * epsilons[i] for i in range(4)) / np.sum(c_vec_4**2)
    v_norm = float(np.linalg.norm(v_fit))
    # β comme projection libre (sans imposer = Δψ_centre)
    betas_free = [float((eps * v_fit).sum()) / (v_norm**2) for eps in epsilons]

    names = [p[0] for p in perturbations]
    print(f"=== (I) Linéarité de α vs Δψ_centre ===\n")
    print(f"  {'cas':<6} {'α':>14} {'Δψ_centre':>14} {'résidu lin':>14}")
    alpha_arr = np.array(alphas)
    dpsi_arr = np.array(dpsi_center)
    # Fit α = κ·Δψ_centre + cst
    slope = float(np.cov(alpha_arr, dpsi_arr, ddof=0)[0,1] / np.var(dpsi_arr))
    intercept = float(alpha_arr.mean() - slope * dpsi_arr.mean())
    pred = slope * dpsi_arr + intercept
    residuals_lin = alpha_arr - pred
    for i, name in enumerate(names):
        print(f"  {name:<6} {alpha_arr[i]:>14.6e} {dpsi_arr[i]:>14.6e} "
              f"{residuals_lin[i]:>+14.4e}")
    R2_lin = 1.0 - np.var(residuals_lin) / np.var(alpha_arr)
    print(f"\n  fit linéaire α = {slope:.4e}·Δψ_centre + {intercept:.4e}")
    print(f"  R² (exact) = {R2_lin:.6f}")
    print(f"  Si R² = 1.0 exactement → Lecture (1) appuyée")
    print(f"  Si R² < 1.0 avec résidus structurés → Lecture (2) appuyée")

    # Examiner si les résidus de fit sont aléatoires ou structurés
    print(f"\n  Résidus de la régression linéaire :")
    print(f"  {'cas':<6} {'résidu':>14} {'résidu / σ_α':>14}")
    sigma_alpha = float(np.std(alpha_arr))
    for i, name in enumerate(names):
        print(f"  {name:<6} {residuals_lin[i]:>+14.4e} "
              f"{residuals_lin[i]/sigma_alpha:>+14.4f}")
    # Ces résidus sont-ils dominés par un cas spécifique ou bien répartis ?
    abs_residuals = np.abs(residuals_lin)
    max_res = float(abs_residuals.max())
    sum_res = float(abs_residuals.sum())
    print(f"\n  Max résidu : {max_res:.4e}, somme |résidus| : {sum_res:.4e}")
    print(f"  Concentration : max/sum = {max_res/sum_res:.4f}")

    # === (II) Régression multiple : α = κ·Δψ_centre + autres ?
    print(f"\n=== (II) α peut-il être mieux prédit par autre chose ===")
    print(f"  ou une combinaison ? ===\n")

    # On essaye des caractéristiques additionnelles à Δψ_centre
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    cc = (N_AXIS - 1) / 2.0
    dist_center_grid = np.sqrt((I-cc)**2 + (J-cc)**2 + (K-cc)**2)

    chars = {
        "delta_center": dpsi_arr,
        "delta_first_shell_mean": np.array([
            float(d[np.isclose(dist_center_grid, 1.0, atol=1e-4)].mean())
            for d in delta_inputs
        ]),
        "delta_corner_mean": np.array([
            float(d[np.isclose(dist_center_grid, 3.464, atol=1e-4)].mean())
            for d in delta_inputs
        ]),
        "L1_norm_input": np.array([float(np.abs(d).sum()) for d in delta_inputs]),
        "L2_squared_input": np.array([float((d**2).sum()) for d in delta_inputs]),
        # Une caractéristique non-linéaire : Δψ_centre × √(masse totale)
        "dpsi_center_x_L1": dpsi_arr * np.array([float(np.abs(d).sum()) for d in delta_inputs]),
    }
    # NOTE : avec 5 points, on ne peut pas faire de régression multiple
    # avec plus de 2 ou 3 régresseurs sans surapprentissage.
    # On compare seulement les R² seuls.
    print(f"  R² seul de chaque caractéristique pour prédire α :")
    R2_results = {}
    for k, v in chars.items():
        if np.std(v) > 1e-30 and np.std(alpha_arr) > 1e-30:
            slope_c = np.cov(alpha_arr, v, ddof=0)[0,1] / np.var(v)
            interc_c = alpha_arr.mean() - slope_c * v.mean()
            pred_c = slope_c * v + interc_c
            res_c = alpha_arr - pred_c
            R2 = 1.0 - np.var(res_c) / np.var(alpha_arr)
            R2_results[k] = R2
            print(f"    {k:<28}: R² = {R2:.4f}")

    # Cherche-t-on un meilleur prédicteur ?
    best_char = max(R2_results, key=R2_results.get)
    print(f"\n  Meilleur prédicteur seul : {best_char} (R² = {R2_results[best_char]:.4f})")

    # Régression bivariée : α = a·Δψ_centre + b·(autre) + c
    # pour voir si une seconde caractéristique réduit encore les résidus
    print(f"\n  Bivarié : α = a·Δψ_centre + b·(autre)·")
    print(f"  Réduction de R² obtenue en ajoutant chaque seconde caractéristique :")
    R2_base = R2_results["delta_center"]
    for k, v in chars.items():
        if k == "delta_center": continue
        # Régression multiple à 2 régresseurs
        X = np.column_stack([dpsi_arr, v, np.ones_like(dpsi_arr)])
        try:
            coefs, _, _, _ = np.linalg.lstsq(X, alpha_arr, rcond=None)
            pred = X @ coefs
            res = alpha_arr - pred
            R2_biv = 1.0 - np.var(res) / np.var(alpha_arr)
            improvement = R2_biv - R2_base
            print(f"    + {k:<25}: R² = {R2_biv:.4f}  (Δ = {improvement:+.4f})")
        except np.linalg.LinAlgError:
            print(f"    + {k:<25}: échec (rang déficient)")

    # === (III) β = projection libre sur v vs β = Δψ_centre ?
    print(f"\n=== (III) β libre (projection sur v) vs β = Δψ_centre ===\n")
    print(f"  {'cas':<6} {'β libre':>14} {'Δψ_centre':>14} {'β/Δψ_c':>14}")
    for i, name in enumerate(names):
        ratio = betas_free[i] / dpsi_arr[i] if abs(dpsi_arr[i]) > 1e-30 else 0
        print(f"  {name:<6} {betas_free[i]:>14.6e} {dpsi_arr[i]:>14.6e} "
              f"{ratio:>14.4f}")

    # Le ratio β/Δψ_centre est-il constant ?
    ratios = np.array([
        betas_free[i] / dpsi_arr[i] if abs(dpsi_arr[i]) > 1e-30 else 0
        for i in range(5)
    ])
    print(f"\n  ratio moyen = {ratios.mean():.4e}, σ = {ratios.std():.4e}")
    print(f"  σ/μ = {ratios.std()/abs(ratios.mean()):.4f}")
    print(f"  Si σ/μ → 0 : β libre = κ·Δψ_centre exactement → identiques")
    print(f"  Si σ/μ > 0 : β libre n'est pas exactement proportionnel à Δψ_centre")

    output = {
        "perturbations": names,
        "alphas": alphas,
        "dpsi_center": dpsi_center,
        "betas_free": betas_free,
        "linear_fit_slope": slope,
        "linear_fit_intercept": intercept,
        "linear_fit_R2": R2_lin,
        "linear_residuals": residuals_lin.tolist(),
        "R2_each_char": R2_results,
        "betas_free_over_dpsi_c": ratios.tolist(),
        "betas_free_over_dpsi_c_cv": float(ratios.std() / abs(ratios.mean())),
    }
    out_path = "/home/claude/mcq_v4/dpsi_center_status.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
