"""
6d-γ — Analyse du résidu β_libre - κ·Δψ_centre.

Asymétrie observée :
- α : R²(Δψ_centre) = 1.000000 (exact)
- β_libre : σ/μ du ratio β/Δψ_centre = 0.22 (22% de variabilité)

Question (Alex) : pourquoi α se ferme exactement alors que β reste
partiellement ouvert ?

Test sur les données existantes :
(1) la structure des résidus β - κ·Δψ_centre est-elle aléatoire ou
    organisée (signes systématiques) ?
(2) ces résidus corrèlent-ils avec une caractéristique identifiable
    de P (autre que Δψ_centre) ?
(3) sont-ils dans la même direction spatiale que v (β mal ajusté) ou
    dans une direction différente (information supplémentaire dans β) ?
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
    c_vec_4 = np.array(dpsi_center[:4])
    v_fit = sum(c_vec_4[i] * epsilons[i] for i in range(4)) / np.sum(c_vec_4**2)
    v_norm_sq = float((v_fit**2).sum())
    betas_free = [float((eps * v_fit).sum()) / v_norm_sq for eps in epsilons]

    names = [p[0] for p in perturbations]
    dpsi_arr = np.array(dpsi_center)
    beta_arr = np.array(betas_free)

    # === (1) Structure des résidus β - κ·Δψ_centre ===
    print(f"=== (1) Structure des résidus de β vs κ·Δψ_centre ===\n")
    # Fit linéaire optimal β = κ·Δψ_centre + b
    slope = float(np.cov(beta_arr, dpsi_arr, ddof=0)[0,1] / np.var(dpsi_arr))
    intercept = float(beta_arr.mean() - slope * dpsi_arr.mean())
    pred = slope * dpsi_arr + intercept
    residuals = beta_arr - pred
    R2 = 1.0 - np.var(residuals) / np.var(beta_arr)
    print(f"  fit linéaire β = {slope:.4e}·Δψ_centre + {intercept:.4e}")
    print(f"  R² = {R2:.6f}")
    print(f"  {'cas':<6} {'β':>14} {'Δψ_c':>14} {'pred':>14} {'résidu':>14}")
    for i in range(5):
        print(f"  {names[i]:<6} {beta_arr[i]:>+14.4e} {dpsi_arr[i]:>+14.4e} "
              f"{pred[i]:>+14.4e} {residuals[i]:>+14.4e}")

    # === (2) Corrélation des résidus avec d'autres caractéristiques ===
    print(f"\n=== (2) Corrélations des résidus β avec autres ===\n")
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    cc = (N_AXIS - 1) / 2.0
    dist_center_grid = np.sqrt((I-cc)**2 + (J-cc)**2 + (K-cc)**2)

    chars = {
        "L1_norm_input": np.array([float(np.abs(d).sum()) for d in delta_inputs]),
        "L2_squared_input": np.array([float((d**2).sum()) for d in delta_inputs]),
        "delta_first_shell_mean": np.array([
            float(d[np.isclose(dist_center_grid, 1.0, atol=1e-4)].mean())
            for d in delta_inputs
        ]),
        "delta_corner_mean": np.array([
            float(d[np.isclose(dist_center_grid, 3.464, atol=1e-4)].mean())
            for d in delta_inputs
        ]),
        "delta_periphery_mean": np.array([
            float(d[dist_center_grid >= 2.5].mean())
            for d in delta_inputs
        ]),
        "ratio_first_to_center": np.array([
            float(d[np.isclose(dist_center_grid, 1.0, atol=1e-4)].mean() / d[2,2,2])
            if abs(d[2,2,2]) > 1e-30 else 0.0
            for d in delta_inputs
        ]),
        "Delta_psi_centre_squared": dpsi_arr**2,
        "abs_Delta_psi_centre": np.abs(dpsi_arr),
    }

    print(f"  Corrélations entre les résidus et caractéristiques de P :")
    for k, v in chars.items():
        if np.std(v) > 1e-30 and np.std(residuals) > 1e-30:
            corr = float(np.corrcoef(residuals, v)[0,1])
            print(f"    {k:<28}: corr = {corr:+.4f}")

    # === (3) Direction spatiale du résidu de β ===
    # Calculer pour chaque i : ε_i - β_pred_i · v_fit (ce qui reste après
    # prédiction de β par κ·Δψ_centre)
    # Et regarder si ces "résidus dans l'espace" sont dans la direction de v
    # ou ailleurs
    print(f"\n=== (3) Direction spatiale du résidu ===\n")
    print(f"  Pour chaque cas, on calcule :")
    print(f"  diff_i = ε_i - (κ·Δψ_centre)_i · v_fit")
    print(f"  Si diff_i ∝ v : β mal ajusté mais bonne direction")
    print(f"  Si diff_i ⊥ v : information supplémentaire dans une autre direction\n")

    v_normed = v_fit / np.sqrt(v_norm_sq)
    print(f"  {'cas':<6} {'||diff_i||':>14} {'cos(diff,v)':>14} {'(diff,v)':>14}")
    spatial_residuals = []
    for i in range(5):
        beta_pred = slope * dpsi_arr[i] + intercept
        diff = epsilons[i] - beta_pred * v_fit
        spatial_residuals.append(diff)
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm > 1e-30:
            cos_to_v = float((diff * v_normed).sum() / diff_norm)
        else:
            cos_to_v = 0.0
        # Décomposer en composante v et orthogonale
        comp_v = float((diff * v_normed).sum())
        print(f"  {names[i]:<6} {diff_norm:>14.4e} {cos_to_v:>+14.4f} "
              f"{comp_v:>+14.4e}")

    # Les diff_i sont-ils dans la même direction entre eux ?
    diffs_normed = []
    for d in spatial_residuals:
        n = np.linalg.norm(d)
        if n > 1e-30:
            diffs_normed.append(d / n)
        else:
            diffs_normed.append(None)

    print(f"\n  cos(angle) entre les diff_i (normalisés) :")
    for i in range(5):
        for j in range(i+1, 5):
            if diffs_normed[i] is not None and diffs_normed[j] is not None:
                cos_ij = float((diffs_normed[i] * diffs_normed[j]).sum())
                print(f"    {names[i]}-{names[j]} : {cos_ij:+.4f}")

    # === Profil radial signé du diff_P5 (le plus éloigné) ===
    print(f"\n=== Profil radial signé de diff_P5 (le résidu le plus net) ===\n")
    diff_P5 = spatial_residuals[4]
    unique_dists = sorted(set(np.round(dist_center_grid.flatten(), 4)))
    for d in unique_dists:
        mask = np.isclose(dist_center_grid, d, atol=1e-4)
        mean_v = diff_P5[mask].mean()
        print(f"  r={d:.3f} : <diff_P5> = {mean_v:+.4e}  ({mask.sum()} cellules)")

    output = {
        "perturbations": names,
        "dpsi_center": dpsi_center,
        "betas_free": betas_free,
        "fit_slope": slope,
        "fit_intercept": intercept,
        "R2": R2,
        "residuals_beta": residuals.tolist(),
        "correlations_residuals_chars": {
            k: float(np.corrcoef(residuals, v)[0,1])
            for k, v in chars.items() if np.std(v) > 1e-30
        },
        "spatial_residuals_norms": [float(np.linalg.norm(d)) for d in spatial_residuals],
        "spatial_residuals_cos_with_v": [
            float((d * v_normed).sum() / max(np.linalg.norm(d), 1e-30))
            for d in spatial_residuals
        ],
    }
    out_path = "/home/claude/mcq_v4/beta_residual_structure.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
