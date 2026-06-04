"""
6d-γ — P6 : test frontière de la réduction P → un seul scalaire.

Construction : booster face i=0, déprimer face i=4 (additif après modulation).
La face i=2 (contenant (2,2,2)) n'est pas touchée.

Propriétés voulues :
- Δψ_centre ≈ 0
- Barycentre décalé selon −x
- Sort du sous-espace centré-radial où vivaient P1-P5

Préinscription :
- r_6 ≈ w_regime → réduction confirmée même hors du sous-espace centré
- r_6 a une structure non triviale → la réduction n'était que locale au
  sous-espace de P1-P5

Pour mesurer "structure de r_6 hors w_regime", on calcule w_regime à
partir de P1-P5 et on regarde r_6 - w_regime - α_6·u - β_6·v.
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
    """Booste face i=0, déprime face i=4. Face i=2 (centre) inchangée.
    Modulation multiplicative pour préserver positivité de ψ_base."""
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
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    def input_amp(P, s):
        return float(np.linalg.norm(P(psi_base, s) - psi_base))
    amp_target = input_amp(P1prime, 0.05)

    s_P5 = brentq(lambda s: input_amp(P5_neighbors_only, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)
    s_P6 = brentq(lambda s: input_amp(P6_face_dipole, s) - amp_target,
                  1e-4, 0.99, xtol=1e-6)

    print(f"=== P6 calibration ===")
    print(f"  amp_target (||P1''(ψ)-ψ||) : {amp_target:.4e}")
    print(f"  s_P6 calibré : {s_P6:.5f}")
    print(f"  ||P6(ψ)-ψ|| obtenu : {input_amp(P6_face_dipole, s_P6):.4e}")

    # Vérifier que Δψ_centre est bien ≈ 0 et barycentre décentré
    delta_P6 = P6_face_dipole(psi_base, s_P6) - psi_base
    print(f"\n  Δψ_centre (2,2,2) pour P6 : {delta_P6[2,2,2]:+.4e}")

    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    # Barycentre de la perturbation = ∑ x⃗ · Δψ
    mass_p = delta_P6.sum()
    bx = (I * delta_P6).sum()
    by = (J * delta_P6).sum()
    bz = (K * delta_P6).sum()
    print(f"  Barycentre signé de ΔP6 : ({bx:+.4e}, {by:+.4e}, {bz:+.4e})")
    print(f"  (sur P1-P5, ces composantes étaient ≈ 0 par symétrie)")

    # Maintenant, calculer les résidus pour P1-P6
    perturbations = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
        ("P5", lambda p: P5_neighbors_only(p, strength=s_P5)),
        ("P6", lambda p: P6_face_dipole(p, strength=s_P6)),
    ]

    r_psi_list = []
    delta_inputs = []
    for name, P in perturbations:
        d = P(psi_base) - psi_base
        delta_inputs.append(d)
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        r_psi_list.append(psi_R - psi_base)

    # Construire u_ref, v, w_regime à partir de P1-P5 seulement (P6 séparé)
    r_15 = r_psi_list[:5]
    r_1_n = r_15[0] / np.linalg.norm(r_15[0])
    aligned = []
    for r in r_15:
        if (r * r_1_n).sum() < 0:
            aligned.append(-r / np.linalg.norm(r))
        else:
            aligned.append(r / np.linalg.norm(r))
    u_ref = np.mean(aligned, axis=0)
    u_ref = u_ref / np.linalg.norm(u_ref)

    alphas_15 = [float((r * u_ref).sum()) for r in r_15]
    epsilons_15 = [r_15[i] - alphas_15[i] * u_ref for i in range(5)]
    dpsi_center_15 = [float(d[2,2,2]) for d in delta_inputs[:5]]
    c_vec_4 = np.array(dpsi_center_15[:4])
    v_fit = sum(c_vec_4[i] * epsilons_15[i] for i in range(4)) / np.sum(c_vec_4**2)

    # η_floor = moyenne des η pour P1-P5
    v_norm_sq = float((v_fit**2).sum())
    eta_list = []
    for i in range(5):
        beta_i = float((epsilons_15[i] * v_fit).sum()) / v_norm_sq
        eta_i = epsilons_15[i] - beta_i * v_fit
        eta_list.append(eta_i)
    w_regime = np.mean(eta_list, axis=0)

    # Maintenant pour P6 :
    r_6 = r_psi_list[5]
    delta_P6_center = float(delta_inputs[5][2,2,2])
    alpha_6 = float((r_6 * u_ref).sum())
    eps_6 = r_6 - alpha_6 * u_ref
    beta_6 = float((eps_6 * v_fit).sum()) / v_norm_sq
    eta_6 = eps_6 - beta_6 * v_fit
    # Reste après retrait de w_regime
    extra_6 = eta_6 - w_regime

    print(f"\n=== Décomposition de r_6 ===")
    print(f"  Δψ_centre pour P6           : {delta_P6_center:+.4e}")
    print(f"  α_6 = <r_6, u_ref>          : {alpha_6:+.4e}")
    print(f"  β_6 = <ε_6, v> / ||v||²     : {beta_6:+.4e}")
    print(f"  ||r_6||                     : {np.linalg.norm(r_6):.4e}")
    print(f"  ||ε_6||                     : {np.linalg.norm(eps_6):.4e}")
    print(f"  ||η_6||                     : {np.linalg.norm(eta_6):.4e}")
    print(f"  ||w_regime||                : {np.linalg.norm(w_regime):.4e}")
    print(f"  ||η_6 - w_regime|| (extra)  : {np.linalg.norm(extra_6):.4e}")
    print(f"  ratio extra / ||r_6||       : "
          f"{np.linalg.norm(extra_6)/np.linalg.norm(r_6):.4f}")

    # Prédiction selon la réduction "P → 1 scalaire = Δψ_centre"
    # Sur P1-P5 : α_i = κ_α · Δψ_centre + b_α  (b_α petit), β_i = κ_β · Δψ_centre + b_β
    alphas_arr_15 = np.array(alphas_15)
    dpsi_arr_15 = np.array(dpsi_center_15)
    betas_15 = []
    for i in range(5):
        b_i = float((epsilons_15[i] * v_fit).sum()) / v_norm_sq
        betas_15.append(b_i)
    betas_arr_15 = np.array(betas_15)

    # Fit α = κ_α·Δψ + b_α (sur P1-P5 seulement)
    kappa_alpha = np.cov(alphas_arr_15, dpsi_arr_15, ddof=0)[0,1] / np.var(dpsi_arr_15)
    b_alpha = alphas_arr_15.mean() - kappa_alpha * dpsi_arr_15.mean()
    kappa_beta = np.cov(betas_arr_15, dpsi_arr_15, ddof=0)[0,1] / np.var(dpsi_arr_15)
    b_beta = betas_arr_15.mean() - kappa_beta * dpsi_arr_15.mean()
    print(f"\n  Régression sur P1-P5 :")
    print(f"  α = {kappa_alpha:.4e}·Δψ_centre + {b_alpha:.4e}")
    print(f"  β = {kappa_beta:.4e}·Δψ_centre + {b_beta:.4e}")

    alpha_6_pred = kappa_alpha * delta_P6_center + b_alpha
    beta_6_pred = kappa_beta * delta_P6_center + b_beta
    print(f"\n  Prédiction pour P6 (extrapolation depuis P1-P5) :")
    print(f"  α_6 prédit   : {alpha_6_pred:+.4e}   mesuré : {alpha_6:+.4e}")
    print(f"  β_6 prédit   : {beta_6_pred:+.4e}   mesuré : {beta_6:+.4e}")
    print(f"  Écart α      : {alpha_6 - alpha_6_pred:+.4e}  "
          f"({(alpha_6 - alpha_6_pred)/max(abs(alpha_6_pred),1e-30):+.2%})")
    print(f"  Écart β      : {beta_6 - beta_6_pred:+.4e}  "
          f"({(beta_6 - beta_6_pred)/max(abs(beta_6_pred),1e-30):+.2%})")

    # Si réduction tient : extra_6 doit être petit comme les déviations
    # de η_i autour de w_regime (3.5e-6 typique)
    typical_deviation = np.mean([np.linalg.norm(eta_list[i] - w_regime)
                                  for i in range(5)])
    print(f"\n  Comparaison de la 'partie extra' :")
    print(f"  Déviation typique des η_i autour de w_regime (sur P1-P5) : {typical_deviation:.4e}")
    print(f"  ||extra_6 = η_6 - w_regime|| pour P6                     : {np.linalg.norm(extra_6):.4e}")
    print(f"  Ratio P6 / typique                                       : "
          f"{np.linalg.norm(extra_6)/typical_deviation:.2f}")

    # === Inspection spatiale de l'éventuelle "partie extra" ===
    print(f"\n=== Inspection spatiale de extra_6 = η_6 - w_regime ===")
    proj_x = np.abs(extra_6).sum(axis=(1,2))
    proj_y = np.abs(extra_6).sum(axis=(0,2))
    proj_z = np.abs(extra_6).sum(axis=(0,1))
    print(f"  projections |extra_6| sur axes :")
    print(f"    x : {proj_x}")
    print(f"    y : {proj_y}")
    print(f"    z : {proj_z}")
    same_xyz = (np.allclose(proj_x, proj_y, atol=1e-12) and
                np.allclose(proj_x, proj_z, atol=1e-12))
    print(f"  Cubiquement symétrique ? {same_xyz}")
    print(f"  (Si NON et l'asymétrie est selon x : extra encode la")
    print(f"   direction décentrée de P6, donc nouvelle variable conservée)")

    # Argmax et signe
    argmax_e = np.unravel_index(int(np.argmax(np.abs(extra_6))), extra_6.shape)
    print(f"  argmax|extra_6| = {argmax_e}, valeur = {extra_6[argmax_e]:+.4e}")

    # Asymétrie selon x : ||extra_6(i,j,k) - extra_6(4-i,j,k)|| / ||extra_6||
    extra_flipped = extra_6[::-1, :, :]
    asym = float(np.linalg.norm(extra_6 - (-extra_flipped))) / max(np.linalg.norm(extra_6), 1e-30)
    sym = float(np.linalg.norm(extra_6 - extra_flipped)) / max(np.linalg.norm(extra_6), 1e-30)
    print(f"  asymétrie selon x (composante impaire) : {asym:.4f}")
    print(f"  symétrie selon x (composante paire)   : {sym:.4f}")

    # === Verdict préinscrit ===
    print(f"\n=== Verdict ===")
    extra_norm = np.linalg.norm(extra_6)
    r_norm = np.linalg.norm(r_6)
    if extra_norm < 2 * typical_deviation:
        verdict = ("RÉDUCTION CONFIRMÉE HORS SOUS-ESPACE CENTRÉ : "
                   "||extra_6|| est du même ordre que les déviations typiques de "
                   "P1-P5 autour de w_regime. Pas de nouvelle variable détectée.")
    elif extra_norm > 10 * typical_deviation:
        verdict = ("NOUVELLE VARIABLE DÉTECTÉE : ||extra_6|| est >> que les "
                   "déviations typiques. P6 a sollicité une dimension conservée "
                   "que P1-P5 ne sollicitaient pas. La réduction P → 1 scalaire "
                   "n'était valable que dans le sous-espace centré.")
    else:
        verdict = (f"LIMITE INTERMÉDIAIRE : extra ~ {extra_norm/typical_deviation:.1f}× "
                   f"la déviation typique. Effet présent mais d'ordre comparable.")
    print(f"  {verdict}")

    output = {
        "amp_target": amp_target,
        "s_P6_calibrated": float(s_P6),
        "delta_P6_center": delta_P6_center,
        "barycenter_P6": [float(bx), float(by), float(bz)],
        "alpha_6": alpha_6,
        "beta_6": beta_6,
        "alpha_6_pred_from_P15": float(alpha_6_pred),
        "beta_6_pred_from_P15": float(beta_6_pred),
        "alpha_residual": float(alpha_6 - alpha_6_pred),
        "beta_residual": float(beta_6 - beta_6_pred),
        "r_6_norm": float(np.linalg.norm(r_6)),
        "extra_6_norm": float(extra_norm),
        "typical_deviation_P15": float(typical_deviation),
        "ratio_extra_to_typical": float(extra_norm / typical_deviation),
        "asym_x_extra": asym,
        "sym_x_extra": sym,
        "cubic_sym_extra": bool(same_xyz),
        "verdict": verdict,
    }
    out_path = "/home/claude/mcq_v4/p6_decentered_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
