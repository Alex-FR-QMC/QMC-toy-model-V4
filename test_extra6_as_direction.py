"""
6d-γ — extra_6 est-il un mode nouveau du régime ou un défaut de la base ?

Méthodologie (Alex) : définir q = extra_6 normalisé, reprojeter P1-P5
sur q, et regarder les γ_i = <r_i, q>.

- γ_i ≈ 0 pour P1-P5 et γ_6 ≠ 0 : mode nouveau sollicité par P6 seul
- γ_i ≠ 0 et non corrélé à Δψ_centre : mode existant masqué par mauvaise base
- γ_i ≠ 0 et corrélé à Δψ_centre : direction mélangée à v/u_ref, P6 décolle
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

    # Construire la base (u, v, w_regime) à partir de P1-P5
    r_15 = r_psi_list[:5]
    r_1_n = r_15[0] / np.linalg.norm(r_15[0])
    aligned = [(-r if (r*r_1_n).sum() < 0 else r) / np.linalg.norm(r)
               for r in r_15]
    u_ref = np.mean(aligned, axis=0)
    u_ref = u_ref / np.linalg.norm(u_ref)

    alphas_15 = [float((r * u_ref).sum()) for r in r_15]
    epsilons_15 = [r_15[i] - alphas_15[i] * u_ref for i in range(5)]
    dpsi_center = [float(d[2,2,2]) for d in delta_inputs]
    c_vec_4 = np.array(dpsi_center[:4])
    v_fit = sum(c_vec_4[i] * epsilons_15[i] for i in range(4)) / np.sum(c_vec_4**2)
    v_norm_sq = float((v_fit**2).sum())

    eta_list = []
    for i in range(5):
        beta_i = float((epsilons_15[i] * v_fit).sum()) / v_norm_sq
        eta_i = epsilons_15[i] - beta_i * v_fit
        eta_list.append(eta_i)
    w_regime = np.mean(eta_list, axis=0)

    # Calculer extra_6 = η_6 - w_regime
    r_6 = r_psi_list[5]
    alpha_6 = float((r_6 * u_ref).sum())
    eps_6 = r_6 - alpha_6 * u_ref
    beta_6 = float((eps_6 * v_fit).sum()) / v_norm_sq
    eta_6 = eps_6 - beta_6 * v_fit
    extra_6 = eta_6 - w_regime

    # q = extra_6 normalisé
    extra_norm = np.linalg.norm(extra_6)
    q = extra_6 / extra_norm

    # === Reprojection : γ_i = <r_i, q> pour TOUS les 6 cas ===
    print(f"=== Test : extra_6 mode nouveau ou défaut de base ? ===\n")
    print(f"  q = extra_6 normalisé (direction candidate à un nouveau mode)")
    print(f"  ||extra_6|| = {extra_norm:.4e}\n")

    names = [p[0] for p in perturbations]
    print(f"  γ_i = <r_i, q> pour les 6 cas :")
    print(f"  {'cas':<6} {'γ_i':>16} {'Δψ_centre':>16} {'γ_i/extra_norm':>16}")

    gammas = []
    for i in range(6):
        g = float((r_psi_list[i] * q).sum())
        gammas.append(g)
        ratio = g / extra_norm
        print(f"  {names[i]:<6} {g:>+16.4e} {dpsi_center[i]:>+16.4e} "
              f"{ratio:>+16.4f}")

    # === Interprétation préinscrite ===
    print(f"\n=== Analyse ===")
    gammas_15 = gammas[:5]
    gamma_6 = gammas[5]
    gamma_15_arr = np.array(gammas_15)
    dpsi_15_arr = np.array(dpsi_center[:5])

    typical_gamma_15 = float(np.std(gamma_15_arr))
    max_abs_gamma_15 = float(np.max(np.abs(gamma_15_arr)))
    ratio_6_to_15 = abs(gamma_6) / max(max_abs_gamma_15, 1e-30)

    print(f"  σ(γ_1..γ_5)   = {typical_gamma_15:.4e}")
    print(f"  max|γ_1..γ_5| = {max_abs_gamma_15:.4e}")
    print(f"  |γ_6|         = {abs(gamma_6):.4e}")
    print(f"  Ratio γ_6 / max(γ_1..γ_5) = {ratio_6_to_15:.2f}")

    # Corrélation γ_15 avec Δψ_centre
    if np.std(gamma_15_arr) > 1e-30 and np.std(dpsi_15_arr) > 1e-30:
        corr = float(np.corrcoef(gamma_15_arr, dpsi_15_arr)[0,1])
        print(f"\n  corr(γ_1..γ_5, Δψ_centre) = {corr:+.4f}")
        if abs(corr) > 0.9:
            print(f"  → γ_i sur P1-P5 corrèle fortement avec Δψ_centre.")
            print(f"  q était déjà mélangé à u_ref ou v dans P1-P5.")
        elif abs(corr) < 0.3:
            print(f"  → γ_i sur P1-P5 ne corrèle pas avec Δψ_centre.")
            print(f"  Si γ_i ne sont pas négligeables, q est un autre mode")
            print(f"  qui existait mais a été classé dans w_regime.")
    else:
        corr = None

    # Verdict préinscrit
    print(f"\n=== Verdict préinscrit ===")
    if ratio_6_to_15 > 100:
        verdict = ("MODE NOUVEAU SOLLICITÉ PAR P6 SEUL : γ_6 >> max|γ_1..γ_5|.")
    elif ratio_6_to_15 < 3:
        verdict = ("DÉFAUT DE BASE : γ_i existaient déjà pour P1-P5 dans le même ordre.")
    else:
        verdict = (f"INTERMÉDIAIRE : γ_6 est {ratio_6_to_15:.1f}× plus grand que")
        verdict += " les γ_i de P1-P5."
    print(f"  {verdict}")

    # Mais : caveat important — la normalisation
    # γ_i / ||r_i|| serait peut-être plus parlant
    print(f"\n=== Caveat : γ_i normalisé par ||r_i|| (mesure relative) ===")
    print(f"  {'cas':<6} {'γ_i':>14} {'||r_i||':>14} {'γ_i/||r_i||':>14}")
    rels = []
    for i in range(6):
        r_norm = np.linalg.norm(r_psi_list[i])
        rel = gammas[i] / r_norm
        rels.append(rel)
        print(f"  {names[i]:<6} {gammas[i]:>+14.4e} {r_norm:>14.4e} "
              f"{rel:>+14.4f}")
    rels_15 = np.array(rels[:5])
    rel_6 = rels[5]
    print(f"\n  Sur P1-P5 : γ_i/||r_i|| varie de {rels_15.min():+.4f} à {rels_15.max():+.4f}")
    print(f"  Pour P6   : γ_6/||r_6|| = {rel_6:+.4f}")
    ratio_rel = abs(rel_6) / max(np.max(np.abs(rels_15)), 1e-30)
    print(f"  Ratio relatif : {ratio_rel:.2f}")

    output = {
        "perturbations": names,
        "gammas": gammas,
        "dpsi_center": dpsi_center,
        "extra_6_norm": float(extra_norm),
        "sigma_gamma_15": typical_gamma_15,
        "max_abs_gamma_15": max_abs_gamma_15,
        "gamma_6": float(gamma_6),
        "ratio_6_to_15": float(ratio_6_to_15),
        "corr_gamma_15_with_dpsi": corr,
        "gammas_normalized_by_r_norm": rels,
        "verdict": verdict,
    }
    out_path = "/home/claude/mcq_v4/extra6_as_direction.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
