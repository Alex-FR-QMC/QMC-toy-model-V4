"""
6d-δ — Structure géométrique du résidu vectoriel r_τ.

Analyse sur les données déjà produites.

Trois questions :

(1) Projection spatiale du résidu par axe : asymétrique selon x
    (comme P6), ou symétrique cubique (comme P1-P5) ?

(2) Comparaison avec extra_6 du cycle 6d-γ : r_τ ressemble-t-il
    au mode lent identifié précédemment ?

(3) Localisation temporelle : la contribution à χ_slow vient-elle
    des premiers instants après P', ou est-elle répartie ?
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

def evolve_with_trajectory(psi, h, D, beta, gamma, h0, dt, n_steps):
    psis = np.zeros((n_steps + 1,) + psi.shape)
    hs = np.zeros((n_steps + 1,) + h.shape)
    psis[0] = psi
    hs[0] = h
    for n in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        psis[n + 1] = psi
        hs[n + 1] = h
    return psis, hs

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


def P6_face_dipole(psi, strength):
    factor = np.ones_like(psi)
    factor[0, :, :] += strength
    factor[4, :, :] -= strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()

def P_prime(psi, strength, sigma_p=0.8):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                factor[i,j,k] += strength * np.exp(-0.5 * r2 / sigma_p**2)
    p = psi * factor
    return p / p.sum()


# Reconstruire extra_6 (référence du cycle 6d-γ)
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


def compute_delta(psi_start, h_start, s_P_prime, D, beta, gamma_v, h0, dt, n_dt):
    psis_ref, hs_ref = evolve_with_trajectory(
        psi_start.copy(), h_start.copy(),
        D, beta, gamma_v, h0, dt, n_dt)
    psi_pp = P_prime(psi_start.copy(), s_P_prime)
    h_pp = h_start.copy()
    psis_with, hs_with = evolve_with_trajectory(
        psi_pp, h_pp, D, beta, gamma_v, h0, dt, n_dt)
    return psis_with - psis_ref, hs_with - hs_ref


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_dt_response = int(100.0 / dt)
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    # Calibrages
    def input_amp(P, s):
        return float(np.linalg.norm(P(psi_base, s) - psi_base))
    amp_target = input_amp(P1prime, 0.05)
    s_P5 = brentq(lambda s: input_amp(P5_neighbors_only, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)
    s_P6 = brentq(lambda s: input_amp(P6_face_dipole, s) - amp_target,
                  1e-4, 0.99, xtol=1e-6)
    s_P_prime = 0.008385

    # === Reconstruction de extra_6 (référence 6d-γ) ===
    print(f"=== Reconstruction de extra_6 du cycle 6d-γ ===")
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
    R_base_psi = np.column_stack([r.flatten() for r in r_15])

    psi_p6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_p6 = h_base.copy()
    psi_R6, h_R6 = evolve(psi_p6, h_p6, D, beta_lock, gamma_v, h0, dt, n_short)
    psi_R6, h_R6 = evolve(psi_R6, h_R6, D, beta_lock, gamma_v, h0, dt, n_long)
    r_6 = psi_R6 - psi_base
    coefs, _, _, _ = np.linalg.lstsq(R_base_psi, r_6.flatten(), rcond=None)
    extra_6_flat = r_6.flatten() - R_base_psi @ coefs
    extra_6 = extra_6_flat.reshape(r_6.shape)
    print(f"  ||extra_6|| = {np.linalg.norm(extra_6):.4e}")

    # === États P6(τ) et trajectoires Δ ===
    delays_target = [0, 50, 100, 200, 400, 800, 1500, 3000]
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_R6_init, h_R6_init = evolve(psi_P6, h_P6,
                                     D, beta_lock, gamma_v, h0, dt, n_short)
    states = {0: (psi_R6_init.copy(), h_R6_init.copy())}
    psi_curr, h_curr = psi_R6_init.copy(), h_R6_init.copy()
    t_current = 0.0
    for tau in delays_target[1:]:
        n_seg = int((tau - t_current) / dt)
        if n_seg > 0:
            psi_curr, h_curr = evolve(psi_curr, h_curr,
                                       D, beta_lock, gamma_v, h0, dt, n_seg)
        t_current = tau
        states[tau] = (psi_curr.copy(), h_curr.copy())

    deltas = {}
    for tau in delays_target:
        psi_t, h_t = states[tau]
        d_psi, d_h = compute_delta(psi_t, h_t, s_P_prime,
                                    D, beta_lock, gamma_v, h0, dt, n_dt_response)
        deltas[tau] = (d_psi, d_h)

    # Référence Δ_3000
    d_psi_ref, d_h_ref = deltas[3000]

    # === ANALYSE (1) : projection spatiale du résidu par axe ===
    print(f"\n=== (1) Symétries spatiales du résidu r_τ ===\n")
    print(f"  Pour chaque τ, on regarde la projection spatiale du résidu")
    print(f"  r_τ(t) intégré en temps :  R_τ(i,j,k) = ∫|r_τ_psi(t,i,j,k)| dt\n")

    print(f"  τ=0 : projections selon les trois axes")
    d_psi_0 = deltas[0][0]
    # Calculer r_τ pour τ=0 (composante orthogonale à Δ_3000)
    # On le fait sur la composante psi uniquement pour la visualisation spatiale
    ref_flat = np.concatenate([d_psi_ref.flatten(), deltas[3000][1].flatten()])
    ref_norm_sq = float(np.dot(ref_flat, ref_flat))

    spatial_data = {}
    for tau in [0, 50, 100, 200]:
        d_psi_tau, d_h_tau = deltas[tau]
        d_flat = np.concatenate([d_psi_tau.flatten(), d_h_tau.flatten()])
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq)
        # Résidu = Δ_τ - a_τ · Δ_ref
        r_psi = d_psi_tau - a_tau * d_psi_ref
        # Intégrer le module en temps
        r_psi_intgr = np.sum(np.abs(r_psi), axis=0)  # somme sur t
        # Projections sur chaque axe
        proj_x = r_psi_intgr.sum(axis=(1,2))
        proj_y = r_psi_intgr.sum(axis=(0,2))
        proj_z = r_psi_intgr.sum(axis=(0,1))
        print(f"\n  τ={tau:>3}")
        print(f"    proj x : {proj_x}")
        print(f"    proj y : {proj_y}")
        print(f"    proj z : {proj_z}")
        # Cubiquement symétrique ?
        # On regarde aussi la symétrie x→4-x
        cubic_sym = (np.allclose(proj_x, proj_y, rtol=0.1) and
                     np.allclose(proj_x, proj_z, rtol=0.1))
        # Symétrie de réflexion en x
        x_refl_diff = np.linalg.norm(proj_x - proj_x[::-1]) / max(np.linalg.norm(proj_x), 1e-30)
        print(f"    cubiquement symétrique ? {cubic_sym}")
        print(f"    déviation à symétrie x→4-x : {x_refl_diff:.4f}")
        spatial_data[tau] = {
            "proj_x": proj_x.tolist(),
            "proj_y": proj_y.tolist(),
            "proj_z": proj_z.tolist(),
            "cubic_symmetric": bool(cubic_sym),
            "x_reflection_deviation": float(x_refl_diff),
        }

    # === ANALYSE (2) : Comparaison avec extra_6 ===
    print(f"\n=== (2) r_τ ressemble-t-il à extra_6 (mode lent 6d-γ) ? ===\n")
    # On compare le résidu à τ=0 (où il est maximal) à extra_6 normalisé
    # Plus précisément, on regarde s'il y a alignement (cos similarité)
    d_psi_0_tau, d_h_0 = deltas[0]
    d_flat_0 = np.concatenate([d_psi_0_tau.flatten(), d_h_0.flatten()])
    a_0 = float(np.dot(d_flat_0, ref_flat) / ref_norm_sq)
    r_psi_0 = d_psi_0_tau - a_0 * d_psi_ref  # array de shape (n_t, 5, 5, 5)

    # extra_6 est un vecteur unique (état asymptotique du cycle 6d-γ)
    # r_psi_0 est une trajectoire (forme temporelle de la réponse)
    # Stratégie : pour chaque t, calculer cos(r_psi_0(t), extra_6)
    extra_6_n = extra_6 / np.linalg.norm(extra_6)
    print(f"  cos(r_τ=0(t), extra_6) à différents instants t :")
    n_t_samples = [0, 50, 100, 250, 499]
    cosines = {}
    for n_t in n_t_samples:
        r_psi_at_t = r_psi_0[n_t]
        norm_r = np.linalg.norm(r_psi_at_t)
        if norm_r > 1e-30:
            cos_val = float(np.sum(r_psi_at_t * extra_6_n) / norm_r)
        else:
            cos_val = 0.0
        t_val = n_t * dt
        cosines[t_val] = cos_val
        print(f"    t ≈ {t_val:>6.2f} : cos = {cos_val:+.4f}  (||r|| = {norm_r:.4e})")

    # Et sur toute la trajectoire intégrée
    r_psi_0_integrated = r_psi_0.sum(axis=0)  # somme temporelle
    n_int = np.linalg.norm(r_psi_0_integrated)
    if n_int > 1e-30:
        cos_int = float(np.sum(r_psi_0_integrated * extra_6_n) / n_int)
    else:
        cos_int = 0.0
    print(f"\n  cos(r_τ=0 intégré sur t, extra_6) = {cos_int:+.4f}")

    # === ANALYSE (3) : Localisation temporelle de χ_slow ===
    print(f"\n=== (3) Localisation temporelle de la contribution à χ_slow ===\n")
    print(f"  Pour τ=0, on regarde où dans la fenêtre [0, Δt] vient")
    print(f"  l'essentiel du résidu r_τ.\n")
    # Norme de r_psi_0(t) à chaque instant t
    norms_in_time = np.array([np.linalg.norm(r_psi_0[n_t])
                               for n_t in range(r_psi_0.shape[0])])
    times = np.arange(r_psi_0.shape[0]) * dt
    auc_total = norms_in_time.sum() * dt
    print(f"  AUC ||r_τ=0(t)|| = {auc_total:.4e}")

    # Cumulé
    cumul = np.cumsum(norms_in_time) * dt
    # Trouver à quel t on atteint 50%, 90%, 99% de l'AUC
    t_50 = times[np.argmax(cumul >= 0.5 * auc_total)]
    t_90 = times[np.argmax(cumul >= 0.9 * auc_total)]
    t_99 = times[np.argmax(cumul >= 0.99 * auc_total)]
    print(f"  50% de l'AUC atteinte à t ≈ {t_50:.2f}")
    print(f"  90% de l'AUC atteinte à t ≈ {t_90:.2f}")
    print(f"  99% de l'AUC atteinte à t ≈ {t_99:.2f}")
    # Comparé à Δt = 100
    print(f"  (Δt total = {n_dt_response * dt:.0f})")

    # Norme à t=0 vs au maximum
    n_initial = norms_in_time[0]
    n_max = norms_in_time.max()
    t_max = times[np.argmax(norms_in_time)]
    print(f"  ||r_τ=0(t=0)|| = {n_initial:.4e}")
    print(f"  ||r_τ=0|| max  = {n_max:.4e} atteint à t ≈ {t_max:.2f}")

    output = {
        "extra_6_norm": float(np.linalg.norm(extra_6)),
        "spatial_data": spatial_data,
        "cosines_with_extra_6": cosines,
        "cos_integrated_with_extra_6": cos_int,
        "t_50_AUC": float(t_50),
        "t_90_AUC": float(t_90),
        "t_99_AUC": float(t_99),
        "norm_initial": float(n_initial),
        "norm_max": float(n_max),
        "t_at_max_norm": float(t_max),
    }
    with open("/home/claude/mcq_v4/6d_delta_residual_structure.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
