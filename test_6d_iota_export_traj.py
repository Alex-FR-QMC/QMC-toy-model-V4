# -*- coding: utf-8 -*-
"""
6d-Î¹ â SensibilitÃ© morphologique locale, perturbation h-only.

Question : h(Î¸) scalaire plein possÃ¨de-t-il une expressivitÃ©
morphologique active ? Plus prÃ©cisÃ©ment : une variation contrÃ´lÃ©e
de h conditionne-t-elle non trivialement la rÃ©ponse dynamique Ã  Pâ' ?

Protocole :
- Ãtat de dÃ©part : X_t0 = (psi_tau0, h_tau0) (Ã©tat P6 relaxÃ©, comme Î·/Î¸)
- Perturbation h : h_Îµ^M = max(h_min, h_tau0 Â· exp(Îµ Â· M_norm))
  avec M_norm centrÃ©, normalisÃ© par max(|M|), sans clip supÃ©rieur h0
- Îµ tel que ||Î´h||/||h_tau0|| â 1%
- Pas de Ît_sep entre h et Pâ' : Pâ' appliquÃ© immÃ©diatement aprÃ¨s h

Quatre trajectoires :
- T1 : X_t0, pas hÎµ, pas Pâ'
- T2 : (psi_tau0, h_Îµ^M), pas Pâ'
- T3 : (Pâ'(psi_tau0), h_tau0)
- T4 : (Pâ'(psi_tau0), h_Îµ^M)

Objet principal :
I_M(Pâ') = || (T4 â T3) â (T2 â T1) || / ||Î´h||
R_I = || (T4 â T3) â (T2 â T1) || / ||T4 â T3||

Masques h :
1. H_centre
2. H_anneau_moyen
3. H_shell
4. H_dipole_face
5. H_random_smooth (contrÃ´le, seed fixe)

P' de lecture :
1. G_standard_centree
2. A_anneau_moyen

Verdict : Î¹-PASS fort / faible / FAIL / PLAFOND / INDETERMINÃ.

Pas de Î. Pas de ð¢ plein. Pas de RTS/RSR. Pas de MCQ.
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

def evolve_traj(psi, h, D, beta, gamma, h0, dt, n_steps):
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

COORDS = np.arange(N_AXIS) * DX
CENTER = (N_AXIS - 1) * DX / 2.0

def _gaussian_factor(cx, cy, cz, sigma_p):
    factor = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((COORDS[i]-cx)**2 + (COORDS[j]-cy)**2 + (COORDS[k]-cz)**2)
                factor[i,j,k] = np.exp(-0.5 * r2 / sigma_p**2)
    return factor

def _radial_mask(r_inner, r_outer):
    mask = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((COORDS[i]-CENTER)**2 + (COORDS[j]-CENTER)**2 + (COORDS[k]-CENTER)**2)
                if r_inner <= r <= r_outer:
                    mask[i,j,k] = 1.0
    return mask

def P_G_standard_centree(psi, s):
    f = 1.0 + s * _gaussian_factor(CENTER, CENTER, CENTER, 0.8)
    return (psi * f) / (psi * f).sum()

def P_A_anneau_moyen(psi, s):
    f = 1.0 + s * _radial_mask(1.3, 1.9)
    return (psi * f) / (psi * f).sum()


# ===== Masques h =====

def mask_H_centre():
    return _gaussian_factor(CENTER, CENTER, CENTER, 0.8)

def mask_H_anneau_moyen():
    return _radial_mask(1.3, 1.9)

def mask_H_shell():
    return _radial_mask(2.5, 100.0)

def mask_H_dipole_face():
    m = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    m[0, :, :] = 1.0
    m[4, :, :] = -1.0
    return m

def mask_H_random_smooth(seed=20260601):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(N_AXIS, N_AXIS, N_AXIS))
    # Lissage par convolution 3D avec noyau gaussien Ï=0.8 cellule
    sigma_lissage = 0.8
    radius = 2
    kernel = np.zeros((2*radius+1, 2*radius+1, 2*radius+1))
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            for dk in range(-radius, radius+1):
                r2 = di*di + dj*dj + dk*dk
                kernel[di+radius, dj+radius, dk+radius] = np.exp(-0.5 * r2 / sigma_lissage**2)
    kernel /= kernel.sum()
    # Convolution avec bords zÃ©ro (Ã©quivalent : padding par 0)
    from scipy.signal import fftconvolve
    smoothed = fftconvolve(noise, kernel, mode='same')
    return smoothed


def normalize_mask(M):
    """Centrer (moyenne nulle) puis normaliser par max(|M_c|)."""
    M_c = M - M.mean()
    max_abs = max(abs(M_c.max()), abs(M_c.min()))
    if max_abs < 1e-30:
        return M_c
    return M_c / max_abs


def perturb_h(h_tau0, M_norm, eps, h_min=1e-6):
    """h_Îµ^M = max(h_min, h_tau0 Â· exp(Îµ Â· M_norm))."""
    return np.maximum(h_min, h_tau0 * np.exp(eps * M_norm))


def calibrate_eps_for_delta_h_norm(h_tau0, M_norm, target_ratio=0.01):
    """Trouver Îµ tel que ||Î´h||/||h_tau0|| = target_ratio."""
    norm_h = float(np.linalg.norm(h_tau0))
    def err(eps):
        h_eps = perturb_h(h_tau0, M_norm, eps)
        return float(np.linalg.norm(h_eps - h_tau0)) / norm_h - target_ratio
    try:
        eps_root = brentq(err, 1e-6, 1.0, xtol=1e-8)
        return eps_root
    except Exception:
        try:
            eps_root = brentq(err, 1e-6, 5.0, xtol=1e-8)
            return eps_root
        except Exception:
            return None


def diag_h(h_eps, h_tau0, h_min, h0):
    """Diagnostic de la perturbation h."""
    delta = h_eps - h_tau0
    return {
        "min_h_eps": float(h_eps.min()),
        "max_h_eps": float(h_eps.max()),
        "max_h_eps_over_h0": float(h_eps.max() / h0),
        "norm_delta_h": float(np.linalg.norm(delta)),
        "rel_norm_delta_h": float(np.linalg.norm(delta) / np.linalg.norm(h_tau0)),
        "n_cells_below_h_min": int(np.sum(h_eps < h_min + 1e-15)),
    }


def safe_div(a, b):
    return a / b if b > 1e-30 else 0.0


def compute_I_metrics(psi_T1, h_T1, psi_T2, h_T2,
                       psi_T3, h_T3, psi_T4, h_T4,
                       norm_delta_h, dt):
    """Calcul des mÃ©triques I_M et R_I."""
    n = min(len(psi_T1), len(psi_T2), len(psi_T3), len(psi_T4))
    psi_T1 = psi_T1[:n]; h_T1 = h_T1[:n]
    psi_T2 = psi_T2[:n]; h_T2 = h_T2[:n]
    psi_T3 = psi_T3[:n]; h_T3 = h_T3[:n]
    psi_T4 = psi_T4[:n]; h_T4 = h_T4[:n]

    # Effet morphologique seul : T2 â T1
    dpsi_morph = psi_T2 - psi_T1
    dh_morph = h_T2 - h_T1
    # Effet morphologique en prÃ©sence de Pâ' : T4 â T3
    dpsi_morph_p = psi_T4 - psi_T3
    dh_morph_p = h_T4 - h_T3
    # Interaction : (T4 â T3) â (T2 â T1)
    Ipsi = dpsi_morph_p - dpsi_morph
    Ih = dh_morph_p - dh_morph

    # Normes par instant
    norm_Ipsi = np.array([np.linalg.norm(Ipsi[t]) for t in range(n)])
    norm_Ih = np.array([np.linalg.norm(Ih[t]) for t in range(n)])
    norm_Iext = np.array([
        np.linalg.norm(np.concatenate([Ipsi[t].flatten(), Ih[t].flatten()]))
        for t in range(n)
    ])

    norm_dmp_psi = np.array([np.linalg.norm(dpsi_morph_p[t]) for t in range(n)])
    norm_dmp_h = np.array([np.linalg.norm(dh_morph_p[t]) for t in range(n)])
    norm_dmp_ext = np.array([
        np.linalg.norm(np.concatenate([dpsi_morph_p[t].flatten(), dh_morph_p[t].flatten()]))
        for t in range(n)
    ])
    norm_dm_psi = np.array([np.linalg.norm(dpsi_morph[t]) for t in range(n)])
    norm_dm_h = np.array([np.linalg.norm(dh_morph[t]) for t in range(n)])
    norm_dm_ext = np.array([
        np.linalg.norm(np.concatenate([dpsi_morph[t].flatten(), dh_morph[t].flatten()]))
        for t in range(n)
    ])

    # AUC (intÃ©grales par trapÃ¨zes)
    auc_I_psi = float(np.trapezoid(norm_Ipsi, dx=dt))
    auc_I_h = float(np.trapezoid(norm_Ih, dx=dt))
    auc_I_ext = float(np.trapezoid(norm_Iext, dx=dt))
    auc_dmp_psi = float(np.trapezoid(norm_dmp_psi, dx=dt))
    auc_dmp_h = float(np.trapezoid(norm_dmp_h, dx=dt))
    auc_dmp_ext = float(np.trapezoid(norm_dmp_ext, dx=dt))
    auc_dm_psi = float(np.trapezoid(norm_dm_psi, dx=dt))
    auc_dm_h = float(np.trapezoid(norm_dm_h, dx=dt))
    auc_dm_ext = float(np.trapezoid(norm_dm_ext, dx=dt))

    # MÃ©triques normalisÃ©es
    # I_M / ||Î´h|| : sensibilitÃ© par unitÃ© de Î´h
    return {
        # AUC bruts (interaction)
        "auc_I_psi": auc_I_psi,
        "auc_I_h": auc_I_h,
        "auc_I_ext": auc_I_ext,
        # AUC bruts (effet morphologique en prÃ©sence de Pâ')
        "auc_dmp_psi": auc_dmp_psi,
        "auc_dmp_h": auc_dmp_h,
        "auc_dmp_ext": auc_dmp_ext,
        # AUC bruts (effet morphologique seul, T2 â T1)
        "auc_dm_psi": auc_dm_psi,
        "auc_dm_h": auc_dm_h,
        "auc_dm_ext": auc_dm_ext,
        # NormalisÃ© par ||Î´h|| : sensibilitÃ© par unitÃ© de Î´h
        "I_AUC_psi_per_dh": safe_div(auc_I_psi, norm_delta_h),
        "I_AUC_h_per_dh": safe_div(auc_I_h, norm_delta_h),
        "I_AUC_ext_per_dh": safe_div(auc_I_ext, norm_delta_h),
        # Ratio relatif R_I : importance de l'interaction par rapport Ã  l'effet morphoÃPâ'
        "R_I_AUC_psi": safe_div(auc_I_psi, auc_dmp_psi),
        "R_I_AUC_h": safe_div(auc_I_h, auc_dmp_h),
        "R_I_AUC_ext": safe_div(auc_I_ext, auc_dmp_ext),
        # Final (secondaire)
        "I_final_ext": float(norm_Iext[-1]),
        "R_I_final_ext": safe_div(float(norm_Iext[-1]), float(norm_dmp_ext[-1])),
        # === Patch P2-lag : trajectoires des normes ===
        "trajectory_a_h": norm_Ih.tolist(),
        "trajectory_a_psi": norm_Ipsi.tolist(),
        "trajectory_a_ext": norm_Iext.tolist(),
        "trajectory_a_dmp_h": norm_dmp_h.tolist(),
        "trajectory_a_dmp_psi": norm_dmp_psi.tolist(),
    }


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    h_min = 1e-8
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    T_final = 800.0
    n_final = int(T_final / dt)

    print(f"=== 6d-Î¹ : sensibilitÃ© morphologique h-only ===\n")
    print(f"  dt = {dt:.5f}, T_final = {T_final}, n_final = {n_final}")
    print(f"  h_min = {h_min}, pas de clip supÃ©rieur h0\n")

    # PrÃ©parer Ã©tat P6 relaxÃ©
    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)
    def P1prime_plateau(psi, strength=0.05):
        coords = np.arange(N_AXIS) * DX
        c = (N_AXIS - 1) * DX / 2.0
        factor = np.ones_like(psi)
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                    if r <= 1.5: factor[i,j,k] += strength
        return (psi * factor) / (psi * factor).sum()
    amp_P1 = float(np.linalg.norm(P1prime_plateau(psi_base, 0.05) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    target_amp = 0.1 * amp_P6
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6, D, beta_lock, gamma_v, h0, dt, n_short)
    print(f"  ||P6|| = {amp_P6:.4e}, target_amp (10%) = {target_amp:.4e}")
    print(f"  ||h_tau0|| = {np.linalg.norm(h_tau0):.4e}")
    print(f"  h_tau0 range : [{h_tau0.min():.4e}, {h_tau0.max():.4e}]\n")

    # Calibrer Pâ'
    def calibrate_strength(P_fn, target_amp):
        return brentq(lambda s: float(np.linalg.norm(P_fn(psi_base, s) - psi_base))
                       - target_amp, 1e-6, 5.0, xtol=1e-8)
    s_G = calibrate_strength(P_G_standard_centree, target_amp)
    s_A = calibrate_strength(P_A_anneau_moyen, target_amp)
    print(f"  s_G_standard = {s_G:.6f}, s_A_anneau_moyen = {s_A:.6f}\n")

    # Masques
    masks = {
        "H_centre": mask_H_centre(),
        "H_anneau_moyen": mask_H_anneau_moyen(),
        "H_shell": mask_H_shell(),
        "H_dipole_face": mask_H_dipole_face(),
        "H_random_smooth": mask_H_random_smooth(seed=20260601),
    }
    masks_norm = {name: normalize_mask(M) for name, M in masks.items()}

    # P' de lecture
    P_prime_list = [
        ("G_standard_centree", P_G_standard_centree, s_G),
        ("A_anneau_moyen", P_A_anneau_moyen, s_A),
    ]

    # Diagnostic des masques
    print(f"--- Diagnostic des masques (calibration Îµ = 1%) ---\n")
    print(f"  {'mask':<22} {'eps':>10} {'rel_||Î´h||':>12} {'min(h_Îµ)':>12} "
          f"{'max(h_Îµ)':>12} {'max/h0':>10} {'n<h_min':>8}")
    eps_by_mask = {}
    h_eps_by_mask = {}
    diag_by_mask = {}
    for name, M_norm in masks_norm.items():
        eps = calibrate_eps_for_delta_h_norm(h_tau0, M_norm, target_ratio=0.01)
        if eps is None:
            print(f"  {name:<22}  CALIBRATION FAILED")
            continue
        h_eps = perturb_h(h_tau0, M_norm, eps, h_min)
        eps_by_mask[name] = eps
        h_eps_by_mask[name] = h_eps
        d = diag_h(h_eps, h_tau0, h_min, h0)
        diag_by_mask[name] = d
        print(f"  {name:<22} {eps:>10.6f} {d['rel_norm_delta_h']:>12.4e} "
              f"{d['min_h_eps']:>12.4e} {d['max_h_eps']:>12.4e} "
              f"{d['max_h_eps_over_h0']:>10.4f} {d['n_cells_below_h_min']:>8}")

    # Floor numÃ©rique : T1 vs T1' (deux runs identiques)
    print(f"\n--- Floor numÃ©rique (T1 vs T1') ---\n")
    psi_T1, h_T1 = evolve_traj(psi_tau0.copy(), h_tau0.copy(),
                                D, beta_lock, gamma_v, h0, dt, n_final)
    psi_T1b, h_T1b = evolve_traj(psi_tau0.copy(), h_tau0.copy(),
                                  D, beta_lock, gamma_v, h0, dt, n_final)
    floor = float(np.linalg.norm(
        np.concatenate([psi_T1[-1].flatten() - psi_T1b[-1].flatten(),
                        h_T1[-1].flatten() - h_T1b[-1].flatten()])))
    print(f"  ||X_T1(T) â X_T1'(T)|| = {floor:.4e}\n")

    # Lancer le test principal : pour chaque (M, Pâ'), calculer T1, T2, T3, T4
    print(f"--- Test principal : (T4 â T3) â (T2 â T1) pour chaque (M, Pâ') ---\n")
    results = {}
    # T1 commun Ã  tous : dÃ©jÃ  calculÃ©
    for P_name, P_fn, s_P in P_prime_list:
        print(f"\n  === P' = {P_name} ===")
        print(f"  {'mask':<22} {'I_AUC_ext':>12} {'I_AUC_ext/||Î´h||':>18} "
              f"{'R_I_AUC_ext':>14} {'I_AUC_Ï/||Î´h||':>18} {'I_AUC_h/||Î´h||':>18} "
              f"{'I_final_ext':>14}")
        # T3 commun Ã  un Pâ' donnÃ©
        psi_t0_P = P_fn(psi_tau0.copy(), s_P)
        psi_T3, h_T3 = evolve_traj(psi_t0_P.copy(), h_tau0.copy(),
                                    D, beta_lock, gamma_v, h0, dt, n_final)
        for mask_name in eps_by_mask:
            h_eps = h_eps_by_mask[mask_name]
            d_diag = diag_by_mask[mask_name]
            # T2 : (psi_tau0, h_eps), pas de Pâ'
            psi_T2, h_T2 = evolve_traj(psi_tau0.copy(), h_eps.copy(),
                                        D, beta_lock, gamma_v, h0, dt, n_final)
            # T4 : (Pâ'(psi_tau0), h_eps)
            psi_T4, h_T4 = evolve_traj(psi_t0_P.copy(), h_eps.copy(),
                                        D, beta_lock, gamma_v, h0, dt, n_final)
            m = compute_I_metrics(psi_T1, h_T1, psi_T2, h_T2,
                                   psi_T3, h_T3, psi_T4, h_T4,
                                   d_diag["norm_delta_h"], dt)
            key = f"{mask_name}_x_{P_name}"
            results[key] = {
                "mask": mask_name, "P_prime": P_name,
                "eps": eps_by_mask[mask_name],
                **d_diag,
                **m,
            }
            print(f"  {mask_name:<22} {m['auc_I_ext']:>12.4e} "
                  f"{m['I_AUC_ext_per_dh']:>18.4e} {m['R_I_AUC_ext']:>14.4e} "
                  f"{m['I_AUC_psi_per_dh']:>18.4e} {m['I_AUC_h_per_dh']:>18.4e} "
                  f"{m['I_final_ext']:>14.4e}")

    # SynthÃ¨se
    print(f"\n--- SynthÃ¨se globale ---\n")
    # Pour chaque P', regarder R_I et I_AUC_ext_per_dh par masque
    summary = {}
    R_I_values_structured = []  # masques structurÃ©s (sans random)
    R_I_values_random = []
    for P_name, _, _ in P_prime_list:
        for mask_name in eps_by_mask:
            key = f"{mask_name}_x_{P_name}"
            R = results[key]["R_I_AUC_ext"]
            if mask_name == "H_random_smooth":
                R_I_values_random.append(R)
            else:
                R_I_values_structured.append(R)
    print(f"  R_I_AUC_ext sur masques structurÃ©s : "
          f"min = {min(R_I_values_structured):.4e}, max = {max(R_I_values_structured):.4e}, "
          f"mean = {np.mean(R_I_values_structured):.4e}")
    print(f"  R_I_AUC_ext sur masque random : "
          f"min = {min(R_I_values_random):.4e}, max = {max(R_I_values_random):.4e}, "
          f"mean = {np.mean(R_I_values_random):.4e}")

    # I_AUC_ext_per_dh par masque (moyennÃ© sur les Pâ')
    print(f"\n  I_AUC_ext_per_dh par masque (moyennÃ© sur Pâ') :")
    sens_by_mask = {}
    for mask_name in eps_by_mask:
        vals = [results[f"{mask_name}_x_{P_name}"]["I_AUC_ext_per_dh"]
                for P_name, _, _ in P_prime_list]
        sens_by_mask[mask_name] = float(np.mean(vals))
        print(f"    {mask_name:<22} {sens_by_mask[mask_name]:.4e}")

    # R_I_AUC_ext par masque (moyennÃ© sur les Pâ')
    print(f"\n  R_I_AUC_ext par masque (moyennÃ© sur Pâ') :")
    R_by_mask = {}
    for mask_name in eps_by_mask:
        vals = [results[f"{mask_name}_x_{P_name}"]["R_I_AUC_ext"]
                for P_name, _, _ in P_prime_list]
        R_by_mask[mask_name] = float(np.mean(vals))
        print(f"    {mask_name:<22} {R_by_mask[mask_name]:.4e}")

    # === Verdict ===
    print(f"\n=== Verdict Î¹ ===\n")
    R_I_max = max([results[k]["R_I_AUC_ext"] for k in results])
    R_I_min_structured = min(R_I_values_structured)
    R_I_max_structured = max(R_I_values_structured)

    # VÃ©rification absence de cells sous h_min
    n_under = sum(d["n_cells_below_h_min"] for d in diag_by_mask.values())
    if n_under > 0:
        verdict = "Î¹-INDETERMINÃ : clipping h_min activÃ©"
    elif R_I_max < 1e-10:
        verdict = "Î¹-PLAFOND : pas d'interaction au-dessus du floor"
    elif R_I_max_structured < 1e-3:
        verdict = "Î¹-FAIL : interaction nÃ©gligeable (R_I < 1e-3)"
    elif R_I_max_structured >= 1e-3:
        # DÃ©pendance au masque ?
        mask_variability = R_I_max_structured / max(R_I_min_structured, 1e-30)
        if mask_variability > 3.0:
            verdict = "Î¹-PASS fort : interaction mesurable et dÃ©pendante du masque"
        else:
            verdict = "Î¹-PASS faible : interaction mesurable mais faible variabilitÃ© de masque"
    else:
        verdict = "Î¹-INDETERMINÃ"

    print(f"  R_I_AUC_ext max (toutes paires) : {R_I_max:.4e}")
    print(f"  R_I_AUC_ext min sur structurÃ©s : {R_I_min_structured:.4e}")
    print(f"  R_I_AUC_ext max sur structurÃ©s : {R_I_max_structured:.4e}")
    print(f"  Floor numÃ©rique : {floor:.4e}")
    print(f"  VariabilitÃ© masques (max/min structurÃ©s) : "
          f"{R_I_max_structured/max(R_I_min_structured, 1e-30):.2f}")
    print(f"\n  VERDICT : {verdict}")

    output = {
        "dt_simulation": float(dt),
        "T_final": T_final,
        "h_min": h_min,
        "no_upper_clip": True,
        "amp_P6": float(amp_P6),
        "target_amp_for_P_prime": float(target_amp),
        "s_G_standard_centree": s_G,
        "s_A_anneau_moyen": s_A,
        "random_seed": 20260601,
        "smoothing_sigma": 0.8,
        "target_relative_delta_h": 0.01,
        "floor_numerique": floor,
        "eps_by_mask": eps_by_mask,
        "diag_by_mask": diag_by_mask,
        "results": results,
        "sens_by_mask_avg": sens_by_mask,
        "R_I_by_mask_avg": R_by_mask,
        "R_I_AUC_ext_max": R_I_max,
        "R_I_min_structured": R_I_min_structured,
        "R_I_max_structured": R_I_max_structured,
        "verdict": verdict,
    }
    with open("/home/claude/mcq_v4/6d_iota_export_traj.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardÃ©.")


if __name__ == "__main__":
    main()
