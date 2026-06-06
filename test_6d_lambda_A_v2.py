# -*- coding: utf-8 -*-
"""
6d-lambda-A-v2 : diagnostic etendu de non-separabilite avec metriques robustes.

Cf. specification 6d_lambda0_h_proj_specification.md et corrections post-v0.

Modifications v0 -> v2 :
- ratios symetriques R_sym = ||a-b|| / (||a|| + ||b|| + 1e-30), bornes dans [0,1]
- 5 strates de masquage : A (all), B (h actif), C_psi (h+psi), C_grad (h+gradpsi),
  D_psi (pondere par psi), D_grad (pondere par |gradpsi|)
- checkpoints temporels t = 0, 10, 50, 100, 200, 400, 800
- normes absolues toujours rapportees
- g_k secondaire en quantiles log sur masque actif
- verdict en trois axes : GEOM / OPER / COMBINE
- n_faces par strate rapporte (garde-fou contre PASS porte par 2 faces)
- seuils : faible 0.005, significatif 0.05, dominant 0.3
- C_grad principal pour verdict flux, C_psi pour coherence
- pas de lambda-B, pas de section §22, document compagnon seulement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import json
from scipy.optimize import brentq

from mcq_v4.factorial_6d import N_AXIS, DX, cfl_dt_max
from mcq_v4.factorial_6d.engine import (
    compute_diffusion_flux, compute_divergence, harmonic_mean,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero

EPS = 1e-30


# ===== Moteur =====
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

def evolve_with_checkpoints(psi, h, D, beta, gamma, h0, dt, checkpoint_steps):
    """Evolue et garde l'etat aux indices specifies (steps depuis t=0)."""
    states = []
    n_max = max(checkpoint_steps)
    next_idx = 0
    for n in range(n_max + 1):
        if next_idx < len(checkpoint_steps) and n == checkpoint_steps[next_idx]:
            states.append((psi.copy(), h.copy()))
            next_idx += 1
        if n < n_max:
            psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return states


# ===== Etats =====
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


# ===== Projection =====
def project_marginal(h_full):
    S = float(h_full.sum())
    if S < EPS:
        return np.zeros_like(h_full)
    h_T = h_full.sum(axis=(1, 2))
    h_M = h_full.sum(axis=(0, 2))
    h_I = h_full.sum(axis=(0, 1))
    return np.einsum('i,j,k->ijk', h_T, h_M, h_I) / (S * S)


# ===== Geom =====
def D_proj_metric(h_full, h_proj):
    n_full = float(np.linalg.norm(h_full))
    n_diff = float(np.linalg.norm(h_full - h_proj))
    return n_diff / (n_full + EPS)


# ===== Flux et strates =====
def compute_flux_components(psi, h, D):
    """Retourne (J_x, J_y, J_z, h_face_x, h_face_y, h_face_z, grad_x, grad_y, grad_z).
    Tous defines aux interfaces. Reproduit fidelement le moteur."""
    h_face_x = harmonic_mean(h[:-1, :, :], h[1:, :, :])
    h_face_y = harmonic_mean(h[:, :-1, :], h[:, 1:, :])
    h_face_z = harmonic_mean(h[:, :, :-1], h[:, :, 1:])
    grad_x = (psi[1:, :, :] - psi[:-1, :, :]) / DX
    grad_y = (psi[:, 1:, :] - psi[:, :-1, :]) / DX
    grad_z = (psi[:, :, 1:] - psi[:, :, :-1]) / DX
    Jx = -h_face_x * D * grad_x
    Jy = -h_face_y * D * grad_y
    Jz = -h_face_z * D * grad_z
    return Jx, Jy, Jz, h_face_x, h_face_y, h_face_z, grad_x, grad_y, grad_z


def psi_face(psi):
    """Moyenne arithmetique de psi aux faces."""
    p_x = 0.5 * (psi[:-1, :, :] + psi[1:, :, :])
    p_y = 0.5 * (psi[:, :-1, :] + psi[:, 1:, :])
    p_z = 0.5 * (psi[:, :, :-1] + psi[:, :, 1:])
    return p_x, p_y, p_z


def flatten_faces(arr_x, arr_y, arr_z):
    return np.concatenate([arr_x.flatten(), arr_y.flatten(), arr_z.flatten()])


def R_sym(a_flat, b_flat, mask_flat=None):
    """Ratio symetrique borne dans [0,1] : ||a-b||/(||a||+||b||+EPS).
    Si mask_flat fourni, ne calcule que sur cellules True."""
    if mask_flat is not None:
        a_flat = a_flat[mask_flat]
        b_flat = b_flat[mask_flat]
    if len(a_flat) == 0:
        return 0.0
    na = float(np.linalg.norm(a_flat))
    nb = float(np.linalg.norm(b_flat))
    nd = float(np.linalg.norm(a_flat - b_flat))
    return nd / (na + nb + EPS)


def weighted_norm_diff(a_flat, b_flat, w_flat):
    """Norme ponderee : sqrt(sum(w * (a-b)^2)) / (sqrt(sum(w * a^2)) + sqrt(sum(w * b^2)) + EPS)"""
    sum_w_a2 = float(np.sum(w_flat * a_flat ** 2))
    sum_w_b2 = float(np.sum(w_flat * b_flat ** 2))
    sum_w_d2 = float(np.sum(w_flat * (a_flat - b_flat) ** 2))
    return np.sqrt(sum_w_d2) / (np.sqrt(sum_w_a2) + np.sqrt(sum_w_b2) + EPS)


def stratified_flux_diagnostics(psi, h_full, h_proj, D, h_face_threshold=1e-6,
                                  psi_face_thresh_rel=0.01, grad_thresh_rel=0.01):
    """Calcule R_sym et normes absolues sur 5 strates : A, B, C_psi, C_grad, D_psi, D_grad."""
    Jx_full, Jy_full, Jz_full, hfx_full, hfy_full, hfz_full, grad_x, grad_y, grad_z = \
        compute_flux_components(psi, h_full, D)
    Jx_proj, Jy_proj, Jz_proj, hfx_proj, hfy_proj, hfz_proj, _, _, _ = \
        compute_flux_components(psi, h_proj, D)

    # Flatten flux + h_face + grad pour avoir vecteurs sur toutes les faces
    J_full_flat = flatten_faces(Jx_full, Jy_full, Jz_full)
    J_proj_flat = flatten_faces(Jx_proj, Jy_proj, Jz_proj)
    h_face_full_flat = flatten_faces(hfx_full, hfy_full, hfz_full)
    p_x, p_y, p_z = psi_face(psi)
    psi_face_flat = flatten_faces(p_x, p_y, p_z)
    grad_flat = flatten_faces(np.abs(grad_x), np.abs(grad_y), np.abs(grad_z))

    # Divergence
    div_full = compute_divergence(Jx_full, Jy_full, Jz_full)
    div_proj = compute_divergence(Jx_proj, Jy_proj, Jz_proj)

    # Masques de strate
    n_total = len(J_full_flat)
    mask_A = np.ones(n_total, dtype=bool)
    mask_B = h_face_full_flat > h_face_threshold
    psi_max = max(float(psi_face_flat.max()), EPS)
    grad_max = max(float(grad_flat.max()), EPS)
    mask_psi_active = psi_face_flat > psi_face_thresh_rel * psi_max
    mask_grad_active = grad_flat > grad_thresh_rel * grad_max
    mask_C_psi = mask_B & mask_psi_active
    mask_C_grad = mask_B & mask_grad_active

    strata = {
        "A_all_interfaces": mask_A,
        "B_h_active": mask_B,
        "C_psi_h_and_psi_active": mask_C_psi,
        "C_grad_h_and_gradpsi_active": mask_C_grad,
    }

    result = {
        "thresholds": {
            "h_face": h_face_threshold,
            "psi_face_rel": psi_face_thresh_rel,
            "grad_rel": grad_thresh_rel,
        },
        "n_total_faces": int(n_total),
        "norms_absolute": {
            "norm_J_full": float(np.linalg.norm(J_full_flat)),
            "norm_J_proj": float(np.linalg.norm(J_proj_flat)),
            "norm_J_diff": float(np.linalg.norm(J_full_flat - J_proj_flat)),
            "norm_div_full": float(np.linalg.norm(div_full)),
            "norm_div_proj": float(np.linalg.norm(div_proj)),
            "norm_div_diff": float(np.linalg.norm(div_full - div_proj)),
        },
        "strata": {},
    }

    for name, mask in strata.items():
        n_kept = int(mask.sum())
        frac = n_kept / max(n_total, 1)
        R_sym_J = R_sym(J_full_flat, J_proj_flat, mask)
        norm_J_full_s = float(np.linalg.norm(J_full_flat[mask])) if n_kept else 0.0
        norm_J_proj_s = float(np.linalg.norm(J_proj_flat[mask])) if n_kept else 0.0
        norm_J_diff_s = float(np.linalg.norm((J_full_flat - J_proj_flat)[mask])) if n_kept else 0.0
        result["strata"][name] = {
            "n_kept": n_kept,
            "fraction": frac,
            "R_sym_J": R_sym_J,
            "norm_J_full": norm_J_full_s,
            "norm_J_proj": norm_J_proj_s,
            "norm_J_diff": norm_J_diff_s,
        }

    # R_sym divergence sur volume entier seulement (pas defini par face)
    R_sym_div_global = R_sym(div_full.flatten(), div_proj.flatten())
    result["R_sym_div_global"] = R_sym_div_global

    # D_psi et D_grad : ponderation
    w_psi = psi_face_flat ** 2  # poids = psi^2 (importance presence)
    w_grad = grad_flat ** 2     # poids = |grad psi|^2 (importance transport)
    R_weighted_J_psi = weighted_norm_diff(J_full_flat, J_proj_flat, w_psi)
    R_weighted_J_grad = weighted_norm_diff(J_full_flat, J_proj_flat, w_grad)
    result["D_psi_weighted"] = {"R_weighted_J": R_weighted_J_psi}
    result["D_grad_weighted"] = {"R_weighted_J": R_weighted_J_grad}

    return result


# ===== g_k =====
def g_k_quantiles(h_full, h_proj, h_threshold=1e-6):
    theta = (np.arange(N_AXIS) + 0.5) * DX
    mask = (h_full > h_threshold) & (h_proj > h_threshold)
    n_kept = int(mask.sum())
    out = {"n_kept": n_kept, "fraction": n_kept / max(N_AXIS ** 3, 1)}
    if n_kept == 0:
        return out
    for axis_name, axis_idx in [("T", 0), ("M", 1), ("I", 2)]:
        # Construire theta_axis(i,j,k)
        shape = [1, 1, 1]
        shape[axis_idx] = N_AXIS
        theta_field = theta.reshape(shape) * np.ones((N_AXIS, N_AXIS, N_AXIS))
        g_full = theta_field / np.where(h_full > 0, h_full, 1.0)
        g_proj = theta_field / np.where(h_proj > 0, h_proj, 1.0)
        ratio = g_full[mask] / g_proj[mask]
        log_ratio = np.log10(np.abs(ratio) + EPS)
        out[f"g_{axis_name}_log10_q05"] = float(np.quantile(log_ratio, 0.05))
        out[f"g_{axis_name}_log10_q50"] = float(np.quantile(log_ratio, 0.50))
        out[f"g_{axis_name}_log10_q95"] = float(np.quantile(log_ratio, 0.95))
        out[f"g_{axis_name}_log10_max"] = float(np.max(np.abs(log_ratio)))
    return out


# ===== Profil h =====
def h_profile_diagnostic(h_full, h0):
    return {
        "h_min": float(h_full.min()),
        "h_max": float(h_full.max()),
        "h_median": float(np.median(h_full)),
        "h_max_over_h0": float(h_full.max() / h0),
        "n_cells_below_1e-6": int(np.sum(h_full < 1e-6)),
        "n_cells_below_1e-12": int(np.sum(h_full < 1e-12)),
        "n_cells_below_1e-30": int(np.sum(h_full < 1e-30)),
        "total_cells": int(N_AXIS ** 3),
    }


# ===== Verdict opératoire =====
def operative_verdict(strata_results):
    """Calcule le verdict OPER selon la grille C_grad / C_psi."""
    R_grad = strata_results["C_grad_h_and_gradpsi_active"]["R_sym_J"]
    R_psi = strata_results["C_psi_h_and_psi_active"]["R_sym_J"]
    n_grad = strata_results["C_grad_h_and_gradpsi_active"]["n_kept"]
    frac_grad = strata_results["C_grad_h_and_gradpsi_active"]["fraction"]
    n_psi = strata_results["C_psi_h_and_psi_active"]["n_kept"]
    frac_psi = strata_results["C_psi_h_and_psi_active"]["fraction"]

    # Garde-fou : si fraction trop faible, PASS local pas robuste
    locality_warning = (frac_grad < 0.05) or (frac_psi < 0.05)

    if R_grad >= 0.05 and R_psi >= 0.05:
        v = "OPER-PASS robuste" if not locality_warning else "OPER-PASS local"
    elif R_grad >= 0.05 and R_psi < 0.05:
        v = "OPER-PASS transport" if not locality_warning else "OPER-PASS transport local"
    elif R_grad < 0.05 and R_psi >= 0.05:
        v = "OPER-INDETERMINE (presence sans flux)"
    elif R_grad < 0.005 and R_psi < 0.005:
        v = "OPER-FAIL"
    else:
        v = "OPER-faible (R_sym entre 0.005 et 0.05)"
    return {
        "verdict": v,
        "R_sym_C_grad": R_grad,
        "R_sym_C_psi": R_psi,
        "frac_C_grad": frac_grad,
        "frac_C_psi": frac_psi,
        "locality_warning": locality_warning,
    }


# ===== Main =====
def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)

    print("=== 6d-lambda-A-v2 : diagnostic robuste ===\n")
    print(f"  dt = {dt:.5f}\n")

    # Prepare etat P6 relaxe
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

    def calibrate(P_fn):
        return brentq(lambda s: float(np.linalg.norm(P_fn(psi_base, s) - psi_base))
                       - target_amp, 1e-6, 5.0, xtol=1e-8)
    s_G = calibrate(P_G_standard_centree)
    s_A = calibrate(P_A_anneau_moyen)
    print(f"  ||P6|| = {amp_P6:.4e}, s_G = {s_G:.6f}, s_A = {s_A:.6f}\n")

    # Checkpoints : 0, 10, 50, 100, 200, 400, 800 unités de temps
    checkpoint_times = [0.0, 10.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    checkpoint_steps = [int(t / dt) for t in checkpoint_times]
    print(f"  Checkpoints : t = {checkpoint_times}\n")

    # Construire E1 et E2 avec checkpoints
    print(f"--- Construction E1 (apres G) avec checkpoints ---")
    psi_init_E1 = P_G_standard_centree(psi_tau0.copy(), s_G)
    E1_states = evolve_with_checkpoints(psi_init_E1, h_tau0.copy(),
                                         D, beta_lock, gamma_v, h0, dt, checkpoint_steps)
    print(f"  {len(E1_states)} etats E1 captures")
    print(f"--- Construction E2 (apres A) avec checkpoints ---")
    psi_init_E2 = P_A_anneau_moyen(psi_tau0.copy(), s_A)
    E2_states = evolve_with_checkpoints(psi_init_E2, h_tau0.copy(),
                                         D, beta_lock, gamma_v, h0, dt, checkpoint_steps)
    print(f"  {len(E2_states)} etats E2 captures\n")

    # E0 unique : (psi_tau0, h_tau0)
    print(f"--- Diagnostic E0 (P6 relaxe) ---")
    h_E0 = h_tau0.copy()
    psi_E0 = psi_tau0.copy()
    h_proj_E0 = project_marginal(h_E0)
    D_proj_E0 = D_proj_metric(h_E0, h_proj_E0)
    profile_E0 = h_profile_diagnostic(h_E0, h0)
    flux_E0 = stratified_flux_diagnostics(psi_E0, h_E0, h_proj_E0, D)
    g_k_E0 = g_k_quantiles(h_E0, h_proj_E0)
    verdict_E0 = operative_verdict(flux_E0["strata"])
    print(f"  D_proj = {D_proj_E0:.4e}, h_min = {profile_E0['h_min']:.4e}")
    print(f"  R_sym C_grad = {verdict_E0['R_sym_C_grad']:.4e} (frac={verdict_E0['frac_C_grad']:.3f})")
    print(f"  R_sym C_psi  = {verdict_E0['R_sym_C_psi']:.4e} (frac={verdict_E0['frac_C_psi']:.3f})")
    print(f"  Verdict OPER : {verdict_E0['verdict']}\n")

    # E1 et E2 par checkpoint
    print(f"--- E1 par checkpoint ---")
    print(f"  {'t':>6} {'D_proj':>10} {'R_C_grad':>10} {'frac_grad':>10} "
          f"{'R_C_psi':>10} {'frac_psi':>10} {'h_min':>10} {'verdict_oper':<30}")
    E1_by_time = {}
    for idx, t in enumerate(checkpoint_times):
        psi_t, h_t = E1_states[idx]
        h_p = project_marginal(h_t)
        D_p = D_proj_metric(h_t, h_p)
        prof = h_profile_diagnostic(h_t, h0)
        fl = stratified_flux_diagnostics(psi_t, h_t, h_p, D)
        gk = g_k_quantiles(h_t, h_p)
        vd = operative_verdict(fl["strata"])
        E1_by_time[t] = {"D_proj": D_p, "profile": prof, "flux": fl, "g_k": gk, "verdict_oper": vd}
        print(f"  {t:>6.1f} {D_p:>10.4e} "
              f"{vd['R_sym_C_grad']:>10.4e} {vd['frac_C_grad']:>10.3f} "
              f"{vd['R_sym_C_psi']:>10.4e} {vd['frac_C_psi']:>10.3f} "
              f"{prof['h_min']:>10.4e} {vd['verdict']:<30}")
    print()

    print(f"--- E2 par checkpoint ---")
    print(f"  {'t':>6} {'D_proj':>10} {'R_C_grad':>10} {'frac_grad':>10} "
          f"{'R_C_psi':>10} {'frac_psi':>10} {'h_min':>10} {'verdict_oper':<30}")
    E2_by_time = {}
    for idx, t in enumerate(checkpoint_times):
        psi_t, h_t = E2_states[idx]
        h_p = project_marginal(h_t)
        D_p = D_proj_metric(h_t, h_p)
        prof = h_profile_diagnostic(h_t, h0)
        fl = stratified_flux_diagnostics(psi_t, h_t, h_p, D)
        gk = g_k_quantiles(h_t, h_p)
        vd = operative_verdict(fl["strata"])
        E2_by_time[t] = {"D_proj": D_p, "profile": prof, "flux": fl, "g_k": gk, "verdict_oper": vd}
        print(f"  {t:>6.1f} {D_p:>10.4e} "
              f"{vd['R_sym_C_grad']:>10.4e} {vd['frac_C_grad']:>10.3f} "
              f"{vd['R_sym_C_psi']:>10.4e} {vd['frac_C_psi']:>10.3f} "
              f"{prof['h_min']:>10.4e} {vd['verdict']:<30}")
    print()

    # Identifier le regime court "physiquement actif partout"
    print(f"--- Identification du regime court actif ---")
    print(f"  E1 : fraction h > 1e-6 par checkpoint :")
    for t in checkpoint_times:
        prof = E1_by_time[t]["profile"]
        frac_h_active = 1.0 - prof["n_cells_below_1e-6"] / prof["total_cells"]
        print(f"    t = {t:>6.1f} : frac(h > 1e-6) = {frac_h_active:.4f}, h_min = {prof['h_min']:.4e}")
    print(f"  E2 : fraction h > 1e-6 par checkpoint :")
    for t in checkpoint_times:
        prof = E2_by_time[t]["profile"]
        frac_h_active = 1.0 - prof["n_cells_below_1e-6"] / prof["total_cells"]
        print(f"    t = {t:>6.1f} : frac(h > 1e-6) = {frac_h_active:.4f}, h_min = {prof['h_min']:.4e}")
    print()

    # Verdict global
    print(f"=== VERDICT GLOBAL lambda-A-v2 ===\n")
    # GEOM : D_proj significatif et stable
    D_proj_values = [D_proj_E0] + [E1_by_time[t]["D_proj"] for t in checkpoint_times] + \
                     [E2_by_time[t]["D_proj"] for t in checkpoint_times]
    D_proj_min = min(D_proj_values)
    D_proj_max = max(D_proj_values)
    D_proj_significant = D_proj_min > 0.01  # stable au-dessus du seuil
    geom_verdict = "GEOM-PASS modere" if D_proj_significant else "GEOM-FAIL"
    print(f"  GEOM :")
    print(f"    D_proj range : [{D_proj_min:.4e}, {D_proj_max:.4e}]")
    print(f"    D_proj stable au-dessus de 0.01 : {D_proj_significant}")
    print(f"    Verdict : {geom_verdict}")

    # OPER : verdict par etat, agrege
    print(f"\n  OPER :")
    all_oper = [verdict_E0]
    for t in checkpoint_times:
        all_oper.append(E1_by_time[t]["verdict_oper"])
        all_oper.append(E2_by_time[t]["verdict_oper"])
    n_pass_robuste = sum(1 for v in all_oper if "PASS robuste" in v["verdict"])
    n_pass_local = sum(1 for v in all_oper if "local" in v["verdict"])
    n_transport = sum(1 for v in all_oper if "transport" in v["verdict"])
    n_fail = sum(1 for v in all_oper if "FAIL" in v["verdict"])
    n_faible = sum(1 for v in all_oper if "faible" in v["verdict"])
    n_indetermine = sum(1 for v in all_oper if "INDETERMINE" in v["verdict"])
    n_total_v = len(all_oper)
    print(f"    Distribution sur {n_total_v} etats :")
    print(f"      PASS robuste : {n_pass_robuste}")
    print(f"      PASS local   : {n_pass_local}")
    print(f"      Transport    : {n_transport}")
    print(f"      Faible       : {n_faible}")
    print(f"      INDETERMINE  : {n_indetermine}")
    print(f"      FAIL         : {n_fail}")

    # Verdict combine
    if n_pass_robuste >= 5:
        oper_verdict = "OPER-PASS robuste (majoritaire)"
    elif n_pass_robuste + n_pass_local + n_transport >= n_total_v / 2:
        oper_verdict = "OPER-PASS faible/transport (majoritaire)"
    elif n_fail >= n_total_v / 2:
        oper_verdict = "OPER-FAIL"
    else:
        oper_verdict = "OPER-INDETERMINE"
    print(f"    Verdict : {oper_verdict}")

    # Combine
    print(f"\n  COMBINE :")
    if "GEOM-PASS" in geom_verdict and "OPER-PASS" in oper_verdict:
        combined = "GEOM-PASS + OPER-PASS : non-separabilite geometrique avec effet operatoire lisible"
        suite = "lambda-B possible apres specification lambda-B-0"
    elif "GEOM-PASS" in geom_verdict and "OPER-FAIL" in oper_verdict:
        combined = "GEOM-PASS + OPER-FAIL : non-separabilite geometrique sans effet dynamique lisible"
        suite = "pas de lambda-B"
    elif "GEOM-PASS" in geom_verdict and "OPER-INDETERMINE" in oper_verdict:
        combined = "GEOM-PASS + OPER-INDETERMINE : diagnostic operatoire a clarifier"
        suite = "pas de lambda-B avant clarification"
    elif "GEOM-FAIL" in geom_verdict:
        combined = "GEOM-FAIL : h plein proche de produit-separable dans la famille testee"
        suite = "pas de lambda-B"
    else:
        combined = "indetermine"
        suite = "pas de lambda-B"
    print(f"    {combined}")
    print(f"    Suite : {suite}")

    print(f"\n  CAVEAT DE PORTEE : lambda-A-v2 teste la non-separabilite sur E0/E1/E2.")
    print(f"  Un verdict ne se generalise pas a h(theta) plein hors de cette famille.")

    # Sauvegarder
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(x) for x in obj]
        return obj

    output = {
        "dt_simulation": float(dt),
        "amp_P6": float(amp_P6),
        "target_amp": float(target_amp),
        "h0": h0,
        "checkpoint_times": checkpoint_times,
        "thresholds_R_sym": {"faible": 0.005, "significatif": 0.05, "dominant": 0.3},
        "E0": {
            "D_proj": D_proj_E0,
            "profile": profile_E0,
            "flux": serialize(flux_E0),
            "g_k": g_k_E0,
            "verdict_oper": serialize(verdict_E0),
        },
        "E1_by_time": serialize({str(t): E1_by_time[t] for t in checkpoint_times}),
        "E2_by_time": serialize({str(t): E2_by_time[t] for t in checkpoint_times}),
        "verdict_global": {
            "GEOM": geom_verdict,
            "OPER": oper_verdict,
            "COMBINE": combined,
            "suite": suite,
            "D_proj_range": [D_proj_min, D_proj_max],
            "n_oper_pass_robuste": n_pass_robuste,
            "n_oper_pass_local": n_pass_local,
            "n_oper_transport": n_transport,
            "n_oper_faible": n_faible,
            "n_oper_indetermine": n_indetermine,
            "n_oper_fail": n_fail,
            "n_total_states": n_total_v,
        },
    }
    with open("/home/claude/mcq_v4/6d_lambda_A_v2.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegarde.")


if __name__ == "__main__":
    main()
