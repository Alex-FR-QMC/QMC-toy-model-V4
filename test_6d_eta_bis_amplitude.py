"""
6d-η-bis — Contrôle L2 ciblé : dépendance de l'axe G/A à l'amplitude commune.

Reprend exactement le protocole 6d-η mais sur les 9 variantes G+A
uniquement (pas de D, car la question est isolée à G/A), à trois
amplitudes : 5%, 10%, 20% de ||P6||.

Moteur, P6, référence P6(3000), Δt long, résidu et observables
temporelles : INCHANGÉS par rapport à 6d-η.

Sorties pour chaque amplitude :
- table calibration (cible, atteinte, statut OK/SATURATED/FAILED)
- table NOISY (||r||, normes observables, statut OK/NOISY)
- matrices cos_total / cos_ss_pure / contrib_cos_ss (9×9)
- A_GA = mean(G,G) - mean(G,A), par observable et mode
- audit permutation exact C(9,5)=126 sur G/A
- statistiques globales

Contrôles :
- 10% doit reproduire qualitativement 6d-η (sur les 9 G+A)
- comparaison séparée : sous-bloc G+A extrait de la matrice 12×12 de 6d-η

Verdict préinscrit : η-bis PASS fort / PASS faible / FAIL / INDETERMINÉ
selon stabilité de l'axe G/A sous variation d'amplitude.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import json
from itertools import combinations
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

# G family
def P_G_etroite_centree(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(CENTER, CENTER, CENTER, 0.5)
    return (psi * factor) / (psi * factor).sum()
def P_G_standard_centree(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(CENTER, CENTER, CENTER, 0.8)
    return (psi * factor) / (psi * factor).sum()
def P_G_large_centree(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(CENTER, CENTER, CENTER, 1.5)
    return (psi * factor) / (psi * factor).sum()
def P_G_decentree_x(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(CENTER + DX, CENTER, CENTER, 0.8)
    return (psi * factor) / (psi * factor).sum()
def P_G_decentree_diag(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(
        CENTER + 0.7*DX, CENTER + 0.7*DX, CENTER + 0.7*DX, 0.8)
    return (psi * factor) / (psi * factor).sum()

# A family
def P_A_anneau_interne(psi, strength):
    factor = 1.0 + strength * _radial_mask(0.7, 1.3)
    return (psi * factor) / (psi * factor).sum()
def P_A_anneau_moyen(psi, strength):
    factor = 1.0 + strength * _radial_mask(1.3, 1.9)
    return (psi * factor) / (psi * factor).sum()
def P_A_anneau_externe(psi, strength):
    factor = 1.0 + strength * _radial_mask(1.9, 2.5)
    return (psi * factor) / (psi * factor).sum()
def P_A_shell_peripherique(psi, strength):
    factor = 1.0 + strength * _radial_mask(2.5, 100.0)
    return (psi * factor) / (psi * factor).sum()


VARIANTS_GA = [
    ("G_etroite_centree",     "G", P_G_etroite_centree),
    ("G_standard_centree",    "G", P_G_standard_centree),
    ("G_large_centree",       "G", P_G_large_centree),
    ("G_decentree_x",         "G", P_G_decentree_x),
    ("G_decentree_diag",      "G", P_G_decentree_diag),
    ("A_anneau_interne",      "A", P_A_anneau_interne),
    ("A_anneau_moyen",        "A", P_A_anneau_moyen),
    ("A_anneau_externe",      "A", P_A_anneau_externe),
    ("A_shell_peripherique",  "A", P_A_shell_peripherique),
]


def calibrate_strength(P_fn, psi_base, target_amp, s_bounds=(1e-6, 5.0)):
    def err(s):
        return float(np.linalg.norm(P_fn(psi_base, s) - psi_base)) - target_amp
    s_min, s_max = s_bounds
    try:
        err_min = err(s_min); err_max = err(s_max)
    except Exception:
        return None, None, "FAILED"
    if err_min >= 0:
        return s_min, float(np.linalg.norm(P_fn(psi_base, s_min) - psi_base)), "SATURATED"
    if err_max < 0:
        amp_at_max = float(np.linalg.norm(P_fn(psi_base, s_max) - psi_base))
        if amp_at_max < 0.5 * target_amp:
            return s_max, amp_at_max, "FAILED"
        return s_max, amp_at_max, "SATURATED"
    try:
        s_root = brentq(err, s_min, s_max, xtol=1e-8)
        amp_obt = float(np.linalg.norm(P_fn(psi_base, s_root) - psi_base))
        status = "OK" if abs(amp_obt - target_amp) / target_amp < 0.02 else "SATURATED"
        return s_root, amp_obt, status
    except Exception:
        return None, None, "FAILED"


def compute_delta_full(psi_start, h_start, P_prime_fn, s_prime,
                       D, beta, gamma_v, h0, dt, n_dt):
    psis_ref, hs_ref = evolve_with_trajectory(
        psi_start.copy(), h_start.copy(),
        D, beta, gamma_v, h0, dt, n_dt)
    psi_pp = P_prime_fn(psi_start.copy(), s_prime)
    h_pp = h_start.copy()
    psis_with, hs_with = evolve_with_trajectory(
        psi_pp, h_pp, D, beta, gamma_v, h0, dt, n_dt)
    return psis_with - psis_ref, hs_with - hs_ref


def get_obs(r_psi, r_h, name):
    if name == "psi_temp_mean_abs":
        return np.mean(np.abs(r_psi), axis=(1,2,3))
    elif name == "psi_temp_norm":
        return np.sqrt(np.sum(r_psi**2, axis=(1,2,3)))
    elif name == "h_temp_mean_abs":
        return np.mean(np.abs(r_h), axis=(1,2,3))
    elif name == "h_temp_norm":
        return np.sqrt(np.sum(r_h**2, axis=(1,2,3)))


def cos_pair(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-30 or n2 < 1e-30: return 0.0
    return float(np.dot(v1.flatten(), v2.flatten()) / (n1 * n2))


def run_one_amplitude(amplitude_pct, target_amp, psi_base, h_base, psi_tau0, h_tau0,
                       psi_tau3000, h_tau3000,
                       D, beta_lock, gamma_v, h0, dt, n_dt_long, permutations):
    """Lance le test complet à une amplitude donnée."""
    print(f"\n{'='*60}")
    print(f"=== Amplitude {amplitude_pct}% de ||P6|| (target = {target_amp:.4e}) ===")
    print(f"{'='*60}\n")

    # Calibration
    print(f"--- Calibration ---")
    print(f"  {'label':<28} {'family':<8} {'s_calib':>12} {'amp_obt':>14} {'status':>10}")
    calibration = []
    for label, family, P_fn in VARIANTS_GA:
        s_calib, amp_obt, status = calibrate_strength(P_fn, psi_base, target_amp)
        calibration.append({
            "label": label, "family": family,
            "s_calib": s_calib, "amp_obt": amp_obt, "status": status,
        })
        print(f"  {label:<28} {family:<8} "
              f"{s_calib if s_calib else 0:>12.6f} "
              f"{amp_obt if amp_obt else 0:>14.4e} "
              f"{status:>10}")

    # Calculer les résidus
    print(f"\n--- Résidus ---")
    residuals = {}
    for c_idx, (label, family, P_fn) in enumerate(VARIANTS_GA):
        c = calibration[c_idx]
        if c["status"] == "FAILED":
            print(f"  {label:<28} FAILED — exclu")
            continue
        d_psi_tau0, d_h_tau0 = compute_delta_full(
            psi_tau0, h_tau0, P_fn, c["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_psi_ref, d_h_ref = compute_delta_full(
            psi_tau3000, h_tau3000, P_fn, c["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_flat = np.concatenate([d_psi_tau0.flatten(), d_h_tau0.flatten()])
        ref_flat = np.concatenate([d_psi_ref.flatten(), d_h_ref.flatten()])
        ref_norm_sq = float(np.dot(ref_flat, ref_flat))
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq) if ref_norm_sq > 1e-30 else 0.0
        r_psi = d_psi_tau0 - a_tau * d_psi_ref
        r_h = d_h_tau0 - a_tau * d_h_ref
        residuals[label] = {"r_psi": r_psi, "r_h": r_h, "family": family}
        norm_rp = float(np.linalg.norm(r_psi))
        norm_rh = float(np.linalg.norm(r_h))
        print(f"  {label:<28} ||r_psi|| = {norm_rp:.4e}, ||r_h|| = {norm_rh:.4e}")

    # Diagnostic NOISY
    print(f"\n--- Diagnostic NOISY ---")
    obs_names = ["psi_temp_mean_abs", "psi_temp_norm",
                 "h_temp_mean_abs", "h_temp_norm"]
    noisy_status = {}
    NOISY_ABS_THR = 1e-9
    NOISY_OBS_THR = 1e-12
    for label, r_dict in residuals.items():
        norm_rp = float(np.linalg.norm(r_dict["r_psi"]))
        norm_rh = float(np.linalg.norm(r_dict["r_h"]))
        obs_norms = {n: float(np.linalg.norm(get_obs(r_dict["r_psi"], r_dict["r_h"], n)))
                     for n in obs_names}
        absolute_noisy = (norm_rp < NOISY_ABS_THR) or (norm_rh < NOISY_ABS_THR)
        relative_noisy = any(v < NOISY_OBS_THR for v in obs_norms.values())
        status = "NOISY" if (absolute_noisy or relative_noisy) else "OK"
        noisy_status[label] = {
            "norm_r_psi": norm_rp,
            "norm_r_h": norm_rh,
            "obs_norms": obs_norms,
            "status": status,
        }
        if status == "NOISY":
            print(f"  {label:<28} NOISY (||r_psi||={norm_rp:.2e}, ||r_h||={norm_rh:.2e})")
        else:
            print(f"  {label:<28} OK")
    n_noisy = sum(1 for s in noisy_status.values() if s["status"] == "NOISY")
    print(f"  Total : {n_noisy} variantes NOISY sur {len(residuals)}")

    # Calculs sur les variantes valides (non FAILED, on garde les NOISY mais on marque)
    labels = list(residuals.keys())
    families = [residuals[l]["family"] for l in labels]
    n = len(labels)
    if n < 9:
        print(f"\n  ATTENTION : seulement {n} variantes valides")

    # Décomposition + matrices, pour chaque observable et mode
    by_obs_mode = {}
    print(f"\n--- Décomposition et matrices ---")
    for name in obs_names:
        for mode in ["brut", "centré"]:
            key = f"{name}_{mode}"
            O = {}
            for lab in labels:
                serie = get_obs(residuals[lab]["r_psi"], residuals[lab]["r_h"], name)
                if mode == "centré":
                    serie = serie - serie.mean()
                O[lab] = serie
            # Ō sur les 9 (option a)
            O_bar = np.mean([O[lab] for lab in labels], axis=0)
            O_bar_norm_sq = float(np.dot(O_bar, O_bar))
            Shared, Specific = {}, {}
            for lab in labels:
                if O_bar_norm_sq > 1e-30:
                    a = float(np.dot(O[lab], O_bar) / O_bar_norm_sq)
                else:
                    a = 0.0
                Shared[lab] = a * O_bar
                Specific[lab] = O[lab] - Shared[lab]
            cos_total = np.zeros((n, n))
            cos_ss_pure = np.zeros((n, n))
            contrib_cos_ss = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    li, lj = labels[i], labels[j]
                    if i == j:
                        cos_total[i,j] = 1.0
                        cos_ss_pure[i,j] = 1.0
                        contrib_cos_ss[i,j] = 0.0
                        continue
                    cos_total[i,j] = cos_pair(O[li], O[lj])
                    cos_ss_pure[i,j] = cos_pair(Specific[li], Specific[lj])
                    norm_i = np.linalg.norm(O[li])
                    norm_j = np.linalg.norm(O[lj])
                    if norm_i > 1e-30 and norm_j > 1e-30:
                        contrib_cos_ss[i,j] = float(np.dot(Specific[li], Specific[lj]) / (norm_i * norm_j))
            by_obs_mode[key] = {
                "labels": labels,
                "families": families,
                "cos_total_matrix": cos_total.tolist(),
                "cos_ss_pure_matrix": cos_ss_pure.tolist(),
                "contrib_cos_ss_matrix": contrib_cos_ss.tolist(),
            }

    # A_GA et audit permutation par cas
    print(f"\n--- Score A_GA et audit permutation ---")
    idx_G = [i for i, f in enumerate(families) if f == "G"]
    idx_A = [i for i, f in enumerate(families) if f == "A"]
    n_G = len(idx_G); n_A = len(idx_A)
    idx_GA = idx_G + idx_A

    def compute_A_GA(matrix, idx_G_set, idx_A_set):
        GG_vals, GA_vals = [], []
        for i in idx_G_set:
            for j in idx_G_set:
                if i >= j: continue
                GG_vals.append(matrix[i][j])
            for j in idx_A_set:
                GA_vals.append(matrix[i][j])
        m_GG = float(np.mean(GG_vals)) if GG_vals else None
        m_GA = float(np.mean(GA_vals)) if GA_vals else None
        if m_GG is None or m_GA is None: return None, None, None
        return m_GG - m_GA, m_GG, m_GA

    # Énumérer permutations (uniquement si nombre G+A correct)
    if n_G == 5 and n_A == 4:
        perm_list = list(combinations(idx_GA, n_G))
    else:
        perm_list = []
        print(f"  ATTENTION : nombre G/A ≠ 5/4 ({n_G}/{n_A}), permutation non énumérée")

    print(f"\n  {'observable_mode':<28} {'mean(G,G)':>10} {'mean(A,A)':>10} "
          f"{'mean(G,A)':>10} {'A_GA':>8} {'p_high':>8} {'cs_pure_p':>10} {'cc_ss_p':>10}")
    audit_by_case = {}
    for key, data in by_obs_mode.items():
        cos_ss_mat = data["cos_ss_pure_matrix"]
        contrib_mat = data["contrib_cos_ss_matrix"]
        A_GA_obs, mGG, mGA = compute_A_GA(cos_ss_mat, idx_G, idx_A)
        # mean(A,A)
        AA_vals = []
        for i in idx_A:
            for j in idx_A:
                if i >= j: continue
                AA_vals.append(cos_ss_mat[i][j])
        mAA = float(np.mean(AA_vals)) if AA_vals else None
        # Permutation
        p_high_cs = None
        p_high_cc = None
        if perm_list:
            eps = 1e-10
            dist_cs = []
            dist_cc = []
            for perm_G in perm_list:
                perm_G_set = list(perm_G)
                perm_A_set = list(set(idx_GA) - set(perm_G))
                a_cs, _, _ = compute_A_GA(cos_ss_mat, perm_G_set, perm_A_set)
                a_cc, _, _ = compute_A_GA(contrib_mat, perm_G_set, perm_A_set)
                if a_cs is not None: dist_cs.append(a_cs)
                if a_cc is not None: dist_cc.append(a_cc)
            dist_cs = np.array(dist_cs)
            dist_cc = np.array(dist_cc)
            n_perms_total = len(dist_cs)
            n_high_cs = int((dist_cs >= A_GA_obs - eps).sum())
            p_high_cs = max(n_high_cs, 1) / n_perms_total
            # Pour contrib, on a besoin du A_GA observé sur contrib
            A_GA_obs_cc, _, _ = compute_A_GA(contrib_mat, idx_G, idx_A)
            n_high_cc = int((dist_cc >= A_GA_obs_cc - eps).sum())
            p_high_cc = max(n_high_cc, 1) / n_perms_total
        else:
            A_GA_obs_cc = None
        audit_by_case[key] = {
            "mean_GG": mGG, "mean_AA": mAA, "mean_GA": mGA,
            "A_GA_cos_ss_pure": A_GA_obs,
            "A_GA_contrib_cos_ss": A_GA_obs_cc,
            "p_high_cos_ss_pure": p_high_cs,
            "p_high_contrib_cos_ss": p_high_cc,
        }
        p_high_str_cs = f"{p_high_cs:.4f}" if p_high_cs is not None else "n/a"
        p_high_str_cc = f"{p_high_cc:.4f}" if p_high_cc is not None else "n/a"
        A_GA_str = f"{A_GA_obs:+.4f}" if A_GA_obs is not None else "n/a"
        mGG_str = f"{mGG:+.4f}" if mGG is not None else "n/a"
        mAA_str = f"{mAA:+.4f}" if mAA is not None else "n/a"
        mGA_str = f"{mGA:+.4f}" if mGA is not None else "n/a"
        # Affichage : p_high principal = cos_ss_pure
        p_high_str = p_high_str_cs
        print(f"  {key:<28} {mGG_str:>10} {mAA_str:>10} {mGA_str:>10} "
              f"{A_GA_str:>8} {p_high_str:>8} {p_high_str_cs:>10} {p_high_str_cc:>10}")

    return {
        "amplitude_pct": amplitude_pct,
        "target_amp": target_amp,
        "calibration": calibration,
        "noisy_status": noisy_status,
        "n_noisy": n_noisy,
        "n_variants_used": n,
        "by_obs_mode_matrices": by_obs_mode,
        "audit_by_case": audit_by_case,
    }


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_dt_long = int(800.0 / dt)

    print(f"=== 6d-η-bis : contrôle L2 amplitude sur axe G/A ===\n")
    print(f"  dt = {dt:.5f}, n_dt_long = {n_dt_long}")

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    # Calibration P6 (référence) — exactement comme 6d-η
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
    amp_P1prime_std = float(np.linalg.norm(P1prime_plateau(psi_base, 0.05) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    print(f"  ||P6|| = {amp_P6:.4e}")

    # États P6(0) et P6(3000)
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6,
                                D, beta_lock, gamma_v, h0, dt, n_short)
    n_3000 = int(3000.0 / dt)
    psi_tau3000, h_tau3000 = evolve(psi_tau0.copy(), h_tau0.copy(),
                                     D, beta_lock, gamma_v, h0, dt, n_3000)

    # Permutations pour 5G + 4A = 9 variantes
    n_G_ref = sum(1 for _, f, _ in VARIANTS_GA if f == "G")
    n_A_ref = sum(1 for _, f, _ in VARIANTS_GA if f == "A")
    perms = list(combinations(range(9), n_G_ref))
    print(f"  Permutations C(9,{n_G_ref}) = {len(perms)}\n")

    # Lancer aux 3 amplitudes
    amplitudes_pct = [5, 10, 20]
    results = {}
    for amp_pct in amplitudes_pct:
        target = (amp_pct / 100.0) * amp_P6
        r = run_one_amplitude(amp_pct, target, psi_base, h_base,
                              psi_tau0, h_tau0, psi_tau3000, h_tau3000,
                              D, beta_lock, gamma_v, h0, dt, n_dt_long, perms)
        results[f"amp_{amp_pct}"] = r

    # ===== Contrôle de reproduction 10% vs 6d-η filtré =====
    print(f"\n{'='*60}")
    print(f"=== Contrôle de reproduction : 10% η-bis vs 6d-η filtré G+A ===")
    print(f"{'='*60}\n")
    try:
        with open("/home/claude/mcq_v4/6d_eta_confirmation.json") as f:
            eta_data = json.load(f)
        eta_labels = eta_data["labels"]
        eta_families = eta_data["families"]
        idx_GA_eta = [i for i, fam in enumerate(eta_families) if fam in ("G", "A")]
        idx_G_eta = [i for i in idx_GA_eta if eta_families[i] == "G"]
        idx_A_eta = [i for i in idx_GA_eta if eta_families[i] == "A"]
        print(f"  6d-η : {len(idx_GA_eta)} variantes G+A extraites du sous-bloc 12×12")
        print(f"  {'observable_mode':<28} {'A_GA(η filtré)':>14} {'A_GA(η-bis 10%)':>16} {'diff':>10}")
        comparison = {}
        for key in results["amp_10"]["audit_by_case"].keys():
            eta_matrix = np.array(eta_data["by_observable"][key]["cos_ss_pure_matrix"])
            # A_GA sur le sous-bloc G+A de la matrice 12×12 (Ō défini sur 12 en η)
            GG_vals = [eta_matrix[i][j] for i in idx_G_eta for j in idx_G_eta if i < j]
            GA_vals = [eta_matrix[i][j] for i in idx_G_eta for j in idx_A_eta]
            mGG_eta = float(np.mean(GG_vals))
            mGA_eta = float(np.mean(GA_vals))
            A_GA_eta_filt = mGG_eta - mGA_eta
            A_GA_bis = results["amp_10"]["audit_by_case"][key]["A_GA_cos_ss_pure"]
            diff = A_GA_bis - A_GA_eta_filt if A_GA_bis is not None else None
            comparison[key] = {
                "A_GA_eta_filtered": A_GA_eta_filt,
                "A_GA_eta_bis_10pct": A_GA_bis,
                "diff": diff,
            }
            diff_str = f"{diff:+.4f}" if diff is not None else "n/a"
            bis_str = f"{A_GA_bis:+.4f}" if A_GA_bis is not None else "n/a"
            print(f"  {key:<28} {A_GA_eta_filt:>+14.4f} {bis_str:>16} {diff_str:>10}")
    except Exception as e:
        print(f"  Erreur de contrôle : {e}")
        comparison = {"error": str(e)}

    # ===== Synthèse globale =====
    print(f"\n{'='*60}")
    print(f"=== Synthèse globale : stabilité de l'axe G/A sous amplitude ===")
    print(f"{'='*60}\n")
    print(f"  {'amplitude':<12} {'n_noisy':>10} {'A_GA_mean':>12} {'A_GA_min':>12} "
          f"{'p<0.05 cs_pure':>16} {'p<0.05 cc_ss':>14}")
    summary_by_amp = {}
    for amp_pct in amplitudes_pct:
        r = results[f"amp_{amp_pct}"]
        cases = r["audit_by_case"]
        A_GA_vals_cs = [c["A_GA_cos_ss_pure"] for c in cases.values()
                         if c["A_GA_cos_ss_pure"] is not None]
        p_cs_list = [c["p_high_cos_ss_pure"] for c in cases.values()
                      if c["p_high_cos_ss_pure"] is not None]
        p_cc_list = [c["p_high_contrib_cos_ss"] for c in cases.values()
                      if c["p_high_contrib_cos_ss"] is not None]
        A_GA_mean = float(np.mean(A_GA_vals_cs)) if A_GA_vals_cs else None
        A_GA_min = float(np.min(A_GA_vals_cs)) if A_GA_vals_cs else None
        n_cs_extreme = sum(1 for p in p_cs_list if p < 0.05)
        n_cc_extreme = sum(1 for p in p_cc_list if p < 0.05)
        summary_by_amp[f"amp_{amp_pct}"] = {
            "n_noisy": r["n_noisy"],
            "A_GA_mean_cos_ss_pure": A_GA_mean,
            "A_GA_min_cos_ss_pure": A_GA_min,
            "n_extreme_cos_ss_pure_top5pct": n_cs_extreme,
            "n_extreme_contrib_cos_ss_top5pct": n_cc_extreme,
        }
        print(f"  {amp_pct}%:".ljust(12) +
              f" {r['n_noisy']:>10}" +
              f" {A_GA_mean:>+12.4f}" +
              f" {A_GA_min:>+12.4f}" +
              f" {n_cs_extreme}/{len(cases):<14}" +
              f" {n_cc_extreme}/{len(cases):<13}")

    # Suivi cas fragile
    print(f"\n  Suivi cas fragile psi_temp_norm_centré :")
    for amp_pct in amplitudes_pct:
        case = results[f"amp_{amp_pct}"]["audit_by_case"].get("psi_temp_norm_centré")
        if case:
            print(f"    {amp_pct}% : A_GA = {case['A_GA_cos_ss_pure']:+.4f}, "
                  f"p_high = {case['p_high_cos_ss_pure']:.4f}")

    # === Verdict ===
    print(f"\n--- Verdict η-bis ---")
    # PASS fort si : pas trop de NOISY, A_GA reste positif aux 3 amplitudes,
    # nombre de cas extrêmes reste élevé (>= 6/8) aux 3 amplitudes
    cond_no_failures = all(
        all(c["status"] != "FAILED" for c in r["calibration"])
        for r in results.values()
    )
    cond_low_noisy = all(r["n_noisy"] <= 2 for r in results.values())
    cond_A_GA_positive = all(s["A_GA_min_cos_ss_pure"] is not None
                              and s["A_GA_mean_cos_ss_pure"] > 0
                              for s in summary_by_amp.values())
    cond_extreme_cs = all(s["n_extreme_cos_ss_pure_top5pct"] >= 6
                           for s in summary_by_amp.values())
    cond_extreme_cc = all(s["n_extreme_contrib_cos_ss_top5pct"] >= 5
                           for s in summary_by_amp.values())

    print(f"  Aucun FAILED : {cond_no_failures}")
    print(f"  NOISY ≤ 2 par amplitude : {cond_low_noisy}")
    print(f"  A_GA_mean > 0 aux 3 amplitudes : {cond_A_GA_positive}")
    print(f"  ≥ 6/8 cas extrêmes cos_ss_pure aux 3 amplitudes : {cond_extreme_cs}")
    print(f"  ≥ 5/8 cas extrêmes contrib_cos_ss aux 3 amplitudes : {cond_extreme_cc}")

    if cond_no_failures and cond_low_noisy and cond_A_GA_positive and cond_extreme_cs and cond_extreme_cc:
        verdict = "η-bis PASS fort"
    elif cond_no_failures and cond_A_GA_positive and (cond_extreme_cs or cond_extreme_cc):
        verdict = "η-bis PASS faible"
    elif not cond_A_GA_positive:
        verdict = "η-bis FAIL"
    else:
        verdict = "η-bis INDETERMINÉ"
    print(f"\n  VERDICT : {verdict}")

    full_output = {
        "amplitudes_tested": amplitudes_pct,
        "amp_P6": amp_P6,
        "results_by_amplitude": results,
        "comparison_eta_filtered": comparison,
        "summary_by_amp": summary_by_amp,
        "verdict": verdict,
        "verdict_criteria": {
            "no_failures": cond_no_failures,
            "low_noisy": cond_low_noisy,
            "A_GA_positive": cond_A_GA_positive,
            "extreme_cs": cond_extreme_cs,
            "extreme_cc": cond_extreme_cc,
        },
    }
    # Nettoyer pour JSON
    for amp_key, r in full_output["results_by_amplitude"].items():
        # Réduire les matrices : on garde, mais on omet labels redondants
        pass
    with open("/home/claude/mcq_v4/6d_eta_bis_amplitude.json", "w") as f:
        json.dump(full_output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else None)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
