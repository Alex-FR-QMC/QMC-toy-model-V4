# -*- coding: utf-8 -*-
"""
6d-P5bis-A : diagnostic complet bassins + chemins + observables fonctionnelles post-λ.

Cf. specification 6d_p5bis0_specification.md.

Familles : A, B1, B2, B3 (heritees de test_5_7_family_B.py / 6d-α §5.7).
beta : 60 principal, 45 et 80 en controles cibles sur A vs B2.
Horizons : T=800 verdict principal, T=3000 controle stabilite (avec diagnostic underflow).

Metriques :
- distances pairwise Dpsi/Dh/Dext (t) + AUC + final
- profils fonctionnels h-active, gradient-active, intersection, Jaccard
- diagnostic underflow
- reactivation fonctionnelle a T=800 (perturbations G_standard, A_anneau)
  avec amplitude absolue 0.1 * ||P6||, recalibree localement par famille

Pas de h_proj. Pas de λ-B. Pas de P4 strict.
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


H_RESOLUTION = 1e-6
H_FUNCTIONAL = 1e-3
EPS = 1e-30


# ========== Moteur ==========
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
    """Evolue et garde les snapshots aux indices specifies (steps depuis t=0)."""
    states = {}
    if 0 in checkpoint_steps:
        states[0] = (psi.copy(), h.copy())
    n_max = max(checkpoint_steps)
    for n in range(1, n_max + 1):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        if n in checkpoint_steps:
            states[n] = (psi.copy(), h.copy())
    return states


# ========== Familles (heritees de 6d-α §5.7) ==========
def make_psi_A(sigma=1.8):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2
                psi[i,j,k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()

def make_psi_B1(sigma=1.8):
    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = 1.0, 2.0, 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i]-cx)**2 + (coords[j]-cy)**2 + (coords[k]-cz)**2
                psi[i,j,k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()

def make_psi_B2(sigma=1.0):
    coords = np.arange(N_AXIS) * DX
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for cx, cy, cz in [(1.5, 2.0, 2.0), (2.5, 2.0, 2.0)]:
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r2 = (coords[i]-cx)**2 + (coords[j]-cy)**2 + (coords[k]-cz)**2
                    psi[i,j,k] += np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()

def make_psi_B3():
    return np.full((N_AXIS, N_AXIS, N_AXIS), 1.0 / (N_AXIS ** 3))


# ========== Perturbations P' (cycles eta/theta/lambda) ==========
COORDS = np.arange(N_AXIS) * DX
CENTER = (N_AXIS - 1) * DX / 2.0

def _gaussian_factor(cx, cy, cz, sigma_p):
    factor = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (COORDS[i]-cx)**2 + (COORDS[j]-cy)**2 + (COORDS[k]-cz)**2
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

def P_G_standard(psi, s):
    f = 1.0 + s * _gaussian_factor(CENTER, CENTER, CENTER, 0.8)
    return (psi * f) / (psi * f).sum()

def P_A_anneau(psi, s):
    f = 1.0 + s * _radial_mask(1.3, 1.9)
    return (psi * f) / (psi * f).sum()

def P6_face_dipole(psi, s):
    factor = np.ones_like(psi)
    factor[0, :, :] += s
    factor[4, :, :] -= s
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()


# ========== Reference globale ||P6|| ==========
def compute_norm_P6_ref(D, beta_lock, gamma_v, h0, dt, n_stab):
    """||P6|| de reference : calibre sur psi_base (gaussienne sigma=1.5 relaxee)
    pour atteindre amp_P1prime_std a strength=0.05.
    Identique aux cycles eta/theta/iota/lambda.
    """
    def make_psi_centered(sigma=1.5):
        coords = np.arange(N_AXIS) * DX
        c = (N_AXIS - 1) * DX / 2.0
        psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r2 = (coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2
                    psi[i,j,k] = np.exp(-0.5 * r2 / sigma**2)
        return psi / psi.sum()

    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_base, _ = evolve(psi0, h0f, D, beta_lock, gamma_v, h0, dt, n_stab)

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
    return amp_P6


def calibrate_local(P_fn, psi_pre, target_amp, s_bounds=(1e-6, 5.0)):
    """Calibre s tel que ||P'(psi_pre) - psi_pre|| = target_amp."""
    def err(s):
        return float(np.linalg.norm(P_fn(psi_pre, s) - psi_pre)) - target_amp
    try:
        s_root = brentq(err, s_bounds[0], s_bounds[1], xtol=1e-8)
        amp_obt = float(np.linalg.norm(P_fn(psi_pre, s_root) - psi_pre))
        if abs(amp_obt - target_amp) / target_amp > 0.05:
            status = "SATURATED"
        else:
            status = "OK"
        return s_root, amp_obt, status
    except Exception as e:
        return None, None, f"FAILED({type(e).__name__})"


# ========== Diagnostic underflow ==========
def underflow_diag(h):
    n = h.size
    n_lt_6 = int(np.sum(h < 1e-6))
    n_lt_12 = int(np.sum(h < 1e-12))
    n_lt_30 = int(np.sum(h < 1e-30))
    n_lt_300 = int(np.sum(h < 1e-300))
    n_eq_0 = int(np.sum(h == 0))
    return {
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "h_median": float(np.median(h)),
        "n_lt_1e-6": n_lt_6,
        "n_lt_1e-12": n_lt_12,
        "n_lt_1e-30": n_lt_30,
        "n_lt_1e-300": n_lt_300,
        "n_eq_0": n_eq_0,
        "frac_lt_1e-6": n_lt_6 / n,
        "frac_lt_1e-12": n_lt_12 / n,
        "frac_lt_1e-30": n_lt_30 / n,
        "frac_lt_1e-300": n_lt_300 / n,
        "frac_eq_0": n_eq_0 / n,
    }


def underflow_status(diag, frac_threshold=0.05):
    """Etiquetage : OK / WARNING_UNDERFLOW / CONVERGENCE_UNDER_FLOOR."""
    if diag["n_eq_0"] > 0 or diag["frac_lt_1e-300"] > frac_threshold:
        return "CONVERGENCE_UNDER_FLOOR"
    if diag["frac_lt_1e-30"] > frac_threshold or diag["frac_lt_1e-12"] > frac_threshold * 2:
        return "WARNING_UNDERFLOW"
    return "OK"


# ========== Diagnostics fonctionnels ==========
def functional_profile(psi, h):
    """Profil h-active, gradient-active, intersection, Jaccard."""
    # Cellules
    n_cells = h.size
    frac_h_res = float(np.sum(h > H_RESOLUTION) / n_cells)
    frac_h_func = float(np.sum(h > H_FUNCTIONAL) / n_cells)

    # Faces : harmonic_mean reproduit moteur
    h_face_x = harmonic_mean(h[:-1, :, :], h[1:, :, :])
    h_face_y = harmonic_mean(h[:, :-1, :], h[:, 1:, :])
    h_face_z = harmonic_mean(h[:, :, :-1], h[:, :, 1:])
    h_face_flat = np.concatenate([h_face_x.flatten(),
                                    h_face_y.flatten(),
                                    h_face_z.flatten()])

    grad_x = (psi[1:, :, :] - psi[:-1, :, :]) / DX
    grad_y = (psi[:, 1:, :] - psi[:, :-1, :]) / DX
    grad_z = (psi[:, :, 1:] - psi[:, :, :-1]) / DX
    grad_flat = np.concatenate([np.abs(grad_x).flatten(),
                                  np.abs(grad_y).flatten(),
                                  np.abs(grad_z).flatten()])

    n_faces = len(h_face_flat)

    # h_active
    h_active = h_face_flat > 1e-6
    frac_h_active = float(h_active.sum() / n_faces)

    # grad_active
    grad_max = float(grad_flat.max())
    if grad_max < 1e-30:
        grad_status = "GRAD_DEGENERATE"
        grad_active = np.zeros(n_faces, dtype=bool)
        frac_grad_active = 0.0
        grad_threshold_value = 0.0
    else:
        grad_status = "OK"
        grad_threshold_value = 0.01 * grad_max
        grad_active = grad_flat > grad_threshold_value
        frac_grad_active = float(grad_active.sum() / n_faces)

    # Intersection
    intersection = h_active & grad_active
    frac_intersection = float(intersection.sum() / n_faces)

    # Jaccard
    union = h_active | grad_active
    n_union = int(union.sum())
    jaccard = float(intersection.sum()) / n_union if n_union > 0 else 0.0

    return {
        "frac_h_resolution": frac_h_res,
        "frac_h_functional": frac_h_func,
        "n_faces_total": n_faces,
        "frac_h_active": frac_h_active,
        "frac_grad_active": frac_grad_active,
        "frac_intersection_h_grad": frac_intersection,
        "jaccard_h_grad": jaccard,
        "grad_max": grad_max,
        "grad_threshold_value": grad_threshold_value,
        "grad_status": grad_status,
    }


# ========== Distances ==========
def pairwise_distances(psi_a, h_a, psi_b, h_b):
    """Distances Dpsi/Dh/Dext entre deux etats."""
    Dpsi = float(np.linalg.norm(psi_b - psi_a) / (np.linalg.norm(psi_a) + EPS))
    Dh = float(np.linalg.norm(h_b - h_a) / (np.linalg.norm(h_a) + EPS))
    ext_a = np.concatenate([psi_a.flatten(), h_a.flatten()])
    ext_b = np.concatenate([psi_b.flatten(), h_b.flatten()])
    Dext = float(np.linalg.norm(ext_b - ext_a) / (np.linalg.norm(ext_a) + EPS))
    return {"Dpsi": Dpsi, "Dh": Dh, "Dext": Dext}


# ========== Run par famille ==========
def run_family_full(name, psi_init, h_init, D, beta, gamma_v, h0, dt,
                     checkpoint_steps):
    """Evolue la famille et collecte snapshots + profils + underflow."""
    states = evolve_with_checkpoints(psi_init, h_init, D, beta, gamma_v, h0, dt,
                                      checkpoint_steps)
    profiles = {}
    underflows = {}
    masses = {}
    for n_step in sorted(states.keys()):
        psi_s, h_s = states[n_step]
        profiles[n_step] = functional_profile(psi_s, h_s)
        underflows[n_step] = underflow_diag(h_s)
        underflows[n_step]["status"] = underflow_status(underflows[n_step])
        masses[n_step] = float(psi_s.sum())
    return {
        "name": name,
        "states": states,  # gardes en memoire pour calculs pairwise
        "profiles": profiles,
        "underflows": underflows,
        "masses": masses,
    }


def compute_pairwise_full(fam_a, fam_b, checkpoint_steps, checkpoint_times, dt):
    """Calcule Dpsi/Dh/Dext aux checkpoints + AUC sur [0, T_max_for_AUC]."""
    pairs_at_t = {}
    for n_step in checkpoint_steps:
        if n_step in fam_a["states"] and n_step in fam_b["states"]:
            psi_a, h_a = fam_a["states"][n_step]
            psi_b, h_b = fam_b["states"][n_step]
            pairs_at_t[n_step] = pairwise_distances(psi_a, h_a, psi_b, h_b)

    # AUC sur tous les checkpoints (approximation trapezoidale)
    n_keys = sorted(pairs_at_t.keys())
    if len(n_keys) >= 2:
        ts = np.array([n * dt for n in n_keys])
        Dpsi_arr = np.array([pairs_at_t[n]["Dpsi"] for n in n_keys])
        Dh_arr = np.array([pairs_at_t[n]["Dh"] for n in n_keys])
        Dext_arr = np.array([pairs_at_t[n]["Dext"] for n in n_keys])
        AUC_Dpsi = float(np.trapezoid(Dpsi_arr, ts))
        AUC_Dh = float(np.trapezoid(Dh_arr, ts))
        AUC_Dext = float(np.trapezoid(Dext_arr, ts))
        max_Dpsi_idx = int(np.argmax(Dpsi_arr))
        max_Dh_idx = int(np.argmax(Dh_arr))
        max_Dext_idx = int(np.argmax(Dext_arr))
        max_info = {
            "max_Dpsi": float(Dpsi_arr[max_Dpsi_idx]),
            "t_max_Dpsi": float(ts[max_Dpsi_idx]),
            "max_Dh": float(Dh_arr[max_Dh_idx]),
            "t_max_Dh": float(ts[max_Dh_idx]),
            "max_Dext": float(Dext_arr[max_Dext_idx]),
            "t_max_Dext": float(ts[max_Dext_idx]),
        }
    else:
        AUC_Dpsi = AUC_Dh = AUC_Dext = None
        max_info = {}

    # Final a T=800 et T=3000
    final_at = {}
    for t_target in [800.0, 3000.0]:
        n_target = int(t_target / dt)
        if n_target in pairs_at_t:
            final_at[f"t_{int(t_target)}"] = pairs_at_t[n_target]

    return {
        "pairs_at_step": {str(k): v for k, v in pairs_at_t.items()},
        "AUC_Dpsi_full_traj": AUC_Dpsi,
        "AUC_Dh_full_traj": AUC_Dh,
        "AUC_Dext_full_traj": AUC_Dext,
        "max_info": max_info,
        "final_at": final_at,
    }


# ========== Test de reactivation ==========
def reactivation_test(family_name, psi_pre, h_pre, P_fn, P_name, s_root, amp_obt,
                       D, beta, gamma_v, h0, dt, n_post):
    """Applique P' a psi_pre, evolue n_post steps, compare avec evolution non perturbee.
    Renvoie metriques de retour metrique, asymptotique, et fonctionnel.
    """
    # Branche non perturbee : evolue psi_pre sur n_post steps
    psi_unp, h_unp = evolve(psi_pre.copy(), h_pre.copy(),
                              D, beta, gamma_v, h0, dt, n_post)
    # Branche perturbee : applique P' a psi_pre puis evolue
    psi_after_P = P_fn(psi_pre.copy(), s_root)
    psi_pert, h_pert = evolve(psi_after_P, h_pre.copy(),
                                D, beta, gamma_v, h0, dt, n_post)

    # A. Retour metrique : distance entre branches a t=fin
    metric_final = pairwise_distances(psi_pert, h_pert, psi_unp, h_unp)

    # B. Retour asymptotique : distance entre etat final perturbe et etat pre-perturbation
    asymp = pairwise_distances(psi_pert, h_pert, psi_pre, h_pre)

    # C. Retour fonctionnel : profils
    prof_pre = functional_profile(psi_pre, h_pre)
    prof_unp = functional_profile(psi_unp, h_unp)
    prof_pert = functional_profile(psi_pert, h_pert)
    # Recuperation : difference de profil intersection h_grad entre perturbe et non perturbe
    delta_jaccard = prof_pert["jaccard_h_grad"] - prof_unp["jaccard_h_grad"]
    delta_intersection = prof_pert["frac_intersection_h_grad"] - prof_unp["frac_intersection_h_grad"]
    delta_h_active = prof_pert["frac_h_active"] - prof_unp["frac_h_active"]

    # Verdict preliminaire de reactivation
    # Seuils empiriques modestes, labels candidats
    metric_threshold = 0.01
    func_threshold = 0.05  # 5 pts de pourcentage
    has_metric_diff = metric_final["Dext"] > metric_threshold
    has_func_diff = (abs(delta_jaccard) > 0.05 or abs(delta_intersection) > func_threshold)
    if has_func_diff and has_metric_diff:
        # verifier si c'est un shift de bassin (asymp Dext eleve persistant) ou functional
        if asymp["Dext"] > 0.1:
            label = "REACT-BASIN_SHIFT-candidate"
        else:
            label = "REACT-FUNCTIONAL-candidate"
    elif has_metric_diff and not has_func_diff:
        label = "REACT-METRIC_ONLY-candidate"
    elif not has_metric_diff and not has_func_diff:
        label = "REACT-NONE-candidate"
    else:
        label = "REACT-UNCLASSIFIED_WITH_METRICS"

    return {
        "perturbation_name": P_name,
        "s_root": s_root,
        "amp_obt": amp_obt,
        "metric_final_vs_unperturbed": metric_final,
        "asymp_vs_pre": asymp,
        "profile_pre": prof_pre,
        "profile_unperturbed": prof_unp,
        "profile_perturbed": prof_pert,
        "delta_intersection_h_grad": delta_intersection,
        "delta_jaccard_h_grad": delta_jaccard,
        "delta_frac_h_active": delta_h_active,
        "label": label,
    }


# ========== Determinism check ==========
def determinism_check(name, psi_init, h_init, D, beta, gamma_v, h0, dt, n_steps):
    psi_1, h_1 = evolve(psi_init.copy(), h_init.copy(),
                          D, beta, gamma_v, h0, dt, n_steps)
    psi_2, h_2 = evolve(psi_init.copy(), h_init.copy(),
                          D, beta, gamma_v, h0, dt, n_steps)
    d = pairwise_distances(psi_1, h_1, psi_2, h_2)
    max_abs_psi = float(np.max(np.abs(psi_1 - psi_2)))
    max_abs_h = float(np.max(np.abs(h_1 - h_2)))
    mass_diff = float(abs(psi_1.sum() - psi_2.sum()))
    h_min_diff = float(abs(h_1.min() - h_2.min()))
    is_deterministic = (max_abs_psi < 1e-14 and max_abs_h < 1e-14)
    return {
        "name": name,
        "Dpsi": d["Dpsi"],
        "Dh": d["Dh"],
        "Dext": d["Dext"],
        "max_abs_diff_psi": max_abs_psi,
        "max_abs_diff_h": max_abs_h,
        "mass_diff": mass_diff,
        "h_min_diff": h_min_diff,
        "verdict": "DETERMINISM_PASS" if is_deterministic else "DETERMINISM_FAIL",
    }


# ========== Labels candidats ==========
def label_pairwise(pairwise_result, T800_only=True):
    """Labels P5bis preliminaires sur paire (A vs B*).
    PATH si AUC eleve mais D_final petit
    BASIN si D_final stable et significatif
    FAIL si tout petit
    """
    if T800_only:
        f_key = "t_800"
        if f_key not in pairwise_result["final_at"]:
            return "P5bis-INDETERMINATE"
        df = pairwise_result["final_at"][f_key]
        Dh_f = df["Dh"]
        Dext_f = df["Dext"]
        AUC_Dh = pairwise_result.get("AUC_Dh_full_traj", None) or 0.0
    else:
        return "P5bis-INDETERMINATE"

    # Seuils empiriques (non figes)
    D_FINAL_LOW = 0.01
    D_FINAL_HIGH = 0.1
    AUC_HIGH = 10.0

    has_high_final = Dh_f > D_FINAL_HIGH
    has_low_final = Dh_f < D_FINAL_LOW
    has_high_AUC = AUC_Dh > AUC_HIGH

    if has_high_final:
        return "P5bis-BASIN-candidate"
    elif has_low_final and has_high_AUC:
        return "P5bis-PATH-candidate"
    elif has_low_final and not has_high_AUC:
        return "P5bis-FAIL-candidate"
    else:
        return "P5bis-INDETERMINATE"


def label_dissociation_per_family(profile_at_t800):
    """Etiquette DISSOC sur profil fonctionnel a T=800."""
    j = profile_at_t800.get("jaccard_h_grad", 0.0)
    fi = profile_at_t800.get("frac_intersection_h_grad", 0.0)
    if profile_at_t800.get("grad_status") == "GRAD_DEGENERATE":
        return "GRAD_DEGENERATE"
    if j > 0.3 and fi > 0.05:
        return "h_grad_overlap_preserved"
    elif j < 0.05 or fi < 0.005:
        return "h_grad_dissociated"
    else:
        return "h_grad_partial"


def label_long_horizon(uf_t3000):
    return uf_t3000.get("status", "OK").replace("WARNING_UNDERFLOW", "LONG_UNDERFLOW_WARNING") \
                                          .replace("CONVERGENCE_UNDER_FLOOR", "LONG_UNDERFLOW_DOMINATED") \
                                          .replace("OK", "LONG_OK")


# ========== Main ==========
def main():
    gamma_v, D, h0 = 1.0, 0.1, 1.0
    beta_lock = 60.0
    psi_max_init = float(make_psi_A().max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max_init + gamma_v))
    n_stab = int(50.0 / dt)
    print(f"=== 6d-P5bis-A : diagnostic bassins/chemins/fonctionnels post-λ ===\n")
    print(f"  dt = {dt:.5f}, h_init uniforme = {h0}\n")

    # Reference globale ||P6||
    norm_P6_ref = compute_norm_P6_ref(D, beta_lock, gamma_v, h0, dt, n_stab)
    target_amp_react = 0.1 * norm_P6_ref
    print(f"  ||P6|| de reference = {norm_P6_ref:.4e}")
    print(f"  target_amp reactivation = {target_amp_react:.4e}\n")

    # Checkpoints : 0, 10, 50, 100, 200, 400, 800, 1500, 3000
    checkpoint_times = [0.0, 10.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]
    checkpoint_steps = [int(t / dt) for t in checkpoint_times]

    # =====================================================
    # Controle determinisme
    # =====================================================
    print(f"--- C1 Controle determinisme ---")
    det_A = determinism_check("A_beta60_T800", make_psi_A(),
                                np.full((N_AXIS, N_AXIS, N_AXIS), h0),
                                D, beta_lock, gamma_v, h0, dt, int(800.0 / dt))
    print(f"  {det_A['name']} : max_abs_psi = {det_A['max_abs_diff_psi']:.2e}, "
          f"max_abs_h = {det_A['max_abs_diff_h']:.2e}, verdict = {det_A['verdict']}")
    det_B2 = determinism_check("B2_beta60_T200", make_psi_B2(),
                                np.full((N_AXIS, N_AXIS, N_AXIS), h0),
                                D, beta_lock, gamma_v, h0, dt, int(200.0 / dt))
    print(f"  {det_B2['name']} : max_abs_psi = {det_B2['max_abs_diff_psi']:.2e}, "
          f"max_abs_h = {det_B2['max_abs_diff_h']:.2e}, verdict = {det_B2['verdict']}")

    if det_A["verdict"] != "DETERMINISM_PASS" or det_B2["verdict"] != "DETERMINISM_PASS":
        print(f"\n  STOP : determinisme casse, audit moteur necessaire.")
        return

    # =====================================================
    # Runs principaux beta=60
    # =====================================================
    print(f"\n--- Runs principaux beta=60 ---")
    families_makers = {
        "A": make_psi_A,
        "B1": make_psi_B1,
        "B2": make_psi_B2,
        "B3": make_psi_B3,
    }
    families_beta60 = {}
    for fname, fmaker in families_makers.items():
        print(f"  Running {fname}...")
        psi_init = fmaker()
        h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
        result = run_family_full(fname, psi_init, h_init,
                                   D, beta_lock, gamma_v, h0, dt,
                                   checkpoint_steps)
        # ajouter etiquette dissociation a T=800
        n_t800 = int(800.0 / dt)
        result["dissociation_label_T800"] = label_dissociation_per_family(
            result["profiles"][n_t800])
        # Etiquette long-horizon
        n_t3000 = int(3000.0 / dt)
        if n_t3000 in result["underflows"]:
            result["long_horizon_label"] = label_long_horizon(result["underflows"][n_t3000])
        else:
            result["long_horizon_label"] = "UNKNOWN"
        families_beta60[fname] = result
        prof_t800 = result["profiles"][n_t800]
        uf_t800 = result["underflows"][n_t800]
        uf_t3000 = result["underflows"][n_t3000] if n_t3000 in result["underflows"] else {}
        print(f"    T=800 : frac_h_active={prof_t800['frac_h_active']:.3f}, "
              f"frac_grad_active={prof_t800['frac_grad_active']:.3f}, "
              f"intersection={prof_t800['frac_intersection_h_grad']:.3f}, "
              f"jaccard={prof_t800['jaccard_h_grad']:.3f}, "
              f"h_min={uf_t800['h_min']:.2e}, "
              f"DISSOC={result['dissociation_label_T800']}")
        print(f"    T=3000 : h_min={uf_t3000.get('h_min', 0):.2e}, "
              f"long_horizon={result['long_horizon_label']}")

    # =====================================================
    # Pairwise A vs B1/B2/B3
    # =====================================================
    print(f"\n--- Pairwise distances ---")
    pairwise_beta60 = {}
    for fname in ["B1", "B2", "B3"]:
        pair_key = f"A_vs_{fname}"
        pwr = compute_pairwise_full(families_beta60["A"], families_beta60[fname],
                                     checkpoint_steps, checkpoint_times, dt)
        pwr["P5bis_candidate_label"] = label_pairwise(pwr, T800_only=True)
        pairwise_beta60[pair_key] = pwr
        f_t800 = pwr["final_at"].get("t_800", {})
        f_t3000 = pwr["final_at"].get("t_3000", {})
        print(f"  {pair_key} :")
        print(f"    T=800  Dpsi={f_t800.get('Dpsi', 0):.4e}, Dh={f_t800.get('Dh', 0):.4e}, "
              f"Dext={f_t800.get('Dext', 0):.4e}")
        print(f"    T=3000 Dpsi={f_t3000.get('Dpsi', 0):.4e}, Dh={f_t3000.get('Dh', 0):.4e}, "
              f"Dext={f_t3000.get('Dext', 0):.4e}")
        print(f"    AUC_Dh={pwr['AUC_Dh_full_traj']:.4e} (sur trajectoire totale)")
        print(f"    label = {pwr['P5bis_candidate_label']}")

    # =====================================================
    # Reactivation a T=800
    # =====================================================
    print(f"\n--- Reactivation fonctionnelle a T=800 ---")
    n_t800 = int(800.0 / dt)
    n_react = int(200.0 / dt)
    react_beta60 = {}
    for fname in ["A", "B1", "B2", "B3"]:
        psi_pre, h_pre = families_beta60[fname]["states"][n_t800]
        norm_psi_pre = float(np.linalg.norm(psi_pre))
        react_beta60[fname] = {"norm_psi_pre": norm_psi_pre}
        for P_name, P_fn in [("G_standard", P_G_standard), ("A_anneau", P_A_anneau)]:
            s_root, amp_obt, status = calibrate_local(P_fn, psi_pre, target_amp_react)
            if status.startswith("FAILED") or s_root is None:
                react_beta60[fname][P_name] = {
                    "calibration_status": status,
                    "s_root": None, "amp_obt": None,
                    "label": "REACT-CALIBRATION_FAILED",
                }
                print(f"  {fname} x {P_name} : CALIBRATION FAILED")
                continue
            relative_amp_eff = amp_obt / (norm_psi_pre + EPS)
            react = reactivation_test(fname, psi_pre, h_pre, P_fn, P_name,
                                       s_root, amp_obt,
                                       D, beta_lock, gamma_v, h0, dt, n_react)
            react["calibration_status"] = status
            react["target_amp"] = target_amp_react
            react["relative_amp_effective"] = relative_amp_eff
            react_beta60[fname][P_name] = react
            print(f"  {fname} x {P_name} : amp_obt={amp_obt:.4e} "
                  f"(rel={relative_amp_eff:.4f}), "
                  f"Dext_unp={react['metric_final_vs_unperturbed']['Dext']:.4e}, "
                  f"Djacc={react['delta_jaccard_h_grad']:+.4f}, "
                  f"label={react['label']}")

    # =====================================================
    # Controles beta=45 et beta=80, A vs B2 minimum
    # =====================================================
    print(f"\n--- Controles beta=45 / beta=80 (A vs B2) ---")
    beta_controls = {}
    for beta_test in [45.0, 80.0]:
        print(f"\n  beta = {beta_test}")
        dt_test = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_test * psi_max_init + gamma_v))
        # Si dt different, recalculer checkpoints
        checkpoint_steps_b = [int(t / dt_test) for t in checkpoint_times]
        fams = {}
        for fname in ["A", "B2", "B1"]:  # A vs B2 obligatoire, B1 si cout raisonnable
            psi_init = families_makers[fname]()
            h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
            print(f"    Running {fname} a beta={beta_test}...")
            result = run_family_full(fname, psi_init, h_init,
                                       D, beta_test, gamma_v, h0, dt_test,
                                       checkpoint_steps_b)
            n_t800_b = int(800.0 / dt_test)
            result["dissociation_label_T800"] = label_dissociation_per_family(
                result["profiles"][n_t800_b])
            n_t3000_b = int(3000.0 / dt_test)
            if n_t3000_b in result["underflows"]:
                result["long_horizon_label"] = label_long_horizon(result["underflows"][n_t3000_b])
            else:
                result["long_horizon_label"] = "UNKNOWN"
            fams[fname] = result

        # Pairwise A vs B2 obligatoire, B1 optionnel
        pairs = {}
        for fname in ["B1", "B2"]:
            pwr = compute_pairwise_full(fams["A"], fams[fname],
                                         checkpoint_steps_b, checkpoint_times, dt_test)
            pwr["P5bis_candidate_label"] = label_pairwise(pwr, T800_only=True)
            pairs[f"A_vs_{fname}"] = pwr
            f_t800 = pwr["final_at"].get("t_800", {})
            print(f"    A vs {fname} T=800 : Dh={f_t800.get('Dh', 0):.4e}, "
                  f"label={pwr['P5bis_candidate_label']}")

        beta_controls[f"beta_{int(beta_test)}"] = {
            "dt": dt_test,
            "families_t800_profiles": {
                fname: {
                    "frac_h_active": fams[fname]["profiles"][int(800.0 / dt_test)]["frac_h_active"],
                    "frac_grad_active": fams[fname]["profiles"][int(800.0 / dt_test)]["frac_grad_active"],
                    "frac_intersection_h_grad": fams[fname]["profiles"][int(800.0 / dt_test)]["frac_intersection_h_grad"],
                    "jaccard_h_grad": fams[fname]["profiles"][int(800.0 / dt_test)]["jaccard_h_grad"],
                    "dissociation_label": fams[fname]["dissociation_label_T800"],
                    "long_horizon": fams[fname]["long_horizon_label"],
                    "h_min_t800": fams[fname]["underflows"][int(800.0 / dt_test)]["h_min"],
                }
                for fname in ["A", "B2", "B1"]
            },
            "pairs": pairs,
        }

    # =====================================================
    # Verdict global candidat
    # =====================================================
    print(f"\n=== Verdict global candidat ===\n")

    # Bassin/chemin sur beta=60
    print(f"  P5bis-candidate labels beta=60 :")
    for pair_key, pwr in pairwise_beta60.items():
        print(f"    {pair_key} : {pwr['P5bis_candidate_label']}")

    # Dissociation par famille beta=60
    print(f"\n  DISSOC labels beta=60 (par famille a T=800) :")
    for fname in ["A", "B1", "B2", "B3"]:
        print(f"    {fname} : {families_beta60[fname]['dissociation_label_T800']}")

    # Reactivation labels
    print(f"\n  REACT labels beta=60 :")
    for fname in ["A", "B1", "B2", "B3"]:
        for P_name in ["G_standard", "A_anneau"]:
            r = react_beta60[fname].get(P_name, {})
            print(f"    {fname} x {P_name} : {r.get('label', 'MISSING')}")

    # Lien DISSOC global / specifique
    dissoc_labels = [families_beta60[f]["dissociation_label_T800"] for f in ["A", "B1", "B2", "B3"]]
    unique_dissoc = set(dissoc_labels)
    if len(unique_dissoc) == 1:
        dissoc_global = "DISSOC-GENERIC-candidate"
    elif len(unique_dissoc) >= 2:
        dissoc_global = "DISSOC-BASIN-SPECIFIC-candidate"
    else:
        dissoc_global = "DISSOC-INDETERMINATE"
    print(f"\n  Lien §22 candidat : {dissoc_global}")

    # =====================================================
    # Serialisation
    # =====================================================
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()
                     if not isinstance(v, np.ndarray) or v.size < 100000}
        if isinstance(obj, list):
            return [serialize(x) for x in obj]
        return obj

    # Construire l'output JSON sans les `states` (arrays trop volumineux)
    def strip_states(fam_dict):
        clean = {}
        for fname, fres in fam_dict.items():
            clean[fname] = {k: v for k, v in fres.items() if k != "states"}
        return clean

    output = {
        "metadata": {
            "script": "test_6d_p5bis_A.py",
            "spec": "6d_p5bis0_specification.md",
            "params": {
                "gamma_v": gamma_v, "D": D, "h0": h0, "beta_lock_main": beta_lock,
                "dt_main": dt,
            },
            "horizons_time": [800, 3000],
            "checkpoint_times": checkpoint_times,
            "norm_P6_ref": norm_P6_ref,
            "target_amp_react": target_amp_react,
            "guardrails": [
                "no h_proj", "no lambda-B", "no P4 strict",
                "no Delta", "no Gscript", "no MCQ observable",
                "no Ch4 metric", "no V5",
            ],
        },
        "determinism_control": {
            "A_T800": det_A,
            "B2_T200": det_B2,
        },
        "beta_60": {
            "families_summary": serialize(strip_states(families_beta60)),
            "pairwise": serialize(pairwise_beta60),
            "reactivation": serialize(react_beta60),
        },
        "beta_controls": serialize(beta_controls),
        "preliminary_summary": {
            "P5bis_candidates": {pk: pw["P5bis_candidate_label"] for pk, pw in pairwise_beta60.items()},
            "DISSOC_global": dissoc_global,
            "DISSOC_per_family": {f: families_beta60[f]["dissociation_label_T800"] for f in ["A", "B1", "B2", "B3"]},
            "long_horizon_per_family": {f: families_beta60[f]["long_horizon_label"] for f in ["A", "B1", "B2", "B3"]},
            "do_not_overinterpret": True,
        },
    }

    out_path = "/home/claude/mcq_v4/6d_p5bis_A.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegarde : {out_path}")


if __name__ == "__main__":
    main()
