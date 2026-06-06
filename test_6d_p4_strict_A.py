# -*- coding: utf-8 -*-
"""
6d-P4-strict-A : cartographie de reconstructibilite sous filtration observable.

Cf. specification 6d_p4_strict_0_specification.md.

Bibliotheque : 108 etats P5bis-A (4 familles A/B1/B2/B3 x 3 beta {45,60,80}
x 9 checkpoints t=0,10,50,100,200,400,800,1500,3000).

Filtration des observables :
- P_min : globaux pauvres (mass, entropy, var, underflow)
- P_functional : + h-active, gradient-active, intersection, Jaccard
- P_trajectory_intrinsic : + L_psi, L_h, L_ext, max speed, temps caracteristiques
- P_rich : + moments spatiaux, dispersion, roughness

Distances etat : D_state_psi et D_state_h separes.
Test : scatter D_state vs D_obs, regression log-log par axe x niveau.
Top paires ambiguies par score robuste (z-score), pas par seuils figes.
Filtrage log-log : exclusion D_state < 1e-12 ET D_obs < 1e-12.

Pas de ML. Pas de psi/h complet. Pas de h_proj. Pas de AUC relatif a A.
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
EPS_FLOOR = 1e-12


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


def evolve_with_traj_observables(psi, h, D, beta, gamma, h0, dt,
                                    checkpoint_steps, max_step):
    """Evolue et accumule observables trajectorielles intrinseques.
    Retourne dict checkpoint -> {(psi, h), traj_obs(integrate up to this step)}.
    """
    snapshots = {}
    # Observables cumulatives
    L_psi = 0.0
    L_h = 0.0
    L_ext = 0.0
    max_speed_psi = 0.0
    max_speed_h = 0.0
    t_max_speed_psi = 0.0
    t_max_speed_h = 0.0
    first_underflow_time = None  # premier t avec frac(h<1e-6)>0
    first_collapse_time = None   # premier t avec frac(h<1e-12)>0
    auc_intersection = 0.0
    last_intersection = None
    last_t = 0.0

    # checkpoint t=0
    if 0 in checkpoint_steps:
        prof0 = compute_functional_profile_quick(psi, h)
        snapshots[0] = {
            "psi": psi.copy(),
            "h": h.copy(),
            "traj_obs": {
                "L_psi": 0.0, "L_h": 0.0, "L_ext": 0.0,
                "max_speed_psi": 0.0, "max_speed_h": 0.0,
                "t_max_speed_psi": 0.0, "t_max_speed_h": 0.0,
                "first_underflow_time": None, "first_collapse_time": None,
                "auc_frac_intersection": 0.0,
            }
        }
        if prof0["frac_h_lt_1e6"] > 0 and first_underflow_time is None:
            first_underflow_time = 0.0
        if prof0["frac_h_lt_1e12"] > 0 and first_collapse_time is None:
            first_collapse_time = 0.0
        last_intersection = prof0["frac_intersection"]

    # Boucle
    for n in range(1, max_step + 1):
        psi_old = psi
        h_old = h
        psi, h = step(psi, h, D, beta, gamma, h0, dt)

        # Increments
        d_psi = psi - psi_old
        d_h = h - h_old
        norm_dpsi = float(np.linalg.norm(d_psi))
        norm_dh = float(np.linalg.norm(d_h))
        norm_dext = float(np.sqrt(norm_dpsi**2 + norm_dh**2))

        L_psi += norm_dpsi
        L_h += norm_dh
        L_ext += norm_dext

        speed_psi = norm_dpsi / dt
        speed_h = norm_dh / dt
        t_now = n * dt
        if speed_psi > max_speed_psi:
            max_speed_psi = speed_psi
            t_max_speed_psi = t_now
        if speed_h > max_speed_h:
            max_speed_h = speed_h
            t_max_speed_h = t_now

        # Diagnostics underflow incrementaux (uniquement quand on est sur un checkpoint)
        if n in checkpoint_steps:
            prof = compute_functional_profile_quick(psi, h)
            if prof["frac_h_lt_1e6"] > 0 and first_underflow_time is None:
                first_underflow_time = t_now
            if prof["frac_h_lt_1e12"] > 0 and first_collapse_time is None:
                first_collapse_time = t_now

            # AUC intersection : trapz
            if last_intersection is not None:
                dt_segment = t_now - last_t
                auc_intersection += 0.5 * (last_intersection + prof["frac_intersection"]) * dt_segment
            last_intersection = prof["frac_intersection"]
            last_t = t_now

            snapshots[n] = {
                "psi": psi.copy(),
                "h": h.copy(),
                "traj_obs": {
                    "L_psi": float(L_psi),
                    "L_h": float(L_h),
                    "L_ext": float(L_ext),
                    "max_speed_psi": float(max_speed_psi),
                    "max_speed_h": float(max_speed_h),
                    "t_max_speed_psi": float(t_max_speed_psi),
                    "t_max_speed_h": float(t_max_speed_h),
                    "first_underflow_time": first_underflow_time,
                    "first_collapse_time": first_collapse_time,
                    "auc_frac_intersection": float(auc_intersection),
                }
            }

    return snapshots


# ========== Familles ==========
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


# ========== Profil fonctionnel ==========
def compute_functional_profile_quick(psi, h):
    """Diagnostic fonctionnel rapide."""
    n = h.size
    frac_h_lt_1e6 = float(np.sum(h < 1e-6) / n)
    frac_h_lt_1e12 = float(np.sum(h < 1e-12) / n)

    # Faces
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
    h_active = h_face_flat > 1e-6
    grad_max = float(grad_flat.max())
    if grad_max < 1e-30:
        grad_active = np.zeros(n_faces, dtype=bool)
        frac_grad_active = 0.0
    else:
        grad_active = grad_flat > 0.01 * grad_max
        frac_grad_active = float(grad_active.sum() / n_faces)
    intersection = h_active & grad_active
    frac_intersection = float(intersection.sum() / n_faces)

    return {
        "frac_h_lt_1e6": frac_h_lt_1e6,
        "frac_h_lt_1e12": frac_h_lt_1e12,
        "frac_h_active": float(h_active.sum() / n_faces),
        "frac_grad_active": frac_grad_active,
        "frac_intersection": frac_intersection,
    }


# ========== Observables 𝒫 par niveau ==========
def compute_P_min(psi, h):
    """Niveau 0 : etat global pauvre."""
    n = h.size
    return {
        "mass_psi": float(psi.sum()),
        "entropy_psi": float(-np.sum(psi * np.log(psi + EPS))),
        "var_psi": float(np.var(psi.flatten())),
        "mean_psi": float(psi.mean()),
        "max_psi": float(psi.max()),
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "h_median": float(np.median(h)),
        "h_mean": float(h.mean()),
        "frac_h_lt_1e6": float(np.sum(h < 1e-6) / n),
        "frac_h_lt_1e12": float(np.sum(h < 1e-12) / n),
        "frac_h_lt_1e30": float(np.sum(h < 1e-30) / n),
        "frac_h_lt_1e300": float(np.sum(h < 1e-300) / n),
        "frac_h_eq_0": float(np.sum(h == 0) / n),
    }


def compute_P_functional(psi, h):
    """Niveau 1 : ajoute observables fonctionnelles."""
    out = compute_P_min(psi, h)
    n_cells = h.size

    # Faces
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

    h_active = h_face_flat > 1e-6
    grad_max = float(grad_flat.max())
    if grad_max < 1e-30:
        grad_status = 0  # GRAD_DEGENERATE
        frac_grad_active = 0.0
        grad_active = np.zeros(n_faces, dtype=bool)
    else:
        grad_status = 1
        threshold = 0.01 * grad_max
        grad_active = grad_flat > threshold
        frac_grad_active = float(grad_active.sum() / n_faces)
    intersection = h_active & grad_active
    union = h_active | grad_active
    frac_intersection = float(intersection.sum() / n_faces)
    n_union = int(union.sum())
    jaccard = float(intersection.sum()) / n_union if n_union > 0 else 0.0

    out["frac_h_resolution_cells"] = float(np.sum(h > 1e-6) / n_cells)
    out["frac_h_functional_cells"] = float(np.sum(h > 1e-3) / n_cells)
    out["frac_h_active_faces"] = float(h_active.sum() / n_faces)
    out["frac_grad_active_faces"] = frac_grad_active
    out["frac_intersection_h_grad"] = frac_intersection
    out["jaccard_h_grad"] = jaccard
    out["grad_max"] = grad_max
    out["grad_status"] = grad_status
    return out


def compute_P_trajectory_intrinsic(psi, h, traj_obs):
    """Niveau 2 : ajoute trajectoire intrinseque."""
    out = compute_P_functional(psi, h)
    out["L_psi_intrinsic"] = traj_obs["L_psi"]
    out["L_h_intrinsic"] = traj_obs["L_h"]
    out["L_ext_intrinsic"] = traj_obs["L_ext"]
    out["max_speed_psi"] = traj_obs["max_speed_psi"]
    out["max_speed_h"] = traj_obs["max_speed_h"]
    out["t_max_speed_psi"] = traj_obs["t_max_speed_psi"]
    out["t_max_speed_h"] = traj_obs["t_max_speed_h"]
    # Encoder None comme -1
    out["first_underflow_time"] = traj_obs["first_underflow_time"] if traj_obs["first_underflow_time"] is not None else -1.0
    out["first_collapse_time"] = traj_obs["first_collapse_time"] if traj_obs["first_collapse_time"] is not None else -1.0
    out["auc_frac_intersection"] = traj_obs["auc_frac_intersection"]
    return out


def compute_P_rich(psi, h, traj_obs):
    """Niveau 3 : ajoute statistiques de forme spatiales."""
    out = compute_P_trajectory_intrinsic(psi, h, traj_obs)

    # Centre de masse psi
    coords = np.arange(N_AXIS) * DX
    psi_norm = psi.sum()
    if psi_norm > EPS:
        cx_psi = float(np.sum(psi * coords[:, None, None]) / psi_norm)
        cy_psi = float(np.sum(psi * coords[None, :, None]) / psi_norm)
        cz_psi = float(np.sum(psi * coords[None, None, :]) / psi_norm)
    else:
        cx_psi = cy_psi = cz_psi = 0.0

    # Dispersion spatiale psi (variance par axe)
    if psi_norm > EPS:
        dx_var_psi = float(np.sum(psi * (coords[:, None, None] - cx_psi)**2) / psi_norm)
        dy_var_psi = float(np.sum(psi * (coords[None, :, None] - cy_psi)**2) / psi_norm)
        dz_var_psi = float(np.sum(psi * (coords[None, None, :] - cz_psi)**2) / psi_norm)
    else:
        dx_var_psi = dy_var_psi = dz_var_psi = 0.0

    # Centre de masse h
    h_norm = h.sum()
    if h_norm > EPS:
        cx_h = float(np.sum(h * coords[:, None, None]) / h_norm)
        cy_h = float(np.sum(h * coords[None, :, None]) / h_norm)
        cz_h = float(np.sum(h * coords[None, None, :]) / h_norm)
    else:
        cx_h = cy_h = cz_h = 0.0

    # Roughness psi : somme des |grad psi|^2 normalisee
    grad_x = (psi[1:, :, :] - psi[:-1, :, :]) / DX
    grad_y = (psi[:, 1:, :] - psi[:, :-1, :]) / DX
    grad_z = (psi[:, :, 1:] - psi[:, :, :-1]) / DX
    roughness_psi = float(np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_z**2))

    # Roughness h
    gh_x = (h[1:, :, :] - h[:-1, :, :]) / DX
    gh_y = (h[:, 1:, :] - h[:, :-1, :]) / DX
    gh_z = (h[:, :, 1:] - h[:, :, :-1]) / DX
    roughness_h = float(np.sum(gh_x**2) + np.sum(gh_y**2) + np.sum(gh_z**2))

    # Skewness psi (moment 3 normalise)
    if psi_norm > EPS and dx_var_psi > EPS:
        skew_x_psi = float(np.sum(psi * (coords[:, None, None] - cx_psi)**3) /
                            (psi_norm * dx_var_psi**1.5))
    else:
        skew_x_psi = 0.0

    out["cx_psi"] = cx_psi
    out["cy_psi"] = cy_psi
    out["cz_psi"] = cz_psi
    out["dx_var_psi"] = dx_var_psi
    out["dy_var_psi"] = dy_var_psi
    out["dz_var_psi"] = dz_var_psi
    out["cx_h"] = cx_h
    out["cy_h"] = cy_h
    out["cz_h"] = cz_h
    out["roughness_psi"] = roughness_psi
    out["roughness_h"] = roughness_h
    out["skew_x_psi"] = skew_x_psi
    # Fractions par seuil supplementaires
    out["frac_h_gt_0p1"] = float(np.sum(h > 0.1) / h.size)
    out["frac_h_gt_0p5"] = float(np.sum(h > 0.5) / h.size)
    return out


# ========== Pipeline ==========
def main():
    gamma_v, D, h0 = 1.0, 0.1, 1.0
    psi_max_init = float(make_psi_A().max())

    families_makers = {
        "A": make_psi_A,
        "B1": make_psi_B1,
        "B2": make_psi_B2,
        "B3": make_psi_B3,
    }
    betas = [45.0, 60.0, 80.0]
    checkpoint_times = [0.0, 10.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]

    print(f"=== 6d-P4-strict-A : cartographie de reconstructibilite ===\n")

    # =============================
    # 1. Recalcul des etats
    # =============================
    print(f"--- Recalcul de la bibliotheque (4 x 3 x 9 = 108 etats) ---\n")
    library = {}  # (family, beta, checkpoint_idx) -> {"psi", "h", "traj_obs", "t_actual"}
    states_metadata = []

    for beta in betas:
        dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max_init + gamma_v))
        checkpoint_steps = [int(t / dt) for t in checkpoint_times]
        max_step = max(checkpoint_steps)
        print(f"  beta={beta} : dt={dt:.5f}, max_step={max_step}")

        for fname, fmaker in families_makers.items():
            psi_init = fmaker()
            h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
            snapshots = evolve_with_traj_observables(
                psi_init, h_init, D, beta, gamma_v, h0, dt,
                checkpoint_steps, max_step)

            for ck_idx, (t_req, n_step) in enumerate(zip(checkpoint_times, checkpoint_steps)):
                key = (fname, beta, ck_idx)
                if n_step in snapshots:
                    snap = snapshots[n_step]
                    library[key] = {
                        "psi": snap["psi"],
                        "h": snap["h"],
                        "traj_obs": snap["traj_obs"],
                        "t_actual": n_step * dt,
                    }
                    states_metadata.append({
                        "key": f"{fname}_beta{int(beta)}_ck{ck_idx}",
                        "family": fname,
                        "beta": beta,
                        "checkpoint_idx": ck_idx,
                        "t_requested": t_req,
                        "step": n_step,
                        "t_actual": n_step * dt,
                    })

    n_states = len(library)
    print(f"  Total etats : {n_states}\n")

    # =============================
    # 2. Controle de reproduction P5bis-A
    # =============================
    print(f"--- Controle de reproduction P5bis-A ---\n")
    # Recuperer A et B2 a beta=60, ck_idx=6 (t=800)
    A_60_t800 = library[("A", 60.0, 6)]
    B1_60_t800 = library[("B1", 60.0, 6)]
    B2_60_t800 = library[("B2", 60.0, 6)]
    B2_60_t3000 = library[("B2", 60.0, 8)]

    # Dh A vs B2 a T=800
    Dh_AB2 = float(np.linalg.norm(B2_60_t800["h"] - A_60_t800["h"]) /
                     (np.linalg.norm(A_60_t800["h"]) + EPS))
    Dh_AB1 = float(np.linalg.norm(B1_60_t800["h"] - A_60_t800["h"]) /
                     (np.linalg.norm(A_60_t800["h"]) + EPS))
    prof_B2 = compute_functional_profile_quick(B2_60_t800["psi"], B2_60_t800["h"])
    prof_B2_3000 = compute_functional_profile_quick(B2_60_t3000["psi"], B2_60_t3000["h"])

    reproduction = {
        "Dh_A_vs_B2_t800_beta60": Dh_AB2,
        "Dh_A_vs_B1_t800_beta60": Dh_AB1,
        "B2_frac_h_active_t800_beta60": prof_B2["frac_h_active"],
        "B2_frac_grad_active_t800_beta60": prof_B2["frac_grad_active"],
        "B2_frac_intersection_t800_beta60": prof_B2["frac_intersection"],
        "B2_h_min_t3000_beta60": float(B2_60_t3000["h"].min()),
    }

    # Verifier
    tol_Dh_B2 = abs(Dh_AB2 - 0.29655) < 1e-3
    tol_Dh_B1 = abs(Dh_AB1 - 4.0028e-8) < 1e-7
    tol_B2_active = abs(prof_B2["frac_h_active"] - 0.880) < 0.01
    tol_B2_grad = abs(prof_B2["frac_grad_active"] - 0.120) < 0.01
    tol_B2_inter = prof_B2["frac_intersection"] < 0.01
    tol_B2_uf = float(B2_60_t3000["h"].min()) < 1e-100

    print(f"  Dh A vs B2 T=800 (beta=60) = {Dh_AB2:.6e} [attendu ~0.29655] : {'OK' if tol_Dh_B2 else 'FAIL'}")
    print(f"  Dh A vs B1 T=800 (beta=60) = {Dh_AB1:.6e} [attendu ~4e-8] : {'OK' if tol_Dh_B1 else 'FAIL'}")
    print(f"  B2 frac_h_active T=800 = {prof_B2['frac_h_active']:.4f} [attendu ~0.880] : {'OK' if tol_B2_active else 'FAIL'}")
    print(f"  B2 frac_grad_active T=800 = {prof_B2['frac_grad_active']:.4f} [attendu ~0.120] : {'OK' if tol_B2_grad else 'FAIL'}")
    print(f"  B2 frac_intersection T=800 = {prof_B2['frac_intersection']:.4f} [attendu = 0] : {'OK' if tol_B2_inter else 'FAIL'}")
    print(f"  B2 h_min T=3000 = {float(B2_60_t3000['h'].min()):.2e} [underflow attendu] : {'OK' if tol_B2_uf else 'FAIL'}")

    all_ok = tol_Dh_B2 and tol_Dh_B1 and tol_B2_active and tol_B2_grad and tol_B2_inter and tol_B2_uf
    reproduction["verdict_reproduction"] = "REPRODUCTION_PASS" if all_ok else "REPRODUCTION_FAIL"
    print(f"\n  Verdict reproduction : {reproduction['verdict_reproduction']}")
    if not all_ok:
        print(f"\n  STOP : audit moteur necessaire.")
        return

    # =============================
    # 3. Calcul des observables par niveau
    # =============================
    print(f"\n--- Calcul des observables par niveau de filtration ---\n")
    keys_ordered = sorted(library.keys(), key=lambda k: (k[1], k[0], k[2]))
    n_states = len(keys_ordered)

    # Pour chaque etat, calculer P_min, P_functional, P_trajectory_intrinsic, P_rich
    P_data = {
        "P_min": {},
        "P_functional": {},
        "P_trajectory_intrinsic": {},
        "P_rich": {},
    }
    for k in keys_ordered:
        st = library[k]
        P_data["P_min"][k] = compute_P_min(st["psi"], st["h"])
        P_data["P_functional"][k] = compute_P_functional(st["psi"], st["h"])
        P_data["P_trajectory_intrinsic"][k] = compute_P_trajectory_intrinsic(st["psi"], st["h"], st["traj_obs"])
        P_data["P_rich"][k] = compute_P_rich(st["psi"], st["h"], st["traj_obs"])

    # Verifier nombre d'observables par niveau
    feature_names = {level: list(P_data[level][keys_ordered[0]].keys()) for level in P_data}
    n_features = {level: len(feature_names[level]) for level in P_data}
    for level in ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]:
        print(f"  {level} : {n_features[level]} features")

    # =============================
    # 4. Standardisation par niveau
    # =============================
    print(f"\n--- Standardisation des observables ---\n")
    standardized = {level: {} for level in P_data}
    excluded_features = {level: [] for level in P_data}
    for level, P_at_level in P_data.items():
        # Matrice n_states x n_features
        features = feature_names[level]
        M = np.array([[P_at_level[k][f] for f in features] for k in keys_ordered])
        mu = M.mean(axis=0)
        sigma = M.std(axis=0)
        # Identifier composantes constantes
        keep = sigma > 1e-15
        if not all(keep):
            excluded_features[level] = [features[i] for i in range(len(features)) if not keep[i]]
        # Standardiser
        mu_kept = mu[keep]
        sigma_kept = sigma[keep]
        M_kept = M[:, keep]
        M_std = (M_kept - mu_kept) / sigma_kept
        for i, k in enumerate(keys_ordered):
            standardized[level][k] = M_std[i]
        print(f"  {level} : {keep.sum()}/{len(features)} features retenues, "
              f"{len(excluded_features[level])} exclues")

    # =============================
    # 5. Distances D_state separees (psi, h) et D_obs par niveau
    # =============================
    print(f"\n--- Calcul des distances pairwise (5778 paires) ---\n")
    pairs = []
    keys_list = keys_ordered
    n_pairs_total = n_states * (n_states - 1) // 2

    # Vecteur des psi et h flatten pour calcul rapide
    psi_norm = {k: float(np.linalg.norm(library[k]["psi"])) for k in keys_list}
    h_norm = {k: float(np.linalg.norm(library[k]["h"])) for k in keys_list}

    # Liste underflow flag par etat
    underflow_dominated = {}
    for k in keys_list:
        h = library[k]["h"]
        # underflow_dominated si frac(h<1e-300) > 0.05 ou n_eq_0 > 0
        n_under = int(np.sum(h < 1e-300))
        n_eq_0 = int(np.sum(h == 0))
        if n_under > 0.05 * h.size or n_eq_0 > 0:
            underflow_dominated[k] = True
        else:
            underflow_dominated[k] = False

    # Calcul des paires
    print(f"  Calcul en cours...")
    for i in range(n_states):
        k_i = keys_list[i]
        psi_i = library[k_i]["psi"]
        h_i = library[k_i]["h"]
        for j in range(i+1, n_states):
            k_j = keys_list[j]
            psi_j = library[k_j]["psi"]
            h_j = library[k_j]["h"]
            D_psi = float(np.linalg.norm(psi_j - psi_i) / (psi_norm[k_i] + EPS))
            D_h = float(np.linalg.norm(h_j - h_i) / (h_norm[k_i] + EPS))
            # D_obs par niveau
            D_obs_lvl = {}
            for level in P_data:
                v_i = standardized[level][k_i]
                v_j = standardized[level][k_j]
                D_obs_lvl[level] = float(np.linalg.norm(v_j - v_i))
            pairs.append({
                "i": i, "j": j,
                "key_i": f"{k_i[0]}_beta{int(k_i[1])}_ck{k_i[2]}",
                "key_j": f"{k_j[0]}_beta{int(k_j[1])}_ck{k_j[2]}",
                "D_state_psi": D_psi,
                "D_state_h": D_h,
                "D_obs": D_obs_lvl,
                "involves_underflow": underflow_dominated[k_i] or underflow_dominated[k_j],
                "involves_B3": (k_i[0] == "B3") or (k_j[0] == "B3"),
            })
    print(f"  {len(pairs)} paires calculees")

    # =============================
    # 6. Regressions log-log par axe x niveau
    # =============================
    print(f"\n--- Regressions log-log ---\n")
    regression_results = {"psi_axis": {}, "h_axis": {}}

    for axis, D_state_key in [("psi_axis", "D_state_psi"), ("h_axis", "D_state_h")]:
        for level in ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]:
            # Filtrer
            xs_full = np.array([p[D_state_key] for p in pairs])
            ys_full = np.array([p["D_obs"][level] for p in pairs])
            mask_state = xs_full > EPS_FLOOR
            mask_obs = ys_full > EPS_FLOOR
            mask = mask_state & mask_obs
            n_used = int(mask.sum())
            n_filtered_state = int((~mask_state).sum())
            n_filtered_obs = int((~mask_obs).sum() - (~mask_state & ~mask_obs).sum())

            if n_used < 10:
                regression_results[axis][level] = {
                    "alpha": None, "c": None, "R2": None,
                    "n_used": n_used, "n_filtered_state": n_filtered_state,
                    "n_filtered_obs": n_filtered_obs,
                }
                continue

            xs = xs_full[mask]
            ys = ys_full[mask]
            log_xs = np.log(xs)
            log_ys = np.log(ys)
            # regression simple
            X = np.column_stack([log_xs, np.ones_like(log_xs)])
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, log_ys, rcond=None)
            alpha = float(coeffs[0])
            c = float(coeffs[1])
            ss_res = float(np.sum((log_ys - (alpha * log_xs + c))**2))
            ss_tot = float(np.sum((log_ys - log_ys.mean())**2))
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Aussi sans underflow
            mask_no_uf = mask & np.array([not p["involves_underflow"] for p in pairs])
            n_no_uf = int(mask_no_uf.sum())
            if n_no_uf >= 10:
                xs_nuf = xs_full[mask_no_uf]
                ys_nuf = ys_full[mask_no_uf]
                log_x_nuf = np.log(xs_nuf)
                log_y_nuf = np.log(ys_nuf)
                X_nuf = np.column_stack([log_x_nuf, np.ones_like(log_x_nuf)])
                coef_nuf, _, _, _ = np.linalg.lstsq(X_nuf, log_y_nuf, rcond=None)
                alpha_nuf = float(coef_nuf[0])
                c_nuf = float(coef_nuf[1])
                ss_res_nuf = float(np.sum((log_y_nuf - (alpha_nuf * log_x_nuf + c_nuf))**2))
                ss_tot_nuf = float(np.sum((log_y_nuf - log_y_nuf.mean())**2))
                R2_nuf = 1.0 - ss_res_nuf / ss_tot_nuf if ss_tot_nuf > 0 else 0.0
            else:
                alpha_nuf, c_nuf, R2_nuf = None, None, None

            regression_results[axis][level] = {
                "alpha": alpha,
                "c": c,
                "R2": R2,
                "n_used": n_used,
                "n_filtered_state": n_filtered_state,
                "n_filtered_obs": n_filtered_obs,
                "alpha_no_underflow": alpha_nuf,
                "R2_no_underflow": R2_nuf,
                "n_used_no_underflow": n_no_uf,
            }
            print(f"  {axis} x {level} : alpha={alpha:+.4f}, R2={R2:.3f}, n={n_used} "
                  f"(no_uf: alpha={alpha_nuf}, R2={R2_nuf})")

    # =============================
    # 7. Top paires ambiguies (z-score)
    # =============================
    print(f"\n--- Top paires ambiguies (z-score, P_functional) ---\n")
    # Pour le diagnostic, utiliser P_functional comme reference
    ref_level = "P_functional"

    # Calculer z-scores pour D_state_psi, D_state_h, D_obs au ref_level
    xs_psi = np.array([p["D_state_psi"] for p in pairs])
    xs_h = np.array([p["D_state_h"] for p in pairs])
    ys = np.array([p["D_obs"][ref_level] for p in pairs])

    # Eviter les paires sous floor pour z-scores
    mask_z = (xs_psi > EPS_FLOOR) & (xs_h > EPS_FLOOR) & (ys > EPS_FLOOR)
    log_xs_psi = np.where(mask_z, np.log(np.clip(xs_psi, EPS_FLOOR, None)), 0)
    log_xs_h = np.where(mask_z, np.log(np.clip(xs_h, EPS_FLOOR, None)), 0)
    log_ys = np.where(mask_z, np.log(np.clip(ys, EPS_FLOOR, None)), 0)
    mu_x_psi = log_xs_psi[mask_z].mean(); sd_x_psi = log_xs_psi[mask_z].std()
    mu_x_h = log_xs_h[mask_z].mean(); sd_x_h = log_xs_h[mask_z].std()
    mu_y = log_ys[mask_z].mean(); sd_y = log_ys[mask_z].std()

    z_x_psi = np.where(mask_z, (log_xs_psi - mu_x_psi) / (sd_x_psi + EPS), 0)
    z_x_h = np.where(mask_z, (log_xs_h - mu_x_h) / (sd_x_h + EPS), 0)
    z_y = np.where(mask_z, (log_ys - mu_y) / (sd_y + EPS), 0)

    # OBS-CLOSE / STATE-FAR (psi) : z_x_psi eleve - z_y bas
    score_ocsf_psi = np.where(mask_z, z_x_psi - z_y, -np.inf)
    score_ocsf_h = np.where(mask_z, z_x_h - z_y, -np.inf)
    score_scof_psi = np.where(mask_z, z_y - z_x_psi, -np.inf)
    score_scof_h = np.where(mask_z, z_y - z_x_h, -np.inf)

    def top_pairs(scores, n=20):
        idx_sorted = np.argsort(scores)[::-1][:n]
        out = []
        for idx in idx_sorted:
            if scores[idx] == -np.inf:
                break
            p = pairs[idx]
            out.append({
                "key_i": p["key_i"], "key_j": p["key_j"],
                "D_state_psi": p["D_state_psi"],
                "D_state_h": p["D_state_h"],
                "D_obs_P_functional": p["D_obs"]["P_functional"],
                "D_obs_P_min": p["D_obs"]["P_min"],
                "D_obs_P_traj": p["D_obs"]["P_trajectory_intrinsic"],
                "D_obs_P_rich": p["D_obs"]["P_rich"],
                "involves_underflow": p["involves_underflow"],
                "involves_B3": p["involves_B3"],
                "score": float(scores[idx]),
            })
        return out

    top_OCSF_psi = top_pairs(score_ocsf_psi, n=20)
    top_OCSF_h = top_pairs(score_ocsf_h, n=20)
    top_SCOF_psi = top_pairs(score_scof_psi, n=20)
    top_SCOF_h = top_pairs(score_scof_h, n=20)

    print(f"  Top 5 OBS-CLOSE / STATE-FAR (psi axis) :")
    for p in top_OCSF_psi[:5]:
        print(f"    {p['key_i']} vs {p['key_j']} : "
              f"D_psi={p['D_state_psi']:.4e}, D_obs={p['D_obs_P_functional']:.4e}, "
              f"underflow={p['involves_underflow']}")
    print(f"  Top 5 OBS-CLOSE / STATE-FAR (h axis) :")
    for p in top_OCSF_h[:5]:
        print(f"    {p['key_i']} vs {p['key_j']} : "
              f"D_h={p['D_state_h']:.4e}, D_obs={p['D_obs_P_functional']:.4e}, "
              f"underflow={p['involves_underflow']}")

    # =============================
    # 8. Verdict candidat
    # =============================
    print(f"\n--- Verdict candidat ---\n")
    # Comparer alpha entre niveaux
    alpha_progression_psi = {
        level: regression_results["psi_axis"][level]["alpha"]
        for level in ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]
    }
    alpha_progression_h = {
        level: regression_results["h_axis"][level]["alpha"]
        for level in ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]
    }
    R2_progression_psi = {
        level: regression_results["psi_axis"][level]["R2"]
        for level in ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]
    }
    R2_progression_h = {
        level: regression_results["h_axis"][level]["R2"]
        for level in ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]
    }
    alpha_nuf_psi = {
        level: regression_results["psi_axis"][level].get("alpha_no_underflow")
        for level in ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]
    }
    alpha_nuf_h = {
        level: regression_results["h_axis"][level].get("alpha_no_underflow")
        for level in ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]
    }

    print(f"  Alpha progression psi axis :")
    for level, alpha in alpha_progression_psi.items():
        nuf = alpha_nuf_psi[level]
        nuf_str = f" (no_uf={nuf:+.4f})" if nuf is not None else ""
        print(f"    {level} : {alpha:+.4f}{nuf_str}")
    print(f"  Alpha progression h axis :")
    for level, alpha in alpha_progression_h.items():
        nuf = alpha_nuf_h[level]
        nuf_str = f" (no_uf={nuf:+.4f})" if nuf is not None else ""
        print(f"    {level} : {alpha:+.4f}{nuf_str}")

    # Verdict heuristique
    # PASS-DISCRIMINANT si alpha ~ 1 et R2 > 0.7 des P_min
    # FILTRATION-SENSITIVE si alpha croit avec niveau, R2 ameliore
    # OBS-SATURATION si alpha << 1 sur tous niveaux
    # TRAJECTORY-REQUIRED si alpha bond a P_trajectory_intrinsic
    # UNDERFLOW-CONFOUNDED si gros ecart entre alpha et alpha_no_uf
    # ANISOTROPIC si forte difference psi/h

    a_min_psi = alpha_progression_psi["P_min"] or 0.0
    a_min_h = alpha_progression_h["P_min"] or 0.0
    a_rich_psi = alpha_progression_psi["P_rich"] or 0.0
    a_rich_h = alpha_progression_h["P_rich"] or 0.0
    a_traj_psi = alpha_progression_psi["P_trajectory_intrinsic"] or 0.0
    a_traj_h = alpha_progression_h["P_trajectory_intrinsic"] or 0.0
    a_func_psi = alpha_progression_psi["P_functional"] or 0.0
    a_func_h = alpha_progression_h["P_functional"] or 0.0

    # Anisotropie
    anisotropy = abs(a_rich_psi - a_rich_h) > 0.3
    anisotropy_label = "P4-ANISOTROPIC-OBSERVABILITY-candidate" if anisotropy else None

    # Verdict principal sur l'axe le plus discriminant
    max_alpha_psi = max(a_min_psi, a_func_psi, a_traj_psi, a_rich_psi)
    max_alpha_h = max(a_min_h, a_func_h, a_traj_h, a_rich_h)

    if max_alpha_psi > 0.7 and max_alpha_h > 0.7:
        verdict = "P4-DISCRIMINANT-candidate"
    elif (a_traj_psi - a_func_psi > 0.2) or (a_traj_h - a_func_h > 0.2):
        verdict = "P4-TRAJECTORY-REQUIRED-candidate"
    elif (a_func_psi - a_min_psi > 0.2) or (a_func_h - a_min_h > 0.2):
        verdict = "P4-FILTRATION-SENSITIVE-candidate"
    elif max_alpha_psi < 0.3 and max_alpha_h < 0.3:
        verdict = "P4-OBS-SATURATION-candidate"
    else:
        verdict = "P4-RELATIVE-ONLY-candidate"

    # Verifier underflow confounding
    if alpha_nuf_psi["P_functional"] is not None:
        delta_uf = abs((alpha_nuf_psi["P_functional"] or 0.0) - a_func_psi) + \
                    abs((alpha_nuf_h["P_functional"] or 0.0) - a_func_h)
        if delta_uf > 0.4:
            verdict = "P4-UNDERFLOW-CONFOUNDED-candidate"

    print(f"\n  Verdict principal : {verdict}")
    if anisotropy_label:
        print(f"  Verdict additionnel : {anisotropy_label}")

    # =============================
    # 9. Sauvegarder JSON
    # =============================
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(x) for x in obj]
        if isinstance(obj, tuple):
            return str(obj)
        return obj

    output = {
        "metadata": {
            "script": "test_6d_p4_strict_A.py",
            "spec": "6d_p4_strict_0_specification.md",
            "n_states": n_states,
            "n_pairs": len(pairs),
            "betas": betas,
            "checkpoint_times": checkpoint_times,
            "guardrails": [
                "no ML", "no full psi/h in P",
                "no h_proj", "no Delta", "no MCQ observable",
                "no AUC relative to A in P",
            ],
        },
        "reproduction_control": reproduction,
        "states_metadata": states_metadata,
        "feature_names": serialize(feature_names),
        "excluded_features": serialize(excluded_features),
        "n_features_per_level": n_features,
        "regression_results": serialize(regression_results),
        "alpha_progression": {
            "psi_axis": serialize(alpha_progression_psi),
            "h_axis": serialize(alpha_progression_h),
            "psi_axis_no_underflow": serialize(alpha_nuf_psi),
            "h_axis_no_underflow": serialize(alpha_nuf_h),
        },
        "R2_progression": {
            "psi_axis": serialize(R2_progression_psi),
            "h_axis": serialize(R2_progression_h),
        },
        "top_pairs": {
            "OBS_CLOSE_STATE_FAR_psi_axis": serialize(top_OCSF_psi),
            "OBS_CLOSE_STATE_FAR_h_axis": serialize(top_OCSF_h),
            "STATE_CLOSE_OBS_FAR_psi_axis": serialize(top_SCOF_psi),
            "STATE_CLOSE_OBS_FAR_h_axis": serialize(top_SCOF_h),
        },
        "verdict_candidate": verdict,
        "anisotropy_label": anisotropy_label,
        "anisotropy_max_diff": float(abs(a_rich_psi - a_rich_h)),
    }

    out_path = "/home/claude/mcq_v4/6d_p4_strict_A.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegarde : {out_path}")


if __name__ == "__main__":
    main()
