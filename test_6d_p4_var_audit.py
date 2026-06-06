# -*- coding: utf-8 -*-
"""
6d-P4-VAR : micro-audit de variance interne du nuage P4-A.

Objet : tester si le "nuage disperse" de P4-A est amorphe ou structure
par des variables deja disponibles (famille, beta, checkpoint, underflow,
B2, dissociation, intra/inter, B3).

Methode :
- recalculer les 108 etats P5bis-A (memes makers, memes parametres)
- recalculer 5778 paires avec D_state_psi/h et D_obs par niveau
- calculer residus log-log r = log(D_obs) - (alpha log(D_state) + c)
  par axe x niveau
- pour chaque classification, comparer Var(r), median(|r|), IQR(|r|)
- bootstrap/permutation 1000 fois
- rapporter taille d'effet : variance_ratio, delta_median_abs_r
- verdict compose : main + modifiers

Pas de nouveau moteur. Pas de nouvelle famille. Pas de ML.
Pas de Delta, Gscript, MCQ, Ch4.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import json

from mcq_v4.factorial_6d import N_AXIS, DX, cfl_dt_max
from mcq_v4.factorial_6d.engine import (
    compute_diffusion_flux, compute_divergence, harmonic_mean,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero

EPS = 1e-30
EPS_FLOOR = 1e-12
N_PERMUTATIONS = 1000
RNG_SEED = 42


# ========== Moteur ==========
def rhs(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi = compute_divergence(Jx, Jy, Jz)
    dh = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi, dh


def step(psi, h, D, beta, gamma, h0, dt):
    dpsi, dh = rhs(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi, h + dt * dh


def evolve_with_traj(psi, h, D, beta, gamma, h0, dt, checkpoint_steps, max_step):
    snapshots = {}
    L_psi = L_h = L_ext = 0.0
    max_speed_psi = max_speed_h = 0.0

    if 0 in checkpoint_steps:
        snapshots[0] = {
            "psi": psi.copy(), "h": h.copy(),
            "L_psi": 0.0, "L_h": 0.0, "L_ext": 0.0,
            "max_speed_psi": 0.0, "max_speed_h": 0.0,
        }

    for n in range(1, max_step + 1):
        psi_old, h_old = psi, h
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        d_psi = psi - psi_old; d_h = h - h_old
        norm_dpsi = float(np.linalg.norm(d_psi))
        norm_dh = float(np.linalg.norm(d_h))
        L_psi += norm_dpsi; L_h += norm_dh
        L_ext += float(np.sqrt(norm_dpsi**2 + norm_dh**2))
        max_speed_psi = max(max_speed_psi, norm_dpsi / dt)
        max_speed_h = max(max_speed_h, norm_dh / dt)
        if n in checkpoint_steps:
            snapshots[n] = {
                "psi": psi.copy(), "h": h.copy(),
                "L_psi": float(L_psi), "L_h": float(L_h), "L_ext": float(L_ext),
                "max_speed_psi": float(max_speed_psi),
                "max_speed_h": float(max_speed_h),
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


# ========== Observables par niveau ==========
def compute_state_features(psi, h, traj):
    """Toutes les features intrinseques pour les 4 niveaux + classification."""
    n_cells = h.size
    # P_min
    P_min = {
        "mass_psi": float(psi.sum()),
        "entropy_psi": float(-np.sum(psi * np.log(psi + EPS))),
        "var_psi": float(np.var(psi.flatten())),
        "mean_psi": float(psi.mean()),
        "max_psi": float(psi.max()),
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "h_median": float(np.median(h)),
        "h_mean": float(h.mean()),
        "frac_h_lt_1e6": float(np.sum(h < 1e-6) / n_cells),
        "frac_h_lt_1e12": float(np.sum(h < 1e-12) / n_cells),
        "frac_h_lt_1e30": float(np.sum(h < 1e-30) / n_cells),
        "frac_h_lt_1e300": float(np.sum(h < 1e-300) / n_cells),
        "frac_h_eq_0": float(np.sum(h == 0) / n_cells),
    }

    # Faces
    h_face_x = harmonic_mean(h[:-1, :, :], h[1:, :, :])
    h_face_y = harmonic_mean(h[:, :-1, :], h[:, 1:, :])
    h_face_z = harmonic_mean(h[:, :, :-1], h[:, :, 1:])
    h_face_flat = np.concatenate([h_face_x.flatten(), h_face_y.flatten(), h_face_z.flatten()])
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
        grad_status = 0
        grad_active = np.zeros(n_faces, dtype=bool)
        frac_grad_active = 0.0
    else:
        grad_status = 1
        grad_active = grad_flat > 0.01 * grad_max
        frac_grad_active = float(grad_active.sum() / n_faces)
    intersection_mask = h_active & grad_active
    union_mask = h_active | grad_active
    frac_intersection = float(intersection_mask.sum() / n_faces)
    n_union = int(union_mask.sum())
    jaccard = float(intersection_mask.sum()) / n_union if n_union > 0 else 0.0

    P_functional = dict(P_min)
    P_functional["frac_h_resolution_cells"] = float(np.sum(h > 1e-6) / n_cells)
    P_functional["frac_h_functional_cells"] = float(np.sum(h > 1e-3) / n_cells)
    P_functional["frac_h_active_faces"] = float(h_active.sum() / n_faces)
    P_functional["frac_grad_active_faces"] = frac_grad_active
    P_functional["frac_intersection_h_grad"] = frac_intersection
    P_functional["jaccard_h_grad"] = jaccard
    P_functional["grad_max"] = grad_max
    P_functional["grad_status"] = float(grad_status)

    # roughness
    roughness_psi = float(np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_z**2))
    gh_x = (h[1:, :, :] - h[:-1, :, :]) / DX
    gh_y = (h[:, 1:, :] - h[:, :-1, :]) / DX
    gh_z = (h[:, :, 1:] - h[:, :, :-1]) / DX
    roughness_h = float(np.sum(gh_x**2) + np.sum(gh_y**2) + np.sum(gh_z**2))

    P_trajectory = dict(P_functional)
    P_trajectory["L_psi"] = traj["L_psi"]
    P_trajectory["L_h"] = traj["L_h"]
    P_trajectory["L_ext"] = traj["L_ext"]
    P_trajectory["max_speed_psi"] = traj["max_speed_psi"]
    P_trajectory["max_speed_h"] = traj["max_speed_h"]

    # P_rich : ajoute moments + roughness
    coords = np.arange(N_AXIS) * DX
    psi_norm = psi.sum()
    cx_psi = float(np.sum(psi * coords[:, None, None]) / psi_norm) if psi_norm > EPS else 0.0
    cy_psi = float(np.sum(psi * coords[None, :, None]) / psi_norm) if psi_norm > EPS else 0.0
    cz_psi = float(np.sum(psi * coords[None, None, :]) / psi_norm) if psi_norm > EPS else 0.0
    dx_var_psi = float(np.sum(psi * (coords[:, None, None] - cx_psi)**2) / psi_norm) if psi_norm > EPS else 0.0
    dy_var_psi = float(np.sum(psi * (coords[None, :, None] - cy_psi)**2) / psi_norm) if psi_norm > EPS else 0.0
    dz_var_psi = float(np.sum(psi * (coords[None, None, :] - cz_psi)**2) / psi_norm) if psi_norm > EPS else 0.0

    P_rich = dict(P_trajectory)
    P_rich["cx_psi"] = cx_psi
    P_rich["cy_psi"] = cy_psi
    P_rich["cz_psi"] = cz_psi
    P_rich["dx_var_psi"] = dx_var_psi
    P_rich["dy_var_psi"] = dy_var_psi
    P_rich["dz_var_psi"] = dz_var_psi
    P_rich["roughness_psi"] = roughness_psi
    P_rich["roughness_h"] = roughness_h
    P_rich["frac_h_gt_0p1"] = float(np.sum(h > 0.1) / n_cells)
    P_rich["frac_h_gt_0p5"] = float(np.sum(h > 0.5) / n_cells)

    # Classification de l'etat
    is_dissociated_state = (frac_intersection < 0.01 and
                            frac_grad_active > 0.05 and
                            grad_status == 1)
    is_underflow_dominated = (P_min["frac_h_lt_1e300"] > 0.05 or
                              P_min["frac_h_eq_0"] > 0)

    classifications = {
        "is_dissociated": is_dissociated_state,
        "is_underflow": is_underflow_dominated,
        "frac_h_active": float(h_active.sum() / n_faces),
        "frac_grad_active": frac_grad_active,
        "frac_intersection": frac_intersection,
        "jaccard_h_grad": jaccard,
        "roughness_h": roughness_h,
        "h_min": float(h.min()),
    }

    return {
        "P_min": P_min,
        "P_functional": P_functional,
        "P_trajectory_intrinsic": P_trajectory,
        "P_rich": P_rich,
        "classifications": classifications,
    }


def standardize(features_dict_list, feature_names):
    """Standardise les vecteurs de features sur la bibliotheque."""
    M = np.array([[f[fn] for fn in feature_names] for f in features_dict_list])
    mu = M.mean(axis=0)
    sigma = M.std(axis=0)
    keep = sigma > 1e-15
    M_std = (M[:, keep] - mu[keep]) / sigma[keep]
    excluded = [feature_names[i] for i in range(len(feature_names)) if not keep[i]]
    return M_std, excluded


# ========== Pipeline ==========
def main():
    gamma_v, D, h0 = 1.0, 0.1, 1.0
    psi_max_init = float(make_psi_A().max())

    families_makers = {"A": make_psi_A, "B1": make_psi_B1, "B2": make_psi_B2, "B3": make_psi_B3}
    betas = [45.0, 60.0, 80.0]
    checkpoint_times = [0.0, 10.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]

    print(f"=== 6d-P4-VAR : audit de variance interne du nuage P4-A ===\n")

    # === Recalcul des 108 etats ===
    print(f"--- Recalcul de la bibliotheque (108 etats) ---")
    library = []  # liste de dicts ordonnee
    for beta in betas:
        dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max_init + gamma_v))
        checkpoint_steps = [int(t / dt) for t in checkpoint_times]
        max_step = max(checkpoint_steps)
        print(f"  beta={beta} : dt={dt:.5f}")
        for fname, fmaker in families_makers.items():
            psi_init = fmaker()
            h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
            snapshots = evolve_with_traj(psi_init, h_init, D, beta, gamma_v, h0, dt,
                                          checkpoint_steps, max_step)
            for ck_idx, n_step in enumerate(checkpoint_steps):
                if n_step in snapshots:
                    snap = snapshots[n_step]
                    feats = compute_state_features(snap["psi"], snap["h"], snap)
                    library.append({
                        "key": f"{fname}_beta{int(beta)}_ck{ck_idx}",
                        "family": fname,
                        "beta": beta,
                        "checkpoint_idx": ck_idx,
                        "t_actual": n_step * dt,
                        "psi": snap["psi"],
                        "h": snap["h"],
                        "features": feats,
                    })
    n_states = len(library)
    print(f"  Total etats : {n_states}\n")

    # === Standardisation par niveau ===
    print(f"--- Standardisation des observables ---")
    levels = ["P_min", "P_functional", "P_trajectory_intrinsic", "P_rich"]
    standardized = {}
    for level in levels:
        feature_names = list(library[0]["features"][level].keys())
        features_list = [s["features"][level] for s in library]
        M_std, excluded = standardize(features_list, feature_names)
        standardized[level] = M_std
        print(f"  {level} : {M_std.shape[1]}/{len(feature_names)} retenues, {len(excluded)} exclues")

    # === Distances pairwise ===
    print(f"\n--- Calcul des 5778 paires ---")
    pairs = []
    psi_norms = [float(np.linalg.norm(s["psi"])) for s in library]
    h_norms = [float(np.linalg.norm(s["h"])) for s in library]
    for i in range(n_states):
        psi_i = library[i]["psi"]; h_i = library[i]["h"]
        for j in range(i+1, n_states):
            psi_j = library[j]["psi"]; h_j = library[j]["h"]
            D_psi = float(np.linalg.norm(psi_j - psi_i) / (psi_norms[i] + EPS))
            D_h = float(np.linalg.norm(h_j - h_i) / (h_norms[i] + EPS))
            D_obs = {level: float(np.linalg.norm(standardized[level][j] - standardized[level][i]))
                     for level in levels}
            # Classifications de paire
            si = library[i]; sj = library[j]
            ci = si["features"]["classifications"]
            cj = sj["features"]["classifications"]
            pairs.append({
                "i": i, "j": j,
                "D_state_psi": D_psi, "D_state_h": D_h,
                "D_obs": D_obs,
                # Classes
                "intra_trajectory": (si["family"] == sj["family"] and si["beta"] == sj["beta"]),
                "inter_family_same_beta": (si["family"] != sj["family"] and si["beta"] == sj["beta"]),
                "inter_beta_same_family": (si["family"] == sj["family"] and si["beta"] != sj["beta"]),
                "B2_involving": (si["family"] == "B2" or sj["family"] == "B2"),
                "B2_vs_nonB2": ((si["family"] == "B2") != (sj["family"] == "B2")),
                "B3_involving": (si["family"] == "B3" or sj["family"] == "B3"),
                "underflow_involving": (ci["is_underflow"] or cj["is_underflow"]),
                "dissoc_involving": (ci["is_dissociated"] or cj["is_dissociated"]),
                "dissoc_vs_nondissoc": (ci["is_dissociated"] != cj["is_dissociated"]),
            })
    print(f"  {len(pairs)} paires calculees")

    # === Residus log-log par axe x niveau ===
    print(f"\n--- Calcul des residus log-log ---")
    residuals = {}  # (axis, level) -> array of residuals
    for axis_name, D_key in [("psi", "D_state_psi"), ("h", "D_state_h")]:
        for level in levels:
            xs = np.array([p[D_key] for p in pairs])
            ys = np.array([p["D_obs"][level] for p in pairs])
            mask = (xs > EPS_FLOOR) & (ys > EPS_FLOOR)
            log_xs = np.log(xs[mask])
            log_ys = np.log(ys[mask])
            X = np.column_stack([log_xs, np.ones_like(log_xs)])
            coeffs, _, _, _ = np.linalg.lstsq(X, log_ys, rcond=None)
            alpha, c = float(coeffs[0]), float(coeffs[1])
            r = log_ys - (alpha * log_xs + c)
            # Stocker resid + mask + alpha
            residuals[(axis_name, level)] = {
                "r": r,
                "mask": mask,
                "alpha": alpha,
                "c": c,
                "n": len(r),
            }
            print(f"  {axis_name} x {level} : alpha={alpha:+.4f}, n_resid={len(r)}")

    # === Tests par classe ===
    print(f"\n--- Tests par classe (axe = D_state_h, niveau = P_functional) ---")
    # Pour ne pas surcharger : focus sur (h, P_functional) qui a alpha le plus eleve
    # et faire les classes ensuite

    target_axis = "h"
    target_level = "P_functional"
    res = residuals[(target_axis, target_level)]
    mask = res["mask"]
    r_full = res["r"]
    abs_r = np.abs(r_full)

    # Construire les labels de classe pour les paires retenues
    pair_indices = [k for k in range(len(pairs)) if mask[k]]
    n_kept = len(pair_indices)
    print(f"  {n_kept} paires retenues apres filtrage")

    def class_mask(class_name):
        return np.array([pairs[k][class_name] for k in pair_indices], dtype=bool)

    classes = {
        "intra_trajectory": class_mask("intra_trajectory"),
        "inter_family_same_beta": class_mask("inter_family_same_beta"),
        "inter_beta_same_family": class_mask("inter_beta_same_family"),
        "B2_involving": class_mask("B2_involving"),
        "B2_vs_nonB2": class_mask("B2_vs_nonB2"),
        "B3_involving": class_mask("B3_involving"),
        "underflow_involving": class_mask("underflow_involving"),
        "dissoc_involving": class_mask("dissoc_involving"),
        "dissoc_vs_nondissoc": class_mask("dissoc_vs_nondissoc"),
    }

    # Stats descriptives par classe
    rng = np.random.default_rng(RNG_SEED)
    class_stats = {}
    for cname, cmask in classes.items():
        if cmask.sum() < 5:
            class_stats[cname] = {"n_in": int(cmask.sum()), "skip": True}
            continue
        r_in = r_full[cmask]
        r_out = r_full[~cmask]
        var_in = float(np.var(r_in))
        var_out = float(np.var(r_out))
        median_abs_in = float(np.median(np.abs(r_in)))
        median_abs_out = float(np.median(np.abs(r_out)))
        iqr_in = float(np.percentile(np.abs(r_in), 75) - np.percentile(np.abs(r_in), 25))
        iqr_out = float(np.percentile(np.abs(r_out), 75) - np.percentile(np.abs(r_out), 25))
        var_ratio = var_in / (var_out + EPS)
        delta_med = median_abs_in - median_abs_out

        # Permutation test sur Var(r)
        observed_diff = var_in - var_out
        n_in = int(cmask.sum())
        perm_diffs = np.zeros(N_PERMUTATIONS)
        for p in range(N_PERMUTATIONS):
            shuffled = rng.permutation(r_full)
            r_in_p = shuffled[:n_in]
            r_out_p = shuffled[n_in:]
            perm_diffs[p] = np.var(r_in_p) - np.var(r_out_p)
        p_value = float(np.mean(np.abs(perm_diffs) >= abs(observed_diff)))

        class_stats[cname] = {
            "n_in": n_in,
            "n_out": int((~cmask).sum()),
            "var_in": var_in,
            "var_out": var_out,
            "median_abs_r_in": median_abs_in,
            "median_abs_r_out": median_abs_out,
            "iqr_in": iqr_in,
            "iqr_out": iqr_out,
            "variance_ratio": var_ratio,
            "delta_median_abs_r": delta_med,
            "p_value_perm": p_value,
            "skip": False,
        }
        print(f"  {cname:30s} n_in={n_in:>5} var_in={var_in:.4f} var_out={var_out:.4f} "
              f"ratio={var_ratio:.3f} p={p_value:.3f}")

    # === Regressions separees par regime ===
    print(f"\n--- Regressions log-log separees par regime (axe h, P_functional) ---")
    regression_by_regime = {}
    for regime_name, regime_filter in [
        ("intra_trajectory_only", lambda p: p["intra_trajectory"]),
        ("inter_family_same_beta_only", lambda p: p["inter_family_same_beta"]),
        ("B2_involving_only", lambda p: p["B2_involving"]),
        ("dissoc_involving_only", lambda p: p["dissoc_involving"]),
        ("no_B3", lambda p: not p["B3_involving"]),
        ("no_underflow", lambda p: not p["underflow_involving"]),
        ("no_B3_no_underflow", lambda p: (not p["B3_involving"]) and (not p["underflow_involving"])),
    ]:
        # filtre + filtrage floor
        kept_pairs = [p for p in pairs if regime_filter(p)]
        if len(kept_pairs) < 10:
            regression_by_regime[regime_name] = {"n": len(kept_pairs), "alpha": None, "R2": None}
            continue
        xs = np.array([p["D_state_h"] for p in kept_pairs])
        ys = np.array([p["D_obs"]["P_functional"] for p in kept_pairs])
        mask_r = (xs > EPS_FLOOR) & (ys > EPS_FLOOR)
        if mask_r.sum() < 10:
            regression_by_regime[regime_name] = {"n": int(mask_r.sum()), "alpha": None, "R2": None}
            continue
        log_xs = np.log(xs[mask_r])
        log_ys = np.log(ys[mask_r])
        X = np.column_stack([log_xs, np.ones_like(log_xs)])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_ys, rcond=None)
        alpha_r = float(coeffs[0]); c_r = float(coeffs[1])
        ss_res = float(np.sum((log_ys - (alpha_r * log_xs + c_r))**2))
        ss_tot = float(np.sum((log_ys - log_ys.mean())**2))
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        regression_by_regime[regime_name] = {
            "n": int(mask_r.sum()), "alpha": alpha_r, "c": c_r, "R2": R2,
        }
        print(f"  {regime_name:30s} n={mask_r.sum():>5} alpha={alpha_r:+.4f} R2={R2:.3f}")

    # === Verdict compose ===
    print(f"\n--- Verdict compose ---")
    # Critere de stratification : variance_ratio > 1.5 ou < 0.67, p < 0.05, et n_in >= 30
    stratifiers = []
    significant_classes = []
    for cname, stats in class_stats.items():
        if stats.get("skip", False):
            continue
        if stats["p_value_perm"] < 0.05 and stats["n_in"] >= 30:
            significant_classes.append(cname)
            if stats["variance_ratio"] >= 1.5 or stats["variance_ratio"] <= 0.67:
                stratifiers.append({
                    "class": cname,
                    "variance_ratio": stats["variance_ratio"],
                    "p_value": stats["p_value_perm"],
                    "delta_median": stats["delta_median_abs_r"],
                    "n_in": stats["n_in"],
                })

    # Tri par taille d'effet
    stratifiers.sort(key=lambda s: -abs(np.log(s["variance_ratio"])))

    # Verdict principal
    if len(stratifiers) == 0:
        if len(significant_classes) > 0:
            verdict_main = "P4VAR-HOMOSCEDASTIC"
            note_main = f"p<0.05 pour {len(significant_classes)} classes mais effet faible (variance_ratio in [0.67, 1.5])"
        else:
            verdict_main = "P4VAR-HOMOSCEDASTIC"
            note_main = "aucune classe ne montre de difference significative"
    else:
        verdict_main = "P4VAR-STRATIFIED"
        note_main = f"{len(stratifiers)} classes structurent la variance avec effet >= 1.5x"

    # Modifiers
    modifiers = []
    for s in stratifiers:
        cname = s["class"]
        if "B2" in cname:
            modifiers.append("B2-DRIVEN")
        if "dissoc" in cname:
            modifiers.append("DISSOC-DRIVEN")
        if "underflow" in cname:
            modifiers.append("UNDERFLOW-CO-DRIVER")
        if "B3" in cname:
            modifiers.append("B3-DEGENERATE-CO-DRIVER")
        if "intra" in cname or "inter" in cname:
            modifiers.append("INTRA-INTER-MIXING")
        if "beta" in cname:
            modifiers.append("BETA-DRIVEN")
    modifiers = list(dict.fromkeys(modifiers))  # uniq preservant l'ordre

    # Weak effect : tous les variance_ratio sont entre 1.5 et 2.0
    if stratifiers and all(1.5 <= s["variance_ratio"] <= 2.0 or 0.5 <= s["variance_ratio"] <= 0.67
                            for s in stratifiers):
        modifiers.append("WEAK-EFFECT")

    print(f"  Verdict principal : {verdict_main}")
    print(f"  Note : {note_main}")
    if modifiers:
        print(f"  Modifiers : {' + '.join(modifiers)}")
    if stratifiers:
        print(f"  Stratificateurs identifies :")
        for s in stratifiers[:5]:
            print(f"    {s['class']:30s} ratio={s['variance_ratio']:.3f} "
                  f"p={s['p_value']:.4f} delta_med={s['delta_median']:+.4f} n_in={s['n_in']}")

    # === Lecture humaine recommandee ===
    print(f"\n--- Lecture recommandee ---")
    if verdict_main == "P4VAR-HOMOSCEDASTIC":
        recommandation = ("§25 renforce : le nuage P4 ne presente pas de stratification "
                          "residuelle detectable. Dette P4 fermee plus solidement.")
    else:
        # Description detaillee des facteurs
        factor_desc = ", ".join(modifiers) if modifiers else "facteurs identifies"
        recommandation = (f"§25 a amender localement (24.4/24.5) : "
                          f"P4 ne donne pas de loi moyenne, mais la dispersion du nuage "
                          f"est stratifiee par {factor_desc}. Cela ne valide pas P4 "
                          f"et ne rouvre pas la self-opacity.")
    print(f"  {recommandation}")

    # === Sauvegarder JSON ===
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size < 50000 else None
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
            "script": "test_6d_p4_var_audit.py",
            "n_states": n_states,
            "n_pairs": len(pairs),
            "n_permutations": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
            "target_axis": target_axis,
            "target_level": target_level,
            "guardrails": [
                "no new engine", "no new family",
                "no ML", "no Delta", "no MCQ", "no Ch4",
                "no self-opacity claim",
            ],
        },
        "regression_global_per_axis_level": {
            f"{axis}_x_{level}": {
                "alpha": residuals[(axis, level)]["alpha"],
                "c": residuals[(axis, level)]["c"],
                "n": residuals[(axis, level)]["n"],
            }
            for axis in ["psi", "h"] for level in levels
        },
        "class_stats": serialize(class_stats),
        "regression_by_regime": serialize(regression_by_regime),
        "stratifiers": serialize(stratifiers),
        "verdict_main": verdict_main,
        "note_main": note_main,
        "modifiers": modifiers,
        "significant_classes": significant_classes,
        "recommandation": recommandation,
    }
    out_path = "/home/claude/mcq_v4/6d_p4_var_audit.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegarde : {out_path}")


if __name__ == "__main__":
    main()
