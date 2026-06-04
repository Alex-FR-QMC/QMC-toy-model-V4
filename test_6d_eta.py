"""
6d-η — Confirmation géométrique de l'axe antagoniste.

Question : l'antagonisme annulaire / gaussiennes observé en 6d-ζ
est-il une structure réelle de l'espace des perturbations P′, ou
seulement un effet contingent de l'échantillon à 4 variantes ?

Méthode : reprendre exactement le protocole 6d-ζ avec 12 variantes
de P′ couvrant trois familles géométriques (G, A, D).

Famille G — gaussiennes :
  1. G_etroite_centree    : σ=0.5, centre (2,2,2)
  2. G_standard_centree   : σ=0.8, centre (2,2,2)
  3. G_large_centree      : σ=1.5, centre (2,2,2)
  4. G_decentree_x        : σ=0.8, centre (3,2,2)
  5. G_decentree_diag     : σ=0.8, centre (2.7,2.7,2.7)

Famille A — anneaux/shells :
  6. A_anneau_interne     : r ∈ [0.7, 1.3]
  7. A_anneau_moyen       : r ∈ [1.3, 1.9]
  8. A_anneau_externe     : r ∈ [1.9, 2.5]
  9. A_shell_peripherique : r ≥ 2.5

Famille D — dipolaires/signées :
  10. D_dipole_radial     : +r∈[0,1], -r≥2
  11. D_dipole_y          : +face j=0, -face j=4
  12. D_double_lobe       : +gauss(3,3,2), -gauss(1,1,2), σ=0.6

Toutes calibrées à 0.1 × ||P6||. Statut OK/SATURATED/FAILED par variante.

Sorties : matrices 12×12 (cos_total, cos_ss_pure, contrib_cos_ss),
f_shared par variante, ||Specific||/||O|| par variante, scores
intra/inter famille, score d'antagonisme A_GA, verdict.

Pas de η-bis d'office (conditionnel selon verdict).
Pas de Δ. Pas de 𝒢. Pas de lecture MCQ.
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


# ---------- 12 perturbations P' ----------

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
    p = psi * factor
    return p / p.sum()

def P_G_standard_centree(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(CENTER, CENTER, CENTER, 0.8)
    p = psi * factor
    return p / p.sum()

def P_G_large_centree(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(CENTER, CENTER, CENTER, 1.5)
    p = psi * factor
    return p / p.sum()

def P_G_decentree_x(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(CENTER + DX, CENTER, CENTER, 0.8)
    p = psi * factor
    return p / p.sum()

def P_G_decentree_diag(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(
        CENTER + 0.7*DX, CENTER + 0.7*DX, CENTER + 0.7*DX, 0.8)
    p = psi * factor
    return p / p.sum()

# A family
def P_A_anneau_interne(psi, strength):
    factor = 1.0 + strength * _radial_mask(0.7, 1.3)
    p = psi * factor
    return p / p.sum()

def P_A_anneau_moyen(psi, strength):
    factor = 1.0 + strength * _radial_mask(1.3, 1.9)
    p = psi * factor
    return p / p.sum()

def P_A_anneau_externe(psi, strength):
    factor = 1.0 + strength * _radial_mask(1.9, 2.5)
    p = psi * factor
    return p / p.sum()

def P_A_shell_peripherique(psi, strength):
    factor = 1.0 + strength * _radial_mask(2.5, 100.0)
    p = psi * factor
    return p / p.sum()

# D family
def P_D_dipole_radial(psi, strength):
    factor = np.ones_like(psi)
    factor += strength * _radial_mask(0.0, 1.0)
    factor -= strength * _radial_mask(2.0, 100.0)
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()

def P_D_dipole_y(psi, strength):
    factor = np.ones_like(psi)
    factor[:, 0, :] += strength
    factor[:, 4, :] -= strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()

def P_D_double_lobe(psi, strength):
    plus = _gaussian_factor(3.0*DX, 3.0*DX, 2.0*DX, 0.6)
    minus = _gaussian_factor(1.0*DX, 1.0*DX, 2.0*DX, 0.6)
    factor = 1.0 + strength * (plus - minus)
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()


VARIANTS = [
    ("G_etroite_centree",     "G", P_G_etroite_centree),
    ("G_standard_centree",    "G", P_G_standard_centree),
    ("G_large_centree",       "G", P_G_large_centree),
    ("G_decentree_x",         "G", P_G_decentree_x),
    ("G_decentree_diag",      "G", P_G_decentree_diag),
    ("A_anneau_interne",      "A", P_A_anneau_interne),
    ("A_anneau_moyen",        "A", P_A_anneau_moyen),
    ("A_anneau_externe",      "A", P_A_anneau_externe),
    ("A_shell_peripherique",  "A", P_A_shell_peripherique),
    ("D_dipole_radial",       "D", P_D_dipole_radial),
    ("D_dipole_y",            "D", P_D_dipole_y),
    ("D_double_lobe",         "D", P_D_double_lobe),
]


def calibrate_strength(P_fn, psi_base, target_amp, s_bounds=(1e-6, 5.0)):
    """Calibre s pour ||P(ψ_base) - ψ_base|| = target_amp.
    Retourne (s, amp_obtenu, status).
    Status: 'OK', 'SATURATED', 'FAILED'.
    """
    def err(s):
        return float(np.linalg.norm(P_fn(psi_base, s) - psi_base)) - target_amp
    s_min, s_max = s_bounds
    try:
        err_min = err(s_min)
        err_max = err(s_max)
    except Exception:
        return None, None, "FAILED"
    if err_min >= 0:
        # même à s minimal on dépasse la cible (cas pathologique)
        return s_min, float(np.linalg.norm(P_fn(psi_base, s_min) - psi_base)), "SATURATED"
    if err_max < 0:
        # même à s_max on n'atteint pas la cible : saturation
        amp_at_max = float(np.linalg.norm(P_fn(psi_base, s_max) - psi_base))
        if amp_at_max < 0.5 * target_amp:
            return s_max, amp_at_max, "FAILED"
        return s_max, amp_at_max, "SATURATED"
    try:
        s_root = brentq(err, s_min, s_max, xtol=1e-8)
        amp_obt = float(np.linalg.norm(P_fn(psi_base, s_root) - psi_base))
        # Vérification : amp dans [0.99, 1.01] × target
        if abs(amp_obt - target_amp) / target_amp < 0.02:
            return s_root, amp_obt, "OK"
        else:
            return s_root, amp_obt, "SATURATED"
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
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-30 or n2 < 1e-30:
        return 0.0
    return float(np.dot(v1.flatten(), v2.flatten()) / (n1 * n2))


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_dt_long = int(800.0 / dt)

    print(f"=== 6d-η : confirmation géométrique sur 12 variantes ===\n")

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    # Référence amplitude : ||P6||
    # Calibrage P6 sur ||P1'_std||
    def P1prime_std(psi, strength=0.05):
        return P_G_standard_centree(psi, strength * 8.0)  # approximatif
    amp_P1prime_std = float(np.linalg.norm(P_G_standard_centree(psi_base, 0.4) - psi_base))
    # En fait on reprend la définition exacte de 6d-ζ : P1'_std est un plateau central
    def P1prime_plateau(psi, strength=0.05):
        coords = np.arange(N_AXIS) * DX
        c = (N_AXIS - 1) * DX / 2.0
        factor = np.ones_like(psi)
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                    if r <= 1.5: factor[i,j,k] += strength
        p = psi * factor
        return p / p.sum()
    amp_P1prime_std = float(np.linalg.norm(P1prime_plateau(psi_base, 0.05) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    target_amp = 0.1 * amp_P6

    print(f"  Référence : ||P6|| = {amp_P6:.4e}, target_amp = {target_amp:.4e}\n")

    # Calibrer toutes les variantes
    print(f"=== Calibration des 12 variantes ===\n")
    print(f"  {'label':<28} {'family':<8} {'s_calib':>12} {'amp_obt':>14} {'status':>10}")
    calibration_results = []
    for label, family, P_fn in VARIANTS:
        s_calib, amp_obt, status = calibrate_strength(P_fn, psi_base, target_amp)
        calibration_results.append({
            "label": label, "family": family,
            "s_calib": s_calib, "amp_obt": amp_obt, "status": status,
        })
        print(f"  {label:<28} {family:<8} "
              f"{s_calib if s_calib else 0:>12.6f} "
              f"{amp_obt if amp_obt else 0:>14.4e} "
              f"{status:>10}")

    # Filtrer FAILED
    valid = [(label, family, P_fn, c["s_calib"], c["status"])
             for (label, family, P_fn), c in zip(VARIANTS, calibration_results)
             if c["status"] != "FAILED"]
    if any(c["status"] == "FAILED" for c in calibration_results):
        print(f"\n  ATTENTION : certaines variantes FAILED et exclues du test principal")

    # Préparer P6(0) et P6(3000)
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6,
                                D, beta_lock, gamma_v, h0, dt, n_short)
    n_3000 = int(3000.0 / dt)
    psi_tau3000, h_tau3000 = evolve(psi_tau0.copy(), h_tau0.copy(),
                                     D, beta_lock, gamma_v, h0, dt, n_3000)

    # Calculer les résidus pour chaque variante valide
    print(f"\n=== Calcul des résidus pour {len(valid)} variantes valides ===\n")
    residuals = {}
    families_by_label = {}
    for label, family, P_fn, s_calib, status in valid:
        print(f"  {label}...")
        d_psi_tau0, d_h_tau0 = compute_delta_full(
            psi_tau0, h_tau0, P_fn, s_calib,
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_psi_ref, d_h_ref = compute_delta_full(
            psi_tau3000, h_tau3000, P_fn, s_calib,
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_flat = np.concatenate([d_psi_tau0.flatten(), d_h_tau0.flatten()])
        ref_flat = np.concatenate([d_psi_ref.flatten(), d_h_ref.flatten()])
        ref_norm_sq = float(np.dot(ref_flat, ref_flat))
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq) if ref_norm_sq > 1e-30 else 0.0
        r_psi = d_psi_tau0 - a_tau * d_psi_ref
        r_h = d_h_tau0 - a_tau * d_h_ref
        residuals[label] = {"r_psi": r_psi, "r_h": r_h, "status": status}
        families_by_label[label] = family

    obs_names = ["psi_temp_mean_abs", "psi_temp_norm",
                 "h_temp_mean_abs", "h_temp_norm"]
    labels = list(residuals.keys())
    n = len(labels)

    # Analyse par observable et mode
    print(f"\n=== Analyses par observable ===\n")
    full_output = {
        "n_variants": n,
        "calibration": calibration_results,
        "labels": labels,
        "families": [families_by_label[l] for l in labels],
        "by_observable": {},
    }

    for name in obs_names:
        for mode in ["brut", "centré"]:
            key = f"{name}_{mode}"
            print(f"\n  --- {key} ---")
            # Observables
            O = {}
            for lab in labels:
                serie = get_obs(residuals[lab]["r_psi"],
                                residuals[lab]["r_h"], name)
                if mode == "centré":
                    serie = serie - serie.mean()
                O[lab] = serie

            # Composante partagée Ō
            O_bar = np.mean([O[lab] for lab in labels], axis=0)
            O_bar_norm_sq = float(np.dot(O_bar, O_bar))

            # Shared / Specific par variante
            Shared = {}
            Specific = {}
            f_shared = {}
            spec_frac_norm = {}
            for lab in labels:
                if O_bar_norm_sq > 1e-30:
                    a = float(np.dot(O[lab], O_bar) / O_bar_norm_sq)
                else:
                    a = 0.0
                Shared[lab] = a * O_bar
                Specific[lab] = O[lab] - Shared[lab]
                norm_O = np.linalg.norm(O[lab])
                norm_shared = np.linalg.norm(Shared[lab])
                norm_spec = np.linalg.norm(Specific[lab])
                f_shared[lab] = (norm_shared / max(norm_O, 1e-30))**2
                spec_frac_norm[lab] = norm_spec / max(norm_O, 1e-30)

            # Matrices n × n
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
                        contrib_cos_ss[i,j] = float(
                            np.dot(Specific[li], Specific[lj])
                            / (norm_i * norm_j))
                    else:
                        contrib_cos_ss[i,j] = 0.0

            # Scores intra/inter famille sur cos_ss_pure
            def family_pair_mean(fam1, fam2):
                vals = []
                for i in range(n):
                    if families_by_label[labels[i]] != fam1: continue
                    for j in range(n):
                        if i == j: continue
                        if families_by_label[labels[j]] != fam2: continue
                        # Pour intra-famille, on ne compte chaque paire qu'une fois
                        if fam1 == fam2 and i >= j: continue
                        vals.append(cos_ss_pure[i,j])
                return float(np.mean(vals)) if vals else None

            mean_ss = {
                "GG": family_pair_mean("G", "G"),
                "AA": family_pair_mean("A", "A"),
                "DD": family_pair_mean("D", "D"),
                "GA": family_pair_mean("G", "A"),
                "GD": family_pair_mean("G", "D"),
                "AD": family_pair_mean("A", "D"),
            }
            print(f"    mean cos_ss(G,G) = {mean_ss['GG']}")
            print(f"    mean cos_ss(A,A) = {mean_ss['AA']}")
            print(f"    mean cos_ss(D,D) = {mean_ss['DD']}")
            print(f"    mean cos_ss(G,A) = {mean_ss['GA']}")
            print(f"    mean cos_ss(G,D) = {mean_ss['GD']}")
            print(f"    mean cos_ss(A,D) = {mean_ss['AD']}")

            # Score d'antagonisme
            A_GA = None
            if mean_ss["GG"] is not None and mean_ss["GA"] is not None:
                A_GA = mean_ss["GG"] - mean_ss["GA"]
                print(f"    Score antagonisme A_GA = mean(G,G) - mean(G,A) = {A_GA:+.4f}")

            full_output["by_observable"][key] = {
                "O_bar_norm": float(np.linalg.norm(O_bar)),
                "f_shared": f_shared,
                "spec_frac_norm": spec_frac_norm,
                "cos_total_matrix": cos_total.tolist(),
                "cos_ss_pure_matrix": cos_ss_pure.tolist(),
                "contrib_cos_ss_matrix": contrib_cos_ss.tolist(),
                "mean_cos_ss_by_family": mean_ss,
                "score_A_GA": A_GA,
            }

    # === Verdict global ===
    print(f"\n=== Verdict global ===\n")

    # Lire les scores A_GA sur les 8 cas
    A_GA_all = []
    GG_all = []
    GA_all = []
    AA_all = []
    for key, data in full_output["by_observable"].items():
        if data["score_A_GA"] is not None:
            A_GA_all.append(data["score_A_GA"])
        if data["mean_cos_ss_by_family"]["GG"] is not None:
            GG_all.append(data["mean_cos_ss_by_family"]["GG"])
        if data["mean_cos_ss_by_family"]["GA"] is not None:
            GA_all.append(data["mean_cos_ss_by_family"]["GA"])
        if data["mean_cos_ss_by_family"]["AA"] is not None:
            AA_all.append(data["mean_cos_ss_by_family"]["AA"])

    A_GA_mean = float(np.mean(A_GA_all))
    A_GA_min = float(min(A_GA_all))
    GG_mean = float(np.mean(GG_all))
    GA_mean = float(np.mean(GA_all))
    AA_mean = float(np.mean(AA_all))

    print(f"  Sur les 8 cas (4 observables × 2 modes) :")
    print(f"  mean(cos_ss G,G) moyenné : {GG_mean:+.4f}")
    print(f"  mean(cos_ss A,A) moyenné : {AA_mean:+.4f}")
    print(f"  mean(cos_ss G,A) moyenné : {GA_mean:+.4f}")
    print(f"  Score A_GA moyenné : {A_GA_mean:+.4f}")
    print(f"  Score A_GA min : {A_GA_min:+.4f}")

    # Critères
    GG_positive_majoritaire = (GG_mean > 0.1)
    GA_negatif_majoritaire = (GA_mean < -0.1)
    A_GA_positif_net = (A_GA_mean > 0.2)
    A_GA_stable = (A_GA_min > 0.0)

    print(f"\n  Critères :")
    print(f"  (1) cos_ss(G,G) globalement positif (>0.1) : {GG_positive_majoritaire}")
    print(f"  (2) cos_ss(G,A) globalement négatif (<-0.1) : {GA_negatif_majoritaire}")
    print(f"  (3) A_GA moyen > 0.2 : {A_GA_positif_net}")
    print(f"  (4) A_GA min > 0 (stable sur tous cas) : {A_GA_stable}")

    n_critera = sum([GG_positive_majoritaire, GA_negatif_majoritaire,
                     A_GA_positif_net, A_GA_stable])
    if n_critera == 4:
        verdict = "η-PASS fort"
    elif n_critera >= 2 and A_GA_stable:
        verdict = "η-PASS faible"
    elif n_critera == 0:
        verdict = "η-FAIL"
    else:
        verdict = "η-INDETERMINÉ"
    print(f"\n  VERDICT : {verdict}")

    full_output["verdict"] = verdict
    full_output["criteria_summary"] = {
        "GG_positive_majoritaire": GG_positive_majoritaire,
        "GA_negatif_majoritaire": GA_negatif_majoritaire,
        "A_GA_positif_net": A_GA_positif_net,
        "A_GA_stable": A_GA_stable,
        "GG_mean": GG_mean,
        "GA_mean": GA_mean,
        "AA_mean": AA_mean,
        "A_GA_mean": A_GA_mean,
        "A_GA_min": A_GA_min,
    }

    with open("/home/claude/mcq_v4/6d_eta_confirmation.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
