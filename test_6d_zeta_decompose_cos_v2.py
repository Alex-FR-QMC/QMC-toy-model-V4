"""
6d-ζ — Décomposition des cos pairwise en quatre termes.

Pour chaque paire de variantes (i, j) et chaque observable temporelle,
décomposer :

  <O_i, O_j> = <Shared_i, Shared_j>
             + <Shared_i, Specific_j>
             + <Specific_i, Shared_j>
             + <Specific_i, Specific_j>

où Shared_v = a_v * Ō et Specific_v = O_v - Shared_v.

L'objectif est de vérifier si :
- le terme S-S porte l'essentiel de la similarité (base commune élevée)
- les termes spécifiques abaissent le cos vers la bande 0.73-0.95
- annulaire et étroite portent le plus de charge spécifique

Note méthodologique : <Shared_i, Specific_j> = a_i * <Ō, Specific_j>
et par construction <Ō, Specific_j> n'est PAS nul (Specific_j = O_j - a_j·Ō,
donc <Ō, Specific_j> = <Ō, O_j> - a_j·||Ō||² = a_j·||Ō||² - a_j·||Ō||² = 0).

DONC les deux termes croisés sont identiquement nuls par construction de
la projection. La décomposition se réduit à :

  <O_i, O_j> = <Shared_i, Shared_j> + <Specific_i, Specific_j>

C'est en réalité plus propre — et c'est ce qu'on va exploiter.
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

def P_prime_gauss(psi, strength, sigma_p):
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

def P_prime_annular(psi, strength, r_inner=1.5, r_outer=2.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r_inner <= r <= r_outer:
                    factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

def P1prime_std(psi, strength=0.05):
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

    print(f"=== 6d-ζ : décomposition des cos pairwise ===\n")

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)
    amp_P1prime_std = float(np.linalg.norm(P1prime_std(psi_base) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    target_amp = 0.1 * amp_P6

    variants = []
    for label, sigma_p in [("étroite", 0.5), ("standard", 0.8), ("large", 1.5)]:
        s_calib = brentq(lambda s: float(np.linalg.norm(P_prime_gauss(psi_base, s, sigma_p) - psi_base))
                          - target_amp, 1e-6, 1.0, xtol=1e-8)
        variants.append({
            "label": label, "s_calib": s_calib,
            "P_fn": (lambda psi, s, sp=sigma_p: P_prime_gauss(psi, s, sp)),
        })
    s_ann = brentq(lambda s: float(np.linalg.norm(P_prime_annular(psi_base, s) - psi_base))
                    - target_amp, 1e-6, 5.0, xtol=1e-8)
    variants.append({
        "label": "annulaire", "s_calib": s_ann,
        "P_fn": (lambda psi, s: P_prime_annular(psi, s)),
    })

    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6,
                                D, beta_lock, gamma_v, h0, dt, n_short)
    n_3000 = int(3000.0 / dt)
    psi_tau3000, h_tau3000 = evolve(psi_tau0.copy(), h_tau0.copy(),
                                     D, beta_lock, gamma_v, h0, dt, n_3000)

    print(f"  Calcul des résidus...")
    residuals = {}
    for v in variants:
        d_psi_tau0, d_h_tau0 = compute_delta_full(
            psi_tau0, h_tau0, v["P_fn"], v["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_psi_ref, d_h_ref = compute_delta_full(
            psi_tau3000, h_tau3000, v["P_fn"], v["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_flat = np.concatenate([d_psi_tau0.flatten(), d_h_tau0.flatten()])
        ref_flat = np.concatenate([d_psi_ref.flatten(), d_h_ref.flatten()])
        ref_norm_sq = float(np.dot(ref_flat, ref_flat))
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq) if ref_norm_sq > 1e-30 else 0.0
        r_psi = d_psi_tau0 - a_tau * d_psi_ref
        r_h = d_h_tau0 - a_tau * d_h_ref
        residuals[v["label"]] = {"r_psi": r_psi, "r_h": r_h}

    labels = ["étroite", "standard", "large", "annulaire"]
    obs_names = ["psi_temp_mean_abs", "psi_temp_norm",
                 "h_temp_mean_abs", "h_temp_norm"]

    # Calculer les observables et leurs décompositions
    def get_obs(r_psi, r_h, name):
        if name == "psi_temp_mean_abs":
            return np.mean(np.abs(r_psi), axis=(1,2,3))
        elif name == "psi_temp_norm":
            return np.sqrt(np.sum(r_psi**2, axis=(1,2,3)))
        elif name == "h_temp_mean_abs":
            return np.mean(np.abs(r_h), axis=(1,2,3))
        elif name == "h_temp_norm":
            return np.sqrt(np.sum(r_h**2, axis=(1,2,3)))

    all_decompositions = {}

    for name in obs_names:
        for mode in ["brut", "centré"]:
            print(f"\n  === {name} ({mode}) ===")
            # Observables
            O = {}
            for lab in labels:
                serie = get_obs(residuals[lab]["r_psi"], residuals[lab]["r_h"], name)
                if mode == "centré":
                    serie = serie - serie.mean()
                O[lab] = serie

            # Composante partagée
            O_bar = np.mean([O[lab] for lab in labels], axis=0)
            O_bar_norm_sq = float(np.dot(O_bar, O_bar))

            # Pour chaque variante, Shared et Specific
            Shared = {}
            Specific = {}
            for lab in labels:
                if O_bar_norm_sq > 1e-30:
                    a = float(np.dot(O[lab], O_bar) / O_bar_norm_sq)
                else:
                    a = 0.0
                Shared[lab] = a * O_bar
                Specific[lab] = O[lab] - Shared[lab]

            # Vérifier l'orthogonalité Shared / Specific (devrait être 0 par construction)
            ortho_check = []
            for lab in labels:
                # <Shared_i, Specific_i> doit être 0 (projection orthogonale)
                ortho = float(np.dot(Shared[lab], Specific[lab]))
                ortho_check.append(ortho)
            max_ortho = max(abs(o) for o in ortho_check)

            # Décomposition des cos pairwise
            decomp = {}
            print(f"  Décomposition des cos hors diagonale :")
            print(f"  {'paire':<22} {'cos':>8} {'SS frac':>10} {'sS frac':>10} "
                  f"{'Ss frac':>10} {'ss frac':>10} {'somme':>10}")
            for i, li in enumerate(labels):
                for j in range(i+1, len(labels)):
                    lj = labels[j]
                    O_i = O[li]
                    O_j = O[lj]
                    norm_i = np.linalg.norm(O_i)
                    norm_j = np.linalg.norm(O_j)
                    if norm_i < 1e-30 or norm_j < 1e-30:
                        continue
                    # Numérateur complet
                    num_total = float(np.dot(O_i, O_j))
                    # Termes
                    SS = float(np.dot(Shared[li], Shared[lj]))
                    sS = float(np.dot(Specific[li], Shared[lj]))
                    Ss = float(np.dot(Shared[li], Specific[lj]))
                    ss = float(np.dot(Specific[li], Specific[lj]))
                    # cos total
                    cos_total = num_total / (norm_i * norm_j)
                    # Fractions du numérateur (peuvent être négatives)
                    frac_SS = SS / num_total if abs(num_total) > 1e-30 else 0.0
                    frac_sS = sS / num_total if abs(num_total) > 1e-30 else 0.0
                    frac_Ss = Ss / num_total if abs(num_total) > 1e-30 else 0.0
                    frac_ss = ss / num_total if abs(num_total) > 1e-30 else 0.0
                    pair_label = f"{li}-{lj}"
                    # Contributions au cos (pas seulement au numérateur)
                    denom = norm_i * norm_j
                    contrib_cos_SS = SS / denom if denom > 1e-30 else 0.0
                    contrib_cos_ss = ss / denom if denom > 1e-30 else 0.0
                    # cos entre composantes pures (pour mémoire)
                    cos_SS_pure = cos_pair(Shared[li], Shared[lj])
                    cos_ss_pure = cos_pair(Specific[li], Specific[lj])
                    decomp[pair_label] = {
                        "cos_total": cos_total,
                        "num_total": num_total,
                        "SS": SS, "sS": sS, "Ss": Ss, "ss": ss,
                        "frac_SS_of_num": frac_SS, "frac_sS_of_num": frac_sS,
                        "frac_Ss_of_num": frac_Ss, "frac_ss_of_num": frac_ss,
                        "contrib_cos_SS": contrib_cos_SS,
                        "contrib_cos_ss": contrib_cos_ss,
                        "cos_SS_pure": cos_SS_pure,
                        "cos_ss_pure": cos_ss_pure,
                    }
                    print(f"  {pair_label:<22} {cos_total:>+8.4f} "
                          f"{frac_SS:>+10.4f} {frac_sS:>+10.4f} "
                          f"{frac_Ss:>+10.4f} {frac_ss:>+10.4f} "
                          f"{frac_SS+frac_sS+frac_Ss+frac_ss:>+10.4f}")

            # Diagnostic global : max des fractions croisées
            max_cross = 0.0
            for d in decomp.values():
                max_cross = max(max_cross, abs(d["frac_sS_of_num"]), abs(d["frac_Ss_of_num"]))

            # cos SS et cos ss (cos des composantes seules, à titre de référence)
            print(f"\n  cos entre composantes pures (référence) :")
            print(f"  {'paire':<22} {'cos(S,S)':>10} {'cos(s,s)':>10}")
            for i, li in enumerate(labels):
                for j in range(i+1, len(labels)):
                    lj = labels[j]
                    cos_SS = cos_pair(Shared[li], Shared[lj])
                    cos_ss = cos_pair(Specific[li], Specific[lj])
                    print(f"  {li}-{lj:<14} {cos_SS:>+10.4f} {cos_ss:>+10.4f}")

            print(f"\n  Diagnostic :")
            print(f"  max |<Shared_v, Specific_v>| = {max_ortho:.4e} (doit être ~0)")
            print(f"  max |frac_sS| ou |frac_Ss|  = {max_cross:.4f}")

            all_decompositions[f"{name}_{mode}"] = {
                "decomp_pairs": decomp,
                "max_orthogonality_residual": float(max_ortho),
                "max_cross_fraction": float(max_cross),
            }

    # Synthèse
    print(f"\n=== Synthèse ===\n")
    print(f"  Pour chaque observable, contribution moyenne SS et ss AU COS (pas au num) :")
    print(f"  contrib_cos_SS = SS / (||O_i|| ||O_j||)")
    print(f"  contrib_cos_ss = ss / (||O_i|| ||O_j||)")
    print(f"  cos_total = contrib_cos_SS + contrib_cos_ss (puisque les termes croisés sont nuls)\n")
    print(f"  {'observable':<28} {'<contrib_SS>':>14} {'<contrib_ss>':>14} {'min cos':>10}")

    summary = {}
    for key, data in all_decompositions.items():
        pairs = data["decomp_pairs"]
        contrib_SS_list = [p["contrib_cos_SS"] for p in pairs.values()]
        contrib_ss_list = [p["contrib_cos_ss"] for p in pairs.values()]
        cos_ss_pure_list = [p["cos_ss_pure"] for p in pairs.values()]
        cos_list = [p["cos_total"] for p in pairs.values()]
        mean_contrib_SS = float(np.mean(contrib_SS_list))
        mean_contrib_ss = float(np.mean(contrib_ss_list))
        mean_cos_ss_pure = float(np.mean(cos_ss_pure_list))
        min_cos_ss_pure = float(min(cos_ss_pure_list))
        max_cos_ss_pure = float(max(cos_ss_pure_list))
        min_cos = float(min(cos_list))
        summary[key] = {
            "mean_contrib_cos_SS": mean_contrib_SS,
            "mean_contrib_cos_ss": mean_contrib_ss,
            "mean_cos_ss_pure": mean_cos_ss_pure,
            "min_cos_ss_pure": min_cos_ss_pure,
            "max_cos_ss_pure": max_cos_ss_pure,
            "min_cos_observed": min_cos,
            "max_cross_fraction": data["max_cross_fraction"],
        }
        print(f"  {key:<28} {mean_contrib_SS:>+14.4f} {mean_contrib_ss:>+14.4f} {min_cos:>+10.4f}")

    print(f"\n  Cos entre composantes spécifiques pures (signe = signe de l'antagonisme) :")
    print(f"  {'observable':<28} {'<cos(s,s)>':>14} {'min':>10} {'max':>10}")
    for key, data in all_decompositions.items():
        s = summary[key]
        print(f"  {key:<28} {s['mean_cos_ss_pure']:>+14.4f} "
              f"{s['min_cos_ss_pure']:>+10.4f} {s['max_cos_ss_pure']:>+10.4f}")

    # Identifier qui porte le plus de spécifique
    print(f"\n=== Variantes les plus 'spécifiques' (norme ||Specific|| / ||O||) ===\n")
    spec_data = {}
    for name in obs_names:
        for mode in ["brut", "centré"]:
            key = f"{name}_{mode}"
            # Reconstruire les données pour calculer ce diagnostic
            O = {}
            for lab in labels:
                serie = get_obs(residuals[lab]["r_psi"], residuals[lab]["r_h"], name)
                if mode == "centré":
                    serie = serie - serie.mean()
                O[lab] = serie
            O_bar = np.mean([O[lab] for lab in labels], axis=0)
            O_bar_norm_sq = float(np.dot(O_bar, O_bar))
            print(f"  {key}")
            for lab in labels:
                if O_bar_norm_sq > 1e-30:
                    a = float(np.dot(O[lab], O_bar) / O_bar_norm_sq)
                else:
                    a = 0.0
                Shared_lab = a * O_bar
                Specific_lab = O[lab] - Shared_lab
                frac_spec_norm = np.linalg.norm(Specific_lab) / max(np.linalg.norm(O[lab]), 1e-30)
                print(f"    {lab:<12} ||Specific||/||O|| = {frac_spec_norm:.4f}")

    output = {
        "obs_names": obs_names,
        "labels": labels,
        "decompositions": all_decompositions,
        "summary": summary,
    }
    with open("/home/claude/mcq_v4/6d_zeta_decompose_cos_v2.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
