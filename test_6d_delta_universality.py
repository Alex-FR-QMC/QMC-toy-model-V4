"""
6d-δ — Universalité du mode limite vs dépendance à P'.

Question : le vecteur limite du résidu r_τ=0 à Δt long est-il
universel pour le régime β=60, ou dépend-il de la géométrie de P' ?

Méthode : reprendre le calcul du résidu à τ=0 avec 4 variantes de P',
toutes calibrées à la même amplitude (10% de P6) et toutes vérifiées
comme ne créant pas de mode lent parasite.

Variantes testées :
- P'_étroite : gaussienne, sigma_p = 0.5
- P'_standard : gaussienne, sigma_p = 0.8 (référence)
- P'_large : gaussienne, sigma_p = 1.5
- P'_annulaire : couche périphérique (r entre 1.5 et 2.5)

Δt = 800 (où on sait que le résidu sature et la direction converge).

Préinscription :
- (U) cos > 0.95 entre toutes les variantes gaussiennes : mode universel
- (D) cos < 0.5 : mode dépendant de P'
- (I) entre les deux : intermédiaire
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


def compute_delta(psi_start, h_start, P_prime_fn, s_prime,
                   D, beta, gamma_v, h0, dt, n_dt):
    psis_ref, hs_ref = evolve_with_trajectory(
        psi_start.copy(), h_start.copy(),
        D, beta, gamma_v, h0, dt, n_dt)
    psi_pp = P_prime_fn(psi_start.copy(), s_prime)
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
    n_dt_long = int(800.0 / dt)
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    # Calibrer P6
    amp_P1prime_std = float(np.linalg.norm(P1prime_std(psi_base) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    target_amp = 0.1 * amp_P6

    # Calibrer chaque variante de P' à amplitude = 10% de P6
    variants = []

    def amp_gauss(s, sigma_p):
        return float(np.linalg.norm(P_prime_gauss(psi_base, s, sigma_p) - psi_base))

    def amp_annular(s):
        return float(np.linalg.norm(P_prime_annular(psi_base, s) - psi_base))

    for label, sigma_p in [("étroite", 0.5), ("standard", 0.8), ("large", 1.5)]:
        s_calib = brentq(lambda s: amp_gauss(s, sigma_p) - target_amp,
                         1e-6, 1.0, xtol=1e-8)
        variants.append({
            "label": label,
            "type": "gauss",
            "sigma_p": sigma_p,
            "s_calib": s_calib,
            "P_fn": (lambda psi, s, sp=sigma_p: P_prime_gauss(psi, s, sp)),
        })

    s_calib_ann = brentq(lambda s: amp_annular(s) - target_amp,
                          1e-6, 5.0, xtol=1e-8)
    variants.append({
        "label": "annulaire",
        "type": "annular",
        "sigma_p": None,
        "s_calib": s_calib_ann,
        "P_fn": (lambda psi, s: P_prime_annular(psi, s)),
    })

    print(f"=== Universalité du mode limite : calibrages ===\n")
    print(f"  amp_P6 = {amp_P6:.4e}")
    print(f"  target_amp = {target_amp:.4e}")
    print(f"  {'variante':<12} {'s_calib':>12} {'amp obtenue':>14}")
    for v in variants:
        if v["type"] == "gauss":
            amp_obt = amp_gauss(v["s_calib"], v["sigma_p"])
        else:
            amp_obt = amp_annular(v["s_calib"])
        print(f"  {v['label']:<12} {v['s_calib']:>12.6f} {amp_obt:>14.4e}")

    # Vérifier que chaque variante ne crée pas de mode lent parasite
    # (mesure extra à t=200 avec base P1-P5 ; epsilon_noise comme dans calibration)
    print(f"\n=== Contrôle : pas de mode lent parasite par variante ===\n")

    # Reconstruire base P1-P5 (comme dans tests précédents)
    def P1prime(psi, strength=0.05):
        return P1prime_std(psi, strength)
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
    def P5_n(psi, strength):
        factor = np.ones_like(psi)
        for (i,j,k) in [(1,2,2),(3,2,2),(2,1,2),(2,3,2),(2,2,1),(2,2,3)]:
            factor[i,j,k] += strength
        p = psi * factor
        return p / p.sum()
    s_P5 = brentq(lambda s: float(np.linalg.norm(P5_n(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 1.0, xtol=1e-6)
    perts_15 = [P1prime, lambda p: P2prime(p, 0.012789),
                P3prime, lambda p: P4(p, 0.03187),
                lambda p: P5_n(p, s_P5)]
    r_15 = []
    for P in perts_15:
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        r_15.append(psi_R - psi_base)
    R_base = np.column_stack([r.flatten() for r in r_15])

    epsilon_noise = 1e-15
    n_t200 = int(200.0 / dt)
    print(f"  {'variante':<12} {'||extra@200||':>16} {'extra/r':>14} {'parasite ?':>14}")
    for v in variants:
        # Appliquer P' seul, relaxer, mesurer extra
        psi_pp = v["P_fn"](psi_base.copy(), v["s_calib"])
        h_pp = h_base.copy()
        psi_R, h_R = evolve(psi_pp, h_pp, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_t200)
        r_pp_flat = (psi_R - psi_base).flatten()
        coefs, _, _, _ = np.linalg.lstsq(R_base, r_pp_flat, rcond=None)
        extra = r_pp_flat - R_base @ coefs
        extra_norm = float(np.linalg.norm(extra))
        r_norm = float(np.linalg.norm(r_pp_flat))
        ratio = extra_norm / r_norm if r_norm > 1e-30 else 0.0
        parasite = "non" if extra_norm < epsilon_noise else "À VÉRIFIER"
        v["extra_norm_at_200"] = extra_norm
        v["ratio_extra_to_r"] = ratio
        v["parasite_flag"] = parasite
        print(f"  {v['label']:<12} {extra_norm:>16.4e} {ratio:>14.4e} {parasite:>14}")

    # Préparer l'état P6(τ=0) (commun à toutes les variantes)
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6,
                               D, beta_lock, gamma_v, h0, dt, n_short)
    # État P6(3000)
    n_3000 = int(3000.0 / dt)
    psi_tau3000, h_tau3000 = evolve(psi_tau0.copy(), h_tau0.copy(),
                                     D, beta_lock, gamma_v, h0, dt, n_3000)

    # Pour chaque variante, calculer r_τ=0 à Δt=800
    print(f"\n=== Calcul du résidu r_τ=0 à Δt=800 pour chaque variante ===\n")
    print(f"  {'variante':<12} {'||r_resid||':>14} {'a_τ':>10}")

    resid_vectors = {}  # variante -> (r_resid normalisé)
    for v in variants:
        # Δ_τ=0 avec cette P'
        d_psi_tau0, d_h_tau0 = compute_delta(
            psi_tau0, h_tau0, v["P_fn"], v["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        # Δ_3000 avec la même P' (référence asymptotique)
        d_psi_ref, d_h_ref = compute_delta(
            psi_tau3000, h_tau3000, v["P_fn"], v["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        # Aplatir et décomposer
        d_flat = np.concatenate([d_psi_tau0.flatten(), d_h_tau0.flatten()])
        ref_flat = np.concatenate([d_psi_ref.flatten(), d_h_ref.flatten()])
        ref_norm_sq = float(np.dot(ref_flat, ref_flat))
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq) if ref_norm_sq > 1e-30 else 0.0
        resid = d_flat - a_tau * ref_flat
        resid_norm = float(np.linalg.norm(resid))
        # Stocker en normalisé pour comparer entre variantes
        if resid_norm > 1e-30:
            resid_normed = resid / resid_norm
        else:
            resid_normed = resid
        # Aussi stocker la projection psi à l'instant final pour visualisation
        r_psi_final = d_psi_tau0[-1] - a_tau * d_psi_ref[-1]
        proj_x_final = np.abs(r_psi_final).sum(axis=(1,2))
        if proj_x_final.sum() > 1e-30:
            proj_x_norm = proj_x_final / proj_x_final.sum()
        else:
            proj_x_norm = proj_x_final
        v["resid_norm"] = resid_norm
        v["a_tau"] = a_tau
        v["resid_normed"] = resid_normed
        v["proj_x_final_normalized"] = proj_x_norm.tolist()
        print(f"  {v['label']:<12} {resid_norm:>14.4e} {a_tau:>10.6f}")

    # === Cosinus entre les vecteurs résiduels normalisés ===
    print(f"\n=== Cosinus entre vecteurs résidu normalisés ===\n")
    pairs = []
    print(f"  {'pair':>30} {'cos':>10}")
    for i in range(len(variants)):
        for j in range(i+1, len(variants)):
            v1 = variants[i]
            v2 = variants[j]
            cos = float(np.dot(v1["resid_normed"], v2["resid_normed"]))
            pairs.append({"v1": v1["label"], "v2": v2["label"], "cos": cos})
            label = f"{v1['label']} ↔ {v2['label']}"
            print(f"  {label:>30} {cos:>+10.4f}")

    # Projections x finales pour visualisation
    print(f"\n=== Projection x normalisée à l'instant final (Δt=800) ===\n")
    for v in variants:
        px = np.array(v["proj_x_final_normalized"])
        print(f"  {v['label']:<12} : {px}")

    # === Verdict ===
    print(f"\n=== Verdict ===\n")
    # Cosinus entre les trois gaussiennes
    gauss_pairs = [p for p in pairs
                    if p["v1"] in ["étroite", "standard", "large"]
                    and p["v2"] in ["étroite", "standard", "large"]]
    min_gauss_cos = min(p["cos"] for p in gauss_pairs)
    print(f"  cos minimal entre gaussiennes : {min_gauss_cos:.4f}")
    # Cosinus annulaire vs gaussiennes
    ann_pairs = [p for p in pairs
                  if p["v1"] == "annulaire" or p["v2"] == "annulaire"]
    if ann_pairs:
        ann_cos = [p["cos"] for p in ann_pairs]
        print(f"  cos annulaire vs gaussiennes : {ann_cos}")

    if min_gauss_cos > 0.95:
        verdict = ("(U) UNIVERSEL parmi gaussiennes : les trois variantes "
                   "gaussiennes produisent le même mode limite (cos > 0.95). "
                   "Le mode est intrinsèque au régime, pas à la forme de P'.")
    elif min_gauss_cos < 0.5:
        verdict = ("(D) DÉPENDANT : les variantes gaussiennes produisent "
                   "des modes limites différents (cos < 0.5). Le mode dépend "
                   "fortement de la forme de P'.")
    else:
        verdict = (f"(I) INTERMÉDIAIRE : cos minimal {min_gauss_cos:.4f} entre 0.5 et 0.95.")
    print(f"\n  {verdict}")

    # Nettoyer avant sauvegarde
    for v in variants:
        del v["P_fn"]
        v["resid_normed"] = "stored separately (vector)"

    output = {
        "amp_P6": amp_P6,
        "target_amp": target_amp,
        "epsilon_noise": epsilon_noise,
        "variants": variants,
        "pairwise_cos": pairs,
        "min_gauss_cos": float(min_gauss_cos),
        "verdict": verdict,
    }
    with open("/home/claude/mcq_v4/6d_delta_universality.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
