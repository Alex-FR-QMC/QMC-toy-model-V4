"""
6d-δ — Localisation de l'universalité : décomposition moyenne/fluctuation.

À partir des résidus r_τ=0 à Δt=800 calculés pour 4 variantes de P'
(étroite, standard, large, annulaire), on construit pour chacune
8 observables :

ψ_spatial_moyen        : mean_t r_ψ(t, i, j, k)
ψ_spatial_fluctuation  : std_t r_ψ(t, i, j, k)
ψ_temporel_moyen       : mean_{i,j,k} r_ψ(t, i, j, k)
ψ_temporel_fluctuation : std_{i,j,k} r_ψ(t, i, j, k)
+ idem pour h

Puis 8 matrices de cosinus 4×4 entre variantes.

Lecture :
- matrice avec tous coefficients > 0.95 → universalité dans cette composante
- matrice avec coefficients < 0.5 → dépendance à P' dans cette composante
- intermédiaire → cas à examiner

Aucune autre opération. Pas de symétrisation, pas de projection x/y/z,
pas de définition de secteur.
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
    """Retourne Δ_psi, Δ_h sous forme (n_dt+1, 5, 5, 5)."""
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

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    # Calibrer P6
    amp_P1prime_std = float(np.linalg.norm(P1prime_std(psi_base) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    target_amp = 0.1 * amp_P6

    # Calibrer variantes
    variants = []
    for label, sigma_p in [("étroite", 0.5), ("standard", 0.8), ("large", 1.5)]:
        s_calib = brentq(lambda s: float(np.linalg.norm(P_prime_gauss(psi_base, s, sigma_p) - psi_base))
                          - target_amp, 1e-6, 1.0, xtol=1e-8)
        variants.append({
            "label": label, "sigma_p": sigma_p, "s_calib": s_calib,
            "P_fn": (lambda psi, s, sp=sigma_p: P_prime_gauss(psi, s, sp)),
        })
    s_ann = brentq(lambda s: float(np.linalg.norm(P_prime_annular(psi_base, s) - psi_base))
                    - target_amp, 1e-6, 5.0, xtol=1e-8)
    variants.append({
        "label": "annulaire", "sigma_p": None, "s_calib": s_ann,
        "P_fn": (lambda psi, s: P_prime_annular(psi, s)),
    })

    print(f"=== Calibrages confirmés (cf. test précédent) ===")
    for v in variants:
        print(f"  {v['label']:<12} s_calib = {v['s_calib']:.6f}")

    # États P6(0) et P6(3000)
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6,
                                D, beta_lock, gamma_v, h0, dt, n_short)
    n_3000 = int(3000.0 / dt)
    psi_tau3000, h_tau3000 = evolve(psi_tau0.copy(), h_tau0.copy(),
                                     D, beta_lock, gamma_v, h0, dt, n_3000)

    # Pour chaque variante : calculer le résidu r = Δ_τ=0 - a · Δ_3000
    # Stocker les composantes ψ et h séparément.
    print(f"\n=== Calcul des résidus r_τ=0 (Δt=800) par variante ===\n")
    residuals = {}  # label -> (r_psi, r_h) avec r_psi shape (n_dt+1, 5, 5, 5)

    for v in variants:
        d_psi_tau0, d_h_tau0 = compute_delta_full(
            psi_tau0, h_tau0, v["P_fn"], v["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_psi_ref, d_h_ref = compute_delta_full(
            psi_tau3000, h_tau3000, v["P_fn"], v["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        # a global sur ψ+h concaténés
        d_flat = np.concatenate([d_psi_tau0.flatten(), d_h_tau0.flatten()])
        ref_flat = np.concatenate([d_psi_ref.flatten(), d_h_ref.flatten()])
        ref_norm_sq = float(np.dot(ref_flat, ref_flat))
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq) if ref_norm_sq > 1e-30 else 0.0
        r_psi = d_psi_tau0 - a_tau * d_psi_ref
        r_h = d_h_tau0 - a_tau * d_h_ref
        residuals[v["label"]] = (r_psi, r_h)
        print(f"  {v['label']:<12} : ||r_psi|| = {np.linalg.norm(r_psi):.4e}, "
              f"||r_h|| = {np.linalg.norm(r_h):.4e}, a = {a_tau:.6f}")

    # Construire les 8 observables pour chaque variante
    print(f"\n=== Décomposition moyenne / fluctuation ===\n")
    observables = {}  # label -> dict avec les 8 observables

    for label, (r_psi, r_h) in residuals.items():
        obs = {}
        # ψ spatial moyen et fluctuation
        obs["psi_spatial_moyen"] = np.mean(r_psi, axis=0)  # (5,5,5)
        obs["psi_spatial_fluct"] = np.std(r_psi, axis=0)   # (5,5,5)
        # ψ temporel moyen et fluctuation
        obs["psi_temporel_moyen"] = np.mean(r_psi, axis=(1,2,3))  # (n_t,)
        obs["psi_temporel_fluct"] = np.std(r_psi, axis=(1,2,3))   # (n_t,)
        # h spatial moyen et fluctuation
        obs["h_spatial_moyen"] = np.mean(r_h, axis=0)
        obs["h_spatial_fluct"] = np.std(r_h, axis=0)
        # h temporel moyen et fluctuation
        obs["h_temporel_moyen"] = np.mean(r_h, axis=(1,2,3))
        obs["h_temporel_fluct"] = np.std(r_h, axis=(1,2,3))
        observables[label] = obs

    # Diagnostic d'amplitude de chaque observable
    print(f"  Amplitudes ||·|| de chaque observable par variante :\n")
    print(f"  {'observable':<26} {'étroite':>12} {'standard':>12} "
          f"{'large':>12} {'annulaire':>12}")
    obs_names = ["psi_spatial_moyen", "psi_spatial_fluct",
                 "psi_temporel_moyen", "psi_temporel_fluct",
                 "h_spatial_moyen", "h_spatial_fluct",
                 "h_temporel_moyen", "h_temporel_fluct"]
    labels = ["étroite", "standard", "large", "annulaire"]
    for name in obs_names:
        amps = [np.linalg.norm(observables[lab][name]) for lab in labels]
        print(f"  {name:<26} " + " ".join(f"{a:>12.4e}" for a in amps))

    # 8 matrices de cosinus 4x4
    print(f"\n=== Matrices de cosinus entre variantes ===\n")
    cos_matrices = {}
    for name in obs_names:
        print(f"\n  --- {name} ---")
        mat = np.zeros((4, 4))
        header = "          " + " ".join(f"{lab:>10}" for lab in labels)
        print(header)
        for i, lab_i in enumerate(labels):
            row_str = f"  {lab_i:<10}"
            for j, lab_j in enumerate(labels):
                c = cos_pair(observables[lab_i][name], observables[lab_j][name])
                mat[i, j] = c
                row_str += f" {c:>+10.4f}"
            print(row_str)
        cos_matrices[name] = mat.tolist()

    # Lecture synthétique : pour chaque observable, donner le min des cosinus
    # hors diagonale (le plus discriminant) et le max
    print(f"\n=== Lecture synthétique : min/max cosinus hors diagonale ===\n")
    print(f"  {'observable':<26} {'min cos':>10} {'max cos':>10} {'lecture':>30}")
    summary = {}
    for name in obs_names:
        mat = np.array(cos_matrices[name])
        # Hors diagonale
        off_diag = mat[~np.eye(4, dtype=bool)]
        min_c = float(off_diag.min())
        max_c = float(off_diag.max())
        if min_c > 0.95:
            lecture = "universel"
        elif min_c < 0.5:
            lecture = "dépendant"
        else:
            lecture = "intermédiaire"
        summary[name] = {"min": min_c, "max": max_c, "lecture": lecture}
        print(f"  {name:<26} {min_c:>+10.4f} {max_c:>+10.4f} {lecture:>30}")

    # Sauvegarder en JSON (les arrays sont convertis en listes pour les
    # observables, sinon trop volumineux ; ici on garde juste les matrices
    # de cos et le résumé)
    output = {
        "amp_P6": amp_P6,
        "target_amp": target_amp,
        "variants_calib": [
            {"label": v["label"], "sigma_p": v["sigma_p"], "s_calib": v["s_calib"]}
            for v in variants
        ],
        "cos_matrices": cos_matrices,
        "summary": summary,
    }
    with open("/home/claude/mcq_v4/6d_delta_localize.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
