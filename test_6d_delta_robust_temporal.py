"""
6d-δ — Vérification de la robustesse de l'universalité temporelle.

L'universalité observée sur ψ_temporel_moyen = mean(r_psi) pourrait
venir d'annulations entre cellules positives et négatives. Il faut
vérifier sur des mesures non sensibles au signe.

Pour chaque variante, calculer trois observables temporelles :
- mean(r_psi) : moyenne signée (déjà calculée)
- mean(|r_psi|) : moyenne des valeurs absolues
- norm_espace(r_psi) : sqrt(sum sur les cellules de r_psi²) à chaque t

Puis matrices de cosinus 4x4 entre variantes pour chaque observable.

Idem pour h.

Si toutes restent > 0.99, l'universalité temporelle est robuste.
Si seule la moyenne signée est universelle, c'est un artefact
d'annulation.
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

    # Calculer les résidus pour chaque variante
    print(f"=== Calcul des résidus r par variante ===\n")
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
        residuals[v["label"]] = (r_psi, r_h)
        print(f"  {v['label']:<12} : ||r_psi|| = {np.linalg.norm(r_psi):.4e}, "
              f"||r_h|| = {np.linalg.norm(r_h):.4e}")

    # 3 observables temporelles pour chaque variante
    print(f"\n=== Trois observables temporelles ===\n")
    observables = {}
    for label, (r_psi, r_h) in residuals.items():
        obs = {}
        # ψ
        obs["psi_temp_mean_signed"] = np.mean(r_psi, axis=(1,2,3))
        obs["psi_temp_mean_abs"] = np.mean(np.abs(r_psi), axis=(1,2,3))
        obs["psi_temp_norm"] = np.sqrt(np.sum(r_psi**2, axis=(1,2,3)))
        # h
        obs["h_temp_mean_signed"] = np.mean(r_h, axis=(1,2,3))
        obs["h_temp_mean_abs"] = np.mean(np.abs(r_h), axis=(1,2,3))
        obs["h_temp_norm"] = np.sqrt(np.sum(r_h**2, axis=(1,2,3)))
        observables[label] = obs

    # Amplitudes pour diagnostic
    obs_names = ["psi_temp_mean_signed", "psi_temp_mean_abs", "psi_temp_norm",
                 "h_temp_mean_signed", "h_temp_mean_abs", "h_temp_norm"]
    labels = ["étroite", "standard", "large", "annulaire"]

    print(f"  Amplitudes ||·|| de chaque observable :\n")
    print(f"  {'observable':<25} " + " ".join(f"{lab:>12}" for lab in labels))
    for name in obs_names:
        amps = [np.linalg.norm(observables[lab][name]) for lab in labels]
        print(f"  {name:<25} " + " ".join(f"{a:>12.4e}" for a in amps))

    # 6 matrices de cosinus
    print(f"\n=== Matrices de cosinus entre variantes ===\n")
    cos_matrices = {}
    for name in obs_names:
        print(f"\n  --- {name} ---")
        mat = np.zeros((4, 4))
        print("          " + " ".join(f"{lab:>10}" for lab in labels))
        for i, lab_i in enumerate(labels):
            row = f"  {lab_i:<10}"
            for j, lab_j in enumerate(labels):
                c = cos_pair(observables[lab_i][name], observables[lab_j][name])
                mat[i, j] = c
                row += f" {c:>+10.4f}"
            print(row)
        cos_matrices[name] = mat.tolist()

    # Synthèse
    print(f"\n=== Lecture synthétique ===\n")
    print(f"  {'observable':<25} {'min cos hors diag':>20} {'verdict':>20}")
    summary = {}
    for name in obs_names:
        mat = np.array(cos_matrices[name])
        off_diag = mat[~np.eye(4, dtype=bool)]
        min_c = float(off_diag.min())
        if min_c > 0.99:
            verdict = "robustement universel"
        elif min_c > 0.95:
            verdict = "universel"
        elif min_c > 0.5:
            verdict = "intermédiaire"
        else:
            verdict = "dépendant"
        summary[name] = {"min_cos": min_c, "verdict": verdict}
        print(f"  {name:<25} {min_c:>+20.4f} {verdict:>20}")

    # Diagnostic d'annulation : ratio mean_signed / mean_abs
    print(f"\n=== Diagnostic d'annulation ===\n")
    print(f"  Si universalité de mean_signed vient d'annulations, alors")
    print(f"  ||mean_signed|| << ||mean_abs|| (annulations entre cellules)")
    print(f"\n  {'variante':<12} {'ψ:|mean_s|/|mean_a|':>22} {'h:|mean_s|/|mean_a|':>22}")
    annul_ratios = {}
    for lab in labels:
        # On compare les normes des deux séries temporelles
        ms_psi = np.linalg.norm(observables[lab]["psi_temp_mean_signed"])
        ma_psi = np.linalg.norm(observables[lab]["psi_temp_mean_abs"])
        ms_h = np.linalg.norm(observables[lab]["h_temp_mean_signed"])
        ma_h = np.linalg.norm(observables[lab]["h_temp_mean_abs"])
        ratio_psi = ms_psi / max(ma_psi, 1e-30)
        ratio_h = ms_h / max(ma_h, 1e-30)
        annul_ratios[lab] = {"ratio_psi": ratio_psi, "ratio_h": ratio_h}
        print(f"  {lab:<12} {ratio_psi:>22.4f} {ratio_h:>22.4f}")
    print(f"\n  (proche de 1 : pas d'annulation ; proche de 0 : forte annulation)")

    output = {
        "cos_matrices": cos_matrices,
        "summary": summary,
        "annulation_ratios": annul_ratios,
    }
    with open("/home/claude/mcq_v4/6d_delta_robust_temporal.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
