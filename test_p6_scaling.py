"""
6d-γ — La composante extra_6 est-elle linéaire ou non linéaire ?

Test : refaire P6 à plusieurs amplitudes (mêmes forme géométrique,
strength variable) et regarder comment extra_6 dépend de l'amplitude
de la perturbation.

- ||extra_6|| ∝ ||r_6|| → lecture linéaire : γ est une coordonnée
  indépendante du même statut que α et β
- ||extra_6|| ∝ ||r_6||² → lecture non linéaire : γ est correction
  d'ordre supérieur, fonction des coordonnées dominantes
- Autre loi → cas plus complexe

L'observation de la symétrisation (P6 antisymétrique → extra_6 pair)
penche vers non linéaire, mais cela reste à confirmer empiriquement.

Amplitudes testées : s_P6_calibrated × {0.25, 0.5, 1.0, 2.0, 3.0}
(une seule géométrie de P6, strength varie)
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


def P1prime(psi, strength=0.05, radius=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= radius: factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

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

def P5_neighbors_only(psi, strength):
    factor = np.ones_like(psi)
    neighbors = [(1,2,2), (3,2,2), (2,1,2), (2,3,2), (2,2,1), (2,2,3)]
    for (i,j,k) in neighbors:
        factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

def P6_face_dipole(psi, strength):
    factor = np.ones_like(psi)
    factor[0, :, :] += strength
    factor[4, :, :] -= strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    def input_amp(P, s): return float(np.linalg.norm(P(psi_base, s) - psi_base))
    amp_target = input_amp(P1prime, 0.05)
    s_P5 = brentq(lambda s: input_amp(P5_neighbors_only, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)

    # Construire la base à partir de P1-P5
    perturbations_15 = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
        ("P5", lambda p: P5_neighbors_only(p, strength=s_P5)),
    ]
    r_15 = []
    for name, P in perturbations_15:
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        r_15.append(psi_R - psi_base)

    # On utilise la matrice des r_1..r_5 comme base de référence
    R_base = np.column_stack([r.flatten() for r in r_15])

    # Tester P6 à plusieurs amplitudes
    # s_P6 standard correspond à amplitude_entree = amp_target (≈ ||P1||)
    s_P6_standard = brentq(lambda s: input_amp(P6_face_dipole, s) - amp_target,
                           1e-4, 0.99, xtol=1e-6)
    scale_factors = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

    print(f"=== Test de scaling de extra_6 ===\n")
    print(f"  s_P6 standard (amplitude entrée = amp_target) : {s_P6_standard:.5f}\n")
    print(f"  {'scale':>7} {'s_P6':>10} {'amp_entree':>14} {'||r_6||':>14} "
          f"{'||extra||':>14} {'extra/||r||':>14}")

    results = []
    for sf in scale_factors:
        s_P6 = s_P6_standard * sf
        amp_in = input_amp(P6_face_dipole, s_P6)
        psi_p = P6_face_dipole(psi_base, s_P6)
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        r_6 = psi_R - psi_base
        r_6_flat = r_6.flatten()

        # Projection sur le span de (r_1..r_5)
        coefs, _, _, _ = np.linalg.lstsq(R_base, r_6_flat, rcond=None)
        r_6_pred = R_base @ coefs
        extra = r_6_flat - r_6_pred

        r_norm = float(np.linalg.norm(r_6_flat))
        extra_norm = float(np.linalg.norm(extra))
        ratio = extra_norm / r_norm
        results.append({
            "scale": sf,
            "s_P6": s_P6,
            "amp_input": amp_in,
            "r_6_norm": r_norm,
            "extra_norm": extra_norm,
            "ratio_extra_to_r": ratio,
        })
        print(f"  {sf:>7.2f} {s_P6:>10.5f} {amp_in:>14.4e} {r_norm:>14.4e} "
              f"{extra_norm:>14.4e} {ratio:>14.4f}")

    # Analyse du scaling
    print(f"\n=== Analyse du scaling ===\n")
    amps = np.array([r["amp_input"] for r in results])
    r_norms = np.array([r["r_6_norm"] for r in results])
    extra_norms = np.array([r["extra_norm"] for r in results])

    # Fit log-log : log(extra) = p · log(amp) + cst
    log_amps = np.log(amps)
    log_extras = np.log(extra_norms)
    slope_amp = float(np.cov(log_amps, log_extras, ddof=0)[0,1] / np.var(log_amps))
    log_rs = np.log(r_norms)
    slope_r = float(np.cov(log_rs, log_extras, ddof=0)[0,1] / np.var(log_rs))

    print(f"  Exposant de scaling de ||extra|| vs ||entrée|| (P6) : {slope_amp:.4f}")
    print(f"  Exposant de scaling de ||extra|| vs ||r_6|| (résidu) : {slope_r:.4f}")
    print(f"\n  Interprétation :")
    print(f"  - exposant ≈ 1.0 → linéaire (γ est une coordonnée indépendante)")
    print(f"  - exposant ≈ 2.0 → quadratique (γ est correction d'ordre 2)")
    print(f"  - exposant intermédiaire → loi de puissance non triviale")

    if 0.9 < slope_amp < 1.1:
        verdict = "LINÉAIRE : extra dépend linéairement de l'amplitude de P6."
        verdict += " Compatible avec γ comme nouvelle coordonnée indépendante."
    elif 1.8 < slope_amp < 2.2:
        verdict = "QUADRATIQUE : extra ∝ amp². Compatible avec γ comme"
        verdict += " correction d'ordre supérieur des coordonnées dominantes."
    else:
        verdict = f"EXPOSANT INTERMÉDIAIRE ({slope_amp:.2f}). Cas plus complexe."
    print(f"\n  Verdict : {verdict}")

    # Et le ratio extra/||r|| en fonction de l'amplitude
    print(f"\n=== Ratio extra/||r|| en fonction de l'amplitude ===")
    print(f"  Si linéaire (extra ∝ amp et r ∝ amp) : ratio constant")
    print(f"  Si quadratique (extra ∝ amp², r ∝ amp) : ratio ∝ amp\n")
    for r in results:
        print(f"  amp={r['amp_input']:.4e} : ratio = {r['ratio_extra_to_r']:.4f}")

    # Fit log-log du ratio
    ratios = np.array([r["ratio_extra_to_r"] for r in results])
    log_ratios = np.log(ratios)
    slope_ratio_amp = float(np.cov(log_amps, log_ratios, ddof=0)[0,1] / np.var(log_amps))
    print(f"\n  Exposant du ratio vs amp : {slope_ratio_amp:.4f}")
    print(f"  (linéaire → 0, quadratique → 1, autre → entre)")

    output = {
        "s_P6_standard": s_P6_standard,
        "scale_factors": scale_factors,
        "results": results,
        "exponent_extra_vs_amp": slope_amp,
        "exponent_extra_vs_r": slope_r,
        "exponent_ratio_vs_amp": slope_ratio_amp,
        "verdict": verdict,
    }
    out_path = "/home/claude/mcq_v4/p6_scaling_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
