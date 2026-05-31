"""
6d-γ contact 10 — couplage géométrie/intensité (continuum, pas verdict).

Contact 9 : P2' (couronne périphérique) sort du corridor d'amplitude
(ratio 3.8 vs P2 originale). Question (Alex) : peut-on modifier une
géométrie sans modifier sa puissance effective dans ce régime ?

Reformulation (Alex) : ne pas trancher (a) "séparables" vs (b)
"non séparables". Mesurer l'ampleur du COUPLAGE RÉSIDUEL après
calibration d'entrée. Continuum, pas verdict.

Protocole :
1. P2 originale (coin (0,0,0))
2. P2' couronne périphérique (non calibrée)
3. P2'' couronne périphérique avec strength réduit pour caler
   ||P2''(ψ) - ψ|| ≈ ||P2(ψ) - ψ||
4. Mesurer pour chacune ||R - état_base|| après court relax

Lecture (sur continuum) :
- ratio1 = ||R(P2')|| / ||R(P2)|| : ampleur de la différence non calibrée
- ratio2 = ||R(P2'')|| / ||R(P2)|| : ampleur résiduelle après calibration
- (ratio2 - 1) / (ratio1 - 1) : fraction de couplage qui PERSISTE après
  calibration (entre 0 = totalement séparable et 1 = couplage non
  réductible par calibration)

Aucun seuil. Position lue sur le continuum.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json
from scipy.optimize import brentq

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mcq_v4.factorial_6d import N_AXIS, DX, cfl_dt_max  # noqa: E402
from mcq_v4.factorial_6d.engine import (  # noqa: E402
    compute_diffusion_flux, compute_divergence,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero  # noqa: E402


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
                r2 = ((coords[i] - c)**2 + (coords[j] - c)**2 +
                      (coords[k] - c)**2)
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()


def P2_corner(psi, strength=0.05):
    """P2 originale : concentration sur le coin (0,0,0)."""
    coords = np.arange(N_AXIS) * DX
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i]**2 + coords[j]**2 + coords[k]**2)
                factor[i, j, k] += strength * np.exp(-0.5 * r2 / 0.8**2)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P2_corona(psi, strength=0.05, radius_inner=2.0):
    """P2' couronne périphérique : boost à distance >= radius_inner."""
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i] - c)**2 + (coords[j] - c)**2 +
                            (coords[k] - c)**2)
                if r >= radius_inner:
                    factor[i, j, k] += strength
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def input_amplitude(P, psi, strength):
    """||P(ψ) - ψ|| pour un strength donné."""
    return float(np.linalg.norm(P(psi, strength) - psi))


def calibrate_strength(P, psi, target_amplitude, s_min=1e-6, s_max=1.0):
    """Trouve le strength tel que ||P(ψ) - ψ|| ≈ target_amplitude."""
    def f(s):
        return input_amplitude(P, psi, s) - target_amplitude
    return brentq(f, s_min, s_max, xtol=1e-8)


def response_amplitude(P, strength, psi_base, h_base,
                       D, beta, gamma, h0, dt, n_short):
    """Mesure ||R - état_base|| après application de P et court relax."""
    psi_p = P(psi_base.copy(), strength)
    h_p = h_base.copy()
    psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
    delta_psi = psi_R - psi_base
    delta_h = h_R - h_base
    return float(np.sqrt(np.linalg.norm(delta_psi)**2 +
                          np.linalg.norm(delta_h)**2))


def run():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))

    print(f"{'='*70}")
    print(f"6d-γ contact 10 — couplage géométrie/intensité")
    print(f"  Mesure de couplage résiduel après calibration d'entrée")
    print(f"  β=60, dt={dt:.5f}")
    print(f"{'='*70}\n")

    # État de base
    n_stab = int(50.0 / dt)
    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)
    print(f"  base : ψ_inhomo="
          f"{psi_base.max()/max(psi_base.min(),1e-30):.3f}, "
          f"h_min={h_base.min():.3e}")

    n_short = int(10.0 / dt)

    # --- Étape 1 : amplitudes d'entrée et de réponse pour P2 et P2' ---
    s_orig = 0.05
    amp_in_P2 = input_amplitude(P2_corner, psi_base, s_orig)
    amp_in_P2p = input_amplitude(P2_corona, psi_base, s_orig)
    amp_resp_P2 = response_amplitude(P2_corner, s_orig, psi_base, h_base,
                                      D, beta, gamma, h0, dt, n_short)
    amp_resp_P2p = response_amplitude(P2_corona, s_orig, psi_base, h_base,
                                       D, beta, gamma, h0, dt, n_short)

    print(f"\n  Amplitudes d'entrée et de réponse (strength=0.05) :")
    print(f"    P2  (coin)      : ||ΔP||={amp_in_P2:.4e}, "
          f"||ΔR||={amp_resp_P2:.4e}")
    print(f"    P2' (couronne)  : ||ΔP||={amp_in_P2p:.4e}, "
          f"||ΔR||={amp_resp_P2p:.4e}")
    ratio_in_non_calib = amp_in_P2p / amp_in_P2
    ratio_resp_non_calib = amp_resp_P2p / amp_resp_P2
    print(f"\n    ratio entrée non calibrée  : {ratio_in_non_calib:.4f}")
    print(f"    ratio réponse non calibrée : {ratio_resp_non_calib:.4f}")

    # --- Étape 2 : calibrer P2'' pour matcher l'entrée de P2 ---
    s_calib = calibrate_strength(P2_corona, psi_base, amp_in_P2)
    amp_in_P2pp = input_amplitude(P2_corona, psi_base, s_calib)
    amp_resp_P2pp = response_amplitude(P2_corona, s_calib, psi_base, h_base,
                                        D, beta, gamma, h0, dt, n_short)
    print(f"\n  P2'' = P2' calibrée à entrée = entrée de P2 :")
    print(f"    strength calibré = {s_calib:.6f} (au lieu de {s_orig})")
    print(f"    ||ΔP|| calibré = {amp_in_P2pp:.4e} (cible {amp_in_P2:.4e})")
    print(f"    ||ΔR|| à entrée calibrée = {amp_resp_P2pp:.4e}")

    ratio_resp_calib = amp_resp_P2pp / amp_resp_P2
    print(f"\n    ratio réponse APRÈS calibration entrée : "
          f"{ratio_resp_calib:.4f}")

    # --- Lecture continue (pas de seuil) ---
    print(f"\n  Lecture sur le continuum (Alex : pas de seuil) :")
    print(f"    pôle 'séparables'       : ratio_resp_calib ≈ 1")
    print(f"    pôle 'non séparables'   : ratio_resp_calib ≈ "
          f"ratio_resp_non_calib = {ratio_resp_non_calib:.4f}")
    print(f"    valeur observée         : {ratio_resp_calib:.4f}")

    # Fraction de couplage persistant après calibration
    # (ratio_resp_calib - 1) / (ratio_resp_non_calib - 1)
    if abs(ratio_resp_non_calib - 1.0) > 1e-6:
        coupling_residual = ((ratio_resp_calib - 1.0) /
                             (ratio_resp_non_calib - 1.0))
        print(f"\n    Fraction de couplage résiduel après calibration :")
        print(f"      (ratio_calib - 1) / (ratio_non_calib - 1) = "
              f"{coupling_residual:.4f}")
        print(f"      (0 = couplage entièrement réductible par calibration,")
        print(f"       1 = couplage entièrement irréductible)")
    else:
        coupling_residual = None

    output = {
        "beta": beta, "dt": dt,
        "strength_original": s_orig,
        "strength_calibrated_P2pp": float(s_calib),
        "amp_in_P2": amp_in_P2,
        "amp_in_P2p": amp_in_P2p,
        "amp_in_P2pp": amp_in_P2pp,
        "amp_resp_P2": amp_resp_P2,
        "amp_resp_P2p": amp_resp_P2p,
        "amp_resp_P2pp": amp_resp_P2pp,
        "ratio_input_non_calibrated": ratio_in_non_calib,
        "ratio_response_non_calibrated": ratio_resp_non_calib,
        "ratio_response_after_input_calibration": ratio_resp_calib,
        "coupling_residual_fraction": coupling_residual,
    }
    out_path = REPO_ROOT / "results" / "phase6d_gamma" / \
        "contact_10_geometry_intensity_coupling.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats : {out_path}")


if __name__ == "__main__":
    run()
