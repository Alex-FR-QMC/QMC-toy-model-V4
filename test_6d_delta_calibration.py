"""
6d-δ — Mini-calibrage de P' et contrôle de mode lent parasite.

Étapes (selon cadrage 6d-δ + spécifications Alex) :

1. Géométrie : P' = P1' atténuée (gaussienne centrale).

2. Calibrage : trouver strength_P' tel que
   ||P'(ψ_base) - ψ_base|| ≈ 0.1 × ||P6(ψ_base) - ψ_base||

3. Contrôle de mode lent parasite : appliquer P' seul, mesurer
   l'extra à t = 200, 400, 800 (comme pour P6 mais sans P6).

Critère d'acceptation :
- extra(P') / r(P') négligeable à t=200
- pas de persistance significative à t=400 ou t=800

Si OK : on enregistre strength_P' calibré pour le test principal χ(τ).
Si non : il faudra réduire le strength ou changer de géométrie.
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


# Perturbations originales (pour construire la base de référence et P6)
def P1prime(psi, strength=0.05, radius=1.5):
    """Plateau central — utilisé pour la BASE."""
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


# P' = variante atténuée de P1' (gaussienne centrale, pas plateau)
# Choix : gaussienne sur la cellule (2,2,2) avec sigma petit, pour
# rester centrée mais avec un profil doux.
def P_prime(psi, strength, sigma_p=0.8):
    """Gaussienne centrale atténuée — perturbation P' du test 6d-δ."""
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


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    def input_amp(P, s, **kw):
        return float(np.linalg.norm(P(psi_base, s, **kw) - psi_base))

    # Calibrer P6 (référence)
    amp_target_base = input_amp(P1prime, 0.05)
    s_P6 = brentq(lambda s: input_amp(P6_face_dipole, s) - amp_target_base,
                  1e-4, 0.99, xtol=1e-6)
    amp_P6 = input_amp(P6_face_dipole, s_P6)
    print(f"=== Calibrages de référence ===")
    print(f"  ||P1' standard||  = {amp_target_base:.4e}")
    print(f"  ||P6 calibré||    = {amp_P6:.4e}  (s_P6 = {s_P6:.5f})")

    # Étape 2 : calibrer P' à amplitude = 0.1 × amplitude P6
    amp_target_P_prime = 0.1 * amp_P6
    print(f"\n=== Étape 2 : calibrage de P' ===")
    print(f"  Cible : ||P'(ψ_base) - ψ_base|| ≈ {amp_target_P_prime:.4e}")
    print(f"  (= 0.1 × ||P6||)")

    s_P_prime = brentq(
        lambda s: input_amp(P_prime, s) - amp_target_P_prime,
        1e-6, 1.0, xtol=1e-8
    )
    amp_P_prime = input_amp(P_prime, s_P_prime)
    print(f"  strength_P' calibré : {s_P_prime:.6f}")
    print(f"  ||P'(ψ_base) - ψ_base|| obtenu : {amp_P_prime:.4e}")
    print(f"  Ratio ||P'|| / ||P6||    : {amp_P_prime/amp_P6:.4f}")

    # Étape 3 : contrôle de mode lent parasite
    print(f"\n=== Étape 3 : contrôle de mode lent parasite ===")
    print(f"  Mesure de extra(P') à t = 200, 400, 800")
    print(f"  Critère : extra(P')/r(P') négligeable et sans persistance\n")

    # Construire la base de référence à partir de P1-P5 (comme dans 6d-γ)
    perturbations_15 = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
        ("P5", lambda p: P5_neighbors_only(p,
            strength=brentq(lambda s: input_amp(P5_neighbors_only, s)
                            - amp_target_base, 1e-4, 1.0, xtol=1e-6))),
    ]
    n_long = int(200.0 / dt)
    r_15 = []
    for name, P in perturbations_15:
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta_lock, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta_lock, gamma_v, h0, dt, n_long)
        r_15.append(psi_R - psi_base)
    R_base = np.column_stack([r.flatten() for r in r_15])

    # Mesurer extra(P') à t = 200, 400, 800
    times_check = [200, 400, 800]
    psi_pp = P_prime(psi_base.copy(), s_P_prime)
    h_pp = h_base.copy()

    # Formation initiale
    psi_curr, h_curr = evolve(psi_pp, h_pp, D, beta_lock, gamma_v, h0, dt, n_short)

    print("  {:>8} {:>14} {:>14} {:>14}".format("t", "||r_P'||", "||extra||", "extra/r"))
    extras_pp = []
    t_current = 0.0
    for t_target in times_check:
        dt_seg = t_target - t_current
        n_seg = int(dt_seg / dt)
        if n_seg > 0:
            psi_curr, h_curr = evolve(psi_curr, h_curr, D, beta_lock, gamma_v, h0, dt, n_seg)
        t_current = t_target

        r_pp = psi_curr - psi_base
        r_pp_flat = r_pp.flatten()
        coefs, _, _, _ = np.linalg.lstsq(R_base, r_pp_flat, rcond=None)
        extra_pp = r_pp_flat - R_base @ coefs

        r_norm = float(np.linalg.norm(r_pp_flat))
        extra_norm = float(np.linalg.norm(extra_pp))
        ratio = extra_norm / r_norm if r_norm > 1e-30 else 0.0
        extras_pp.append({"t": t_target, "r_norm": r_norm,
                          "extra_norm": extra_norm, "ratio": ratio})
        print(f"  {t_target:>8.0f} {r_norm:>14.4e} {extra_norm:>14.4e} {ratio:>14.4f}")

    # Critère d'acceptation
    ratio_200 = extras_pp[0]["ratio"]
    extra_200 = extras_pp[0]["extra_norm"]
    extra_400 = extras_pp[1]["extra_norm"]
    extra_800 = extras_pp[2]["extra_norm"]

    # Seuil de bruit numérique : si extra est déjà à ce niveau, le critère
    # de décroissance n'est pas applicable (on compare deux fluctuations
    # numériques autour de zéro).
    epsilon_noise = 1e-15

    if extra_200 < epsilon_noise:
        # extra numériquement nul : pas de mode lent, indépendamment de
        # ce que le ratio t=400/t=200 montre.
        print(f"\n  extra(t=200) = {extra_200:.4e} < epsilon_noise = {epsilon_noise:.4e}")
        print(f"  → extra numériquement nul, test de décroissance non applicable")
        accept_amp = ratio_200 < 0.01
        accept_decay = True  # non applicable, considéré comme satisfait
        accept_noise_floor = True
        print(f"\n  Critères :")
        print(f"    extra(P')/r(P') à t=200      = {ratio_200:.4e}")
        print(f"    extra(P') au seuil de bruit  : oui")
    else:
        # Cas standard : tester la décroissance
        if extra_200 > 1e-30:
            decay_200_800 = extra_800 / extra_200
            decay_200_400 = extra_400 / extra_200
        else:
            decay_200_800 = 0.0
            decay_200_400 = 0.0
        print(f"\n  Critères :")
        print(f"    ratio à t=200 (extra/r)  = {ratio_200:.6f}")
        print(f"    extra à t=400 / t=200    = {decay_200_400:.4f}")
        print(f"    extra à t=800 / t=200    = {decay_200_800:.4f}")
        accept_amp = ratio_200 < 0.01
        accept_decay = decay_200_800 < 0.5
        accept_noise_floor = False

    accepted = accept_amp and accept_decay

    print(f"\n  Acceptation amplitude (ratio < 1%)         : {accept_amp}")
    if not accept_noise_floor:
        print(f"  Acceptation décroissance (extra/200 < 0.5) : {accept_decay}")
    else:
        print(f"  Décroissance non applicable (extra < bruit numérique)")
    print(f"  → P' acceptable pour test χ(τ) ?            : {accepted}")

    if not accepted:
        print(f"\n  ATTENTION : P' génère un mode lent non négligeable.")
        print(f"  Recommandations :")
        if not accept_amp:
            print(f"    - réduire strength_P' (amplitude trop forte)")
        if not accept_decay and not accept_noise_floor:
            print(f"    - changer la géométrie de P' (mode lent persistant)")

    output = {
        "s_P6": float(s_P6),
        "amp_P6": amp_P6,
        "amp_target_P_prime": amp_target_P_prime,
        "s_P_prime": float(s_P_prime),
        "amp_P_prime": amp_P_prime,
        "ratio_amp_P_prime_over_P6": amp_P_prime / amp_P6,
        "control_at_times": extras_pp,
        "epsilon_noise": epsilon_noise,
        "extra_at_noise_floor": accept_noise_floor,
        "accept_amplitude": accept_amp,
        "accept_decay": accept_decay,
        "accepted": accepted,
    }
    with open("/home/claude/mcq_v4/6d_delta_calibration.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
