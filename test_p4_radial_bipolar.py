"""
6d-γ — test P4 (gradient radial bipolaire signé).

Question (Alex) :
Quelles propriétés continuent à être comprimées vers une morphologie
commune, et quelles propriétés refusent encore cette compression ?

P4 = booste centre (cellules à distance < r_inner du centre, +s)
   ET déprime périphérie (cellules à distance > r_outer du centre, -s)

PAS de projection sur u (cela traiterait u comme référence canonique,
ce que les données ne soutiennent pas encore).

Observables (dans l'ordre, sans présupposer un axe) :
1. Morphologie spatiale brute de r_4 = R_4(∞) − R_base
2. Différences r_4 − r_i pour i = 1, 2, 3
3. Localisation des écarts persistants

Verdicts préinscrits :
- Si r_4 ressemble morphologiquement à r_1/r_2/r_3 (même structure
  centrée cubiquement symétrique, juste amplitude/signe différent)
  → P4 est compressé vers la même morphologie. Le gradient signé
  bipolaire a été projeté sur la morphologie commune. Quelle
  propriété de P4 a été oubliée ?
- Si r_4 a une morphologie différente (par exemple lui-même
  bipolaire radial, ou présentant une structure que r_1/r_2/r_3
  n'ont pas)
  → P4 résiste à la compression. Quelque chose du gradient signé
  a survécu.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
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


# Perturbations précédentes (pour comparaison)
def P1prime_plateau(psi, strength=0.05, radius=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= radius:
                    factor[i,j,k] += strength
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P2prime_corona(psi, strength, radius_inner=2.0):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r >= radius_inner:
                    factor[i,j,k] += strength
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P3prime_anisotropic_z(psi, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for k in range(N_AXIS):
        dist_z = abs(k - c)
        factor[:,:,k] *= 1.0 + strength * (2.0 * dist_z / c - 1.0)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def P4_radial_bipolar(psi, strength, r_inner=1.0, r_outer=2.5):
    """Gradient radial bipolaire signé :
    +strength dans les cellules à distance < r_inner du centre,
    -strength dans les cellules à distance > r_outer du centre."""
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= r_inner:
                    factor[i,j,k] += strength
                elif r >= r_outer:
                    factor[i,j,k] -= strength
    psi_new = psi * factor
    psi_new = np.maximum(psi_new, 0)  # garde-fou positivité
    return psi_new / psi_new.sum()


def input_amplitude(P, psi, strength):
    return float(np.linalg.norm(P(psi, strength) - psi))


def main():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0
    strength_P2pp = 0.012789

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)

    # Calibration de P4 pour avoir amplitude d'entrée dans le corridor
    amp_target = input_amplitude(P1prime_plateau, psi_base, 0.05)
    print(f"Amplitude cible (= ||P1''||) : {amp_target:.4e}")
    print(f"Amplitudes des autres : ||P2''||={input_amplitude(P2prime_corona, psi_base, strength_P2pp):.4e}, "
          f"||P3''||={input_amplitude(P3prime_anisotropic_z, psi_base, 0.05):.4e}")

    # Calibrer strength de P4 pour matcher amp_target
    try:
        s_P4 = brentq(lambda s: input_amplitude(P4_radial_bipolar, psi_base, s) - amp_target,
                      1e-4, 0.5, xtol=1e-6)
    except ValueError:
        print(f"Calibration P4 a échoué — strength=0.05 par défaut")
        s_P4 = 0.05
    amp_P4 = input_amplitude(P4_radial_bipolar, psi_base, s_P4)
    print(f"Strength P4 calibré : {s_P4:.5f}, ||P4(ψ)-ψ|| = {amp_P4:.4e}")
    # Vérifier que P4 est bien dans le corridor
    in_corridor = 0.5 < amp_P4 / amp_target < 2.0
    print(f"Amplitude P4 dans corridor [0.5, 2.0]×cible ? {in_corridor}")

    # Évolution des 4 résidus jusqu'à stationnarité
    perturbations = {
        "P1''": lambda p: P1prime_plateau(p, strength=0.05),
        "P2''": lambda p: P2prime_corona(p, strength=strength_P2pp),
        "P3''": lambda p: P3prime_anisotropic_z(p, strength=0.05),
        "P4 ": lambda p: P4_radial_bipolar(p, strength=s_P4),
    }
    R = {}
    for name, P in perturbations.items():
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta, gamma, h0, dt, n_long)
        R[name] = (psi_R - psi_base, h_R - h_base)

    print(f"\n=== Étape 1 : morphologie spatiale brute de r_4 ===")
    r_psi_4, r_h_4 = R["P4 "]
    print(f"  ||r_ψ_4||={np.linalg.norm(r_psi_4):.4e}, "
          f"||r_h_4||={np.linalg.norm(r_h_4):.4e}")
    argmax_psi = np.unravel_index(int(np.argmax(np.abs(r_psi_4))), r_psi_4.shape)
    argmax_h = np.unravel_index(int(np.argmax(np.abs(r_h_4))), r_h_4.shape)
    print(f"  argmax|r_ψ_4|={argmax_psi}, valeur={r_psi_4[argmax_psi]:+.4e}")
    print(f"  argmax|r_h_4|={argmax_h}, valeur={r_h_4[argmax_h]:+.4e}")

    proj_x = np.abs(r_psi_4).sum(axis=(1,2))
    proj_y = np.abs(r_psi_4).sum(axis=(0,2))
    proj_z = np.abs(r_psi_4).sum(axis=(0,1))
    print(f"  projections |r_ψ_4| : x={proj_x}")
    print(f"                        y={proj_y}")
    print(f"                        z={proj_z}")
    same_xyz = (np.allclose(proj_x, proj_y, atol=1e-10) and
                np.allclose(proj_x, proj_z, atol=1e-10))
    print(f"  Projections identiques sur les 3 axes ? {same_xyz}")
    print(f"  → si OUI, le résidu est cubiquement symétrique comme r_1, r_2, r_3")

    # Profil radial du résidu : moyenne de r_ψ_4 par distance au centre
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    c = (N_AXIS - 1) / 2.0
    dist_center = np.sqrt((I-c)**2 + (J-c)**2 + (K-c)**2)
    unique_dists = sorted(set(np.round(dist_center.flatten(), 4)))
    print(f"\n  Profil radial r_ψ_4 (moyenne par couche radiale, sans abs) :")
    for d in unique_dists:
        mask = np.isclose(dist_center, d, atol=1e-4)
        mean_r = r_psi_4[mask].mean()
        print(f"    distance {d:.3f} : <r_ψ> = {mean_r:+.4e}  ({mask.sum()} cellules)")

    print(f"\n=== Étape 2 : différences r_4 − r_i ===")
    for name in ["P1''", "P2''", "P3''"]:
        r_psi_i, r_h_i = R[name]
        diff_psi = r_psi_4 - r_psi_i
        diff_h = r_h_4 - r_h_i
        print(f"  r_4 − r_{name} :")
        print(f"    ||diff_ψ||={np.linalg.norm(diff_psi):.4e}, "
              f"||diff_h||={np.linalg.norm(diff_h):.4e}")
        # Localisation de la différence
        argmax_diff = np.unravel_index(int(np.argmax(np.abs(diff_psi))), diff_psi.shape)
        print(f"    argmax|diff_ψ|={argmax_diff}, valeur={diff_psi[argmax_diff]:+.4e}")
        # Est-ce une simple différence d'amplitude ou une différence structurelle ?
        # Si r_4 = α·r_i + ε avec ε petit, alors diff = (α-1)·r_i + ε
        # On peut estimer α par moindres carrés et regarder ce qui reste
        denom = float((r_psi_i**2).sum())
        if denom > 1e-30:
            alpha = float((r_psi_i * r_psi_4).sum() / denom)
            residual = r_psi_4 - alpha * r_psi_i
            print(f"    si r_4 = α·r_{name} + ε : α={alpha:+.4f}, "
                  f"||ε||/||r_4||={np.linalg.norm(residual)/np.linalg.norm(r_psi_4):.4f}")

    print(f"\n=== Étape 3 : localisation des écarts persistants ===")
    print(f"  Sur la base des composantes ε ci-dessus, où se concentre la")
    print(f"  partie de r_4 NON représentable comme α·r_i ?")
    # On prend ε par rapport à P1'' comme référence (le résidu le plus "propre")
    r_psi_1, _ = R["P1''"]
    denom = float((r_psi_1**2).sum())
    alpha_1 = float((r_psi_1 * r_psi_4).sum() / denom)
    epsilon = r_psi_4 - alpha_1 * r_psi_1
    epsilon_proj_x = np.abs(epsilon).sum(axis=(1,2))
    epsilon_proj_y = np.abs(epsilon).sum(axis=(0,2))
    epsilon_proj_z = np.abs(epsilon).sum(axis=(0,1))
    print(f"  ε = r_4 − {alpha_1:.4f}·r_1")
    print(f"  ||ε|| = {np.linalg.norm(epsilon):.4e}  "
          f"(||r_4|| = {np.linalg.norm(r_psi_4):.4e}, fraction = "
          f"{np.linalg.norm(epsilon)/np.linalg.norm(r_psi_4):.4f})")
    print(f"  projections |ε| : x={epsilon_proj_x}")
    print(f"                    y={epsilon_proj_y}")
    print(f"                    z={epsilon_proj_z}")
    # Le profil radial de ε
    print(f"\n  Profil radial de ε (la partie de r_4 non absorbée par r_1) :")
    for d in unique_dists:
        mask = np.isclose(dist_center, d, atol=1e-4)
        mean_eps = epsilon[mask].mean()
        print(f"    distance {d:.3f} : <ε> = {mean_eps:+.4e}")


if __name__ == "__main__":
    main()
