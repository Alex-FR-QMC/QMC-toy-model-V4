"""
6d-γ — P5, test de décorrélation des proxies.

Construction : booster les 6 cellules voisines à distance 1 du centre,
laisser la cellule centrale (2,2,2) avec un facteur 1 (donc inchangée
avant renormalisation).

Après renormalisation par la somme :
- Δψ(centre) sera petit (la cellule centrale décroît parce que la
  masse totale a augmenté, mais elle décroît proportionnellement
  à toutes les autres cellules)
- Δψ(voisines à r=1) sera grand (boost direct par le facteur)
- les autres couches sont inchangées avant renorm, donc décroissent
  toutes proportionnellement après renorm

Si Δψ_centre / Δψ_voisines << 1, on a une décorrélation utile.

Prédictions concurrentes :
- w_center : c ≈ 0  → ε ≈ 0
- w_central_ball : c grand (les 6 voisines sont dans la boule r ≤ 1)
                 → ε grand
- w_gaussian : c intermédiaire
- w_signed_radial : c dépend de la périphérie aussi (les autres
                   couches décroissent un peu)
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


def P5_neighbors_only(psi, strength):
    """Booste les 6 cellules à distance 1 du centre (cubiquement symétrique).
    Le centre (2,2,2) garde son facteur de 1.0.
    Après renormalisation, Δψ_centre ≠ 0 mais petit par rapport à Δψ_voisines."""
    factor = np.ones_like(psi)
    # Les 6 voisines à distance 1 (cellule centre = (2,2,2))
    neighbors = [(1,2,2), (3,2,2), (2,1,2), (2,3,2), (2,2,1), (2,2,3)]
    for (i,j,k) in neighbors:
        factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()


def input_amplitude(P, psi, strength):
    return float(np.linalg.norm(P(psi, strength) - psi))


def main():
    gamma, D, h0, beta = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma, h0, dt, n_stab)

    # Calibrer P5 pour avoir amplitude d'entrée dans le corridor
    amp_target = input_amplitude(P1prime, psi_base, 0.05)
    s_P5 = brentq(lambda s: input_amplitude(P5_neighbors_only, psi_base, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)
    amp_P5 = input_amplitude(P5_neighbors_only, psi_base, s_P5)
    print(f"Strength P5 calibré : {s_P5:.5f}")
    print(f"||P5(ψ)-ψ|| = {amp_P5:.4e}  (cible : {amp_target:.4e})")

    # Mesurer Δψ d'entrée
    delta_P5 = P5_neighbors_only(psi_base, s_P5) - psi_base
    print(f"\n=== Δψ d'entrée pour P5 ===")
    print(f"  Δψ(centre 2,2,2)     = {delta_P5[2,2,2]:+.4e}")
    print(f"  Δψ(voisine 1,2,2)    = {delta_P5[1,2,2]:+.4e}")
    print(f"  Δψ(diagonal 1,1,2)   = {delta_P5[1,1,2]:+.4e}  (distance √2)")
    print(f"  Δψ(périphérie 0,0,0) = {delta_P5[0,0,0]:+.4e}")
    print(f"  Ratio |Δψ_centre/Δψ_voisine| = "
          f"{abs(delta_P5[2,2,2]/delta_P5[1,2,2]):.4f}")
    print(f"  Si << 1 : décorrélation utile.")

    # Calculer les c_i pour chaque proxy
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    cc = (N_AXIS - 1) / 2.0
    dist_center = np.sqrt((I-cc)**2 + (J-cc)**2 + (K-cc)**2)

    weights = {
        "w_center": (dist_center == 0).astype(float),
        "w_central_ball": (dist_center <= 1.0).astype(float),
        "w_gaussian": np.exp(-0.5 * dist_center**2),
        "w_signed_radial": (dist_center <= 1.0).astype(float) - (dist_center >= 2.5).astype(float),
    }

    print(f"\n=== c_i selon chaque proxy pour P5 ===")
    c_P5 = {}
    for wname, w in weights.items():
        c_P5[wname] = float((delta_P5 * w).sum())
        print(f"  {wname:<18}: c_P5 = {c_P5[wname]:+.4e}")

    # Rappel : ε(centre) pour les 4 précédentes
    # Si les proxies sont d'accord sur P5, ils donnent tous le même signe et
    # le même ordre de grandeur. S'ils divergent, c'est la décorrélation.

    # Calculer le résidu de P5
    psi_p = P5_neighbors_only(psi_base.copy(), s_P5)
    h_p = h_base.copy()
    psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
    psi_R, h_R = evolve(psi_R, h_R, D, beta, gamma, h0, dt, n_long)
    r_psi_5 = psi_R - psi_base
    r_h_5 = h_R - h_base

    print(f"\n=== Résidu de P5 ===")
    print(f"  ||r_psi_5|| = {np.linalg.norm(r_psi_5):.4e}")
    print(f"  ||r_h_5|| = {np.linalg.norm(r_h_5):.4e}")
    print(f"  r_psi_5 au centre (2,2,2) = {r_psi_5[2,2,2]:+.4e}")

    # Pour comparer : recalculer u_ref et la "forme universelle" v à partir
    # des 4 résidus précédents, puis calculer α_5 et ε_5
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

    perturbations = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
    ]
    r_psi_list_4 = []
    for name, P in perturbations:
        psi_p_i = P(psi_base.copy())
        h_p_i = h_base.copy()
        psi_R_i, h_R_i = evolve(psi_p_i, h_p_i, D, beta, gamma, h0, dt, n_short)
        psi_R_i, h_R_i = evolve(psi_R_i, h_R_i, D, beta, gamma, h0, dt, n_long)
        r_psi_list_4.append(psi_R_i - psi_base)

    r_1_normed = r_psi_list_4[0] / np.linalg.norm(r_psi_list_4[0])
    aligned = []
    for r in r_psi_list_4:
        if (r * r_1_normed).sum() < 0:
            aligned.append(-r / np.linalg.norm(r))
        else:
            aligned.append(r / np.linalg.norm(r))
    u_ref = np.mean(aligned, axis=0)
    u_ref = u_ref / np.linalg.norm(u_ref)

    # v de la forme universelle, ajusté sur les 4 précédents
    c_vec_center = np.array([
        (perturbations[i][1](psi_base) - psi_base)[2,2,2]
        for i in range(4)
    ])
    epsilons_4 = []
    for r in r_psi_list_4:
        alpha = float((r * u_ref).sum())
        epsilons_4.append(r - alpha * u_ref)
    v_fit = sum(c_vec_center[i] * epsilons_4[i] for i in range(4)) / np.sum(c_vec_center**2)

    alpha_5 = float((r_psi_5 * u_ref).sum())
    epsilon_5 = r_psi_5 - alpha_5 * u_ref

    print(f"\n=== Comparaison résidu P5 vs prédictions des proxies ===")
    print(f"  α_5 (projection sur u_ref)    = {alpha_5:+.4e}")
    print(f"  ε_5 au centre (2,2,2)         = {epsilon_5[2,2,2]:+.4e}")
    print(f"  ||ε_5||                        = {np.linalg.norm(epsilon_5):.4e}")

    print(f"\n  Prédictions des proxies pour ε_5 :")
    # Si ε = c · v_fit, alors ε(centre) = c · v_fit(centre)
    v_at_center = v_fit[2,2,2]
    print(f"  (v(centre) ajusté sur P1-P4 = {v_at_center:+.4e})")
    for wname, c_val in c_P5.items():
        pred_eps_center = c_val * v_at_center
        print(f"  {wname:<18}: c={c_val:+.4e}  → ε(centre) prédit = "
              f"{pred_eps_center:+.4e}")

    print(f"\n  Mesuré : ε(centre) = {epsilon_5[2,2,2]:+.4e}")
    # Quel proxy est le plus proche ?
    best_proxy = None
    best_diff = float('inf')
    for wname, c_val in c_P5.items():
        pred = c_val * v_at_center
        diff = abs(pred - epsilon_5[2,2,2])
        print(f"  {wname:<18}: |prédit - mesuré| = {diff:.4e}")
        if diff < best_diff:
            best_diff = diff
            best_proxy = wname
    print(f"\n  Meilleur proxy pour P5 : {best_proxy}")

    # Lecture par modèle complet (ajustement du résidu entier, pas juste au centre)
    print(f"\n=== Ajustement de ε_5 complet par chaque proxy ===")
    for wname, c_val in c_P5.items():
        predicted_eps = c_val * v_fit
        residual = epsilon_5 - predicted_eps
        rel_err = np.linalg.norm(residual) / np.linalg.norm(epsilon_5)
        print(f"  {wname:<18}: ||ε_5 - c·v|| / ||ε_5|| = {rel_err:.4f}")


if __name__ == "__main__":
    main()
