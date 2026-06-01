"""
Test de l'hypothèse de compatibilité (Alex).

ε est-il une mémoire de la structure géométrique de P,
ou seulement une mesure de l'effet net de P sur la concentration
centrale (compatibilité avec l'attracteur) ?

Test :
1. Pour chaque Pᵢ, calculer un scalaire c_i = ∫ (Pᵢ(ψ) − ψ) · w(x) dV
   où w(x) est une pondération centrée (par exemple gaussienne)
2. Comparer c_i aux amplitudes et signes des ε_i
3. Si ε_i / c_i est ~constant en amplitude ET en forme, c'est de la
   compatibilité scalaire, pas une mémoire géométrique.

Variantes de w(x) testées (Alex a parlé de "concentration centrale" sans
définition précise) :
- w_center : 1 seulement sur la cellule (2,2,2)
- w_central_ball : 1 sur les cellules à distance ≤ 1 du centre
- w_gaussian : gaussienne centrée
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

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

    perturbations = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
    ]

    # Pondérations w(x) candidates pour le scalaire de compatibilité
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    cc = (N_AXIS - 1) / 2.0
    dist_center = np.sqrt((I-cc)**2 + (J-cc)**2 + (K-cc)**2)

    w_center = (dist_center == 0).astype(float)  # cellule (2,2,2) seule
    w_central_ball = (dist_center <= 1.0).astype(float)  # boule r<=1
    w_gaussian = np.exp(-0.5 * dist_center**2)  # gaussienne σ=1
    # Aussi : signature radiale signée (centre +, périphérie -)
    w_signed_radial = (dist_center <= 1.0).astype(float) - (dist_center >= 2.5).astype(float)

    weights = {
        "w_center": w_center,
        "w_central_ball": w_central_ball,
        "w_gaussian": w_gaussian,
        "w_signed_radial": w_signed_radial,
    }

    # === Étape 1 : calculer les scalaires c_i pour chaque P et chaque w ===
    print(f"=== Scalaires de compatibilité c_i = ∫ (P(ψ) − ψ)·w dV ===\n")
    print(f"  Per Pi, on calcule la 'masse signée déplacée vers le centre'")
    print(f"  selon différentes pondérations w(x).\n")

    c_values = {wname: {} for wname in weights}
    for name, P in perturbations:
        delta_psi = P(psi_base) - psi_base
        for wname, w in weights.items():
            c_i = float((delta_psi * w).sum())
            c_values[wname][name] = c_i

    print(f"  {'w':<20}" + "".join(f"{n:>14}" for n,_ in perturbations))
    for wname in weights:
        row = f"  {wname:<20}"
        for n,_ in perturbations:
            row += f"{c_values[wname][n]:>14.4e}"
        print(row)

    # === Étape 2 : calculer les résidus, puis comparer ε_i / c_i ===
    print(f"\n=== Calcul des résidus r_i et des projections ===\n")
    r_psi_list = []
    for name, P in perturbations:
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta, gamma, h0, dt, n_long)
        r_psi_list.append(psi_R - psi_base)

    # u_ref signe-harmonisé comme avant
    r_norms = [np.linalg.norm(r) for r in r_psi_list]
    r_1_normed = r_psi_list[0] / r_norms[0]
    aligned = []
    for r in r_psi_list:
        if (r * r_1_normed).sum() < 0:
            aligned.append(-r / np.linalg.norm(r))
        else:
            aligned.append(r / np.linalg.norm(r))
    u_ref = np.mean(aligned, axis=0)
    u_ref = u_ref / np.linalg.norm(u_ref)

    eps_list = []
    for i, (name, _) in enumerate(perturbations):
        r = r_psi_list[i]
        alpha = float((r * u_ref).sum())
        eps = r - alpha * u_ref
        eps_list.append(eps)

    # Signature : ε_i évalué au centre
    eps_at_center = [float(e[2,2,2]) for e in eps_list]
    print(f"  {'metric':<25}" + "".join(f"{n:>14}" for n,_ in perturbations))
    row = f"  {'ε(2,2,2)':<25}"
    for v in eps_at_center: row += f"{v:>+14.4e}"
    print(row)
    eps_norms = [float(np.linalg.norm(e)) for e in eps_list]
    row = f"  {'||ε||':<25}"
    for v in eps_norms: row += f"{v:>14.4e}"
    print(row)

    # === Étape 3 : tester la corrélation ε ↔ c pour chaque w ===
    print(f"\n=== Test de l'hypothèse de compatibilité ===")
    print(f"  Pour chaque pondération w, regarder si ε(centre) ∝ c_i")
    print(f"  c.-à-d. si signes et amplitudes sont cohérents.\n")

    eps_center_vec = np.array(eps_at_center)
    for wname in weights:
        c_vec = np.array([c_values[wname][n] for n,_ in perturbations])
        # Régression linéaire ε(centre) = β · c + intercept
        if np.linalg.norm(c_vec) > 1e-30:
            beta_fit = float(np.dot(eps_center_vec, c_vec) / np.dot(c_vec, c_vec))
            pred = beta_fit * c_vec
            residual = eps_center_vec - pred
            rel_residual = float(np.linalg.norm(residual) /
                                  max(np.linalg.norm(eps_center_vec), 1e-30))
            corr = float(np.corrcoef(eps_center_vec, c_vec)[0,1])
            print(f"  {wname:<20}: corr={corr:+.4f}, β_fit={beta_fit:+.4e}, "
                  f"||résidus||/||ε(centre)|| = {rel_residual:.4f}")

    # === Étape 4 : tester si la FORME de ε est la même pour les 4 cas ===
    print(f"\n=== Forme normalisée de ε (test : ε_i = c_i · forme_universelle ?) ===")
    print(f"  Normaliser chaque ε par sa valeur au centre, et comparer.")
    print(f"  Si les formes normalisées sont quasi identiques, ε est ~ c · u_2.")
    eps_normalized = []
    for e in eps_list:
        v_at_center = float(e[2,2,2])
        if abs(v_at_center) > 1e-30:
            eps_normalized.append(e / v_at_center)
        else:
            eps_normalized.append(None)

    if all(e is not None for e in eps_normalized):
        # Comparer les profils radiaux normalisés
        unique_dists = sorted(set(np.round(dist_center.flatten(), 4)))
        print(f"\n  {'r':>8}" + "".join(f"{n+' norm':>14}" for n,_ in perturbations))
        for d in unique_dists:
            mask = np.isclose(dist_center, d, atol=1e-4)
            row = f"  {d:>8.3f}"
            for e_n in eps_normalized:
                row += f"{e_n[mask].mean():>+14.4e}"
            print(row)

        # Mesurer la similarité entre paires normalisées
        print(f"\n  Différences ||e_i_norm − e_j_norm|| / ||e_i_norm|| :")
        for i, (n_i, _) in enumerate(perturbations):
            for j, (n_j, _) in enumerate(perturbations):
                if i < j:
                    diff = eps_normalized[i] - eps_normalized[j]
                    rel = (np.linalg.norm(diff) /
                           max(np.linalg.norm(eps_normalized[i]), 1e-30))
                    print(f"    {n_i}_norm vs {n_j}_norm : {rel:.4f}")


if __name__ == "__main__":
    main()
