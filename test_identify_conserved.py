"""
Identification de la variable réellement conservée par la projection P → r.

Test : pour chaque proxy candidat c_i, ajuster r = α · u_ref + c · v
et mesurer la qualité d'ajustement sur l'ensemble du résidu.

Si plusieurs proxies sont quasi-équivalents, ils sont corrélés et on
ne peut pas trancher sans P5. Si un proxy est strictement meilleur,
c'est le meilleur candidat pour la variable conservée.

Quatre proxies candidats (déjà calculés au test précédent) :
- w_center (Δψ à la cellule centrale)
- w_central_ball (Δψ sommé sur boule r ≤ 1)
- w_gaussian (Δψ pondéré gaussienne σ=1)
- w_signed_radial (Δψ pondéré (centre - périphérie))
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

    # Calcul des résidus
    r_psi_list = []
    delta_inputs = []
    for name, P in perturbations:
        delta_inputs.append(P(psi_base) - psi_base)
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta, gamma, h0, dt, n_long)
        r_psi_list.append(psi_R - psi_base)

    # Calcul des c_i pour chaque proxy
    c_by_proxy = {}
    for wname, w in weights.items():
        c_by_proxy[wname] = np.array([float((d * w).sum()) for d in delta_inputs])

    # Construire u_ref (moyenne signée des résidus normalisés)
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

    # Pour chaque proxy w, ajuster r_i = α_i · u_ref + c_i · v_w
    # avec α_i = <r_i, u_ref> et c_i fixé par le proxy
    # v_w est ajusté par moindres carrés sur les 4 cas
    # Erreur d'ajustement = ||r_i - α_i·u_ref - c_i·v_w|| sommée
    print(f"=== Identification de la variable conservée ===")
    print(f"  Modèle : r_i = α_i·u_ref + c_i·v")
    print(f"  α_i = <r_i, u_ref>, c_i fixé par chaque proxy")
    print(f"  v est ajusté par moindres carrés sur les 4 cas")
    print(f"  Métrique : erreur totale ||r_i - prédiction|| sommée\n")

    # Pour chaque proxy, ajuster v et calculer l'erreur
    results = {}
    for wname in weights:
        c_vec = c_by_proxy[wname]
        # Calculer ε_i = r_i - α_i · u_ref (déjà fait au test précédent)
        epsilons = []
        alphas = []
        for r in r_psi_list:
            alpha = float((r * u_ref).sum())
            alphas.append(alpha)
            epsilons.append(r - alpha * u_ref)

        # Maintenant on cherche v tel que ε_i ≈ c_i · v
        # v optimal au sens des moindres carrés : v = ∑(c_i·ε_i) / ∑(c_i^2)
        if np.sum(c_vec**2) > 1e-30:
            v_fit = sum(c_vec[i] * epsilons[i] for i in range(len(c_vec))) / np.sum(c_vec**2)
        else:
            v_fit = np.zeros_like(epsilons[0])

        # Erreur résiduelle par cas
        total_err = 0.0
        total_norm = 0.0
        for i, eps_i in enumerate(epsilons):
            err = eps_i - c_vec[i] * v_fit
            total_err += float(np.linalg.norm(err)**2)
            total_norm += float(np.linalg.norm(eps_i)**2)
        rel_err = (total_err / total_norm)**0.5
        results[wname] = {
            "c_vec": c_vec.tolist(),
            "rel_err": rel_err,
            "v_norm": float(np.linalg.norm(v_fit)),
        }
        print(f"  {wname:<20}: erreur relative = {rel_err:.6f}  "
              f"||v|| = {np.linalg.norm(v_fit):.4e}")

    print(f"\n  Tableau des c_i par proxy :")
    print(f"  {'proxy':<20}" + "".join(f"{n:>14}" for n,_ in perturbations))
    for wname in weights:
        row = f"  {wname:<20}"
        for v in c_by_proxy[wname]:
            row += f"{v:>+14.4e}"
        print(row)

    # Test discriminant : si tous les proxies donnent quasi la même erreur,
    # ils sont équivalents et il faut une P5 pour trancher
    errs = [results[w]["rel_err"] for w in weights]
    print(f"\n  Erreur min : {min(errs):.6f}, max : {max(errs):.6f}")
    print(f"  Ratio max/min : {max(errs)/min(errs):.4f}")
    if max(errs) / min(errs) < 1.1:
        print(f"  → Tous les proxies sont quasi équivalents. On ne peut pas")
        print(f"    discriminer entre eux. Il faut une P5 pour trancher.")
    else:
        best = min(weights, key=lambda w: results[w]["rel_err"])
        print(f"  → Meilleur proxy : {best} (erreur {results[best]['rel_err']:.6f})")

    # Test additionnel : corrélations entre les proxies eux-mêmes
    # Si tous les c_vec sont quasi colinéaires, c'est qu'ils mesurent la même chose
    print(f"\n  Corrélations entre les c_vec (proxies entre eux) :")
    wnames = list(weights.keys())
    for i, w1 in enumerate(wnames):
        for w2 in wnames[i+1:]:
            corr = float(np.corrcoef(c_by_proxy[w1], c_by_proxy[w2])[0,1])
            print(f"    {w1:<18} vs {w2:<18} : {corr:+.6f}")


if __name__ == "__main__":
    main()
