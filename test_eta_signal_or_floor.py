"""
6d-γ — Distinction Lecture A vs Lecture B sur η.

Question (Alex) : la direction w (commune aux 5 cas) transporte-t-elle
de l'information sur P (Lecture A), ou est-elle un fond résiduel du
régime verrouillé β=60 (Lecture B) ?

Test : regarder non seulement ||η_i|| mais le SIGNE et la VALEUR
de γ_i = <η_i, w_normalized>. 

- Si γ_i varie en signe et amplitude selon P → Lecture A
- Si γ_i est quasi constant en valeur (pas seulement en amplitude)
  → Lecture B (fond résiduel)

Et de manière plus directe : mesurer ||η_i - η_j|| pour les paires.
- Si grand par rapport à ||η||_typique → Lecture A (les η diffèrent)
- Si très petit → Lecture B (fond commun identique)
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


def main():
    gamma_v, D, h0, beta = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_long = int(200.0 / dt)

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta, gamma_v, h0, dt, n_stab)

    def input_amp(P, s):
        return float(np.linalg.norm(P(psi_base, s) - psi_base))
    amp_target = input_amp(P1prime, 0.05)
    s_P5 = brentq(lambda s: input_amp(P5_neighbors_only, s) - amp_target,
                  1e-4, 1.0, xtol=1e-6)

    perturbations = [
        ("P1", lambda p: P1prime(p, strength=0.05)),
        ("P2", lambda p: P2prime(p, strength=0.012789)),
        ("P3", lambda p: P3prime(p, strength=0.05)),
        ("P4", lambda p: P4(p, strength=0.03187)),
        ("P5", lambda p: P5_neighbors_only(p, strength=s_P5)),
    ]

    r_psi_list = []
    dpsi_center = []
    for name, P in perturbations:
        d = P(psi_base) - psi_base
        dpsi_center.append(float(d[2,2,2]))
        psi_p = P(psi_base.copy())
        h_p = h_base.copy()
        psi_R, h_R = evolve(psi_p, h_p, D, beta, gamma_v, h0, dt, n_short)
        psi_R, h_R = evolve(psi_R, h_R, D, beta, gamma_v, h0, dt, n_long)
        r_psi_list.append(psi_R - psi_base)

    # u_ref, v, et calcul des η
    r_1_n = r_psi_list[0] / np.linalg.norm(r_psi_list[0])
    aligned = []
    for r in r_psi_list:
        if (r * r_1_n).sum() < 0:
            aligned.append(-r / np.linalg.norm(r))
        else:
            aligned.append(r / np.linalg.norm(r))
    u_ref = np.mean(aligned, axis=0)
    u_ref = u_ref / np.linalg.norm(u_ref)

    eps_list = []
    alphas = []
    for r in r_psi_list:
        a = float((r * u_ref).sum())
        alphas.append(a)
        eps_list.append(r - a * u_ref)

    c_vec_4 = np.array(dpsi_center[:4])
    v_fit = sum(c_vec_4[i] * eps_list[i] for i in range(4)) / np.sum(c_vec_4**2)

    eta_list = []
    for i in range(5):
        eta = eps_list[i] - dpsi_center[i] * v_fit
        eta_list.append(eta)

    # === TEST DIRECT : ||η_i - η_j|| pour toutes les paires ===
    print(f"=== Test direct : les η sont-ils identiques entre eux ? ===\n")
    print(f"  Si Lecture B (fond résiduel) : ||η_i - η_j|| ≈ 0")
    print(f"  Si Lecture A (coordonnée γ varie) : ||η_i - η_j|| ~ ||η_i||\n")

    names = [perturbations[i][0] for i in range(5)]
    eta_norms = [np.linalg.norm(e) for e in eta_list]
    avg_eta_norm = float(np.mean(eta_norms))

    print(f"  Norme moyenne des η : {avg_eta_norm:.4e}\n")
    print(f"  Matrice des ||η_i - η_j|| (et fraction de ||η_typique||) :")
    print(f"  {'':>5} " + " ".join(f"{n:>14}" for n in names))
    diffs = np.zeros((5, 5))
    for i in range(5):
        row = f"  {names[i]:>5} "
        for j in range(5):
            d = float(np.linalg.norm(eta_list[i] - eta_list[j]))
            diffs[i,j] = d
            if i == j:
                row += f"{'-':>14}"
            else:
                ratio = d / avg_eta_norm
                row += f" {d:>8.2e}({ratio:>4.2f})"
        print(row)

    # Verdict global
    upper_diffs = []
    for i in range(5):
        for j in range(i+1, 5):
            upper_diffs.append(diffs[i,j])
    mean_diff = np.mean(upper_diffs)
    print(f"\n  Différence moyenne ||η_i - η_j|| (i≠j) = {mean_diff:.4e}")
    print(f"  Ratio à ||η_typique||             = {mean_diff/avg_eta_norm:.4f}")

    if mean_diff < 0.1 * avg_eta_norm:
        print(f"  → Lecture B FORTEMENT APPUYÉE : les η_i sont presque")
        print(f"    identiques. Le 'fond résiduel' explique mieux que")
        print(f"    une coordonnée γ qui varierait.")
    elif mean_diff > 0.5 * avg_eta_norm:
        print(f"  → Lecture A APPUYÉE : les η_i diffèrent significativement.")
        print(f"    γ transporte de l'information sur P.")
    else:
        print(f"  → Intermédiaire. Les η ont une composante commune ET")
        print(f"    une composante variable selon P.")

    # === Décomposition : η = η_floor + η_variable ===
    print(f"\n=== Décomposition η_i = η_floor + (η_i - η_floor) ===\n")
    print(f"  Estimateur de η_floor : moyenne des η_i")
    eta_floor = np.mean(eta_list, axis=0)
    print(f"  ||η_floor|| = {np.linalg.norm(eta_floor):.4e}")

    print(f"\n  Décomposition par perturbation :")
    print(f"  {'cas':<5} {'||η_i||':>14} {'||η_floor||':>14} "
          f"{'||η_i-η_floor||':>16} {'fraction var':>14}")
    deviations = []
    for i, name in enumerate(names):
        floor_norm = np.linalg.norm(eta_floor)
        eta_dev = eta_list[i] - eta_floor
        dev_norm = float(np.linalg.norm(eta_dev))
        deviations.append(eta_dev)
        var_frac = dev_norm / eta_norms[i]
        print(f"  {name:<5} {eta_norms[i]:>14.4e} {floor_norm:>14.4e} "
              f"{dev_norm:>16.4e} {var_frac:>14.4f}")

    # Verdict sur la décomposition
    avg_dev = np.mean([np.linalg.norm(d) for d in deviations])
    print(f"\n  Moyenne ||η_i - η_floor|| = {avg_dev:.4e}")
    print(f"  Comparé à ||η_floor|| = {np.linalg.norm(eta_floor):.4e}")
    if avg_dev < 0.1 * np.linalg.norm(eta_floor):
        verdict = "Lecture B confirmée : η_i ≈ η_floor pour tous les i."
        verdict += " La 'troisième direction' w est en fait un FOND du régime."
    elif avg_dev > 0.5 * np.linalg.norm(eta_floor):
        verdict = "Lecture A confirmée : grande variabilité autour de η_floor."
    else:
        verdict = "Intermédiaire : composante commune + variation modérée."
    print(f"\n  {verdict}")

    # === Si Lecture B, η_floor est intéressant en soi ===
    print(f"\n=== Profil radial de η_floor ===")
    coords = np.arange(N_AXIS)
    I, J, K = np.meshgrid(coords, coords, coords, indexing='ij')
    cc = (N_AXIS - 1) / 2.0
    dist_center = np.sqrt((I-cc)**2 + (J-cc)**2 + (K-cc)**2)
    unique_dists = sorted(set(np.round(dist_center.flatten(), 4)))

    for d in unique_dists:
        mask = np.isclose(dist_center, d, atol=1e-4)
        mean_v = eta_floor[mask].mean()
        std_v = eta_floor[mask].std()
        n_c = mask.sum()
        print(f"  r={d:.3f} : <η_floor> = {mean_v:+.4e}  σ = {std_v:.4e}  ({n_c})")

    # === Quantification finale ===
    print(f"\n=== Quantification finale ===")
    print(f"  Si on écrit r_i = α_i·u + β_i·v + (η_floor + résidu_i),")
    print(f"  alors la 'troisième direction' apparente est composée de :")
    print(f"  - η_floor (constant ≈ {np.linalg.norm(eta_floor):.4e})")
    print(f"  - résidu_i (variable ≈ {avg_dev:.4e})")
    print(f"  Rapport résidu/floor = {avg_dev/max(np.linalg.norm(eta_floor),1e-30):.4f}")

    output = {
        "perturbations": names,
        "dpsi_center": dpsi_center,
        "eta_norms": eta_norms,
        "avg_eta_norm": avg_eta_norm,
        "matrix_diff_eta_ij": diffs.tolist(),
        "mean_diff_eta": float(mean_diff),
        "ratio_mean_diff_to_typical": float(mean_diff / avg_eta_norm),
        "eta_floor_norm": float(np.linalg.norm(eta_floor)),
        "deviations_norm": [float(np.linalg.norm(d)) for d in deviations],
        "avg_deviation_norm": float(avg_dev),
        "verdict": verdict,
    }
    with open("/home/claude/mcq_v4/eta_signal_or_floor.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
