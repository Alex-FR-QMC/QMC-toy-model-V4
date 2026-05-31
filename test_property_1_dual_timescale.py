"""
6d-β cycle 2 — test direct propriété 1 (dual-timescale memory).

Référence : CADRAGE_6d.md §7, Ch3 §3.6.11.

Question : ψ et h relaxent-ils sur des échelles de temps séparées,
avec τ_h >> τ_ψ ?

Protocole minimal :
1. Faire évoluer le système jusqu'à un état stationnaire approximatif
2. Injecter une perturbation localisée
3. Mesurer la décroissance de l'écart sur ψ et sur h séparément
4. Comparer les deux temps de relaxation

Pas de sophistication. Un test, un résultat, on verra.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

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


def estimate_relaxation_time(times, signal, threshold_ratio=0.37):
    """Temps pour décroître à threshold_ratio (≈1/e) de la valeur initiale.
    Si jamais atteint, retourne None."""
    if signal[0] <= 0:
        return None
    target = signal[0] * threshold_ratio
    for i, s in enumerate(signal):
        if s < target:
            return float(times[i])
    return None


def run_test():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0

    # Phase de stabilisation
    t_stab = 200.0
    # Phase d'observation post-perturbation
    t_observe = 300.0

    psi = make_psi_centered(sigma=1.5)
    h = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))

    print(f"{'='*70}")
    print(f"Test propriété 1 — Dual-timescale memory")
    print(f"  β={beta}, D={D}, γ={gamma}, h0={h0}")
    print(f"  dt={dt:.5f}, t_stab={t_stab}, t_observe={t_observe}")
    print(f"{'='*70}\n")

    # 1. Stabilisation
    n_stab = int(t_stab / dt)
    for _ in range(n_stab):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)

    psi_ref = psi.copy()
    h_ref = h.copy()
    print(f"État après stabilisation :")
    print(f"  ψ: min={psi.min():.4e}, max={psi.max():.4e}, sum={psi.sum():.6f}")
    print(f"  h: min={h.min():.4e}, max={h.max():.4e}, mean={h.mean():.4e}")

    # 2. Branche A : pas de perturbation
    psi_A = psi.copy()
    h_A = h.copy()

    # 3. Branche B : perturbation localisée sur ψ
    psi_B = psi.copy()
    h_B = h.copy()
    # Perturbation : 1% relatif sur cellule (2,2,2)
    eps_rel = 0.01
    psi_B[2, 2, 2] *= (1.0 + eps_rel)
    psi_B /= psi_B.sum()  # renormalisation

    delta_psi_init = float(np.linalg.norm(psi_B - psi_A))
    delta_h_init = float(np.linalg.norm(h_B - h_A))
    print(f"\nAprès perturbation (ψ[2,2,2] +{eps_rel*100}% renormalisé) :")
    print(f"  ‖Δψ‖ initial = {delta_psi_init:.4e}")
    print(f"  ‖Δh‖ initial = {delta_h_init:.4e} (devrait être ~0)")

    # 4. Évolution parallèle, mesure des écarts
    n_obs = int(t_observe / dt)
    snapshot_every = max(1, n_obs // 300)

    times = [0.0]
    delta_psi = [delta_psi_init]
    delta_h = [delta_h_init]

    for k in range(1, n_obs + 1):
        psi_A, h_A = step(psi_A, h_A, D, beta, gamma, h0, dt)
        psi_B, h_B = step(psi_B, h_B, D, beta, gamma, h0, dt)
        if k % snapshot_every == 0:
            times.append(k * dt)
            delta_psi.append(float(np.linalg.norm(psi_B - psi_A)))
            delta_h.append(float(np.linalg.norm(h_B - h_A)))

    times = np.array(times)
    delta_psi = np.array(delta_psi)
    delta_h = np.array(delta_h)

    # 5. Analyse : temps caractéristiques
    delta_psi_max = float(delta_psi.max())
    delta_h_max = float(delta_h.max())
    idx_psi_max = int(np.argmax(delta_psi))
    idx_h_max = int(np.argmax(delta_h))

    print(f"\nÉvolution des écarts :")
    print(f"  max ‖Δψ‖ = {delta_psi_max:.4e} à t={times[idx_psi_max]:.2f}")
    print(f"  max ‖Δh‖ = {delta_h_max:.4e} à t={times[idx_h_max]:.2f}")

    # Temps de décroissance depuis le maximum
    if idx_psi_max < len(delta_psi) - 1:
        post_psi = delta_psi[idx_psi_max:]
        post_times_psi = times[idx_psi_max:] - times[idx_psi_max]
        tau_psi = estimate_relaxation_time(post_times_psi, post_psi)
    else:
        tau_psi = None

    if idx_h_max < len(delta_h) - 1:
        post_h = delta_h[idx_h_max:]
        post_times_h = times[idx_h_max:] - times[idx_h_max]
        tau_h = estimate_relaxation_time(post_times_h, post_h)
    else:
        tau_h = None

    print(f"\nTemps caractéristiques (décroissance à 1/e du max post-pic) :")
    print(f"  τ_ψ ≈ {tau_psi if tau_psi else 'non atteint dans la fenêtre'}")
    print(f"  τ_h ≈ {tau_h if tau_h else 'non atteint dans la fenêtre'}")

    if tau_psi and tau_h:
        ratio = tau_h / tau_psi
        print(f"  Ratio τ_h / τ_ψ = {ratio:.2f}")
        print(f"\nProp. 1 dual-timescale : τ_h >> τ_ψ ?")
        if ratio > 3.0:
            print(f"  → ratio > 3 : présence d'une séparation d'échelles")
        elif ratio > 1.5:
            print(f"  → ratio modeste : séparation faible")
        else:
            print(f"  → ratio proche de 1 : pas de séparation claire")

    # Valeurs finales
    print(f"\nValeurs finales (t={times[-1]:.1f}) :")
    print(f"  ‖Δψ‖ = {delta_psi[-1]:.4e}")
    print(f"  ‖Δh‖ = {delta_h[-1]:.4e}")
    print(f"  ratio ‖Δh‖/‖Δψ‖ = "
          f"{delta_h[-1] / max(delta_psi[-1], 1e-30):.4e}")

    # Affichage des trajectoires aux points clés
    print(f"\nQuelques points de la trajectoire :")
    print(f"  {'t':>8} {'‖Δψ‖':>14} {'‖Δh‖':>14} {'ratio Δh/Δψ':>14}")
    sample_indices = [0, 5, 10, 20, 40, 80, len(times) // 4,
                      len(times) // 2, 3 * len(times) // 4, len(times) - 1]
    sample_indices = sorted(set(i for i in sample_indices if i < len(times)))
    for i in sample_indices:
        ratio = (delta_h[i] / max(delta_psi[i], 1e-30)
                 if delta_psi[i] > 0 else 0)
        print(f"  {times[i]:>8.2f} {delta_psi[i]:>14.4e} "
              f"{delta_h[i]:>14.4e} {ratio:>14.4e}")

    return {
        "params": {
            "beta": beta, "D": D, "gamma": gamma, "h0": h0,
            "dt": dt, "t_stab": t_stab, "t_observe": t_observe,
            "perturbation": f"psi[2,2,2] *= 1.01 then renormalize",
        },
        "delta_psi_init": delta_psi_init,
        "delta_h_init": delta_h_init,
        "delta_psi_max": delta_psi_max,
        "delta_h_max": delta_h_max,
        "t_max_psi": float(times[idx_psi_max]),
        "t_max_h": float(times[idx_h_max]),
        "tau_psi": tau_psi,
        "tau_h": tau_h,
        "ratio_tau_h_over_tau_psi":
            (tau_h / tau_psi) if (tau_psi and tau_h) else None,
        "times": [float(t) for t in times],
        "delta_psi": [float(x) for x in delta_psi],
        "delta_h": [float(x) for x in delta_h],
    }


if __name__ == "__main__":
    out = run_test()
    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_property_1_dual_timescale.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
