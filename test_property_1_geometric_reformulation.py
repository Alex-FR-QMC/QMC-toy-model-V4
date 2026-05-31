"""
6d-β cycle 2 — test propriété 1, reformulation géométrique.

Reformulation (Alex) :
"La trajectoire perturbée ressemble-t-elle davantage à une trajectoire
soeur non perturbée qu'à une relaxation perturbative classique ?"

Protocole :
1. Stabilisation jusqu'à t_stab
2. À partir de l'état stationnaire approché, construire :
   - 1 trajectoire de référence (continue sans perturbation)
   - 1 trajectoire perturbée (psi[2,2,2] += 1% renormalisé)
   - K trajectoires soeurs (micro-variation d'initialisation à t=0,
     même ordre de grandeur que la perturbation, à des endroits
     différents ou aléatoires)
3. Mesurer les distances ||Δψ|| et ||Δh|| entre :
   - perturbée vs référence
   - soeurs vs référence
4. Comparer : la perturbée se range-t-elle dans l'ensemble des soeurs,
   ou s'en distingue-t-elle ?

Si perturbée ~ soeurs : le déplacement est géométrique, pas perturbatif.
Si perturbée >> soeurs : il y a quelque chose de spécifique à
la perturbation post-stabilisation.

Statut : hypothèse, pas verdict. Une seule mesure.
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


def make_psi_micro_varied(sigma_base=1.5, seed=0):
    """Initialisation 'soeur' : même structure gaussienne centrée,
    mais avec une micro-variation aléatoire au niveau de chaque cellule.
    L'amplitude de variation est calibrée pour être du même ordre
    que la perturbation 1% post-stabilisation."""
    psi = make_psi_centered(sigma=sigma_base)
    rng = np.random.default_rng(seed)
    # Variation 1% sur chaque cellule
    noise = rng.uniform(-0.01, 0.01, size=psi.shape)
    psi = psi * (1.0 + noise)
    psi /= psi.sum()
    return psi


def evolve(psi, h, D, beta, gamma, h0, dt, n_steps, snapshot_every):
    """Évolue le système et retourne snapshots de psi et h."""
    psi_list = [psi.copy()]
    h_list = [h.copy()]
    t_list = [0.0]
    for k in range(1, n_steps + 1):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        if k % snapshot_every == 0:
            psi_list.append(psi.copy())
            h_list.append(h.copy())
            t_list.append(k * dt)
    return np.array(t_list), psi_list, h_list, psi, h


def run_test():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0

    t_stab = 200.0
    t_observe = 300.0
    K_sisters = 8  # 8 trajectoires soeurs

    psi_ref_init = make_psi_centered(sigma=1.5)
    h_ref_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi_ref_init.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))

    print(f"{'='*70}")
    print(f"Test propriété 1 — reformulation géométrique")
    print(f"  β={beta}, dt={dt:.5f}")
    print(f"  K_sisters={K_sisters}, t_stab={t_stab}, t_observe={t_observe}")
    print(f"{'='*70}\n")

    # === Référence : on suit la trajectoire complète depuis l'init ===
    n_stab = int(t_stab / dt)
    n_obs = int(t_observe / dt)
    n_total = n_stab + n_obs
    snapshot_every = max(1, n_total // 200)

    print("Évolution de la trajectoire de référence...")
    times_ref, psi_ref_traj, h_ref_traj, psi_ref_end, h_ref_end = evolve(
        psi_ref_init.copy(), h_ref_init.copy(),
        D, beta, gamma, h0, dt, n_total, snapshot_every,
    )

    # Trouver l'index correspondant à t_stab (début perturbation)
    idx_stab = int(np.searchsorted(times_ref, t_stab))
    print(f"  idx_stab dans trajectoire ref = {idx_stab}")
    print(f"  times_ref[idx_stab] = {times_ref[idx_stab]:.2f}")

    # === Branche perturbée : repart de l'état ref à t_stab, perturbée ===
    psi_at_stab = psi_ref_traj[idx_stab].copy()
    h_at_stab = h_ref_traj[idx_stab].copy()
    # Perturbation 1% sur cellule (2,2,2)
    psi_pert = psi_at_stab.copy()
    psi_pert[2, 2, 2] *= 1.01
    psi_pert /= psi_pert.sum()

    print("\nÉvolution de la trajectoire perturbée...")
    times_pert, psi_pert_traj, h_pert_traj, _, _ = evolve(
        psi_pert, h_at_stab.copy(),
        D, beta, gamma, h0, dt, n_obs, snapshot_every,
    )

    # Distances perturbée vs ref (à partir de t_stab)
    delta_psi_pert = []
    delta_h_pert = []
    n_compare = min(len(psi_pert_traj), len(psi_ref_traj) - idx_stab)
    for k in range(n_compare):
        delta_psi_pert.append(
            float(np.linalg.norm(psi_pert_traj[k] - psi_ref_traj[idx_stab + k]))
        )
        delta_h_pert.append(
            float(np.linalg.norm(h_pert_traj[k] - h_ref_traj[idx_stab + k]))
        )
    delta_psi_pert = np.array(delta_psi_pert)
    delta_h_pert = np.array(delta_h_pert)

    # === Trajectoires soeurs : init micro-variée, évolution complète ===
    print(f"\nÉvolution de {K_sisters} trajectoires soeurs "
          f"(init micro-variée seed=1..{K_sisters})...")
    sisters_delta_psi = []
    sisters_delta_h = []

    for seed in range(1, K_sisters + 1):
        psi_sis_init = make_psi_micro_varied(sigma_base=1.5, seed=seed)
        # Vérification : écart initial du même ordre que perturbation
        delta_init = float(np.linalg.norm(psi_sis_init - psi_ref_init))
        print(f"  seed={seed}: ‖psi_sis - psi_ref‖_init = {delta_init:.4e}")

        h_sis_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
        _, psi_sis_traj, h_sis_traj, _, _ = evolve(
            psi_sis_init, h_sis_init,
            D, beta, gamma, h0, dt, n_total, snapshot_every,
        )

        # Distances soeur vs ref à partir de t_stab
        d_psi = []
        d_h = []
        for k in range(n_compare):
            d_psi.append(float(np.linalg.norm(
                psi_sis_traj[idx_stab + k] - psi_ref_traj[idx_stab + k])))
            d_h.append(float(np.linalg.norm(
                h_sis_traj[idx_stab + k] - h_ref_traj[idx_stab + k])))
        sisters_delta_psi.append(d_psi)
        sisters_delta_h.append(d_h)

    sisters_delta_psi = np.array(sisters_delta_psi)  # K × n_compare
    sisters_delta_h = np.array(sisters_delta_h)

    # === Analyse comparative ===
    times_compare = times_ref[idx_stab:idx_stab + n_compare] - t_stab

    sisters_psi_mean = sisters_delta_psi.mean(axis=0)
    sisters_psi_min = sisters_delta_psi.min(axis=0)
    sisters_psi_max = sisters_delta_psi.max(axis=0)
    sisters_h_mean = sisters_delta_h.mean(axis=0)
    sisters_h_min = sisters_delta_h.min(axis=0)
    sisters_h_max = sisters_delta_h.max(axis=0)

    print(f"\n{'─'*70}")
    print(f"Comparaison : perturbée vs ensemble soeurs (post-t_stab)")
    print(f"{'─'*70}")
    print(f"\n  À t_post = 0 (= juste après perturbation/init) :")
    print(f"    ‖Δψ_pert‖     = {delta_psi_pert[0]:.4e}")
    print(f"    soeurs ‖Δψ‖   = "
          f"[{sisters_psi_min[0]:.4e}, {sisters_psi_max[0]:.4e}], "
          f"mean={sisters_psi_mean[0]:.4e}")
    print(f"    ‖Δh_pert‖     = {delta_h_pert[0]:.4e}")
    print(f"    soeurs ‖Δh‖   = "
          f"[{sisters_h_min[0]:.4e}, {sisters_h_max[0]:.4e}], "
          f"mean={sisters_h_mean[0]:.4e}")

    # Quelques points dans la trajectoire
    sample_t = [10.0, 50.0, 100.0, 200.0, 290.0]
    print(f"\n  Évolution comparative :")
    print(f"  {'t_post':>8} {'‖Δψ_pert‖':>12} "
          f"{'soeurs_Δψ [min,max]':>30} "
          f"{'‖Δh_pert‖':>12} {'soeurs_Δh [min,max]':>30}")
    for t_target in sample_t:
        idx = int(np.searchsorted(times_compare, t_target))
        if idx >= len(times_compare):
            continue
        t_actual = times_compare[idx]
        print(f"  {t_actual:>8.2f} "
              f"{delta_psi_pert[idx]:>12.4e} "
              f"{f'[{sisters_psi_min[idx]:.2e}, {sisters_psi_max[idx]:.2e}]':>30} "
              f"{delta_h_pert[idx]:>12.4e} "
              f"{f'[{sisters_h_min[idx]:.2e}, {sisters_h_max[idx]:.2e}]':>30}")

    # Question clé : ‖Δh_pert‖ tombe-t-il dans [min, max] des soeurs ?
    print(f"\n  Position de la perturbée vis-à-vis de l'ensemble soeurs :")
    n_in_psi = sum(
        1 for k in range(n_compare)
        if sisters_psi_min[k] <= delta_psi_pert[k] <= sisters_psi_max[k]
    )
    n_in_h = sum(
        1 for k in range(n_compare)
        if sisters_h_min[k] <= delta_h_pert[k] <= sisters_h_max[k]
    )
    print(f"    fraction d'instants où ‖Δψ_pert‖ ∈ [soeurs] : "
          f"{n_in_psi}/{n_compare} = {n_in_psi/n_compare:.2%}")
    print(f"    fraction d'instants où ‖Δh_pert‖ ∈ [soeurs] : "
          f"{n_in_h}/{n_compare} = {n_in_h/n_compare:.2%}")

    # Aussi : ratio amplitude perturbée / amplitude moyenne soeurs
    ratio_psi_final = delta_psi_pert[-1] / max(sisters_psi_mean[-1], 1e-30)
    ratio_h_final = delta_h_pert[-1] / max(sisters_h_mean[-1], 1e-30)
    print(f"\n  Ratio à t_post = {times_compare[-1]:.1f} :")
    print(f"    ‖Δψ_pert‖ / mean(soeurs_Δψ) = {ratio_psi_final:.3f}")
    print(f"    ‖Δh_pert‖ / mean(soeurs_Δh) = {ratio_h_final:.3f}")

    return {
        "params": {
            "beta": beta, "D": D, "gamma": gamma, "h0": h0,
            "dt": dt, "t_stab": t_stab, "t_observe": t_observe,
            "K_sisters": K_sisters,
        },
        "times_compare": [float(t) for t in times_compare],
        "delta_psi_pert": [float(x) for x in delta_psi_pert],
        "delta_h_pert": [float(x) for x in delta_h_pert],
        "sisters_psi_mean": [float(x) for x in sisters_psi_mean],
        "sisters_psi_min": [float(x) for x in sisters_psi_min],
        "sisters_psi_max": [float(x) for x in sisters_psi_max],
        "sisters_h_mean": [float(x) for x in sisters_h_mean],
        "sisters_h_min": [float(x) for x in sisters_h_min],
        "sisters_h_max": [float(x) for x in sisters_h_max],
        "fraction_psi_in_sisters_range": n_in_psi / n_compare,
        "fraction_h_in_sisters_range": n_in_h / n_compare,
        "ratio_psi_final": ratio_psi_final,
        "ratio_h_final": ratio_h_final,
    }


if __name__ == "__main__":
    out = run_test()
    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_property_1_geometric_reformulation.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
