"""
6d-β cycle 2 — h_min(t) à β=55.

Hypothèse à éprouver (formulée par Alex) :
La transition entre régime uniforme et régime structuré pourrait
être corrélée à l'effondrement local de h_min vers ~0.

Mécanisme possible : la structuration apparaît quand certaines
régions de h deviennent morphologiquement quasi-éteintes.

Test minimal :
- β = 55 (un seul point intermédiaire, entre 50 uniforme et 60 structuré)
- Suivre h_min(t), h_max(t), h_mean(t), ψ_min(t), ψ_max(t) dans le temps
- Observer s'il y a un décrochage local de h_min

Pas de balayage. Un seul test mécanistique.
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


def run_test(beta):
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    t_total = 500.0

    psi = make_psi_centered(sigma=1.5)
    h = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max_init = float(psi.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max_init + gamma))
    n_total = int(t_total / dt)
    snapshot_every = max(1, n_total // 500)  # dense pour bien voir h_min(t)

    times = [0.0]
    h_min_list = [float(h.min())]
    h_max_list = [float(h.max())]
    h_mean_list = [float(h.mean())]
    psi_min_list = [float(psi.min())]
    psi_max_list = [float(psi.max())]
    # Localiser quelle cellule devient le min de h
    h_argmin_history = []

    for k in range(1, n_total + 1):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        if k % snapshot_every == 0 or k == n_total:
            times.append(k * dt)
            h_min_list.append(float(h.min()))
            h_max_list.append(float(h.max()))
            h_mean_list.append(float(h.mean()))
            psi_min_list.append(float(psi.min()))
            psi_max_list.append(float(psi.max()))
            argmin_flat = int(np.argmin(h))
            argmin = np.unravel_index(argmin_flat, h.shape)
            h_argmin_history.append(tuple(int(x) for x in argmin))

    return {
        "beta": beta,
        "dt": dt,
        "times": times,
        "h_min": h_min_list,
        "h_max": h_max_list,
        "h_mean": h_mean_list,
        "psi_min": psi_min_list,
        "psi_max": psi_max_list,
        "h_argmin_history": h_argmin_history,
    }


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"h_min(t) à β=55 — chercher un décrochage morphologique local")
    print(f"{'='*70}")

    out = run_test(beta=55.0)
    times = out["times"]
    h_min = out["h_min"]
    h_max = out["h_max"]
    h_mean = out["h_mean"]
    psi_min = out["psi_min"]
    psi_max = out["psi_max"]

    print(f"\n  β=55, dt={out['dt']:.5f}")
    print(f"  {len(times)} snapshots sur t ∈ [0, {times[-1]:.1f}]")

    # Aperçu temporel
    print(f"\n  Évolution :")
    print(f"  {'t':>8} {'h_min':>14} {'h_max':>10} {'h_mean':>10} "
          f"{'ψ_min':>12} {'ψ_max':>12}")
    sample_t = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 499.0]
    for t_target in sample_t:
        idx = int(np.searchsorted(times, t_target))
        if idx >= len(times):
            continue
        print(f"  {times[idx]:>8.2f} {h_min[idx]:>14.4e} "
              f"{h_max[idx]:>10.4f} {h_mean[idx]:>10.4f} "
              f"{psi_min[idx]:>12.4e} {psi_max[idx]:>12.4e}")

    # Caractérisation du décrochage de h_min
    print(f"\n  Analyse de h_min(t) :")
    h_min_array = np.array(h_min)
    log_h_min = np.log10(np.maximum(h_min_array, 1e-300))
    print(f"    log10(h_min) initial  = {log_h_min[0]:.2f}")
    print(f"    log10(h_min) final    = {log_h_min[-1]:.2f}")
    print(f"    chute totale en log10 = {log_h_min[0] - log_h_min[-1]:.2f}")

    # Identifier moment de décrochage : où la dérivée de log_h_min est maximale
    log_h_min_diffs = np.diff(log_h_min)
    # Décrochage = saut négatif important
    idx_strongest_drop = int(np.argmin(log_h_min_diffs))
    if idx_strongest_drop < len(times) - 1:
        t_drop = times[idx_strongest_drop]
        drop_amount = log_h_min_diffs[idx_strongest_drop]
        print(f"\n    Plus fort décrochage (sur 1 pas snapshot) :")
        print(f"      à t ≈ {t_drop:.2f}")
        print(f"      Δlog10(h_min) = {drop_amount:.3f}")

    # Le décrochage est-il brusque ou progressif ?
    # On compare la pente moyenne entre première et deuxième moitié
    n_half = len(log_h_min) // 2
    slope_first_half = ((log_h_min[n_half] - log_h_min[0]) /
                        (times[n_half] - times[0] + 1e-30))
    slope_second_half = ((log_h_min[-1] - log_h_min[n_half]) /
                         (times[-1] - times[n_half] + 1e-30))
    print(f"\n    Pente log10(h_min) / unité de temps :")
    print(f"      première moitié : {slope_first_half:.4f}")
    print(f"      seconde moitié  : {slope_second_half:.4f}")

    # Trace de la cellule qui devient h_min
    if out["h_argmin_history"]:
        last_argmin = out["h_argmin_history"][-1]
        print(f"\n    Position du h_min final : cellule {last_argmin}")
        # Stabilité de l'argmin
        all_argmins = out["h_argmin_history"]
        unique_argmins = list(set(all_argmins))
        print(f"    Nombre de cellules distinctes argmin sur l'évolution : "
              f"{len(unique_argmins)}")
        if len(unique_argmins) < 10:
            print(f"    Cellules argmin observées : {unique_argmins}")

    # Lecture
    print(f"\n{'='*70}")
    print(f"LECTURE")
    print(f"{'='*70}")
    final_h_min = h_min[-1]
    final_psi_inhomo = psi_max[-1] / max(psi_min[-1], 1e-30)
    print(f"\n  État final β=55 :")
    print(f"    h_min          = {final_h_min:.4e}")
    print(f"    ψ_inhomogénéité = {final_psi_inhomo:.2f}")

    # Comparaison aux deux extrêmes connus :
    # β=50 : ψ uniforme (inhomo=1), h uniforme (~0.60)
    # β=60 : ψ structuré (inhomo=2.58), h_min ~ 1e-50
    print(f"\n  Comparaison aux régimes connus :")
    print(f"    β=50 : ψ_inhomo=1.00, h_min ≈ 0.60")
    print(f"    β=55 (test): ψ_inhomo={final_psi_inhomo:.2f}, "
          f"h_min = {final_h_min:.4e}")
    print(f"    β=60 : ψ_inhomo=2.58, h_min ≈ 1e-50")

    if final_h_min > 0.1:
        regime = "homogène (proche β=50)"
    elif final_h_min < 1e-30:
        regime = "structuré quasi-singulier (proche β=60)"
    elif final_h_min < 0.01:
        regime = "intermédiaire structuré"
    else:
        regime = "intermédiaire"
    print(f"\n  Régime apparent : {regime}")

    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_h_min_dynamics_beta55.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
