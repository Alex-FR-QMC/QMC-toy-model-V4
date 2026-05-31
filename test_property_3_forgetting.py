"""
6d-β cycle 2 — Propriété 3 : "productive forgetting".

Question (formulée avec Alex) :
Les perturbations passées modifient-elles la sensibilité ultérieure
du système ?

Réopérationnalisation possible (tension active) :
- "trace persistante mais morphologiquement neutre" plutôt que
  "disparition de trace"
- saturation locale vs oubli distribué

Protocole :
1. Stabilisation à β=60 sur 100 unités de temps
2. Trois perturbations identiques en amplitude :
   - P1 à t=100 sur cellule (2,2,2) — centre, là où ψ_argmax
   - P2 à t=200 sur cellule (2,2,2) — même cellule
   - P3 à t=300 sur cellule (0,0,0) — coin, lieu différent
3. Comparaison à trajectoire de référence (sans perturbation)
4. Mesure de Δψ(t) et Δh(t) à chaque instant

Lectures multiples possibles :
- P2 = P1 ET P3 = P1 → oubli distribué (réel)
- P2 ≈ 0 ET P3 = P1 → saturation locale (pas oubli)
- P2 + P1 vs P1 → accumulation (pas oubli)
- évolution de |diff| entre perturbations → trace active vs morte

β=60 (régime structuré, là où la mémoire géométrique apparaît)
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


def apply_perturbation_psi(psi, indices, eps_relative=0.01):
    """Perturbation : psi[indices] *= (1+eps), puis renormalisation."""
    psi_pert = psi.copy()
    i, j, k = indices
    psi_pert[i, j, k] *= (1.0 + eps_relative)
    psi_pert /= psi_pert.sum()
    return psi_pert


def run_test():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0
    t_total = 500.0
    eps_rel = 0.01

    # Instants des 3 perturbations
    t_P1 = 100.0
    t_P2 = 200.0
    t_P3 = 300.0
    loc_P1 = (2, 2, 2)
    loc_P2 = (2, 2, 2)
    loc_P3 = (0, 0, 0)

    psi_init = make_psi_centered(sigma=1.5)
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max_init = float(psi_init.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max_init + gamma))
    n_total = int(t_total / dt)
    n_P1 = int(t_P1 / dt)
    n_P2 = int(t_P2 / dt)
    n_P3 = int(t_P3 / dt)
    snapshot_every = max(1, n_total // 600)

    print(f"{'='*70}")
    print(f"Propriété 3 — Productive forgetting (test β)")
    print(f"  β={beta}, dt={dt:.5f}")
    print(f"  P1: t={t_P1} sur {loc_P1}")
    print(f"  P2: t={t_P2} sur {loc_P2} (même cellule que P1)")
    print(f"  P3: t={t_P3} sur {loc_P3} (cellule différente)")
    print(f"  Amplitude relative : {eps_rel*100}%")
    print(f"{'='*70}\n")

    # === Référence : pas de perturbation ===
    psi_R = psi_init.copy()
    h_R = h_init.copy()
    # === Branche perturbée : avec les 3 perturbations ===
    psi_P = psi_init.copy()
    h_P = h_init.copy()

    times = [0.0]
    delta_psi = [0.0]
    delta_h = [0.0]
    psi_min_R = [float(psi_R.min())]
    psi_max_R = [float(psi_R.max())]
    h_min_R = [float(h_R.min())]

    # Marqueurs des perturbations
    perturbation_events = []

    for k in range(1, n_total + 1):
        # Référence
        psi_R, h_R = step(psi_R, h_R, D, beta, gamma, h0, dt)
        # Perturbée
        psi_P, h_P = step(psi_P, h_P, D, beta, gamma, h0, dt)

        # Application des perturbations à des pas spécifiques
        if k == n_P1:
            psi_P = apply_perturbation_psi(psi_P, loc_P1, eps_rel)
            perturbation_events.append(("P1", k * dt, loc_P1,
                                         float(np.linalg.norm(psi_P - psi_R))))
        elif k == n_P2:
            delta_before = float(np.linalg.norm(psi_P - psi_R))
            psi_P = apply_perturbation_psi(psi_P, loc_P2, eps_rel)
            delta_after = float(np.linalg.norm(psi_P - psi_R))
            perturbation_events.append(("P2", k * dt, loc_P2,
                                         delta_after,
                                         delta_before))
        elif k == n_P3:
            delta_before = float(np.linalg.norm(psi_P - psi_R))
            psi_P = apply_perturbation_psi(psi_P, loc_P3, eps_rel)
            delta_after = float(np.linalg.norm(psi_P - psi_R))
            perturbation_events.append(("P3", k * dt, loc_P3,
                                         delta_after,
                                         delta_before))

        if k % snapshot_every == 0 or k == n_total:
            times.append(k * dt)
            delta_psi.append(float(np.linalg.norm(psi_P - psi_R)))
            delta_h.append(float(np.linalg.norm(h_P - h_R)))
            psi_min_R.append(float(psi_R.min()))
            psi_max_R.append(float(psi_R.max()))
            h_min_R.append(float(h_R.min()))

    times = np.array(times)
    delta_psi = np.array(delta_psi)
    delta_h = np.array(delta_h)

    # === Mesures aux instants clés ===
    def get_at_time(t_target, arr):
        idx = int(np.searchsorted(times, t_target))
        if idx >= len(arr):
            return None
        return float(arr[idx])

    # Plateaux entre perturbations
    print(f"  Évolution de |Δψ| et |Δh| :")
    print(f"  {'t':>8} {'|Δψ|':>14} {'|Δh|':>14} {'phase':>20}")

    key_t = [50.0, 99.5, t_P1, t_P1 + 5, t_P1 + 50, t_P1 + 99,
             t_P2, t_P2 + 5, t_P2 + 50, t_P2 + 99,
             t_P3, t_P3 + 5, t_P3 + 50, t_P3 + 99,
             499.0]
    for t_t in key_t:
        idx = int(np.searchsorted(times, t_t))
        if idx >= len(times):
            continue
        if times[idx] < t_P1:
            phase = "avant P1"
        elif times[idx] < t_P2:
            phase = "P1 → P2"
        elif times[idx] < t_P3:
            phase = "P2 → P3"
        else:
            phase = "après P3"
        print(f"  {times[idx]:>8.2f} {delta_psi[idx]:>14.4e} "
              f"{delta_h[idx]:>14.4e} {phase:>20}")

    # Saturations entre perturbations (moyenne sur la 2ème moitié de chaque intervalle)
    print(f"\n  Niveaux de saturation par phase :")

    def mean_in_window(t_low, t_high):
        mask = (times >= t_low) & (times <= t_high)
        if mask.sum() == 0:
            return None, None
        return (float(delta_psi[mask].mean()),
                float(delta_h[mask].mean()))

    # Phase P1 → P2 (moitié finale, donc déjà saturée)
    psi_lvl_P1, h_lvl_P1 = mean_in_window(t_P1 + 50, t_P2 - 1)
    # Phase P2 → P3 (moitié finale)
    psi_lvl_P2, h_lvl_P2 = mean_in_window(t_P2 + 50, t_P3 - 1)
    # Phase après P3 (moitié finale)
    psi_lvl_P3, h_lvl_P3 = mean_in_window(t_P3 + 50, t_total - 1)

    print(f"    saturation P1→P2 : |Δψ| ≈ {psi_lvl_P1:.4e}, "
          f"|Δh| ≈ {h_lvl_P1:.4e}")
    print(f"    saturation P2→P3 : |Δψ| ≈ {psi_lvl_P2:.4e}, "
          f"|Δh| ≈ {h_lvl_P2:.4e}")
    print(f"    saturation après P3 : |Δψ| ≈ {psi_lvl_P3:.4e}, "
          f"|Δh| ≈ {h_lvl_P3:.4e}")

    # Effet cumulatif ?
    print(f"\n  Effets cumulatifs :")
    if psi_lvl_P1 and psi_lvl_P2:
        ratio_P2_P1 = psi_lvl_P2 / psi_lvl_P1
        print(f"    |Δψ|(P2→P3) / |Δψ|(P1→P2) = {ratio_P2_P1:.3f}")
        if 0.9 < ratio_P2_P1 < 1.1:
            print(f"      → P2 (même cellule que P1) ne cumule pas")
            print(f"      → soit oubli, soit saturation locale")
        elif ratio_P2_P1 > 1.5:
            print(f"      → P2 cumule significativement")
            print(f"      → trace de P1 conservée")
        elif ratio_P2_P1 < 0.5:
            print(f"      → P2 effacée plus que P1 (inattendu)")

    if psi_lvl_P2 and psi_lvl_P3:
        # Si la perturbation P3 sur cellule différente produit un effet
        # plus grand que P2 sur même cellule, c'est de la saturation locale
        ratio_P3_P2 = psi_lvl_P3 / psi_lvl_P2
        print(f"    |Δψ|(après P3) / |Δψ|(P2→P3) = {ratio_P3_P2:.3f}")
        if ratio_P3_P2 > 1.3:
            print(f"      → P3 ajoute plus que P2 (cellule différente)")
            print(f"      → DISCRIMINATION : la 2ème perturbation au même")
            print(f"        endroit n'avait pas d'effet additionnel,")
            print(f"        mais une perturbation ailleurs en a un")
            print(f"      → SATURATION LOCALE plutôt qu'OUBLI distribué")
        elif 0.9 < ratio_P3_P2 < 1.1:
            print(f"      → P3 (cellule différente) ne cumule pas non plus")
            print(f"      → OUBLI distribué : le système réintègre toute")
            print(f"        perturbation locale au même état stationnaire")
        else:
            print(f"      → comportement intermédiaire")

    # Évolution entre perturbations : la trace est-elle active ou morte ?
    print(f"\n  Évolution de |Δψ| entre P1 et P2 :")
    print(f"    |Δψ| juste après P1 : {get_at_time(t_P1 + 1, delta_psi):.4e}")
    print(f"    |Δψ| mi-phase       : {get_at_time((t_P1+t_P2)/2, delta_psi):.4e}")
    print(f"    |Δψ| juste avant P2 : {get_at_time(t_P2 - 1, delta_psi):.4e}")

    return {
        "params": {
            "beta": beta, "dt": dt, "t_total": t_total,
            "perturbations": [
                {"label": "P1", "t": t_P1, "loc": list(loc_P1)},
                {"label": "P2", "t": t_P2, "loc": list(loc_P2)},
                {"label": "P3", "t": t_P3, "loc": list(loc_P3)},
            ],
            "eps_relative": eps_rel,
        },
        "times": [float(t) for t in times],
        "delta_psi": [float(x) for x in delta_psi],
        "delta_h": [float(x) for x in delta_h],
        "saturation_levels": {
            "P1_to_P2": {"psi": psi_lvl_P1, "h": h_lvl_P1},
            "P2_to_P3": {"psi": psi_lvl_P2, "h": h_lvl_P2},
            "after_P3": {"psi": psi_lvl_P3, "h": h_lvl_P3},
        },
    }


if __name__ == "__main__":
    out = run_test()
    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_property_3_forgetting.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
