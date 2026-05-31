"""
6d-β cycle 2 — Propriété 2 : structure temporelle relationnelle ψ↔h.

Question (formulée avec Alex) :
Quelle structure temporelle relationnelle existe entre les
transformations de ψ et celles de h pendant la sélection de
l'attracteur ?

PAS : "h suit-il ψ avec un retard τ ?"
PAS : "quel est τ_h vs τ_ψ ?"

On ne présuppose ni causalité orientée, ni hiérarchie temporelle.
On observe.

Régime : β=55 (le seul où on a vu une dynamique non monotone
sans figement quasi-singulier — phase de sélection observable).

Mesures :
- Sψ(t) = ||ψ(t) − mean(ψ(t))||  (inhomogénéité globale de ψ)
- Sh(t) = ||h(t) − mean(h(t))||  (inhomogénéité globale de h)

Caveat noté : ces signatures sont des choix non neutres. Elles
mesurent l'inhomogénéité globale et peuvent manquer des
réorganisations locales. Un résultat négatif (pas de structure
temporelle relationnelle observable) peut venir soit du système,
soit du choix de signature.

Observation des données brutes d'abord. Lecture après.
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


def signature_inhomogeneity(field):
    """||field - mean(field)|| (signature scalaire d'inhomogénéité)."""
    return float(np.linalg.norm(field - field.mean()))


def run_test():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 55.0
    t_total = 200.0  # phase de sélection ~ [0, 100], + marge

    psi = make_psi_centered(sigma=1.5)
    h = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_total = int(t_total / dt)
    # Snapshots denses pour bien voir la phase de sélection
    snapshot_every = max(1, n_total // 800)

    print(f"{'='*70}")
    print(f"Propriété 2 — structure temporelle relationnelle ψ↔h")
    print(f"  β=55, dt={dt:.5f}, t_total={t_total}")
    print(f"  snapshots tous les {snapshot_every} pas, "
          f"≈ tous les {dt * snapshot_every:.3f} unités")
    print(f"{'='*70}\n")

    times = [0.0]
    S_psi = [signature_inhomogeneity(psi)]
    S_h = [signature_inhomogeneity(h)]
    h_min_list = [float(h.min())]

    for k in range(1, n_total + 1):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        if k % snapshot_every == 0 or k == n_total:
            times.append(k * dt)
            S_psi.append(signature_inhomogeneity(psi))
            S_h.append(signature_inhomogeneity(h))
            h_min_list.append(float(h.min()))

    times = np.array(times)
    S_psi = np.array(S_psi)
    S_h = np.array(S_h)
    h_min_arr = np.array(h_min_list)

    # === Observations brutes ===
    print(f"  Évolution des signatures :")
    print(f"  {'t':>8} {'Sψ':>14} {'Sh':>14} {'h_min':>14}")
    sample_t = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0,
                75.0, 100.0, 150.0, 199.0]
    for t_target in sample_t:
        idx = int(np.searchsorted(times, t_target))
        if idx >= len(times):
            continue
        print(f"  {times[idx]:>8.2f} {S_psi[idx]:>14.4e} "
              f"{S_h[idx]:>14.4e} {h_min_arr[idx]:>14.4e}")

    # === Extrema locaux : maxima de Sψ et de Sh ===
    # Argmax (sur la phase initiale, avant éventuelle stabilisation)
    idx_max_psi = int(np.argmax(S_psi))
    idx_max_h = int(np.argmax(S_h))
    t_max_psi = float(times[idx_max_psi])
    t_max_h = float(times[idx_max_h])
    S_psi_max = float(S_psi[idx_max_psi])
    S_h_max = float(S_h[idx_max_psi]) if idx_max_psi < len(S_h) else None
    S_h_at_psi_max = float(S_h[idx_max_psi])

    print(f"\n  Extrema :")
    print(f"    max Sψ = {S_psi_max:.4e} à t = {t_max_psi:.3f}")
    print(f"    max Sh = {float(S_h[idx_max_h]):.4e} à t = {t_max_h:.3f}")
    decalage_max = t_max_h - t_max_psi
    print(f"    décalage temporel des maxima : t_max_h − t_max_psi "
          f"= {decalage_max:+.3f}")

    # === Cross-correlation entre dSψ/dt et dSh/dt ===
    # Pour mesurer la corrélation des taux de transformation
    dS_psi = np.diff(S_psi) / np.diff(times)
    dS_h = np.diff(S_h) / np.diff(times)

    # Normalisation
    if dS_psi.std() > 0 and dS_h.std() > 0:
        dS_psi_n = (dS_psi - dS_psi.mean()) / dS_psi.std()
        dS_h_n = (dS_h - dS_h.mean()) / dS_h.std()

        # Cross-correlation pour lags entre -50 et +50 snapshots
        max_lag = min(50, len(dS_psi) // 4)
        lags = np.arange(-max_lag, max_lag + 1)
        cross_corr = []
        for lag in lags:
            if lag < 0:
                # dS_h précède dS_psi
                a = dS_psi_n[-lag:]
                b = dS_h_n[:len(a)]
            elif lag > 0:
                # dS_psi précède dS_h
                a = dS_psi_n[:len(dS_psi_n) - lag]
                b = dS_h_n[lag:lag + len(a)]
            else:
                a = dS_psi_n
                b = dS_h_n
            if len(a) > 0 and len(b) == len(a):
                cross_corr.append(float(np.mean(a * b)))
            else:
                cross_corr.append(0.0)
        cross_corr = np.array(cross_corr)

        idx_best = int(np.argmax(np.abs(cross_corr)))
        best_lag = int(lags[idx_best])
        best_corr = float(cross_corr[idx_best])
        best_lag_time = best_lag * float(np.mean(np.diff(times)))

        print(f"\n  Cross-correlation des taux dSψ/dt et dSh/dt :")
        print(f"    pic à lag = {best_lag} snapshots "
              f"≈ {best_lag_time:.3f} unités de temps")
        print(f"    corrélation au pic = {best_corr:+.4f}")
        if best_lag > 0:
            print(f"    interprétation : dSψ/dt précède dSh/dt de "
                  f"{best_lag_time:.3f}")
        elif best_lag < 0:
            print(f"    interprétation : dSh/dt précède dSψ/dt de "
                  f"{-best_lag_time:.3f}")
        else:
            print(f"    interprétation : synchronie au pas snapshot près")

        # Profil de cross-correlation
        print(f"\n    Profil de cross-corrélation autour du pic :")
        for offset in [-10, -5, -2, -1, 0, 1, 2, 5, 10]:
            li = idx_best + offset
            if 0 <= li < len(lags):
                print(f"      lag={lags[li]:+4d} : "
                      f"corr={cross_corr[li]:+.4f}")
    else:
        cross_corr = None
        best_lag = None
        best_corr = None
        best_lag_time = None
        print(f"\n  Cross-correlation non calculable (variances nulles)")

    # === Monotonicité de chaque signature ===
    # h_min était non monotone (creux puis remontée). Et Sψ, Sh ?
    psi_monotone_decrease_after_max = True
    for i in range(idx_max_psi + 1, len(S_psi)):
        if S_psi[i] > S_psi[i-1] + 1e-10:
            psi_monotone_decrease_after_max = False
            break

    h_monotone_decrease_after_max = True
    for i in range(idx_max_h + 1, len(S_h)):
        if S_h[i] > S_h[i-1] + 1e-10:
            h_monotone_decrease_after_max = False
            break

    print(f"\n  Monotonicité post-pic :")
    print(f"    Sψ monotone décroissante après son max ? "
          f"{psi_monotone_decrease_after_max}")
    print(f"    Sh monotone décroissante après son max ? "
          f"{h_monotone_decrease_after_max}")

    # Valeurs finales
    print(f"\n  Valeurs finales :")
    print(f"    Sψ(t={times[-1]:.1f}) = {S_psi[-1]:.4e}")
    print(f"    Sh(t={times[-1]:.1f}) = {S_h[-1]:.4e}")
    print(f"    h_min(t={times[-1]:.1f}) = {h_min_arr[-1]:.4e}")
    print(f"\n  Rappel : β=55 converge vers attracteur HOMOGÈNE")
    print(f"    donc Sψ et Sh doivent toutes deux décroître vers 0")

    return {
        "params": {
            "beta": beta, "D": D, "gamma": gamma, "h0": h0,
            "dt": dt, "t_total": t_total,
        },
        "times": [float(t) for t in times],
        "S_psi": [float(x) for x in S_psi],
        "S_h": [float(x) for x in S_h],
        "h_min": [float(x) for x in h_min_arr],
        "extrema": {
            "t_max_S_psi": t_max_psi,
            "t_max_S_h": t_max_h,
            "S_psi_max": S_psi_max,
            "S_h_max": float(S_h[idx_max_h]),
            "decalage_max_S_h_minus_S_psi": decalage_max,
        },
        "cross_correlation": {
            "best_lag_snapshots": best_lag,
            "best_lag_time": best_lag_time,
            "best_corr": best_corr,
        } if cross_corr is not None else None,
        "monotonicity_after_peak": {
            "S_psi_monotone_decrease": bool(psi_monotone_decrease_after_max),
            "S_h_monotone_decrease": bool(h_monotone_decrease_after_max),
        },
    }


if __name__ == "__main__":
    out = run_test()
    output_dir = REPO_ROOT / "results" / "phase6d_beta_cycle_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_property_2_relational_beta55.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
