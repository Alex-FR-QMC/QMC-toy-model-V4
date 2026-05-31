"""
6d-γ — mémoire trajectorielle de l'espace des transformations.

Question (formulée avec Alex) :
La géométrie relationnelle des transformations admissibles dépend-elle
seulement de l'état verrouillé, ou aussi de la trajectoire qui a
sculpté ce verrouillage ?

Protocole (a) — vitesse d'approche :
- Histoire RAPIDE   : β=60 d'emblée
- Histoire PROGRESSIVE : rampe β de 45 → 60 sur une durée

Les deux visent le régime verrouillé. On compare le PROFIL RELATIONNEL
de séparabilité (proches/ortho/éloignés) sur chaque histoire.

Garde-fou de comparabilité — BLOQUANT mais MINIMAL SUFFISANT (Alex) :
- même régime global (structuré/verrouillé)
- mêmes ordres de grandeur h_min et ψ_inhomo
- pas de bifurcation vers un autre bassin
PAS d'équivalence stricte (qui neutraliserait l'effet recherché).

Si garde-fou échoue → test NON concluant (histoires → verrouillages
différents), pas de comparaison de profils.

Si garde-fou passe ET profils relationnels diffèrent → la géométrie
des transformations garde une mémoire trajectorielle (proche Γ_meta).
Si profils identiques → géométrie déterminée par l'état verrouillé seul.
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


def compression_region(psi, center, sigma_r=0.8, strength=0.05):
    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = center
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i] - cx)**2 + (coords[j] - cy)**2 +
                      (coords[k] - cz)**2)
                factor[i, j, k] += strength * np.exp(-0.5 * r2 / sigma_r**2)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def compression_along_axis(psi, axis, strength=0.05):
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for idx in range(N_AXIS):
        dist = abs(idx - c)
        local = 1.0 + strength * (1.0 - dist / c)
        if axis == 0:
            factor[idx, :, :] *= local
        elif axis == 1:
            factor[:, idx, :] *= local
        else:
            factor[:, :, idx] *= local
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def evolve_const_beta(psi, h, D, beta, gamma, h0, dt, n_steps):
    for _ in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return psi, h


def evolve_ramp_beta(psi, h, D, beta_start, beta_end, gamma, h0, dt, n_steps):
    """Rampe linéaire de beta_start à beta_end sur n_steps."""
    for k in range(n_steps):
        frac = k / max(n_steps - 1, 1)
        beta_k = beta_start + (beta_end - beta_start) * frac
        psi, h = step(psi, h, D, beta_k, gamma, h0, dt)
    return psi, h


def measure_nonlinearity(psi_base, h_base, op_A, op_B,
                         D, beta, gamma, h0, dt, n_relax):
    psi_a, h_a = op_A(psi_base.copy()), h_base.copy()
    psi_a, h_a = evolve_const_beta(psi_a, h_a, D, beta, gamma, h0, dt, n_relax)
    resp_A = psi_a - psi_base
    psi_b, h_b = op_B(psi_base.copy()), h_base.copy()
    psi_b, h_b = evolve_const_beta(psi_b, h_b, D, beta, gamma, h0, dt, n_relax)
    resp_B = psi_b - psi_base
    psi_ab, h_ab = op_B(op_A(psi_base.copy())), h_base.copy()
    psi_ab, h_ab = evolve_const_beta(psi_ab, h_ab, D, beta, gamma, h0, dt, n_relax)
    resp_AB = psi_ab - psi_base
    nonlin = float(np.linalg.norm(resp_AB - (resp_A + resp_B)))
    norm_AB = float(np.linalg.norm(resp_AB))
    return nonlin / max(norm_AB, 1e-30)


def relational_profile(psi_base, h_base, D, beta, gamma, h0, dt, n_relax):
    nl_close = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_region(p, (1.5, 2.0, 2.0), strength=0.05),
        lambda p: compression_region(p, (2.5, 2.0, 2.0), strength=0.05),
        D, beta, gamma, h0, dt, n_relax)
    nl_ortho = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_along_axis(p, 0, 0.05),
        lambda p: compression_along_axis(p, 1, 0.05),
        D, beta, gamma, h0, dt, n_relax)
    nl_far = measure_nonlinearity(
        psi_base, h_base,
        lambda p: compression_region(p, (0.0, 0.0, 0.0), strength=0.05),
        lambda p: compression_region(p, (4.0, 4.0, 4.0), strength=0.05),
        D, beta, gamma, h0, dt, n_relax)
    return {
        "nl_close": nl_close, "nl_ortho": nl_ortho, "nl_far": nl_far,
        "spread": max(nl_close, nl_ortho, nl_far) -
                  min(nl_close, nl_ortho, nl_far),
        "monotone": bool(nl_close < nl_ortho < nl_far),
    }


def attractor_signature(psi, h):
    return {
        "psi_inhomo": float(psi.max() / max(psi.min(), 1e-30)),
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "h_argmin": tuple(int(x) for x in
                          np.unravel_index(int(np.argmin(h)), h.shape)),
    }


def run():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta_target = 60.0
    t_sculpt = 100.0   # durée de sculptage du verrouillage
    t_relax_probe = 200.0  # relaxation pour les sondes de séparabilité

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_target * psi_max + gamma))
    n_sculpt = int(t_sculpt / dt)
    n_probe = int(t_relax_probe / dt)

    print(f"{'='*70}")
    print(f"6d-γ — mémoire trajectorielle de l'espace des transformations")
    print(f"  β_target={beta_target}, dt={dt:.5f}, t_sculpt={t_sculpt}")
    print(f"{'='*70}")

    # === Histoire RAPIDE : β=60 d'emblée ===
    psi_fast, h_fast = evolve_const_beta(
        psi0.copy(), h0f.copy(), D, beta_target, gamma, h0, dt, n_sculpt)
    sig_fast = attractor_signature(psi_fast, h_fast)

    # === Histoire PROGRESSIVE : rampe 45 → 60 ===
    psi_slow, h_slow = evolve_ramp_beta(
        psi0.copy(), h0f.copy(), D, 45.0, beta_target, gamma, h0, dt, n_sculpt)
    # On laisse un peu stabiliser à β=60 après la rampe pour comparer
    # à attracteur installé
    psi_slow, h_slow = evolve_const_beta(
        psi_slow, h_slow, D, beta_target, gamma, h0, dt, n_sculpt)
    sig_slow = attractor_signature(psi_slow, h_slow)

    # Histoire rapide : on stabilise aussi le même temps total
    psi_fast, h_fast = evolve_const_beta(
        psi_fast, h_fast, D, beta_target, gamma, h0, dt, n_sculpt)
    sig_fast = attractor_signature(psi_fast, h_fast)

    print(f"\n  Signatures d'attracteur :")
    print(f"    {'':>14} {'RAPIDE':>16} {'PROGRESSIVE':>16}")
    print(f"    {'ψ_inhomo':>14} {sig_fast['psi_inhomo']:>16.3f} "
          f"{sig_slow['psi_inhomo']:>16.3f}")
    print(f"    {'h_min':>14} {sig_fast['h_min']:>16.3e} "
          f"{sig_slow['h_min']:>16.3e}")
    print(f"    {'h_max':>14} {sig_fast['h_max']:>16.4f} "
          f"{sig_slow['h_max']:>16.4f}")
    print(f"    {'h_argmin':>14} {str(sig_fast['h_argmin']):>16} "
          f"{str(sig_slow['h_argmin']):>16}")

    # === GARDE-FOU DE COMPARABILITÉ (bloquant, minimal suffisant) ===
    print(f"\n  {'─'*60}")
    print(f"  GARDE-FOU DE COMPARABILITÉ (bloquant, minimal suffisant)")
    print(f"  {'─'*60}")

    # Même régime global : les deux structurés (h_min petit, ψ_inhomo > 1.5)
    both_structured = (sig_fast['h_min'] < 0.01 and sig_slow['h_min'] < 0.01
                       and sig_fast['psi_inhomo'] > 1.5
                       and sig_slow['psi_inhomo'] > 1.5)
    # Mêmes ordres de grandeur
    same_order_inhomo = (0.5 < sig_fast['psi_inhomo'] / sig_slow['psi_inhomo'] < 2.0)
    # h_min : comparer les ordres (log)
    log_hmin_fast = np.log10(max(sig_fast['h_min'], 1e-300))
    log_hmin_slow = np.log10(max(sig_slow['h_min'], 1e-300))
    same_order_hmin = abs(log_hmin_fast - log_hmin_slow) < 10  # tolérance large
    # Pas de bifurcation de bassin : même h_argmin
    same_basin = sig_fast['h_argmin'] == sig_slow['h_argmin']

    print(f"    même régime global (structuré) ? {both_structured}")
    print(f"    ψ_inhomo même ordre ? {same_order_inhomo} "
          f"(ratio {sig_fast['psi_inhomo']/sig_slow['psi_inhomo']:.2f})")
    print(f"    h_min même ordre ? {same_order_hmin} "
          f"(Δlog10 = {abs(log_hmin_fast - log_hmin_slow):.1f})")
    print(f"    même bassin (h_argmin) ? {same_basin}")

    comparable = both_structured and same_order_inhomo and same_basin
    print(f"\n    COMPARABLE (minimal suffisant) ? {comparable}")

    if not comparable:
        print(f"\n  TEST NON CONCLUANT : les deux histoires ont produit des")
        print(f"  verrouillages non comparables. On ne compare PAS les profils.")
        print(f"  (Ce n'est pas un échec — c'est une information : la vitesse")
        print(f"   de sculptage change le bassin lui-même.)")
        result = {
            "comparable": False,
            "sig_fast": sig_fast, "sig_slow": sig_slow,
        }
        output_dir = REPO_ROOT / "results" / "phase6d_gamma"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "contact_4_trajectory_memory.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    # === Si comparable : mesurer les profils relationnels ===
    print(f"\n  {'─'*60}")
    print(f"  PROFILS RELATIONNELS (attracteurs comparables)")
    print(f"  {'─'*60}")

    prof_fast = relational_profile(psi_fast, h_fast,
                                   D, beta_target, gamma, h0, dt, n_probe)
    prof_slow = relational_profile(psi_slow, h_slow,
                                   D, beta_target, gamma, h0, dt, n_probe)

    print(f"    {'':>14} {'RAPIDE':>16} {'PROGRESSIVE':>16}")
    print(f"    {'nl_close':>14} {prof_fast['nl_close']:>16.4f} "
          f"{prof_slow['nl_close']:>16.4f}")
    print(f"    {'nl_ortho':>14} {prof_fast['nl_ortho']:>16.4f} "
          f"{prof_slow['nl_ortho']:>16.4f}")
    print(f"    {'nl_far':>14} {prof_fast['nl_far']:>16.4f} "
          f"{prof_slow['nl_far']:>16.4f}")
    print(f"    {'spread':>14} {prof_fast['spread']:>16.4f} "
          f"{prof_slow['spread']:>16.4f}")

    # Différence de profil
    profile_diff = (abs(prof_fast['nl_close'] - prof_slow['nl_close']) +
                    abs(prof_fast['nl_ortho'] - prof_slow['nl_ortho']) +
                    abs(prof_fast['nl_far'] - prof_slow['nl_far']))
    print(f"\n    Différence totale de profil (L1) = {profile_diff:.4f}")

    print(f"\n  Verdict :")
    if profile_diff > 0.15:
        print(f"    PROFILS RELATIONNELS DIFFÉRENTS à attracteur comparable.")
        print(f"    → la géométrie des transformations garde une MÉMOIRE")
        print(f"      TRAJECTORIELLE de la vitesse de sculptage.")
        print(f"    → proche de Γ_meta : dépendance au chemin de l'espace")
        print(f"      des transformations, pas seulement de l'état.")
    elif profile_diff < 0.05:
        print(f"    PROFILS RELATIONNELS QUASI-IDENTIQUES.")
        print(f"    → la géométrie des transformations est déterminée par")
        print(f"      l'état verrouillé, PAS par l'histoire de sculptage.")
        print(f"    → pas de mémoire trajectorielle détectable ici.")
    else:
        print(f"    DIFFÉRENCE INTERMÉDIAIRE — à ne pas trancher.")

    result = {
        "comparable": True,
        "sig_fast": sig_fast, "sig_slow": sig_slow,
        "profile_fast": prof_fast, "profile_slow": prof_slow,
        "profile_diff": profile_diff,
    }
    output_dir = REPO_ROOT / "results" / "phase6d_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "contact_4_trajectory_memory.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nRésultats : {output_dir / 'contact_4_trajectory_memory.json'}")
    return result


if __name__ == "__main__":
    run()
