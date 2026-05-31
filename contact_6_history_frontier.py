"""
6d-γ — contact 6 : frontière expérimentale de la persistance d'histoire.

Question (Alex) :
"Quelle est la plus grande divergence trajectorielle que le garde-fou
de comparabilité autorise encore ?"

Reformulation conceptuelle (Alex) :
Ce qui se hiérarchise ici n'est peut-être pas la mémoire mais la
RÉACTIVABILITÉ. Le contact 4 montre une couche très réactive à
l'histoire (accessibilité du bassin) ; le contact 5 montre une couche
peu réactive (dynamique interne). Cela ressemble plus à 𝒢 qu'à la
mémoire classique.

Protocole :
- H_FAST : β=60 d'emblée (référence du contact 5)
- H_DIVERGENT : rampe β=50 → 60 sur 5 unités, puis β=60 jusqu'à t_sculpt

Plus divergent que contact 5 (qui était 55→60 sur 2 unités).
Pas une cartographie. Un test à un point plus extrême.

Garde-fou de comparabilité (identique au contact 5) :
- structuré ? h_min < 0.01 et ψ_inhomo > 1.5 dans les deux
- même ordre de grandeur
- même bassin (h_argmin)

Si garde-fou bloque → frontière trouvée entre 55→60 (passe) et 50→60
(échoue). C'est une information sur où la sensibilité au timing
devient critique.

Si garde-fou passe → mesurer le profil relationnel et comparer à
FAST. Si différence > seuil → persistance d'histoire détectée pour
cette paire plus contrastée. Si pas de différence → l'absence de
persistance vaut pour une plus large plage d'histoires, et la
distinction entre "réactivabilité de l'accessibilité" et "non-
réactivabilité de la dynamique interne" est renforcée.
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


def evolve_const(psi, h, D, beta, gamma, h0, dt, n_steps):
    for _ in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return psi, h


def evolve_ramp(psi, h, D, beta_start, beta_end, gamma, h0, dt, n_steps):
    for k in range(n_steps):
        frac = k / max(n_steps - 1, 1)
        beta_k = beta_start + (beta_end - beta_start) * frac
        psi, h = step(psi, h, D, beta_k, gamma, h0, dt)
    return psi, h


def attractor_signature(psi, h):
    return {
        "psi_inhomo": float(psi.max() / max(psi.min(), 1e-30)),
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "h_argmin": tuple(int(x) for x in
                          np.unravel_index(int(np.argmin(h)), h.shape)),
    }


def measure_nonlinearity(psi_base, h_base, op_A, op_B,
                         D, beta, gamma, h0, dt, n_relax):
    psi_a, h_a = op_A(psi_base.copy()), h_base.copy()
    psi_a, h_a = evolve_const(psi_a, h_a, D, beta, gamma, h0, dt, n_relax)
    resp_A = psi_a - psi_base
    psi_b, h_b = op_B(psi_base.copy()), h_base.copy()
    psi_b, h_b = evolve_const(psi_b, h_b, D, beta, gamma, h0, dt, n_relax)
    resp_B = psi_b - psi_base
    psi_ab, h_ab = op_B(op_A(psi_base.copy())), h_base.copy()
    psi_ab, h_ab = evolve_const(psi_ab, h_ab, D, beta, gamma, h0, dt, n_relax)
    resp_AB = psi_ab - psi_base
    nonlin = float(np.linalg.norm(resp_AB - (resp_A + resp_B)))
    norm_AB = float(np.linalg.norm(resp_AB))
    return nonlin / max(norm_AB, 1e-30)


def relational_profile(psi_base, h_base, D, beta, gamma, h0, dt, n_relax):
    return {
        "nl_close": measure_nonlinearity(
            psi_base, h_base,
            lambda p: compression_region(p, (1.5, 2.0, 2.0), strength=0.05),
            lambda p: compression_region(p, (2.5, 2.0, 2.0), strength=0.05),
            D, beta, gamma, h0, dt, n_relax),
        "nl_ortho": measure_nonlinearity(
            psi_base, h_base,
            lambda p: compression_along_axis(p, 0, 0.05),
            lambda p: compression_along_axis(p, 1, 0.05),
            D, beta, gamma, h0, dt, n_relax),
        "nl_far": measure_nonlinearity(
            psi_base, h_base,
            lambda p: compression_region(p, (0.0, 0.0, 0.0), strength=0.05),
            lambda p: compression_region(p, (4.0, 4.0, 4.0), strength=0.05),
            D, beta, gamma, h0, dt, n_relax),
    }


def run():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta_target = 60.0
    t_sculpt = 100.0
    t_ramp = 5.0      # plus long que contact 5 (qui était 2.0)
    beta_start_ramp = 50.0  # plus bas que contact 5 (qui était 55)

    psi0 = make_psi_centered(sigma=1.5)
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_target * psi_max + gamma))
    n_sculpt = int(t_sculpt / dt)
    n_ramp = int(t_ramp / dt)

    print(f"{'='*70}")
    print(f"6d-γ contact 6 — frontière expérimentale (paire plus divergente)")
    print(f"  H_FAST       : β=60 d'emblée")
    print(f"  H_DIVERGENT  : rampe β={beta_start_ramp}→{beta_target} sur t={t_ramp},")
    print(f"                  puis β={beta_target}")
    print(f"  Comparé à contact 5 : rampe 55→60 sur t=2.0 (passait le garde-fou)")
    print(f"{'='*70}\n")

    # H_FAST
    psi_fast, h_fast = evolve_const(psi0.copy(), h0f.copy(),
                                     D, beta_target, gamma, h0, dt, n_sculpt)
    sig_fast = attractor_signature(psi_fast, h_fast)

    # H_DIVERGENT
    psi_div, h_div = evolve_ramp(psi0.copy(), h0f.copy(),
                                  D, beta_start_ramp, beta_target,
                                  gamma, h0, dt, n_ramp)
    psi_div, h_div = evolve_const(psi_div, h_div,
                                   D, beta_target, gamma, h0, dt,
                                   n_sculpt - n_ramp)
    sig_div = attractor_signature(psi_div, h_div)

    print(f"  Signatures d'attracteur :")
    print(f"    {'':>14} {'H_FAST':>16} {'H_DIVERGENT':>16}")
    print(f"    {'ψ_inhomo':>14} {sig_fast['psi_inhomo']:>16.3f} "
          f"{sig_div['psi_inhomo']:>16.3f}")
    print(f"    {'h_min':>14} {sig_fast['h_min']:>16.3e} "
          f"{sig_div['h_min']:>16.3e}")
    print(f"    {'h_max':>14} {sig_fast['h_max']:>16.4f} "
          f"{sig_div['h_max']:>16.4f}")
    print(f"    {'h_argmin':>14} {str(sig_fast['h_argmin']):>16} "
          f"{str(sig_div['h_argmin']):>16}")

    # Garde-fou
    print(f"\n  Garde-fou de comparabilité :")
    both_structured = (sig_fast['h_min'] < 0.01 and sig_div['h_min'] < 0.01
                       and sig_fast['psi_inhomo'] > 1.5
                       and sig_div['psi_inhomo'] > 1.5)
    inhomo_ratio = sig_fast['psi_inhomo'] / sig_div['psi_inhomo']
    same_order_inhomo = 0.5 < inhomo_ratio < 2.0
    log_hmin_fast = np.log10(max(sig_fast['h_min'], 1e-300))
    log_hmin_div = np.log10(max(sig_div['h_min'], 1e-300))
    same_order_hmin = abs(log_hmin_fast - log_hmin_div) < 10
    same_basin = sig_fast['h_argmin'] == sig_div['h_argmin']
    print(f"    structuré ? {both_structured} | "
          f"ψ_inhomo ratio={inhomo_ratio:.3f} ({same_order_inhomo}) | "
          f"h_min Δlog10={abs(log_hmin_fast - log_hmin_div):.1f} "
          f"({same_order_hmin}) | bassin {same_basin}")

    comparable = both_structured and same_order_inhomo and same_basin
    print(f"\n    COMPARABLE ? {comparable}")

    if not comparable:
        print(f"\n  Garde-fou bloque. Frontière trouvée entre :")
        print(f"  - rampe 55→60 sur 2 (contact 5, passait le garde-fou)")
        print(f"  - rampe {beta_start_ramp}→{beta_target} sur {t_ramp} (échoue ici)")
        print(f"  La sensibilité au timing devient critique entre ces deux.")
        result = {"comparable": False,
                  "sig_fast": sig_fast, "sig_divergent": sig_div,
                  "ramp_params": {"beta_start": beta_start_ramp,
                                   "t_ramp": t_ramp}}
    else:
        n_relax = int(200.0 / dt)
        print(f"\n  Mesure des profils relationnels (n_relax = "
              f"{int(200.0)} unités) ...")
        prof_fast = relational_profile(psi_fast, h_fast,
                                       D, beta_target, gamma, h0, dt, n_relax)
        prof_div = relational_profile(psi_div, h_div,
                                      D, beta_target, gamma, h0, dt, n_relax)
        print(f"\n  Profils relationnels :")
        print(f"    {'':>10} {'H_FAST':>16} {'H_DIVERGENT':>16}")
        for key in ['nl_close', 'nl_ortho', 'nl_far']:
            print(f"    {key:>10} {prof_fast[key]:>16.4f} "
                  f"{prof_div[key]:>16.4f}")
        diff_total = (abs(prof_fast['nl_close'] - prof_div['nl_close']) +
                      abs(prof_fast['nl_ortho'] - prof_div['nl_ortho']) +
                      abs(prof_fast['nl_far'] - prof_div['nl_far']))
        print(f"\n    Différence totale (L1) = {diff_total:.4f}")
        print(f"    (rappel contact 5 : diff = 0.0007)")

        print(f"\n  Verdict :")
        if diff_total > 0.15:
            print(f"    PERSISTANCE D'HISTOIRE détectée pour cette paire plus")
            print(f"    contrastée. La frontière du contact 5 était une limite,")
            print(f"    pas un résultat général.")
        elif diff_total < 0.05:
            print(f"    PAS DE PERSISTANCE même pour cette paire plus")
            print(f"    contrastée. L'absence vaut pour une plus large plage")
            print(f"    d'histoires. La hiérarchie de réactivabilités")
            print(f"    (accessibilité ↑ / dynamique interne ↓) est renforcée.")
        else:
            print(f"    DIFFÉRENCE INTERMÉDIAIRE — frontière trouvée mais")
            print(f"    sans verdict net.")

        result = {"comparable": True,
                  "sig_fast": sig_fast, "sig_divergent": sig_div,
                  "psi_inhomo_ratio": inhomo_ratio,
                  "profile_fast": prof_fast, "profile_divergent": prof_div,
                  "diff_total": diff_total,
                  "ramp_params": {"beta_start": beta_start_ramp,
                                   "t_ramp": t_ramp}}

    output_dir = REPO_ROOT / "results" / "phase6d_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "contact_6_history_frontier.json"
    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    with open(out_path, "w") as f:
        json.dump(serialize(result), f, indent=2)
    print(f"\nRésultats : {out_path}")
    return result


if __name__ == "__main__":
    run()
