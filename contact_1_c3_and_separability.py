"""
6d-γ — premier contact empirique.

Garde-fou négatif : voir 6d-gamma-c3-negative-guardrail.md
Trace de transition : voir 6d-beta-to-gamma-transition-trace.md

Compromis d'ouverture (validé avec Alex) :
- un seul régime (β=60, structuré)
- deux observables côte à côte
- pas de gradient construit a priori
- pas de cartographie

Observables :
1. C3 brut : ||État_AB − État_BA|| (deux ordres de contraction)
2. Proxy provisoire de séparabilité : indépendance des réponses
   directionnelles A et B

PLUS : les six exclusions négatives du garde-fou, mesurées, pour que
C3 doive SURVIVRE à ce qu'il n'est pas.

Ce qu'est une "contraction directionnelle" ici :
On définit deux directions de transformation A et B comme deux
perturbations localisées dans des régions spatiales distinctes de la
grille, appliquées comme une compression locale de ψ (concentration
accrue). C'est une proxy provisoire, pas une définition de
"contraction morphodynamique".
- Direction A : compression vers le plan x=centre (axe x)
- Direction B : compression vers le plan y=centre (axe y)

PAS de définition forte de "séparabilité". On mesure seulement si la
réponse à (A puis B) est prédictible à partir des réponses à A et B
séparément.
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


def contraction_A(psi, strength=0.05):
    """Direction A : compression le long de l'axe x vers le plan central.
    Augmente ψ dans les cellules proches de x=centre, diminue ailleurs.
    Conserve la masse (renormalisation)."""
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        # plus proche de x=centre → facteur > 1
        dist_x = abs(i - c)
        factor[i, :, :] = 1.0 + strength * (1.0 - dist_x / c)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def contraction_B(psi, strength=0.05):
    """Direction B : compression le long de l'axe y vers le plan central."""
    c = (N_AXIS - 1) / 2.0
    factor = np.ones_like(psi)
    for j in range(N_AXIS):
        dist_y = abs(j - c)
        factor[:, j, :] = 1.0 + strength * (1.0 - dist_y / c)
    psi_new = psi * factor
    return psi_new / psi_new.sum()


def evolve(psi, h, D, beta, gamma, h0, dt, n_steps):
    for _ in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return psi, h


def run_contact():
    gamma = 1.0
    D = 0.1
    h0 = 1.0
    beta = 60.0
    t_relax = 50.0  # relaxation entre contractions et après

    psi0 = make_psi_centered(sigma=1.5)
    h0_field = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta * psi_max + gamma))
    n_relax = int(t_relax / dt)

    print(f"{'='*70}")
    print(f"6d-γ — premier contact empirique")
    print(f"  β={beta}, dt={dt:.5f}, t_relax={t_relax}")
    print(f"  Régime structuré. Deux observables côte à côte.")
    print(f"{'='*70}\n")

    # État de base : on laisse le système se stabiliser un peu
    psi_base, h_base = evolve(psi0.copy(), h0_field.copy(),
                              D, beta, gamma, h0, dt, n_relax)
    print(f"  État de base (après {t_relax} relaxation) :")
    print(f"    ψ_inhomo = {psi_base.max()/max(psi_base.min(),1e-30):.3f}")
    print(f"    h_min = {h_base.min():.4e}")

    # ===== OBSERVABLE 1 : C3 (non-commutativité) =====
    print(f"\n  {'─'*60}")
    print(f"  OBSERVABLE 1 : C3 (ordre des contractions)")
    print(f"  {'─'*60}")

    # AB : contraction A, relaxation, contraction B, relaxation
    psi_ab, h_ab = contraction_A(psi_base.copy()), h_base.copy()
    psi_ab, h_ab = evolve(psi_ab, h_ab, D, beta, gamma, h0, dt, n_relax)
    psi_ab = contraction_B(psi_ab)
    psi_ab, h_ab = evolve(psi_ab, h_ab, D, beta, gamma, h0, dt, n_relax)

    # BA : contraction B, relaxation, contraction A, relaxation
    psi_ba, h_ba = contraction_B(psi_base.copy()), h_base.copy()
    psi_ba, h_ba = evolve(psi_ba, h_ba, D, beta, gamma, h0, dt, n_relax)
    psi_ba = contraction_A(psi_ba)
    psi_ba, h_ba = evolve(psi_ba, h_ba, D, beta, gamma, h0, dt, n_relax)

    c3_psi = float(np.linalg.norm(psi_ab - psi_ba))
    c3_h = float(np.linalg.norm(h_ab - h_ba))
    print(f"    ||ψ_AB − ψ_BA|| = {c3_psi:.4e}")
    print(f"    ||h_AB − h_BA|| = {c3_h:.4e}")

    # ===== OBSERVABLE 2 : proxy de séparabilité =====
    print(f"\n  {'─'*60}")
    print(f"  OBSERVABLE 2 : proxy de séparabilité (réponses directionnelles)")
    print(f"  {'─'*60}")

    # Réponse à A seul
    psi_a, h_a = contraction_A(psi_base.copy()), h_base.copy()
    psi_a, h_a = evolve(psi_a, h_a, D, beta, gamma, h0, dt, 2 * n_relax)
    resp_A_psi = psi_a - psi_base  # réponse en ψ à A seul

    # Réponse à B seul
    psi_b, h_b = contraction_B(psi_base.copy()), h_base.copy()
    psi_b, h_b = evolve(psi_b, h_b, D, beta, gamma, h0, dt, 2 * n_relax)
    resp_B_psi = psi_b - psi_base

    # Réponse à A+B simultané
    psi_ab_sim = contraction_B(contraction_A(psi_base.copy()))
    h_ab_sim = h_base.copy()
    psi_ab_sim, h_ab_sim = evolve(psi_ab_sim, h_ab_sim, D, beta, gamma, h0,
                                  dt, 2 * n_relax)
    resp_AB_psi = psi_ab_sim - psi_base

    # Séparabilité : resp(A+B) ≈ resp(A) + resp(B) ?
    resp_sum = resp_A_psi + resp_B_psi
    nonlinearity = float(np.linalg.norm(resp_AB_psi - resp_sum))
    norm_resp_AB = float(np.linalg.norm(resp_AB_psi))
    nonlinearity_relative = nonlinearity / max(norm_resp_AB, 1e-30)

    print(f"    ||resp(A)|| = {np.linalg.norm(resp_A_psi):.4e}")
    print(f"    ||resp(B)|| = {np.linalg.norm(resp_B_psi):.4e}")
    print(f"    ||resp(A+B)|| = {norm_resp_AB:.4e}")
    print(f"    ||resp(A+B) − [resp(A)+resp(B)]|| = {nonlinearity:.4e}")
    print(f"    non-linéarité relative = {nonlinearity_relative:.4f}")
    print(f"    (0 = séparable additif ; >0 = couplage directionnel)")

    # ===== EXCLUSIONS NÉGATIVES (garde-fou) =====
    print(f"\n  {'─'*60}")
    print(f"  EXCLUSIONS NÉGATIVES — C3 survit-il à ce qu'il n'est pas ?")
    print(f"  {'─'*60}")

    diff_psi = psi_ab - psi_ba
    diff_h = h_ab - h_ba

    # (1) Écart scalaire final ? La différence est-elle juste une amplitude ?
    # Test : la différence normalisée a-t-elle une structure spatiale,
    # ou est-elle proportionnelle à l'état moyen ?
    mean_state = 0.5 * (psi_ab + psi_ba)
    if np.linalg.norm(mean_state) > 1e-30:
        # corrélation entre diff et état moyen (si forte → diff ~ scaling global)
        corr_diff_mean = float(np.corrcoef(diff_psi.flatten(),
                                            mean_state.flatten())[0, 1])
    else:
        corr_diff_mean = 0.0
    exclusion_1 = abs(corr_diff_mean) > 0.95  # True = c'est un écart scalaire
    print(f"    (1) écart scalaire final ? corr(diff, état_moyen) = "
          f"{corr_diff_mean:+.4f}")
    print(f"        → {'OUI (réductible à scaling)' if exclusion_1 else 'NON (structure propre)'}")

    # (2) Non-identité numérique triviale ? (au bruit machine ?)
    exclusion_2 = c3_psi < 1e-14
    print(f"    (2) bruit machine ? ||ψ_AB−ψ_BA|| = {c3_psi:.2e}")
    print(f"        → {'OUI (trivial)' if exclusion_2 else 'NON (significatif)'}")

    # (5) Signature spatiale structurée ?
    diff_argmax = np.unravel_index(int(np.argmax(np.abs(diff_psi))),
                                   diff_psi.shape)
    diff_inhomo = float(np.abs(diff_psi).max() /
                        max(np.abs(diff_psi).mean(), 1e-30))
    exclusion_5 = diff_inhomo < 2.0  # True = pas de structure (uniforme)
    print(f"    (5) signature relationnelle ? inhomo(|diff|) = "
          f"{diff_inhomo:.2f}, argmax = {diff_argmax}")
    print(f"        → {'pas de structure' if exclusion_5 else 'structure présente'}")

    # Verdict de survie
    print(f"\n  {'─'*60}")
    print(f"  VERDICT DE SURVIE")
    print(f"  {'─'*60}")
    survives = (not exclusion_1) and (not exclusion_2) and (not exclusion_5)
    if survives:
        print(f"    C3 SURVIT aux exclusions minimales testées.")
        print(f"    → différence AB/BA non scalaire, non triviale, structurée")
        print(f"    → premier objet à étudier (PAS non-commutativité démontrée)")
    else:
        print(f"    C3 NE SURVIT PAS à au moins une exclusion.")
        print(f"    → réduit à artefact ou différence non morphologique")
        print(f"    → réviser lecture L1/L2/L3 de la trace de transition")

    # Mise en relation C3 ↔ séparabilité (côte à côte, PAS de gradient)
    print(f"\n  {'─'*60}")
    print(f"  DEUX OBSERVABLES CÔTE À CÔTE (pas de gradient)")
    print(f"  {'─'*60}")
    print(f"    C3 (non-commutativité)    : ||ψ_AB−ψ_BA|| = {c3_psi:.4e}")
    print(f"    Séparabilité (non-lin.)   : {nonlinearity:.4e} "
          f"(relatif {nonlinearity_relative:.4f})")
    print(f"\n    [Relation non interprétée à ce stade — un seul régime,")
    print(f"     pas de gradient. Co-variation testable seulement plus tard.]")

    return {
        "params": {"beta": beta, "dt": dt, "t_relax": t_relax},
        "c3_psi": c3_psi,
        "c3_h": c3_h,
        "separability_nonlinearity": nonlinearity,
        "separability_nonlinearity_relative": nonlinearity_relative,
        "resp_A_norm": float(np.linalg.norm(resp_A_psi)),
        "resp_B_norm": float(np.linalg.norm(resp_B_psi)),
        "resp_AB_norm": norm_resp_AB,
        "exclusions": {
            "1_scalar_final": {"corr_diff_mean": corr_diff_mean,
                               "excluded": bool(exclusion_1)},
            "2_machine_noise": {"c3_psi": c3_psi,
                                "excluded": bool(exclusion_2)},
            "5_no_structure": {"diff_inhomo": diff_inhomo,
                               "diff_argmax": [int(x) for x in diff_argmax],
                               "excluded": bool(exclusion_5)},
        },
        "survives_negative_guardrail": bool(survives),
    }


if __name__ == "__main__":
    out = run_contact()
    output_dir = REPO_ROOT / "results" / "phase6d_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "contact_1_c3_and_separability.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {output_path}")
