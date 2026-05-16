"""
Test 6d-α micro-étape 4a-η — re-injection unique de ψ après quasi-collapse.

Question pivot validée Alex post-4a-ζ :
> Un support quasi-collapsé peut-il redevenir transformable si ψ
  repasse dessus ?

Protocole :

Phase 1 (t = 0 → t_phase1 = 100) :
- ψ initial centré en (1, 2, 2), h uniforme h₀ = 1.0
- Régime C (β=125, γ=1, D=0.1) — produit quasi-collapse stabilisé
- À t = t_phase1, capturer pre_reinject_collapsed_mask
  (cellules sous h_resolution juste avant re-injection)

Phase 2 (t = t_phase1 → t_phase1 + t_phase2 = 500) :
- Re-injecter ψ : remplacer ψ par gaussienne centrée en (3, 2, 2),
  normalisée Σψ = 1
- h NON modifié (garde la cicatrice morphologique)
- Simuler jusqu'à t_total

Mesures spécifiques sur la zone ancienne :
- ψ_mass_on_old_collapsed_region(t) : flux d'arrivée de ψ
- h_mean_old_collapsed_region(t) : récupération éventuelle de h
- n_reactivated_old_collapsed_cells
- τ_return_old_collapsed_cells

Classification du résultat sur les cellules anciennement collapsées :

- ACCESS_FAILURE : ψ_mass_on_old_collapsed_region reste ~0
  (ψ ne parvient même pas à revisiter la zone — auto-renforcement
  du collapse via h bas qui freine la diffusion)

- REACTIVATION_FAILURE : ψ arrive (mass > seuil) mais h ne remonte
  pas (érosion γ·h·(1-h/h₀) n'arrive pas à redresser h tant que
  h reste proche de 0, parce que dh/dt → 0 quand h → 0)

- PARTIAL_REACTIVATION : certaines cellules réactivent (h passe
  au-dessus de h_resolution) mais pas toutes

- FULL_REACTIVATION : toutes les cellules anciennement collapsées
  reviennent au-dessus de h_resolution

CAVEAT (rappel) : le verdict est sur la fenêtre simulée t_phase2 = 400,
PAS sur le comportement asymptotique. "REACTIVATION_FAILURE à t=400"
≠ "réactivation impossible".
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from mcq_v4.factorial_6d import (  # noqa: E402
    N_AXIS, DX, DIM, cfl_dt_max,
)
from mcq_v4.factorial_6d.engine import (  # noqa: E402
    compute_diffusion_flux, compute_divergence,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero  # noqa: E402


H_RESOLUTION = 1e-6
H_FUNCTIONAL = 1e-3
# Seuil pour considérer que ψ a "revisité" une cellule (présence non triviale)
PSI_REVISIT_THRESHOLD = 1e-4  # mass locale significative
# Seuil pour identifier les cellules verrouillées numériquement
# (h sous ce seuil → dh/dt logistique négligeable, structurellement piégé)
H_FLOAT_FLOOR = 1e-100  # bien au-dessus de denormal_min, mais "numériquement zéro"
# Seuil pour identifier "h_pre_reinject était faible mais récupérable"
# (entre H_FLOAT_FLOOR et H_RESOLUTION)
H_WEAK_PRESENCE_MIN = H_FLOAT_FLOOR  # borne basse de la "shell"


def rhs_coupled(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)
    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi_dt, dh_dt


def step_engine_euler(psi, h, D, beta, gamma, h0, dt):
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi_dt, h + dt * dh_dt


def make_psi_centered_at(center, sigma_0=1.5):
    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = center
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i]-cx)**2 + (coords[j]-cy)**2 + (coords[k]-cz)**2
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma_0**2)
    psi /= psi.sum()
    return psi


def classify_reactivation(
    psi_visited_old_zone: bool,
    n_reactivated: int,
    n_dynamically_locked: int,
    n_numerical_floor_core: int,
    n_collapsed_pre: int,
) -> dict:
    """
    Classification raffinée du résultat sur les cellules anciennement
    collapsées (post-audit Alex 4a-η).

    Quatre labels principaux :
    - ACCESS_FAILURE : ψ_mass dans zone ancienne reste sous seuil
      (ψ ne parvient pas à revisiter — auto-renforcement du collapse)
    - REACTIVATION_FAILURE : ψ arrive (ACCESS_CONFIRMED) mais h ne
      remonte pour aucune cellule
    - STRATIFIED_REACTIVATION : ACCESS_CONFIRMED + au moins une
      cellule récupère ET au moins une reste verrouillée → double
      régime selon profondeur résiduelle et bifurcation locale
    - FULL_REACTIVATION : toutes les cellules anciennement collapsées
      reviennent au-dessus de h_resolution

    Sous-composantes pour le verrouillage (mécanisme bifurcationnel) :
    - REACTIVATED_SHELL : cellules récupérées (h_post ≥ h_resolution).
      h_pre était suffisant ET régime local γ - β·ψ_local sous-critique.
    - DYNAMICALLY_LOCKED_DEEP : cellules non récupérées.
      Mécanisme principal : régime local β·ψ_local ≥ γ pendant la
      revisite maintient sédimentation, OU profondeur trop basse
      pour que la croissance logistique remonte h dans la fenêtre.
    - NUMERICAL_FLOOR_CORE : sous-catégorie de DYNAMICALLY_LOCKED_DEEP
      où h_pre est strictement au plancher numérique (h < h_float_floor).
      Mécanisme additionnel : dh/dt logistique numériquement nul.

    CAVEAT (rappel) :
    - DYNAMICALLY_LOCKED_DEEP n'est pas une irréversibilité ontologique.
    - Le verrouillage est dans la fenêtre simulée, sous dynamique
      minimale déterministe sans bruit ni h_min explicite.
    """
    if not psi_visited_old_zone:
        label = "ACCESS_FAILURE"
    elif n_reactivated == 0 and n_dynamically_locked == n_collapsed_pre:
        label = "REACTIVATION_FAILURE"
    elif n_reactivated == n_collapsed_pre:
        label = "FULL_REACTIVATION"
    elif n_reactivated > 0 and n_dynamically_locked > 0:
        label = "STRATIFIED_REACTIVATION"
    elif n_reactivated > 0:
        label = "STRATIFIED_REACTIVATION"  # cas limite
    else:
        label = "REACTIVATION_FAILURE"

    return {
        "label": label,
        "subcomponents": {
            "ACCESS_CONFIRMED": bool(psi_visited_old_zone),
            "REACTIVATED_SHELL_count": int(n_reactivated),
            "DYNAMICALLY_LOCKED_DEEP_count": int(n_dynamically_locked),
            "NUMERICAL_FLOOR_CORE_count": int(n_numerical_floor_core),
        },
    }


def run_test_4a_eta(
    beta: float = 125.0,
    gamma: float = 1.0,
    D: float = 0.1,
    h0_target: float = 1.0,
    t_phase1: float = 100.0,
    t_phase2: float = 400.0,
    sigma_psi: float = 1.5,
    psi_init_center: tuple[float, float, float] = (1.0, 2.0, 2.0),
    psi_reinject_center: tuple[float, float, float] = (3.0, 2.0, 2.0),
) -> dict:
    """Test 4a-η avec re-injection unique."""
    print(f"{'='*60}")
    print(f"Test 4a-η : re-injection unique de ψ")
    print(f"  β={beta}, γ={gamma}, D={D}")
    print(f"  Phase 1 : ψ en {psi_init_center}, t = 0 → {t_phase1}")
    print(f"  Phase 2 : ψ ré-injecté en {psi_reinject_center}, "
          f"t = {t_phase1} → {t_phase1 + t_phase2}")
    print(f"{'='*60}")

    psi = make_psi_centered_at(psi_init_center, sigma_psi)
    h = np.full((N_AXIS, N_AXIS, N_AXIS), h0_target)
    psi_max_init = float(psi.max())

    rate_h_max = beta * psi_max_init + gamma
    dt_cfl_diff = cfl_dt_max(h0_target, D)
    dt_cfl_h = 1.0 / rate_h_max
    dt = 0.5 * min(dt_cfl_diff, dt_cfl_h)
    n_steps_phase1 = max(1, int(np.ceil(t_phase1 / dt)))
    n_steps_phase2 = max(1, int(np.ceil(t_phase2 / dt)))
    print(f"\n  dt = {dt:.4f}, n_steps_phase1 = {n_steps_phase1}, "
          f"n_steps_phase2 = {n_steps_phase2}")

    psi_uniform = 1.0 / psi.size

    # Logs
    logs = {
        "t": [], "step": [],
        "phase": [],
        "psi_total": [], "psi_min": [], "psi_max": [],
        "h_min": [], "h_max": [], "h_mean": [],
        "frac_under_resolution": [],
        "frac_under_functional": [],
        "n_components_resolved": [],
        # Spécifique zone ancienne (logs phase 2 surtout)
        "psi_mass_old_collapsed_region": [],
        "h_mean_old_collapsed_region": [],
        "h_min_old_collapsed_region": [],
        "n_currently_reactivated_in_old_zone": [],
    }

    # Programmer le logging : dense aux frontières de phase, normal sinon
    log_every_phase1 = max(1, n_steps_phase1 // 20)
    log_every_phase2 = max(1, n_steps_phase2 // 40)

    pre_reinject_collapsed_mask = None  # Sera défini à la fin de phase 1

    def log_state(step, t, phase, psi, h):
        logs["step"].append(step)
        logs["t"].append(t)
        logs["phase"].append(phase)
        logs["psi_total"].append(float(psi.sum()))
        logs["psi_min"].append(float(psi.min()))
        logs["psi_max"].append(float(psi.max()))
        logs["h_min"].append(float(h.min()))
        logs["h_max"].append(float(h.max()))
        logs["h_mean"].append(float(h.mean()))
        logs["frac_under_resolution"].append(float(np.sum(h < H_RESOLUTION) / h.size))
        logs["frac_under_functional"].append(float(np.sum(h < H_FUNCTIONAL) / h.size))
        # Composantes connexes du support résolu
        mask_active = (h >= H_RESOLUTION).astype(int)
        struct = ndimage.generate_binary_structure(3, 1)
        _, n_comp = ndimage.label(mask_active, structure=struct)
        logs["n_components_resolved"].append(int(n_comp))
        # Métriques sur zone ancienne (uniquement si masque défini)
        if pre_reinject_collapsed_mask is not None:
            psi_old = float(psi[pre_reinject_collapsed_mask].sum())
            h_old_mean = float(h[pre_reinject_collapsed_mask].mean())
            h_old_min = float(h[pre_reinject_collapsed_mask].min())
            n_reactiv = int(np.sum(h[pre_reinject_collapsed_mask] >= H_RESOLUTION))
            logs["psi_mass_old_collapsed_region"].append(psi_old)
            logs["h_mean_old_collapsed_region"].append(h_old_mean)
            logs["h_min_old_collapsed_region"].append(h_old_min)
            logs["n_currently_reactivated_in_old_zone"].append(n_reactiv)
        else:
            logs["psi_mass_old_collapsed_region"].append(None)
            logs["h_mean_old_collapsed_region"].append(None)
            logs["h_min_old_collapsed_region"].append(None)
            logs["n_currently_reactivated_in_old_zone"].append(None)

    # === PHASE 1 ===
    log_state(0, 0.0, "phase1", psi, h)
    for step in range(1, n_steps_phase1 + 1):
        psi, h = step_engine_euler(psi, h, D, beta, gamma, h0_target, dt)
        if step % log_every_phase1 == 0:
            log_state(step, step * dt, "phase1", psi, h)

    # Capture du masque collapsé juste avant re-injection
    pre_reinject_collapsed_mask = h < H_RESOLUTION
    n_collapsed_pre = int(np.sum(pre_reinject_collapsed_mask))
    print(f"\nFin de phase 1 (t = {n_steps_phase1 * dt:.4f}) :")
    print(f"  cellules sous h_resolution : {n_collapsed_pre}/{h.size}")
    print(f"  h_min phase 1              : {h.min():.4e}")
    print(f"  h dans zone à collapser : min={h[pre_reinject_collapsed_mask].min() if n_collapsed_pre > 0 else 'N/A'}")

    # Snapshot complet de h à la fin de phase 1
    h_at_reinject = h.copy()
    psi_at_reinject = psi.copy()

    # Log spécifique au moment de la transition (pre_reinject_collapsed_mask désormais défini)
    log_state(n_steps_phase1, n_steps_phase1 * dt, "phase1_end_remeasure", psi, h)

    # === RE-INJECTION ===
    psi_new = make_psi_centered_at(psi_reinject_center, sigma_psi)
    psi = psi_new
    print(f"\nRé-injection ψ → {psi_reinject_center}")
    print(f"  ψ_max après ré-injection : {psi.max():.6f}")
    print(f"  ψ_mass dans zone ancienne avant Phase 2 : "
          f"{psi[pre_reinject_collapsed_mask].sum():.6e}")
    print(f"  (mass attendue depuis position {psi_reinject_center} : faible)")

    # === PHASE 2 ===
    log_state(n_steps_phase1, n_steps_phase1 * dt, "phase2_start", psi, h)
    for step in range(1, n_steps_phase2 + 1):
        psi, h = step_engine_euler(psi, h, D, beta, gamma, h0_target, dt)
        if step % log_every_phase2 == 0:
            log_state(
                n_steps_phase1 + step, (n_steps_phase1 + step) * dt,
                "phase2", psi, h
            )

    # === Analyse finale ===
    print(f"\n{'='*60}")
    print(f"Analyse fin de Phase 2 (t = {(n_steps_phase1 + n_steps_phase2) * dt:.4f})")
    print(f"{'='*60}")

    psi_final = psi.copy()
    h_final = h.copy()

    # ψ a-t-il visité significativement la zone ancienne ?
    psi_mass_max_old = max(
        m for m in logs["psi_mass_old_collapsed_region"]
        if m is not None
    )
    psi_visited = psi_mass_max_old > PSI_REVISIT_THRESHOLD

    # === Géométrie du masque collapsé pré-réinjection ===
    # Pour ne pas sur-interpréter "ψ_mass = 0.52 dans zone ancienne"
    # comme une traversée totale, on documente où se trouve le masque.
    collapsed_coords = np.argwhere(pre_reinject_collapsed_mask)
    # Centre de masse du masque collapsé (purement géométrique)
    if len(collapsed_coords) > 0:
        mask_com = collapsed_coords.mean(axis=0)
        mask_extent = collapsed_coords.max(axis=0) - collapsed_coords.min(axis=0)
        # Distance du masque à la position de réinjection
        reinj_arr = np.array(psi_reinject_center) / DX
        distances_to_reinj = np.linalg.norm(
            collapsed_coords - reinj_arr, axis=1
        )
        dist_min_mask_to_reinj = float(distances_to_reinj.min())
        dist_max_mask_to_reinj = float(distances_to_reinj.max())
        dist_mean_mask_to_reinj = float(distances_to_reinj.mean())
    else:
        mask_com = np.array([0, 0, 0])
        mask_extent = np.array([0, 0, 0])
        dist_min_mask_to_reinj = float("nan")
        dist_max_mask_to_reinj = float("nan")
        dist_mean_mask_to_reinj = float("nan")

    # === Distribution h pré-réinjection (sur cellules collapsées) ===
    h_pre_in_mask = h_at_reinject[pre_reinject_collapsed_mask]
    h_pre_dist = {
        "min": float(h_pre_in_mask.min()) if len(h_pre_in_mask) > 0 else None,
        "max": float(h_pre_in_mask.max()) if len(h_pre_in_mask) > 0 else None,
        "median": float(np.median(h_pre_in_mask)) if len(h_pre_in_mask) > 0 else None,
        "n_below_float_floor": int(np.sum(h_pre_in_mask < H_FLOAT_FLOOR)),
        "n_above_float_floor": int(np.sum(h_pre_in_mask >= H_FLOAT_FLOOR)),
    }

    # === Distribution h post-réinjection (sur même masque) ===
    h_post_in_mask = h_final[pre_reinject_collapsed_mask]
    h_post_dist = {
        "min": float(h_post_in_mask.min()),
        "max": float(h_post_in_mask.max()),
        "median": float(np.median(h_post_in_mask)),
    }

    # === Stratification : REACTIVABLE_SHELL vs NUMERICALLY_LOCKED_CORE ===
    # Cellule "shell" = h_pre était > float_floor (présence résiduelle)
    # Cellule "core" = h_pre était ≤ float_floor (verrouillée)
    shell_mask_within_collapsed = h_pre_in_mask >= H_FLOAT_FLOOR
    core_mask_within_collapsed = h_pre_in_mask < H_FLOAT_FLOOR

    # Sur cellules shell : combien ont récupéré (h_post ≥ h_resolution) ?
    n_shell_total = int(shell_mask_within_collapsed.sum())
    n_shell_reactivated = int(np.sum(
        (h_post_in_mask >= H_RESOLUTION) & shell_mask_within_collapsed
    ))

    # Sur cellules core : combien sont restées verrouillées vs ont récupéré ?
    n_core_total = int(core_mask_within_collapsed.sum())
    n_core_reactivated = int(np.sum(
        (h_post_in_mask >= H_RESOLUTION) & core_mask_within_collapsed
    ))
    n_core_still_locked = n_core_total - n_core_reactivated

    # === Stratification raffinée post-audit Alex ===
    # n_reactivated_total : cellules dont h_post ≥ h_resolution
    # n_dynamically_locked : cellules non réactivées (mécanisme principal :
    #   bifurcation locale γ - β·ψ_local ≤ 0 maintient sédimentation)
    # n_numerical_floor_core : sous-catégorie marginale où h_pre < float_floor
    #   (verrouillage numérique strict additionnel — sous-catégorie de
    #   DYNAMICALLY_LOCKED_DEEP, pas un label séparé)
    n_reactivated_total = int(np.sum(h_post_in_mask >= H_RESOLUTION))
    n_non_reactivated_total = n_collapsed_pre - n_reactivated_total
    # Sous-catégorie : cellules au plancher numérique strict (h_pre < float_floor)
    # et non réactivées
    n_numerical_floor_core = int(np.sum(
        (h_post_in_mask < H_RESOLUTION) & core_mask_within_collapsed
    ))
    # DYNAMICALLY_LOCKED_DEEP = toutes non réactivées (incluant la sous-cat
    # numerical_floor_core qui est juste une fraction marginale)
    n_dynamically_locked = n_non_reactivated_total

    # === Délai de réactivation post-réinjection (correction Alex) ===
    # Filtrer uniquement sur phases "phase2_start" et "phase2" pour ne pas
    # confondre la mesure phase1_end_remeasure (qui mesure avant réinjection
    # avec masque déjà défini) avec l'accès post-réinjection réel.
    t_first_access_post_reinject = None
    t_first_reactivation_in_old = None
    for i, (t, phase, mass, n_reactiv) in enumerate(zip(
        logs["t"], logs["phase"],
        logs["psi_mass_old_collapsed_region"],
        logs["n_currently_reactivated_in_old_zone"]
    )):
        # Ne considérer que les états POST-réinjection
        if phase not in ("phase2_start", "phase2"):
            continue
        if (mass is not None and mass > PSI_REVISIT_THRESHOLD
                and t_first_access_post_reinject is None):
            t_first_access_post_reinject = t
        if (n_reactiv is not None and n_reactiv > 0
                and t_first_reactivation_in_old is None):
            t_first_reactivation_in_old = t

    # Pour information aussi : timing global (incluant phase1_end_remeasure)
    t_first_access_global = None
    for i, (t, mass) in enumerate(zip(
        logs["t"], logs["psi_mass_old_collapsed_region"]
    )):
        if mass is not None and mass > PSI_REVISIT_THRESHOLD and t_first_access_global is None:
            t_first_access_global = t

    # === Mesure du régime bifurcationnel local pendant Phase 2 ===
    # Pour les cellules réactivées vs non-réactivées, mesurer β·⟨ψ⟩/γ
    # où ⟨ψ⟩ est moyennée sur le début de Phase 2 (premières fractions
    # significatives post-réinjection).
    # On utilise ψ_final comme proxy stationnaire post-réinjection.
    reactivated_indices = np.argwhere(
        h_final >= H_RESOLUTION
    )
    non_reactivated_indices = np.argwhere(
        (h_final < H_RESOLUTION) & pre_reinject_collapsed_mask
    )
    # Filtrer reactivated_indices pour ne garder que les cellules
    # qui étaient dans pre_reinject_collapsed_mask
    reactivated_in_mask_mask = (h_final >= H_RESOLUTION) & pre_reinject_collapsed_mask
    reactivated_in_mask_indices = np.argwhere(reactivated_in_mask_mask)

    if len(reactivated_in_mask_indices) > 0:
        psi_at_reactivated = psi_final[reactivated_in_mask_mask]
        beta_psi_over_gamma_reactivated = beta * psi_at_reactivated / gamma
        beta_psi_g_react_min = float(beta_psi_over_gamma_reactivated.min())
        beta_psi_g_react_max = float(beta_psi_over_gamma_reactivated.max())
        beta_psi_g_react_median = float(np.median(beta_psi_over_gamma_reactivated))
    else:
        beta_psi_g_react_min = beta_psi_g_react_max = beta_psi_g_react_median = None

    non_reactivated_in_mask_mask = (h_final < H_RESOLUTION) & pre_reinject_collapsed_mask
    if int(non_reactivated_in_mask_mask.sum()) > 0:
        psi_at_non_react = psi_final[non_reactivated_in_mask_mask]
        beta_psi_over_gamma_non_react = beta * psi_at_non_react / gamma
        beta_psi_g_nonreact_min = float(beta_psi_over_gamma_non_react.min())
        beta_psi_g_nonreact_max = float(beta_psi_over_gamma_non_react.max())
        beta_psi_g_nonreact_median = float(np.median(beta_psi_over_gamma_non_react))
    else:
        beta_psi_g_nonreact_min = beta_psi_g_nonreact_max = beta_psi_g_nonreact_median = None

    # === Temps logistique attendu sous γ pur (γ - β·ψ ≈ γ si ψ=0 supposé) ===
    # Pour cellules qui SERAIENT sous-critiques (γ - β·ψ_local > 0),
    # le temps pour passer de h_pre à h_resolution sous logistique pure
    # vaut approximativement (1/γ)·ln(h_resolution/h_pre).
    # On le calcule comme borne inférieure (en supposant γ pur, pas net).
    # C'est purement diagnostique — pas un critère.
    estimated_recovery_times = []
    for h_pre_val in h_pre_in_mask:
        if h_pre_val > 0 and gamma > 0:
            t_rec = (1.0 / gamma) * np.log(H_RESOLUTION / h_pre_val) if h_pre_val < H_RESOLUTION else 0.0
            estimated_recovery_times.append(float(t_rec))
        else:
            estimated_recovery_times.append(float("inf"))

    est_recovery_median = float(np.median(estimated_recovery_times)) \
        if estimated_recovery_times else None
    est_recovery_max = float(np.max(estimated_recovery_times)) \
        if estimated_recovery_times else None

    classification = classify_reactivation(
        psi_visited_old_zone=psi_visited,
        n_reactivated=n_reactivated_total,
        n_dynamically_locked=n_dynamically_locked,
        n_numerical_floor_core=n_numerical_floor_core,
        n_collapsed_pre=n_collapsed_pre,
    )

    print(f"\nψ_mass max dans zone ancienne : {psi_mass_max_old:.4e}")
    print(f"  (seuil 'revisite' = {PSI_REVISIT_THRESHOLD})")
    print(f"  ψ a-t-il revisité ? {'OUI' if psi_visited else 'NON'}")
    if t_first_access_post_reinject is not None:
        print(f"  t_first_access_post_reinject = {t_first_access_post_reinject:.4f}")
    if t_first_access_global is not None:
        print(f"  t_first_access_global = {t_first_access_global:.4f} "
              f"(inclut phase1_end_remeasure)")

    print(f"\n=== Géométrie du masque collapsé (pré-réinjection) ===")
    print(f"  Centre de masse (en cellules) : {mask_com}")
    print(f"  Extension (cellules)          : {mask_extent}")
    print(f"  Distance au point de réinjection {psi_reinject_center} :")
    print(f"    min  : {dist_min_mask_to_reinj:.4f}")
    print(f"    mean : {dist_mean_mask_to_reinj:.4f}")
    print(f"    max  : {dist_max_mask_to_reinj:.4f}")
    print(f"  Note : si dist_min ≈ 0, le masque touche la zone de réinjection,")
    print(f"         donc une partie de ψ_mass_old vient de recouvrement initial,")
    print(f"         pas de traversée par diffusion.")

    print(f"\n=== Distribution h pré-réinjection (dans masque) ===")
    print(f"  min    : {h_pre_dist['min']:.4e}")
    print(f"  median : {h_pre_dist['median']:.4e}")
    print(f"  max    : {h_pre_dist['max']:.4e}")
    print(f"  n cellules h ≥ float_floor ({H_FLOAT_FLOOR}) : {h_pre_dist['n_above_float_floor']}")
    print(f"  n cellules h <  float_floor : {h_pre_dist['n_below_float_floor']}")

    print(f"\n=== Distribution h post-réinjection (même masque) ===")
    print(f"  min    : {h_post_dist['min']:.4e}")
    print(f"  median : {h_post_dist['median']:.4e}")
    print(f"  max    : {h_post_dist['max']:.4e}")

    print(f"\n=== Stratification réactivation (raffinée post-audit) ===")
    print(f"  Total réactivées (h_post ≥ h_resolution) : {n_reactivated_total}/{n_collapsed_pre}")
    print(f"  DYNAMICALLY_LOCKED_DEEP : {n_dynamically_locked}/{n_collapsed_pre}")
    print(f"    dont NUMERICAL_FLOOR_CORE (h_pre < {H_FLOAT_FLOOR}) : "
          f"{n_numerical_floor_core}/{n_collapsed_pre}")
    print(f"  Indicateur secondaire 'shell vs core' par h_pre :")
    print(f"    h_pre ≥ float_floor : {n_shell_total} cellules "
          f"({n_shell_reactivated} réactivées)")
    print(f"    h_pre <  float_floor : {n_core_total} cellules "
          f"({n_core_reactivated} réactivées)")
    if t_first_reactivation_in_old is not None:
        print(f"  t_first_reactivation_in_old = {t_first_reactivation_in_old:.4f}")

    print(f"\n=== Régime bifurcationnel local (β·ψ_local / γ) ===")
    print(f"  Cellules réactivées (n={n_reactivated_total}) :")
    if beta_psi_g_react_min is not None:
        print(f"    range = [{beta_psi_g_react_min:.4f}, {beta_psi_g_react_max:.4f}]")
        print(f"    median = {beta_psi_g_react_median:.4f}")
    print(f"  Cellules NON réactivées (n={n_dynamically_locked}) :")
    if beta_psi_g_nonreact_min is not None:
        print(f"    range = [{beta_psi_g_nonreact_min:.4f}, {beta_psi_g_nonreact_max:.4f}]")
        print(f"    median = {beta_psi_g_nonreact_median:.4f}")
    print(f"  Interprétation : médiane non-réact > 1 → sur-critique local, "
          f"sédimentation maintient verrouillage")

    print(f"\n=== Temps logistique estimé sous γ pur (diagnostic) ===")
    if est_recovery_median is not None:
        print(f"  médiane = {est_recovery_median:.4e}")
        print(f"  max    = {est_recovery_max:.4e}")
        print(f"  À comparer à fenêtre simulée t_phase2 = {t_phase2}")

    print(f"\n=== CLASSIFICATION ===")
    print(f"  {classification['label']}")
    print(f"  Sub-components :")
    for k, v in classification["subcomponents"].items():
        print(f"    {k} = {v}")

    # Invariants
    mass_drift = max(abs(m - 1.0) for m in logs["psi_total"])
    psi_min_global = min(logs["psi_min"])

    print(f"\n=== Invariants ===")
    print(f"  Σψ drift           : {mass_drift:.4e}")
    print(f"  positivité ψ min   : {psi_min_global:.4e}")

    verdict = "PASS"
    reasons = []
    if mass_drift > 1e-10:
        verdict = "FAIL"
        reasons.append(f"Σψ drift = {mass_drift:.4e}")
    if psi_min_global < -1e-12:
        verdict = "FAIL"
        reasons.append(f"positivité ψ : min = {psi_min_global:.4e}")

    print(f"\n=== VERDICT ===")
    if verdict == "PASS":
        print(f"  PASS ✓")
    else:
        print(f"  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "classification": classification,
        "params": {
            "beta": beta, "gamma": gamma, "D": D, "h0_target": h0_target,
            "t_phase1": t_phase1, "t_phase2": t_phase2,
            "dt": dt,
            "psi_init_center": list(psi_init_center),
            "psi_reinject_center": list(psi_reinject_center),
            "h_resolution": H_RESOLUTION,
            "h_functional": H_FUNCTIONAL,
            "h_float_floor": H_FLOAT_FLOOR,
            "psi_revisit_threshold": PSI_REVISIT_THRESHOLD,
        },
        "invariants": {
            "mass_drift": float(mass_drift),
            "psi_min_global": float(psi_min_global),
        },
        "pre_reinject": {
            "n_collapsed": n_collapsed_pre,
            "n_collapsed_fraction": float(n_collapsed_pre / 125),
            "mask_geometry": {
                "center_of_mass_cells": [float(x) for x in mask_com],
                "extent_cells": [float(x) for x in mask_extent],
                "dist_to_reinject_min": dist_min_mask_to_reinj,
                "dist_to_reinject_mean": dist_mean_mask_to_reinj,
                "dist_to_reinject_max": dist_max_mask_to_reinj,
            },
            "h_distribution": h_pre_dist,
        },
        "phase2_access": {
            "psi_visited_old_zone": bool(psi_visited),
            "psi_mass_max_old": float(psi_mass_max_old),
            "t_first_access_post_reinject": (
                float(t_first_access_post_reinject)
                if t_first_access_post_reinject is not None else None
            ),
            "t_first_access_global": (
                float(t_first_access_global)
                if t_first_access_global is not None else None
            ),
        },
        "phase2_h_distribution": h_post_dist,
        "phase2_stratification": {
            "n_reactivated_total": n_reactivated_total,
            "n_dynamically_locked_deep": n_dynamically_locked,
            "n_numerical_floor_core": n_numerical_floor_core,
            # Indicateurs secondaires (h_pre based)
            "n_shell_total_hpre_above_floor": n_shell_total,
            "n_shell_reactivated": n_shell_reactivated,
            "n_core_total_hpre_below_floor": n_core_total,
            "n_core_reactivated": n_core_reactivated,
        },
        "phase2_bifurcation_local": {
            "beta_psi_over_gamma_reactivated": {
                "min": beta_psi_g_react_min,
                "max": beta_psi_g_react_max,
                "median": beta_psi_g_react_median,
            },
            "beta_psi_over_gamma_non_reactivated": {
                "min": beta_psi_g_nonreact_min,
                "max": beta_psi_g_nonreact_max,
                "median": beta_psi_g_nonreact_median,
            },
        },
        "phase2_estimated_recovery_diagnostic": {
            "estimated_recovery_time_under_pure_gamma_median": est_recovery_median,
            "estimated_recovery_time_under_pure_gamma_max": est_recovery_max,
            "t_phase2_simulated": t_phase2,
        },
        "phase2_reactivation_timing": {
            "t_first_reactivation_in_old": (
                float(t_first_reactivation_in_old)
                if t_first_reactivation_in_old is not None else None
            ),
        },
    }


if __name__ == "__main__":
    summary = run_test_4a_eta()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_4a_eta_reinjection.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["verdict"] == "PASS" else 1)
