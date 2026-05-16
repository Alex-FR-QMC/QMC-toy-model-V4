"""
Test 6d-α A.5 — Calibration β/γ pour test pivot §5.

acquired 4a: morphologic memory, feedback ψ↔h, stratified reactivation.
No strong MCQ conclusion yet.

Plan validé Alex :
- γ = 1 fixé, β ∈ {30, 45, 60, 80, 100} balayé
- Famille A uniquement (gaussienne centrée σ_0 = 1.8·dx, h uniforme h₀=1.0)
- Le but n'est PAS de réussir §5, mais d'identifier des candidats
  non triviaux, non collapsés, quasi-stationnaires avec co-production
  visible.

Spec §4.3 : 
- σ_0 = 1.8·dx
- h = h₀ uniforme initial
- 𝔊^sed et 𝔊^ero actifs
- pas de drift, pas de bruit, pas de couplage inter-modulaire
- Stationnarité : ‖∂_t ψ‖, ‖∂_t h‖ < ε_stat

Critère stationnaire raffiné post-discussion Alex :
- rel_norm(dψ/dt) < ε_ψ  (1e-3 screening, 1e-4 strict)
- rel_norm(dh/dt) sur cellules non saturées < ε_h
- stabilité observables sur fenêtre W

Classification rejets explicite :
- REJECT_COLLAPSED : frac_collapsed trop élevée
- REJECT_TRIVIAL_H_NEAR_H0 : mean(h/h₀) > 0.8 (système trivial)
- REJECT_NOT_STATIONARY_PSI : rel_norm(dψ/dt) > ε_ψ
- REJECT_NOT_STATIONARY_H : rel_norm(dh/dt active) > ε_h
- REJECT_TOO_SATURATED : frac_saturated > 30%
- REJECT_BAD_PIVOT_PRECHECK : L2_rel, corr ou L∞_rel pre-test très mauvais
- CANDIDATE : pré-conditions §5 satisfaites

Caveats :
- Ce sweep ne prouve pas l'invariance par changement de γ. Robustesse
  temporelle (cβ, cγ) à tester APRÈS identification d'un candidat.
- A.5 ≠ §5. Le pré-check des métriques §5 ici est informatif, pas un
  test bloquant. Le test §5 strict se fera après, sur le candidat
  retenu, avec validation famille B en plus (§5.7).
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np

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


# Seuils (post-discussion Alex)
EPS_STAT_PSI = 1e-3   # screening
EPS_STAT_H = 1e-3
EPS_SAT = 0.05        # ε_sat fraction (cellule saturée si h_obs > h₀ - ε_sat·h₀)
H_RESOLUTION = 1e-6
H_FUNCTIONAL = 1e-3
H_MIN_POSTULATED = 0.1   # h_min postulé MCQ-théorique (non clipé, juste seuil)

# Constants for filters durs spec
FILTRE_SATURE_MAX = 0.30
FILTRE_MEAN_H_MIN = 0.20
FILTRE_MEAN_H_MAX = 0.80

# Pré-check pivot (informatif, pas critère bloquant)
PRECHECK_CORR_MIN = 0.50  # si corr<0.5 le pivot a peu de chances
PRECHECK_L2_MAX = 0.50    # si L2 > 0.5 idem


def rhs_coupled(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)
    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi_dt, dh_dt


def step_engine_euler(psi, h, D, beta, gamma, h0, dt):
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi_dt, h + dt * dh_dt


def make_gaussian_centered(sigma_0=1.8):
    """Gaussienne centrée selon spec §4.3 : σ_0 = 1.8·dx."""
    coords = np.arange(N_AXIS) * DX
    center = (N_AXIS - 1) * DX / 2.0  # = 2.0 pour N_AXIS=5
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = (coords[i] - center) ** 2 + \
                     (coords[j] - center) ** 2 + \
                     (coords[k] - center) ** 2
                psi[i, j, k] = np.exp(-0.5 * r2 / sigma_0 ** 2)
    psi /= psi.sum()
    return psi


def rel_norm(v, ref):
    """||v||_2 / max(||ref||_2, eps)."""
    n_v = float(np.linalg.norm(v))
    n_ref = float(np.linalg.norm(ref))
    return n_v / max(n_ref, 1e-30)


def compute_pivot_metrics(psi, h, beta, gamma, h0_target, eps_sat_h):
    """
    Calcule L2_rel, corr, Linf_rel comparant ρ_obs et ρ_pred sur
    cellules non saturées.

    eps_sat_h = ε_sat dimensionné en h, = ε_sat·h₀
    """
    rho_obs = psi  # dx=1, donc ρ_obs ≡ ψ numériquement
    rho_pred = (gamma / beta) * (1.0 - h / h0_target)

    # Cellules non saturées : h > h_min + ε_sat (selon §5.2)
    # Note : sur grille minimaliste, on n'a pas h_min strict, donc
    # on utilise h > eps_sat (proche de 0 exclu aussi) ET h < h0 - eps_sat
    saturated_low = h <= H_MIN_POSTULATED + eps_sat_h  # proche de h_min
    saturated_high = h >= h0_target - eps_sat_h  # proche de h₀
    mask_S = ~(saturated_low | saturated_high)

    n_in_S = int(mask_S.sum())
    if n_in_S < 3:
        # Trop peu de cellules pour mesure valide
        return {
            "n_cells_in_S": n_in_S,
            "L2_rel": float("nan"),
            "corr": float("nan"),
            "Linf_rel": float("nan"),
        }

    rho_obs_S = rho_obs[mask_S]
    rho_pred_S = rho_pred[mask_S]
    R = rho_obs_S - rho_pred_S

    L2_rel = float(np.linalg.norm(R) / max(np.linalg.norm(rho_obs_S), 1e-30))
    Linf_rel = float(np.max(np.abs(R)) / max(np.max(rho_obs_S), 1e-30))

    if rho_obs_S.std() > 0 and rho_pred_S.std() > 0:
        corr = float(np.corrcoef(rho_obs_S, rho_pred_S)[0, 1])
    else:
        corr = float("nan")

    return {
        "n_cells_in_S": n_in_S,
        "L2_rel": L2_rel,
        "corr": corr,
        "Linf_rel": Linf_rel,
    }


def simulate_to_quasi_stationary(
    psi_init, h_init, D, beta, gamma, h0_target,
    t_sim, dt, log_every,
):
    """
    Simule jusqu'à t_sim, retourne logs incluant taux de variation
    pour évaluation stationnarité.
    """
    psi = psi_init.copy()
    h = h_init.copy()
    n_steps = max(1, int(np.ceil(t_sim / dt)))

    logs = {
        "t": [], "step": [],
        "psi_total": [],
        "psi_min": [], "psi_max": [], "psi_typ": [],
        "h_min": [], "h_max": [], "h_mean": [],
        "frac_collapsed": [],
        "frac_saturated_near_h0": [],
        "beta_psi_typ_over_gamma": [],
        "beta_psi_max_over_gamma": [],
    }

    def log_state(step, t, psi, h):
        logs["step"].append(step)
        logs["t"].append(t)
        logs["psi_total"].append(float(psi.sum()))
        logs["psi_min"].append(float(psi.min()))
        logs["psi_max"].append(float(psi.max()))
        psi_typ = float(np.median(psi))
        logs["psi_typ"].append(psi_typ)
        logs["h_min"].append(float(h.min()))
        logs["h_max"].append(float(h.max()))
        logs["h_mean"].append(float(h.mean()))
        logs["frac_collapsed"].append(float(np.sum(h < H_RESOLUTION) / h.size))
        logs["frac_saturated_near_h0"].append(
            float(np.sum(h > h0_target - EPS_SAT * h0_target) / h.size)
        )
        logs["beta_psi_typ_over_gamma"].append(beta * psi_typ / gamma if gamma > 0 else float("inf"))
        logs["beta_psi_max_over_gamma"].append(beta * float(psi.max()) / gamma if gamma > 0 else float("inf"))

    log_state(0, 0.0, psi, h)

    for step in range(1, n_steps + 1):
        psi, h = step_engine_euler(psi, h, D, beta, gamma, h0_target, dt)
        if step % log_every == 0 or step == n_steps:
            log_state(step, step * dt, psi, h)

    return psi, h, logs


def assess_stationarity(psi, h, D, beta, gamma, h0_target, dt):
    """
    Évalue rel_norm(dψ/dt) et rel_norm(dh/dt) sur cellules non saturées.
    """
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0_target)

    rel_dpsi = rel_norm(dpsi_dt, psi)

    # Cellules non saturées pour dh : on exclut collapsées et trop proches h₀
    eps_h = EPS_SAT * h0_target
    mask_active = (h > H_RESOLUTION) & (h < h0_target - eps_h)
    if mask_active.sum() >= 3:
        rel_dh_active = rel_norm(dh_dt[mask_active], h[mask_active])
    else:
        rel_dh_active = float("nan")

    return {
        "rel_norm_dpsi": rel_dpsi,
        "rel_norm_dh_active": rel_dh_active,
        "n_cells_active": int(mask_active.sum()),
    }


def classify_candidate(
    beta, gamma,
    rel_dpsi, rel_dh_active,
    frac_collapsed, frac_saturated, mean_h,
    pivot_pre,
) -> dict:
    """
    Classification du résultat du sweep pour un β donné.
    Ordre des rejets : du plus structurel au plus fin.
    """
    label = "CANDIDATE"
    reasons = []

    # 1. REJECT_COLLAPSED : collapse trop important
    if frac_collapsed > 0.20:
        label = "REJECT_COLLAPSED"
        reasons.append(f"frac_collapsed={frac_collapsed:.3f} > 0.20")

    # 2. REJECT_TRIVIAL_H_NEAR_H0 : système trop trivial
    if mean_h > FILTRE_MEAN_H_MAX:
        if label == "CANDIDATE":
            label = "REJECT_TRIVIAL_H_NEAR_H0"
        reasons.append(f"mean(h/h₀)={mean_h:.3f} > {FILTRE_MEAN_H_MAX}")

    if mean_h < FILTRE_MEAN_H_MIN:
        if label == "CANDIDATE":
            label = "REJECT_COLLAPSED"  # mean trop bas = système quasi-mort
        reasons.append(f"mean(h/h₀)={mean_h:.3f} < {FILTRE_MEAN_H_MIN}")

    # 3. REJECT_TOO_SATURATED : zones saturées h₀ ou h_min trop nombreuses
    if frac_saturated > FILTRE_SATURE_MAX:
        if label == "CANDIDATE":
            label = "REJECT_TOO_SATURATED"
        reasons.append(f"frac_saturated={frac_saturated:.3f} > {FILTRE_SATURE_MAX}")

    # 4. REJECT_NOT_STATIONARY_PSI
    if rel_dpsi > EPS_STAT_PSI:
        if label == "CANDIDATE":
            label = "REJECT_NOT_STATIONARY_PSI"
        reasons.append(f"rel_norm(dψ/dt)={rel_dpsi:.3e} > {EPS_STAT_PSI}")

    # 5. REJECT_NOT_STATIONARY_H
    if not np.isnan(rel_dh_active) and rel_dh_active > EPS_STAT_H:
        if label == "CANDIDATE":
            label = "REJECT_NOT_STATIONARY_H"
        reasons.append(f"rel_norm(dh/dt active)={rel_dh_active:.3e} > {EPS_STAT_H}")

    # 6. REJECT_BAD_PIVOT_PRECHECK : pré-conditions §5 manifestement mauvaises
    if not np.isnan(pivot_pre["corr"]) and pivot_pre["corr"] < PRECHECK_CORR_MIN:
        if label == "CANDIDATE":
            label = "REJECT_BAD_PIVOT_PRECHECK"
        reasons.append(f"corr_precheck={pivot_pre['corr']:.3f} < {PRECHECK_CORR_MIN}")

    if not np.isnan(pivot_pre["L2_rel"]) and pivot_pre["L2_rel"] > PRECHECK_L2_MAX:
        if label == "CANDIDATE":
            label = "REJECT_BAD_PIVOT_PRECHECK"
        reasons.append(f"L2_precheck={pivot_pre['L2_rel']:.3f} > {PRECHECK_L2_MAX}")

    return {
        "label": label,
        "reasons": reasons,
    }


def run_one_beta(beta, gamma, D, h0_target, sigma_0, t_sim):
    """Lance §4.3 pour un β/γ donné et classifie."""
    print(f"\n{'='*60}")
    print(f"A.5 — β = {beta}, γ = {gamma} (β/γ = {beta/gamma:.1f})")
    print(f"{'='*60}")

    psi_init = make_gaussian_centered(sigma_0)
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0_target)

    psi_max_init = float(psi_init.max())
    rate_h_max = beta * psi_max_init + gamma
    dt_cfl_diff = cfl_dt_max(h0_target, D)
    dt_cfl_h = 1.0 / rate_h_max
    dt = 0.5 * min(dt_cfl_diff, dt_cfl_h)
    n_steps = int(np.ceil(t_sim / dt))
    log_every = max(1, n_steps // 20)

    print(f"  dt = {dt:.4f}, n_steps = {n_steps}, t_sim = {t_sim}")
    print(f"  σ_0 = {sigma_0}, ψ_max init = {psi_max_init:.4e}")
    print(f"  β·ψ_max init / γ = {beta * psi_max_init / gamma:.4f}")

    psi_final, h_final, logs = simulate_to_quasi_stationary(
        psi_init, h_init, D, beta, gamma, h0_target, t_sim, dt, log_every
    )

    # Évaluation finale
    stat = assess_stationarity(psi_final, h_final, D, beta, gamma, h0_target, dt)
    eps_h = EPS_SAT * h0_target
    pivot_pre = compute_pivot_metrics(psi_final, h_final, beta, gamma, h0_target, eps_h)

    psi_typ_final = float(np.median(psi_final))
    psi_max_final = float(psi_final.max())
    mean_h_final = float(h_final.mean() / h0_target)
    h_min_final = float(h_final.min())
    frac_coll = float(np.sum(h_final < H_RESOLUTION) / h_final.size)
    frac_sat = float(np.sum(h_final > h0_target - eps_h) / h_final.size)

    beta_psi_typ_g = beta * psi_typ_final / gamma
    beta_psi_max_g = beta * psi_max_final / gamma

    print(f"\n  --- Fin de simulation (t={t_sim}) ---")
    print(f"  β·ψ_typ/γ          = {beta_psi_typ_g:.4f}")
    print(f"  β·ψ_max/γ          = {beta_psi_max_g:.4f}")
    print(f"  mean(h/h₀)         = {mean_h_final:.4f}")
    print(f"  h_min              = {h_min_final:.4e}")
    print(f"  frac_collapsed     = {frac_coll:.4f}")
    print(f"  frac_saturated     = {frac_sat:.4f}")
    print(f"  rel_norm(dψ/dt)    = {stat['rel_norm_dpsi']:.4e}  (seuil {EPS_STAT_PSI})")
    print(f"  rel_norm(dh/dt act) = {stat['rel_norm_dh_active']:.4e}  (seuil {EPS_STAT_H})")
    print(f"  n_cells_active     = {stat['n_cells_active']}")
    print(f"\n  Pré-check pivot §5 (informatif) :")
    print(f"    n cells in S (non saturées) : {pivot_pre['n_cells_in_S']}")
    print(f"    L2_rel  = {pivot_pre['L2_rel']:.4f}")
    print(f"    corr    = {pivot_pre['corr']:.4f}")
    print(f"    Linf_rel = {pivot_pre['Linf_rel']:.4f}")

    # Classification
    classif = classify_candidate(
        beta, gamma,
        stat["rel_norm_dpsi"], stat["rel_norm_dh_active"],
        frac_coll, frac_sat, mean_h_final, pivot_pre,
    )

    print(f"\n  CLASSIFICATION : {classif['label']}")
    if classif["reasons"]:
        for r in classif["reasons"]:
            print(f"    - {r}")

    return {
        "beta": beta,
        "gamma": gamma,
        "beta_over_gamma": beta / gamma,
        "classification": classif,
        "metrics": {
            "beta_psi_typ_over_gamma": beta_psi_typ_g,
            "beta_psi_max_over_gamma": beta_psi_max_g,
            "mean_h_over_h0": mean_h_final,
            "h_min": h_min_final,
            "h_max": float(h_final.max()),
            "frac_collapsed": frac_coll,
            "frac_saturated": frac_sat,
            "rel_norm_dpsi": stat["rel_norm_dpsi"],
            "rel_norm_dh_active": stat["rel_norm_dh_active"],
            "n_cells_active": stat["n_cells_active"],
            "L2_rel_precheck": pivot_pre["L2_rel"],
            "corr_precheck": pivot_pre["corr"],
            "Linf_rel_precheck": pivot_pre["Linf_rel"],
            "n_cells_in_S": pivot_pre["n_cells_in_S"],
        },
        "params": {
            "D": D,
            "h0_target": h0_target,
            "sigma_0": sigma_0,
            "t_sim": t_sim,
            "dt": dt,
            "n_steps": n_steps,
            "eps_stat_psi": EPS_STAT_PSI,
            "eps_stat_h": EPS_STAT_H,
            "eps_sat": EPS_SAT,
        },
    }


def run_A5():
    """Sweep complet A.5."""
    gamma = 1.0
    D = 0.1
    h0_target = 1.0
    sigma_0 = 1.8
    t_sim = 500.0

    beta_values = [30.0, 45.0, 60.0, 80.0, 100.0]

    print(f"{'='*70}")
    print(f"A.5 SWEEP — γ={gamma}, β ∈ {beta_values}")
    print(f"σ_0 = {sigma_0}, t_sim = {t_sim}")
    print(f"{'='*70}")

    results = {}
    for beta in beta_values:
        result = run_one_beta(beta, gamma, D, h0_target, sigma_0, t_sim)
        results[f"beta_{int(beta)}"] = result

    # Synthèse tabulaire
    print(f"\n\n{'='*100}")
    print(f"SYNTHÈSE A.5")
    print(f"{'='*100}")
    print(f"{'β':>5} {'β/γ':>5} {'class':<30} {'mean_h':>8} {'frac_col':>9} {'frac_sat':>9} {'corr':>7} {'L2_rel':>7}")
    print(f"{'-'*100}")
    for label, r in results.items():
        m = r["metrics"]
        c = r["classification"]
        corr_str = f"{m['corr_precheck']:.3f}" if not np.isnan(m['corr_precheck']) else "NaN"
        l2_str = f"{m['L2_rel_precheck']:.3f}" if not np.isnan(m['L2_rel_precheck']) else "NaN"
        print(f"{r['beta']:>5.0f} {r['beta_over_gamma']:>5.1f} {c['label']:<30} "
              f"{m['mean_h_over_h0']:>8.4f} {m['frac_collapsed']:>9.4f} "
              f"{m['frac_saturated']:>9.4f} {corr_str:>7} {l2_str:>7}")

    # Identifier candidats
    candidates = [
        label for label, r in results.items()
        if r["classification"]["label"] == "CANDIDATE"
    ]
    print(f"\n{'='*70}")
    print(f"CANDIDATS pour test §5 : {len(candidates)}")
    print(f"{'='*70}")
    for label in candidates:
        r = results[label]
        print(f"  β = {r['beta']:.0f} (β/γ = {r['beta_over_gamma']:.1f})")
        print(f"    corr_precheck = {r['metrics']['corr_precheck']:.4f}, "
              f"L2_rel_precheck = {r['metrics']['L2_rel_precheck']:.4f}")

    if not candidates:
        print(f"\n  AUCUN CANDIDAT identifié dans ce sweep.")
        print(f"  Voir spec §A.5.7 — lecture en cas d'échec.")

    # Verdict global du sweep
    if candidates:
        global_verdict = "PASS"
        print(f"\n  Verdict global A.5 : PASS — {len(candidates)} candidat(s) identifié(s)")
    else:
        global_verdict = "REVISION"
        print(f"\n  Verdict global A.5 : REVISION — aucun candidat")

    return {
        "global_verdict": global_verdict,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "results": results,
    }


if __name__ == "__main__":
    summary = run_A5()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_A5_sweep_beta.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
