"""
Test 6d-α micro-étape 4a-δ — co-évolution ψ↔h symétrique minimale.

Premier vrai test de la co-évolution :
    ∂_t ψ = ∇·(h·D·∇ψ)                  [diffusion conforme]
    ∂_t h = -β·ψ·h + γ·h·(1-h/h₀)       [sédimentation + érosion]

Sans drift, sans bruit, sans coupling supplémentaire.

Plan validé Alex avec 6 garde-fous :
1. Trois régimes A/B/C calibrés en termes de rapports d'échelles
   τ_ψ ~ dx²/D ; τ_sed ~ 1/(β·ψ_typ) ; τ_ero ~ 1/γ
2. RK4 référence indépendante à dt petit
3. Test d'ordre temporel : erreur Euler doit décroître ∝ dt
4. Trace memory mesurée à deux échelles : corr(h_final, ∫ψ·dt)
   et corr(-log(h_final/h0), ∫ψ·dt) (plus linéaire)
5. C_T_proxy_h = ||h(t)-h(t-T)||_L2 — pas 𝒞_T plein au sens Ch3,
   juste un proxy
6. Logging local βψ_max(t)/γ et global βψ_typ/γ

CAVEAT : aucun co-attracteur observé n'est promu comme "structure MCQ".

INVARIANTS hérités §4a-0 à vérifier :
(A1) Σψ = 1 conservation ✓ attendue exacte
(A2) Antisymétrie flux ψ ✓ attendue exacte
(A3) h ∈ [h_min, h₀] postulé, à mesurer empiriquement
(B1) Engine vs RK4 référence à erreur Euler O(dt)
(C1) Positivité ψ ≥ 0 mesurée
(C2) CFL respecté

Nouveau invariant attendu :
- Symétrie x/y/z préservée à machine precision (ψ centré, h uniforme initiaux)
- Mémoire intégrale : corr(-log(h/h0), ∫ψdt) significative et positive
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
    State6dMinimal,
    make_gaussian_state,
    N_AXIS,
    DX,
    DIM,
    cfl_dt_max,
)
from mcq_v4.factorial_6d.engine import (  # noqa: E402
    compute_diffusion_flux,
    compute_divergence,
)
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero  # noqa: E402


# ============================================================
# RHS du système couplé (ψ, h)
# ============================================================

def rhs_coupled(
    psi: np.ndarray,
    h: np.ndarray,
    D: float,
    beta: float,
    gamma: float,
    h0_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule dψ/dt et dh/dt pour le système couplé.

    dψ/dt = -∇·J_diff avec J_diff = -h_face·D·∇ψ
          = +∇·(h_face·D·∇ψ)
    dh/dt = -β·ψ·h + γ·h·(1 - h/h0)

    Note : compute_divergence renvoie -(J_{i+1/2} - J_{i-1/2})/dx, ce qui
    donne -∇·J = dψ/dt directement quand on passe les flux J_diff.
    """
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)

    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0_target)

    return dpsi_dt, dh_dt


def step_engine_euler(
    psi: np.ndarray,
    h: np.ndarray,
    D: float,
    beta: float,
    gamma: float,
    h0_target: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Step Euler explicite du système couplé."""
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0_target)
    psi_new = psi + dt * dpsi_dt
    h_new = h + dt * dh_dt
    return psi_new, h_new


def step_rk4(
    psi: np.ndarray,
    h: np.ndarray,
    D: float,
    beta: float,
    gamma: float,
    h0_target: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Step RK4 du système couplé (référence indépendante d'ordre 4)."""
    k1_psi, k1_h = rhs_coupled(psi, h, D, beta, gamma, h0_target)

    psi2 = psi + 0.5 * dt * k1_psi
    h2 = h + 0.5 * dt * k1_h
    k2_psi, k2_h = rhs_coupled(psi2, h2, D, beta, gamma, h0_target)

    psi3 = psi + 0.5 * dt * k2_psi
    h3 = h + 0.5 * dt * k2_h
    k3_psi, k3_h = rhs_coupled(psi3, h3, D, beta, gamma, h0_target)

    psi4 = psi + dt * k3_psi
    h4 = h + dt * k3_h
    k4_psi, k4_h = rhs_coupled(psi4, h4, D, beta, gamma, h0_target)

    psi_new = psi + dt / 6.0 * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
    h_new = h + dt / 6.0 * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)
    return psi_new, h_new


def simulate(
    psi_init: np.ndarray,
    h_init: np.ndarray,
    D: float,
    beta: float,
    gamma: float,
    h0_target: float,
    n_steps: int,
    dt: float,
    method: str,
    log_every: int,
    track_integral_psi: bool,
) -> dict:
    """Simule le système (ψ,h) avec engine Euler ou RK4."""
    if method == "engine":
        step_fn = step_engine_euler
    elif method == "rk4":
        step_fn = step_rk4
    else:
        raise ValueError(method)

    psi = psi_init.copy()
    h = h_init.copy()

    integral_psi = np.zeros_like(psi) if track_integral_psi else None

    logs = {
        "t": [], "step": [],
        "psi": [],  # snapshots
        "h": [],
        "psi_total": [], "psi_min": [], "psi_max": [],
        "h_min": [], "h_max": [], "h_mean": [],
        "var_x": [], "var_y": [], "var_z": [],
        "beta_psi_max_over_gamma": [],
    }

    def log_state(step, t, psi, h):
        logs["t"].append(t)
        logs["step"].append(step)
        logs["psi"].append(psi.copy())
        logs["h"].append(h.copy())
        logs["psi_total"].append(float(psi.sum()))
        logs["psi_min"].append(float(psi.min()))
        logs["psi_max"].append(float(psi.max()))
        logs["h_min"].append(float(h.min()))
        logs["h_max"].append(float(h.max()))
        logs["h_mean"].append(float(h.mean()))
        # Variance par axe
        coords = np.arange(N_AXIS) * DX
        psi_x = psi.sum(axis=(1, 2))
        psi_y = psi.sum(axis=(0, 2))
        psi_z = psi.sum(axis=(0, 1))
        def v1d(p):
            tot = p.sum()
            if tot <= 0:
                return 0.0
            m = (coords * p).sum() / tot
            return float(((coords - m) ** 2 * p).sum() / tot)
        logs["var_x"].append(v1d(psi_x))
        logs["var_y"].append(v1d(psi_y))
        logs["var_z"].append(v1d(psi_z))
        # Ratio dynamique local
        if gamma > 0:
            logs["beta_psi_max_over_gamma"].append(
                float(beta * psi.max() / gamma)
            )
        else:
            logs["beta_psi_max_over_gamma"].append(float("inf"))

    log_state(0, 0.0, psi, h)

    for step in range(1, n_steps + 1):
        if track_integral_psi:
            # Intégrale par méthode trapèze : on accumule (ψ(t)+ψ(t+dt))/2 · dt
            psi_before = psi.copy()
        psi, h = step_fn(psi, h, D, beta, gamma, h0_target, dt)
        if track_integral_psi:
            integral_psi += 0.5 * dt * (psi_before + psi)

        if step % log_every == 0:
            log_state(step, step * dt, psi, h)

    logs["psi_final"] = psi.copy()
    logs["h_final"] = h.copy()
    logs["integral_psi"] = integral_psi

    return logs


# ============================================================
# Test une config
# ============================================================

def test_one_config(
    beta: float,
    gamma: float,
    label: str,
    D: float = 0.1,
    h0_target: float = 1.0,
    sigma_psi_init: float = 1.5,
) -> dict:
    """Test une config 4a-δ : engine vs RK4, mémoire intégrale, symétrie."""
    print(f"\n{'='*60}")
    print(f"Test 4a-δ : {label}")
    print(f"β = {beta}, γ = {gamma}, D = {D}")
    print(f"{'='*60}")

    # État initial : ψ gaussien centré, h uniforme
    state_template = make_gaussian_state(
        sigma_0=sigma_psi_init, center=(2.0, 2.0, 2.0), h_uniform=h0_target
    )
    psi_init = state_template.psi.copy()
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0_target)

    psi_typ = 1.0 / (N_AXIS ** 3)  # ψ_typ uniforme = 1/125 ≈ 0.008
    psi_max_init = float(psi_init.max())

    # Échelles de temps
    tau_psi = DX**2 / D
    tau_sed_typ = 1.0 / (beta * psi_typ) if beta * psi_typ > 0 else float("inf")
    tau_sed_max = 1.0 / (beta * psi_max_init) if beta * psi_max_init > 0 else float("inf")
    tau_ero = 1.0 / gamma if gamma > 0 else float("inf")

    print(f"\nÉchelles de temps :")
    print(f"  τ_ψ        = dx²/D       = {tau_psi:.4f}")
    print(f"  τ_sed_typ  = 1/(β·ψ_typ) = {tau_sed_typ:.4f}")
    print(f"  τ_sed_max  = 1/(β·ψ_max) = {tau_sed_max:.4f}")
    print(f"  τ_ero      = 1/γ         = {tau_ero:.4f}")

    print(f"\nRapports :")
    print(f"  β·ψ_typ / γ = {beta * psi_typ / gamma:.4f}")
    print(f"  β·ψ_max / γ = {beta * psi_max_init / gamma:.4f}")
    print(f"  τ_h / τ_ψ   = {min(tau_sed_typ, tau_ero) / tau_psi:.4f}")

    # Temps de simulation : couvrir au moins quelques τ pour chaque échelle
    t_target = 5.0 * max(tau_psi, min(tau_sed_typ, tau_ero) if min(tau_sed_typ, tau_ero) < 1000 else tau_psi)
    t_target = min(t_target, 200.0)  # plafond

    # CFL
    h_max_init = h0_target
    rate_h_max = beta * psi_max_init + gamma
    dt_cfl_diff = cfl_dt_max(h_max_init, D)
    dt_cfl_h = 1.0 / rate_h_max if rate_h_max > 0 else float("inf")
    dt_cfl = 0.5 * min(dt_cfl_diff, dt_cfl_h)

    print(f"\nCFL :")
    print(f"  dt_cfl_diff = {dt_cfl_diff:.4f}")
    print(f"  dt_cfl_h    = {dt_cfl_h:.4f}")
    print(f"  dt_cfl      = {dt_cfl:.4f}")
    print(f"  t_target    = {t_target:.4f}")

    # === Test 1 : Engine vs RK4 référence à même dt
    dt_engine = dt_cfl
    n_steps_engine = max(1, int(np.ceil(t_target / dt_engine)))
    print(f"\n--- Simulation engine (Euler) ---")
    print(f"dt = {dt_engine:.4f}, n_steps = {n_steps_engine}")
    logs_engine = simulate(
        psi_init, h_init, D, beta, gamma, h0_target,
        n_steps_engine, dt_engine, method="engine",
        log_every=max(1, n_steps_engine // 20), track_integral_psi=True,
    )

    # RK4 à dt très petit (référence "vraie")
    dt_rk4 = dt_cfl / 10.0  # plus petit pour minimiser erreur RK4
    n_steps_rk4 = max(1, int(np.ceil(t_target / dt_rk4)))
    # Aligner sur même temps total
    n_steps_rk4 = int(np.ceil(t_target / dt_rk4))
    print(f"\n--- Simulation RK4 référence ---")
    print(f"dt_rk4 = {dt_rk4:.4f}, n_steps_rk4 = {n_steps_rk4}")
    logs_rk4 = simulate(
        psi_init, h_init, D, beta, gamma, h0_target,
        n_steps_rk4, dt_rk4, method="rk4",
        log_every=max(1, n_steps_rk4 // 20), track_integral_psi=False,
    )

    # Comparaison aux temps finaux
    psi_engine_final = logs_engine["psi_final"]
    h_engine_final = logs_engine["h_final"]
    psi_rk4_final = logs_rk4["psi_final"]
    h_rk4_final = logs_rk4["h_final"]

    diff_psi = float(np.max(np.abs(psi_engine_final - psi_rk4_final)))
    diff_h = float(np.max(np.abs(h_engine_final - h_rk4_final)))

    # Erreur Euler théorique attendue : ~ rate · dt · t_total · |state|
    # Pour ψ : rate ~ D·DIM/dx² (diffusion), state ~ psi_max
    # Pour h : rate ~ β·ψ_max + γ
    rate_psi = D * DIM / DX**2
    rate_h_eff = beta * psi_max_init + gamma
    err_euler_psi_attendu = rate_psi * dt_engine * t_target * psi_max_init
    err_euler_h_attendu = rate_h_eff * dt_engine * t_target * h0_target

    print(f"\n=== Comparaison engine vs RK4 ===")
    print(f"  diff_psi (engine vs RK4)     : {diff_psi:.4e}")
    print(f"  diff_h   (engine vs RK4)     : {diff_h:.4e}")
    print(f"  err_Euler ψ attendue         : {err_euler_psi_attendu:.4e}")
    print(f"  err_Euler h attendue         : {err_euler_h_attendu:.4e}")

    # === Test 2 : Ordre temporel d'Euler (engine vs RK4 à dt/2)
    print(f"\n--- Test ordre temporel : engine dt vs engine dt/2 ---")
    dt_engine_half = dt_engine / 2.0
    n_steps_engine_half = max(1, int(np.ceil(t_target / dt_engine_half)))
    logs_engine_half = simulate(
        psi_init, h_init, D, beta, gamma, h0_target,
        n_steps_engine_half, dt_engine_half, method="engine",
        log_every=max(1, n_steps_engine_half // 20),
        track_integral_psi=False,
    )

    diff_psi_dt = float(np.max(np.abs(logs_engine["psi_final"] - psi_rk4_final)))
    diff_psi_dt_half = float(np.max(np.abs(logs_engine_half["psi_final"] - psi_rk4_final)))
    if diff_psi_dt_half > 1e-15:
        ratio_psi = diff_psi_dt / diff_psi_dt_half
    else:
        ratio_psi = float("nan")
    diff_h_dt = float(np.max(np.abs(logs_engine["h_final"] - h_rk4_final)))
    diff_h_dt_half = float(np.max(np.abs(logs_engine_half["h_final"] - h_rk4_final)))
    if diff_h_dt_half > 1e-15:
        ratio_h = diff_h_dt / diff_h_dt_half
    else:
        ratio_h = float("nan")

    print(f"  diff_ψ engine(dt) vs RK4 : {diff_psi_dt:.4e}")
    print(f"  diff_ψ engine(dt/2) vs RK4 : {diff_psi_dt_half:.4e}")
    print(f"  ratio = {ratio_psi:.4f} (attendu ≈ 2 pour ordre 1)")
    print(f"  diff_h engine(dt) vs RK4 : {diff_h_dt:.4e}")
    print(f"  diff_h engine(dt/2) vs RK4 : {diff_h_dt_half:.4e}")
    print(f"  ratio = {ratio_h:.4f} (attendu ≈ 2 pour ordre 1)")

    # === Test 3 : Invariants exacts
    mass_drift = max(abs(m - 1.0) for m in logs_engine["psi_total"])
    psi_min = min(logs_engine["psi_min"])
    h_min = min(logs_engine["h_min"])
    h_max = max(logs_engine["h_max"])

    print(f"\n=== Invariants exacts ===")
    print(f"  Σψ conservation : drift max = {mass_drift:.4e}")
    print(f"  positivité ψ    : min ψ     = {psi_min:.4e}")
    print(f"  positivité h    : min h     = {h_min:.4e}")
    print(f"  h_max global    : {h_max:.4f} (h₀ = {h0_target})")

    # === Test 4 : Symétrie x/y/z (engine)
    var_x = np.array(logs_engine["var_x"])
    var_y = np.array(logs_engine["var_y"])
    var_z = np.array(logs_engine["var_z"])
    sym_xy = float(np.max(np.abs(var_x - var_y)))
    sym_xz = float(np.max(np.abs(var_x - var_z)))
    sym_yz = float(np.max(np.abs(var_y - var_z)))
    sym_max = max(sym_xy, sym_xz, sym_yz)

    print(f"\n=== Symétrie x/y/z ===")
    print(f"  max |Var_x - Var_y| = {sym_xy:.4e}")
    print(f"  max |Var_x - Var_z| = {sym_xz:.4e}")
    print(f"  max |Var_y - Var_z| = {sym_yz:.4e}")
    print(f"  symétrie max = {sym_max:.4e}")

    # === Test 5 : Trace memory
    integral_psi = logs_engine["integral_psi"]
    h_final = logs_engine["h_final"]

    # Corrélation Pearson (engine final h vs ∫ψ dt)
    h_flat = h_final.flatten()
    I_flat = integral_psi.flatten()
    if h_flat.std() > 0 and I_flat.std() > 0:
        corr_h_I = float(np.corrcoef(h_flat, I_flat)[0, 1])
    else:
        corr_h_I = float("nan")

    # Corrélation Pearson (-log(h/h0) vs ∫ψ dt)
    log_ratio = -np.log(np.maximum(h_flat, 1e-30) / h0_target)
    if log_ratio.std() > 0 and I_flat.std() > 0:
        corr_logh_I = float(np.corrcoef(log_ratio, I_flat)[0, 1])
    else:
        corr_logh_I = float("nan")

    print(f"\n=== Trace memory ===")
    print(f"  corr(h_final, ∫ψ·dt)         = {corr_h_I:+.4f} (attendu négatif fort)")
    print(f"  corr(-log(h/h₀), ∫ψ·dt)      = {corr_logh_I:+.4f} (attendu positif fort)")

    # === Test 6 : C_T proxy
    # Prend T = t_target/4 et compare h(t_final) vs h(t_final - T)
    T_window = t_target / 4.0
    if len(logs_engine["t"]) >= 2:
        idx_final = -1
        t_final = logs_engine["t"][-1]
        t_target_window = t_final - T_window
        # Trouver index proche
        idx_window = min(
            range(len(logs_engine["t"])),
            key=lambda i: abs(logs_engine["t"][i] - t_target_window),
        )
        h_at_t = logs_engine["h"][idx_final]
        h_at_t_minus_T = logs_engine["h"][idx_window]
        C_T_proxy_h = float(np.sqrt(np.sum((h_at_t - h_at_t_minus_T) ** 2)))
    else:
        C_T_proxy_h = float("nan")

    print(f"\n=== C_T proxy ===")
    print(f"  C_T_proxy_h (T={T_window:.2f}) = {C_T_proxy_h:.4e}")
    print(f"  (NB : pas 𝒞_T plein au sens Ch3, juste proxy L2 sur h)")

    # === VERDICT ===
    verdict = "PASS"
    reasons = []

    if mass_drift > 1e-10:
        verdict = "FAIL"
        reasons.append(f"Conservation Σψ : drift = {mass_drift:.4e}")

    if psi_min < -1e-12:
        verdict = "FAIL"
        reasons.append(f"Positivité ψ violée : min = {psi_min:.4e}")

    if h_min < -1e-12:
        verdict = "FAIL"
        reasons.append(f"Positivité h violée : min = {h_min:.4e}")

    if sym_max > 1e-12:
        verdict = "FAIL" if sym_max > 1e-9 else "REVISION"
        reasons.append(f"Symétrie x/y/z cassée : max = {sym_max:.4e}")

    # Engine vs RK4 cohérent avec erreur Euler attendue
    seuil_psi = max(err_euler_psi_attendu * 10, 1e-8)
    if diff_psi > seuil_psi:
        verdict = "FAIL" if diff_psi > seuil_psi * 10 else "REVISION"
        reasons.append(f"Engine vs RK4 (ψ) : diff={diff_psi:.4e} > {seuil_psi:.4e}")

    # Ratio d'ordre temporel : attendu ~ 2
    if not np.isnan(ratio_psi) and (ratio_psi < 1.3 or ratio_psi > 4.0):
        verdict = "REVISION"
        reasons.append(f"Ordre temporel ψ : ratio={ratio_psi:.2f} hors [1.3, 4.0]")

    print(f"\n=== VERDICT ===")
    if verdict == "PASS":
        print(f"  PASS ✓ pour {label}")
    else:
        print(f"  Verdict : {verdict}")
        for r in reasons:
            print(f"  - {r}")

    return {
        "label": label,
        "verdict": verdict,
        "reasons": reasons,
        "params": {
            "beta": beta, "gamma": gamma, "D": D, "h0_target": h0_target,
            "tau_psi": tau_psi, "tau_sed_typ": tau_sed_typ,
            "tau_sed_max": tau_sed_max, "tau_ero": tau_ero,
            "ratio_betapsi_typ_over_gamma": beta * psi_typ / gamma if gamma > 0 else float("inf"),
            "ratio_betapsi_max_over_gamma": beta * psi_max_init / gamma if gamma > 0 else float("inf"),
            "dt_engine": dt_engine, "n_steps_engine": n_steps_engine,
            "dt_rk4": dt_rk4, "n_steps_rk4": n_steps_rk4,
            "t_target": t_target,
        },
        "metrics": {
            "mass_drift_max": float(mass_drift),
            "psi_min": float(psi_min),
            "h_min": float(h_min),
            "h_max": float(h_max),
            "diff_engine_vs_rk4_psi": diff_psi,
            "diff_engine_vs_rk4_h": diff_h,
            "err_euler_psi_attendue": float(err_euler_psi_attendu),
            "err_euler_h_attendue": float(err_euler_h_attendu),
            "ratio_temporal_order_psi": float(ratio_psi) if not np.isnan(ratio_psi) else None,
            "ratio_temporal_order_h": float(ratio_h) if not np.isnan(ratio_h) else None,
            "symmetry_max": float(sym_max),
            "trace_memory_corr_h_I": corr_h_I,
            "trace_memory_corr_logh_I": corr_logh_I,
            "C_T_proxy_h": C_T_proxy_h,
            "T_window": float(T_window),
        },
    }


def run_step_4a_delta() -> dict:
    """3 configs A/B/C selon les rapports d'échelles."""
    configs = [
        # (beta, gamma, label)
        # τ_ψ = 10 (D=0.1, dx=1)
        (1.0, 0.01, "A_h_lent_psi_rapide"),       # τ_sed=125, τ_ero=100 >> 10
        (12.5, 0.1, "B_echelles_comparables"),    # τ_sed=10, τ_ero=10 ~ 10
        (125.0, 1.0, "C_h_rapide_psi_lent"),      # τ_sed=1, τ_ero=1 << 10
    ]

    results = {}
    for beta, gamma, label in configs:
        result = test_one_config(beta=beta, gamma=gamma, label=label)
        results[label] = result

    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*60}")
    print(f"VERDICT GLOBAL 4a-δ : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    # Tableau récap des métriques clés
    print(f"\n{'='*70}")
    print(f"RÉCAP métriques clés")
    print(f"{'='*70}")
    print(f"{'config':<30} {'sym_max':>10} {'corr_logh_I':>12} {'h_min':>10}")
    for label, r in results.items():
        m = r["metrics"]
        print(f"{label:<30} {m['symmetry_max']:>10.2e} "
              f"{m['trace_memory_corr_logh_I']:>+12.4f} "
              f"{m['h_min']:>10.4f}")

    return {"global_verdict": global_verdict, "cases": results}


if __name__ == "__main__":
    summary = run_step_4a_delta()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_4a_delta_coupled.json"

    # NumPy arrays cannot be JSON-serialized directly
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return None  # ne pas sauvegarder les snapshots arrays
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    summary_serializable = make_serializable(summary)
    with open(output_path, "w") as f:
        json.dump(summary_serializable, f, indent=2)
    print(f"\nRésultats sauvegardés : {output_path}")
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
