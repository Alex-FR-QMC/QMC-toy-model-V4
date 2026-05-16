"""
Test 6d-α micro-étape 4a-ε — co-évolution ψ↔h avec ψ asymétrique.

Plan validé Alex post-4a-δ :
- Même système couplé que 4a-δ : ∂_t ψ = ∇·(h·D·∇ψ), ∂_t h = -βψh + γh(1-h/h₀)
- ψ initial GAUSSIEN DÉCENTRÉ à (1, 2, 2)
- Mêmes 3 régimes A/B/C (h lent, comparable, h rapide)

Mesures supplémentaires post-4a-δ (instrumentation collapse) :
- fraction_cells_under_h_resolution(t)
- connected_components_h_active(t)
- temps_de_reactivation par cellule

Invariants attendus :
- (A1) Σψ conservation : ✓ exacte
- (A2) Antisymétrie flux ✓
- Symétrie y↔z (h n'a pas de raison de différencier y et z) ✓
- Symétrie x brisée par construction (ψ décentré en x)
- (B1) Engine vs RK4 cohérent erreur Euler
- Mémoire morphologique : corr(-log(h/h₀), ∫ψdt) toujours forte
- h doit tracer asymétriquement le passage de ψ

CAVEAT (réitéré §4a-0) : h_resolution est un SEUIL NUMÉRIQUE, pas un
critère MCQ. h < h_resolution = sortie du domaine numériquement
résoluble, pas violation MCQ. Pas de clipping appliqué.
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


# Seuil numérique de résolution h
H_RESOLUTION = 1e-6


# ============================================================
# RHS et steps (importés de 4a-δ — réutilisés)
# ============================================================

def rhs_coupled(psi, h, D, beta, gamma, h0_target):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)
    dh_dt = G_sed(psi, h, beta) + G_ero(h, gamma, h0_target)
    return dpsi_dt, dh_dt


def step_engine_euler(psi, h, D, beta, gamma, h0_target, dt):
    dpsi_dt, dh_dt = rhs_coupled(psi, h, D, beta, gamma, h0_target)
    return psi + dt * dpsi_dt, h + dt * dh_dt


def step_rk4(psi, h, D, beta, gamma, h0_target, dt):
    k1p, k1h = rhs_coupled(psi, h, D, beta, gamma, h0_target)
    k2p, k2h = rhs_coupled(psi + 0.5*dt*k1p, h + 0.5*dt*k1h, D, beta, gamma, h0_target)
    k3p, k3h = rhs_coupled(psi + 0.5*dt*k2p, h + 0.5*dt*k2h, D, beta, gamma, h0_target)
    k4p, k4h = rhs_coupled(psi + dt*k3p, h + dt*k3h, D, beta, gamma, h0_target)
    psi_new = psi + dt/6.0 * (k1p + 2*k2p + 2*k3p + k4p)
    h_new = h + dt/6.0 * (k1h + 2*k2h + 2*k3h + k4h)
    return psi_new, h_new


def make_psi_decentered(center: tuple[float, float, float], sigma_0: float = 1.5) -> np.ndarray:
    """ψ gaussien décentré, normalisé Σψ=1."""
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


def variance_per_axis_arr(psi: np.ndarray) -> tuple[float, float, float]:
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
    return v1d(psi_x), v1d(psi_y), v1d(psi_z)


def com_per_axis(psi: np.ndarray) -> tuple[float, float, float]:
    coords = np.arange(N_AXIS) * DX
    tot = psi.sum()
    if tot <= 0:
        return 0.0, 0.0, 0.0
    psi_x = psi.sum(axis=(1, 2))
    psi_y = psi.sum(axis=(0, 2))
    psi_z = psi.sum(axis=(0, 1))
    return (
        float((coords * psi_x).sum() / tot),
        float((coords * psi_y).sum() / tot),
        float((coords * psi_z).sum() / tot),
    )


def count_connected_components_h_active(h: np.ndarray, threshold: float) -> int:
    """Nombre de composantes connexes dans le sous-domaine h >= threshold."""
    mask = (h >= threshold).astype(int)
    # 6-connectivity en 3D (face-sharing)
    structure = ndimage.generate_binary_structure(3, 1)
    _, n_components = ndimage.label(mask, structure=structure)
    return int(n_components)


def simulate(
    psi_init, h_init, D, beta, gamma, h0_target,
    n_steps, dt, method, log_every, track_collapse_metrics,
):
    """Simule (ψ, h) avec instrumentation collapse."""
    if method == "engine":
        step_fn = step_engine_euler
    elif method == "rk4":
        step_fn = step_rk4
    else:
        raise ValueError(method)

    psi = psi_init.copy()
    h = h_init.copy()
    integral_psi = np.zeros_like(psi)

    # Tracking réactivation : pour chaque cellule, t_down (premier passage
    # sous seuil) et t_up (premier passage au-dessus après down)
    t_first_down = np.full(psi.shape, -1.0)  # -1 = pas encore descendu
    t_first_reactivation = np.full(psi.shape, -1.0)  # -1 = pas encore remonté

    logs = {
        "t": [],
        "psi_total": [], "psi_min": [], "psi_max": [],
        "h_min": [], "h_max": [], "h_mean": [],
        "var_x": [], "var_y": [], "var_z": [],
        "com_x": [], "com_y": [], "com_z": [],
        "fraction_cells_under_h_resolution": [],
        "n_connected_components_h_active": [],
    }

    def log_state(t, psi, h):
        logs["t"].append(t)
        logs["psi_total"].append(float(psi.sum()))
        logs["psi_min"].append(float(psi.min()))
        logs["psi_max"].append(float(psi.max()))
        logs["h_min"].append(float(h.min()))
        logs["h_max"].append(float(h.max()))
        logs["h_mean"].append(float(h.mean()))
        vx, vy, vz = variance_per_axis_arr(psi)
        logs["var_x"].append(vx)
        logs["var_y"].append(vy)
        logs["var_z"].append(vz)
        cx, cy, cz = com_per_axis(psi)
        logs["com_x"].append(cx)
        logs["com_y"].append(cy)
        logs["com_z"].append(cz)
        if track_collapse_metrics:
            frac = float(np.sum(h < H_RESOLUTION) / h.size)
            n_comp = count_connected_components_h_active(h, H_RESOLUTION)
            logs["fraction_cells_under_h_resolution"].append(frac)
            logs["n_connected_components_h_active"].append(n_comp)

    log_state(0.0, psi, h)

    for step in range(1, n_steps + 1):
        psi_before = psi.copy()
        psi, h = step_fn(psi, h, D, beta, gamma, h0_target, dt)
        # Intégrale ψ par trapèze
        integral_psi += 0.5 * dt * (psi_before + psi)

        # Détection passage sous seuil et réactivation
        if track_collapse_metrics:
            below_now = h < H_RESOLUTION
            # Premier passage sous seuil
            first_down_mask = below_now & (t_first_down < 0)
            t_first_down[first_down_mask] = step * dt
            # Réactivation : était sous seuil ET maintenant au-dessus
            # ET pas encore enregistré
            reactivation_mask = (
                (t_first_down >= 0)
                & (t_first_reactivation < 0)
                & (h >= H_RESOLUTION)
            )
            t_first_reactivation[reactivation_mask] = step * dt

        if step % log_every == 0:
            log_state(step * dt, psi, h)

    logs["psi_final"] = psi.copy()
    logs["h_final"] = h.copy()
    logs["integral_psi"] = integral_psi
    logs["t_first_down"] = t_first_down
    logs["t_first_reactivation"] = t_first_reactivation
    return logs


def test_one_config(beta, gamma, label, D=0.1, h0_target=1.0, sigma_psi=1.5):
    print(f"\n{'='*60}")
    print(f"Test 4a-ε : {label}")
    print(f"β = {beta}, γ = {gamma}, D = {D}")
    print(f"ψ INITIAL DÉCENTRÉ à (1, 2, 2)")
    print(f"{'='*60}")

    psi_init = make_psi_decentered(center=(1.0, 2.0, 2.0), sigma_0=sigma_psi)
    h_init = np.full((N_AXIS, N_AXIS, N_AXIS), h0_target)
    psi_max_init = float(psi_init.max())
    psi_typ = 1.0 / (N_AXIS ** 3)

    tau_psi = DX**2 / D
    tau_sed_typ = 1.0 / (beta * psi_typ)
    tau_ero = 1.0 / gamma
    print(f"\nÉchelles : τ_ψ={tau_psi:.4f}, τ_sed_typ={tau_sed_typ:.4f}, τ_ero={tau_ero:.4f}")

    rate_h_max = beta * psi_max_init + gamma
    dt_cfl_diff = cfl_dt_max(h0_target, D)
    dt_cfl_h = 1.0 / rate_h_max
    dt = 0.5 * min(dt_cfl_diff, dt_cfl_h)

    t_target = 5.0 * max(tau_psi, min(tau_sed_typ, tau_ero))
    t_target = min(t_target, 200.0)
    n_steps = max(1, int(np.ceil(t_target / dt)))

    print(f"dt = {dt:.4f}, n_steps = {n_steps}, t_target = {t_target:.4f}")

    # === Simulation engine
    logs_engine = simulate(
        psi_init, h_init, D, beta, gamma, h0_target,
        n_steps, dt, method="engine",
        log_every=max(1, n_steps // 30),
        track_collapse_metrics=True,
    )

    # === Simulation RK4 référence
    dt_rk4 = dt / 10.0
    n_steps_rk4 = max(1, int(np.ceil(t_target / dt_rk4)))
    logs_rk4 = simulate(
        psi_init, h_init, D, beta, gamma, h0_target,
        n_steps_rk4, dt_rk4, method="rk4",
        log_every=max(1, n_steps_rk4 // 10),
        track_collapse_metrics=False,
    )

    # === Comparaisons
    diff_psi = float(np.max(np.abs(logs_engine["psi_final"] - logs_rk4["psi_final"])))
    diff_h = float(np.max(np.abs(logs_engine["h_final"] - logs_rk4["h_final"])))

    rate_psi = D * DIM / DX**2
    rate_h_eff = beta * psi_max_init + gamma
    err_euler_psi = rate_psi * dt * t_target * psi_max_init
    err_euler_h = rate_h_eff * dt * t_target * h0_target

    print(f"\n=== Engine vs RK4 ===")
    print(f"  diff_ψ = {diff_psi:.4e} (attendu O({err_euler_psi:.2e}))")
    print(f"  diff_h = {diff_h:.4e} (attendu O({err_euler_h:.2e}))")

    # === Invariants
    mass_drift = max(abs(m - 1.0) for m in logs_engine["psi_total"])
    psi_min = min(logs_engine["psi_min"])
    h_min_global = min(logs_engine["h_min"])
    h_max_global = max(logs_engine["h_max"])

    # Symétrie y↔z : doit être préservée (h ne distingue pas y et z)
    var_y_arr = np.array(logs_engine["var_y"])
    var_z_arr = np.array(logs_engine["var_z"])
    sym_yz = float(np.max(np.abs(var_y_arr - var_z_arr)))
    # Symétrie x↔y : DOIT être brisée (ψ décentré en x)
    var_x_arr = np.array(logs_engine["var_x"])
    sym_xy_max = float(np.max(np.abs(var_x_arr - var_y_arr)))

    com_x_arr = np.array(logs_engine["com_x"])
    com_y_arr = np.array(logs_engine["com_y"])
    com_z_arr = np.array(logs_engine["com_z"])

    print(f"\n=== Invariants ===")
    print(f"  Σψ : drift = {mass_drift:.4e}")
    print(f"  positivité ψ : min = {psi_min:.4e}")
    print(f"  h_min global : {h_min_global:.4e}")
    print(f"  h_max global : {h_max_global:.4f}")

    print(f"\n=== Symétries ===")
    print(f"  Symétrie y↔z (doit tenir) : max|Vy-Vz| = {sym_yz:.4e}")
    print(f"  Symétrie x↔y (brisée par construction) : max|Vx-Vy| = {sym_xy_max:.4e}")
    print(f"  COM x initial = {com_x_arr[0]:.4f}, COM x final = {com_x_arr[-1]:.4f}")
    print(f"  COM y initial = {com_y_arr[0]:.4f}, COM y final = {com_y_arr[-1]:.4f}")
    print(f"  COM z initial = {com_z_arr[0]:.4f}, COM z final = {com_z_arr[-1]:.4f}")

    # === Trace memory
    integral_psi = logs_engine["integral_psi"]
    h_final = logs_engine["h_final"]
    h_flat = h_final.flatten()
    I_flat = integral_psi.flatten()
    corr_h_I = float(np.corrcoef(h_flat, I_flat)[0, 1]) if h_flat.std() > 0 and I_flat.std() > 0 else float("nan")
    log_ratio = -np.log(np.maximum(h_flat, 1e-30) / h0_target)
    corr_logh_I = float(np.corrcoef(log_ratio, I_flat)[0, 1]) if log_ratio.std() > 0 and I_flat.std() > 0 else float("nan")

    print(f"\n=== Trace memory ===")
    print(f"  corr(h_final, ∫ψ·dt)         = {corr_h_I:+.4f}")
    print(f"  corr(-log(h/h₀), ∫ψ·dt)      = {corr_logh_I:+.4f}")

    # === Instrumentation collapse (NEW post-4a-δ)
    frac_under = logs_engine["fraction_cells_under_h_resolution"]
    n_comp_arr = logs_engine["n_connected_components_h_active"]
    frac_final = frac_under[-1] if frac_under else 0.0
    frac_max = max(frac_under) if frac_under else 0.0
    n_comp_final = n_comp_arr[-1] if n_comp_arr else 1
    n_comp_min = min(n_comp_arr) if n_comp_arr else 1
    n_comp_max = max(n_comp_arr) if n_comp_arr else 1

    # Réactivation
    t_first_down = logs_engine["t_first_down"]
    t_first_reactivation = logs_engine["t_first_reactivation"]
    n_cells_ever_down = int(np.sum(t_first_down >= 0))
    n_cells_reactivated = int(np.sum(t_first_reactivation >= 0))
    if n_cells_reactivated > 0:
        delays = t_first_reactivation[t_first_reactivation >= 0] - t_first_down[t_first_reactivation >= 0]
        delay_mean = float(np.mean(delays))
        delay_max = float(np.max(delays))
    else:
        delay_mean = float("nan")
        delay_max = float("nan")

    print(f"\n=== Instrumentation collapse ===")
    print(f"  fraction cellules sous h_resolution ({H_RESOLUTION}) :")
    print(f"    final = {frac_final:.4f}, max sur trajectoire = {frac_max:.4f}")
    print(f"  composantes connexes h actif :")
    print(f"    final = {n_comp_final}, min = {n_comp_min}, max = {n_comp_max}")
    print(f"  cellules passées sous seuil : {n_cells_ever_down}/125")
    print(f"  cellules réactivées         : {n_cells_reactivated}/125")
    if n_cells_reactivated > 0:
        print(f"  délai réactivation moyen : {delay_mean:.4f}, max : {delay_max:.4f}")

    # === VERDICT
    verdict = "PASS"
    reasons = []

    if mass_drift > 1e-10:
        verdict = "FAIL"
        reasons.append(f"Σψ : drift = {mass_drift:.4e}")

    if psi_min < -1e-12:
        verdict = "FAIL"
        reasons.append(f"Positivité ψ : min = {psi_min:.4e}")

    if sym_yz > 1e-12:
        verdict = "FAIL" if sym_yz > 1e-9 else "REVISION"
        reasons.append(f"Symétrie y↔z violée : {sym_yz:.4e}")

    seuil_psi = max(err_euler_psi * 10, 1e-8)
    if diff_psi > seuil_psi:
        verdict = "FAIL" if diff_psi > seuil_psi * 10 else "REVISION"
        reasons.append(f"Engine vs RK4 ψ : {diff_psi:.4e} > {seuil_psi:.4e}")

    # Note : h_min très bas n'est PAS un échec (cf §4a-0 stratification α/β/γ)

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
            "psi_center_init": [1.0, 2.0, 2.0],
            "tau_psi": tau_psi, "tau_sed_typ": tau_sed_typ, "tau_ero": tau_ero,
            "dt": dt, "n_steps": n_steps, "t_target": t_target,
            "h_resolution": H_RESOLUTION,
        },
        "metrics": {
            "mass_drift": float(mass_drift),
            "psi_min": float(psi_min),
            "h_min_global": float(h_min_global),
            "h_max_global": float(h_max_global),
            "diff_engine_vs_rk4_psi": diff_psi,
            "diff_engine_vs_rk4_h": diff_h,
            "err_euler_psi_attendue": float(err_euler_psi),
            "err_euler_h_attendue": float(err_euler_h),
            "symmetry_yz_max": sym_yz,
            "asymmetry_xy_max": sym_xy_max,
            "com_x_final": float(com_x_arr[-1]),
            "com_y_final": float(com_y_arr[-1]),
            "com_z_final": float(com_z_arr[-1]),
            "trace_corr_h_I": corr_h_I,
            "trace_corr_logh_I": corr_logh_I,
            "fraction_under_h_resolution_final": float(frac_final),
            "fraction_under_h_resolution_max": float(frac_max),
            "n_connected_components_final": n_comp_final,
            "n_connected_components_min": n_comp_min,
            "n_connected_components_max": n_comp_max,
            "n_cells_ever_under_resolution": n_cells_ever_down,
            "n_cells_reactivated": n_cells_reactivated,
            "delay_reactivation_mean": delay_mean if not np.isnan(delay_mean) else None,
            "delay_reactivation_max": delay_max if not np.isnan(delay_max) else None,
        },
    }


def run_step_4a_epsilon():
    configs = [
        (1.0, 0.01, "A_h_lent_psi_rapide"),
        (12.5, 0.1, "B_echelles_comparables"),
        (125.0, 1.0, "C_h_rapide_psi_lent"),
    ]
    results = {}
    for beta, gamma, label in configs:
        results[label] = test_one_config(beta, gamma, label)

    verdicts = [r["verdict"] for r in results.values()]
    if all(v == "PASS" for v in verdicts):
        global_verdict = "PASS"
    elif "FAIL" in verdicts:
        global_verdict = "FAIL"
    else:
        global_verdict = "REVISION"

    print(f"\n\n{'='*60}")
    print(f"VERDICT GLOBAL 4a-ε : {global_verdict}")
    print(f"{'='*60}")
    for label, r in results.items():
        print(f"  {label} : {r['verdict']}")

    print(f"\n{'='*80}")
    print(f"RÉCAP collapse instrumentation")
    print(f"{'='*80}")
    print(f"{'config':<30} {'frac_max':>10} {'n_comp_min':>10} {'n_react':>10}")
    for label, r in results.items():
        m = r["metrics"]
        print(f"{label:<30} {m['fraction_under_h_resolution_max']:>10.4f} "
              f"{m['n_connected_components_min']:>10d} "
              f"{m['n_cells_reactivated']:>10d}")

    return {"global_verdict": global_verdict, "cases": results}


if __name__ == "__main__":
    summary = run_step_4a_epsilon()
    output_dir = REPO_ROOT / "results" / "phase6d_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_4a_epsilon_asymmetric.json"

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
    sys.exit(0 if summary["global_verdict"] == "PASS" else 1)
