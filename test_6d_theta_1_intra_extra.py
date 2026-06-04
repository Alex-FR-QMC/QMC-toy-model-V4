"""
6d-θ-1 — Test principal : G/A vs intra-familles à Δt_sep ∈ {25, 100}.

Question : le petit signal post-composition observé en θ-0b est-il
spécifique à l'axe centre/couronne G/A, ou seulement une non-linéarité
générique du moteur qui produit du commutateur pour n'importe quelle
paire de perturbations ?

Δt_peak = 100 (pic de K_inst+100 en θ-0b)
Δt_control = 25 (signal post-composition minimal)
T_final = 800 depuis t0
Amplitude = 10% de ||P6||

8 paires :
G/A (inter-famille, axe testé) :
  1. G_standard_centree ↔ A_anneau_moyen
  2. G_large_centree ↔ A_shell_peripherique
  3. G_decentree_x ↔ A_anneau_externe
  4. G_etroite_centree ↔ A_anneau_interne

G/G (contrôle intra-famille gaussiennes) :
  5. G_standard_centree ↔ G_large_centree
  6. G_etroite_centree ↔ G_decentree_diag

A/A (contrôle intra-famille anneaux) :
  7. A_anneau_moyen ↔ A_anneau_externe
  8. A_anneau_interne ↔ A_shell_peripherique

Critères de verdict (sur métriques principales K_AUC_post_ext et K_inst+100) :
- θ-PASS fort : K(G/A) nettement > K(G/G, A/A), signal au-dessus du floor
- θ-PASS faible : K(G/A) > K(intra) mais marginalement, ou seulement sur certaines paires
- θ-FAIL : K(G/A) ≈ K(intra) (effet générique du moteur)
- θ-FLOOR : tout au floor numérique

K_AUC_total et K_AUC_pre rapportés mais hors verdict (effet d'intégration).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import json
from scipy.optimize import brentq

from mcq_v4.factorial_6d import N_AXIS, DX, cfl_dt_max
from mcq_v4.factorial_6d.engine import compute_diffusion_flux, compute_divergence
from mcq_v4.factorial_6d.h_dynamics import G_sed, G_ero


def rhs(psi, h, D, beta, gamma, h0):
    Jx, Jy, Jz = compute_diffusion_flux(psi, h, D)
    dpsi = compute_divergence(Jx, Jy, Jz)
    dh = G_sed(psi, h, beta) + G_ero(h, gamma, h0)
    return dpsi, dh

def step(psi, h, D, beta, gamma, h0, dt):
    dpsi, dh = rhs(psi, h, D, beta, gamma, h0)
    return psi + dt * dpsi, h + dt * dh

def evolve(psi, h, D, beta, gamma, h0, dt, n_steps):
    for _ in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
    return psi, h

def make_psi_centered(sigma=1.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    psi = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                psi[i,j,k] = np.exp(-0.5 * r2 / sigma**2)
    return psi / psi.sum()

def P6_face_dipole(psi, strength):
    factor = np.ones_like(psi)
    factor[0, :, :] += strength
    factor[4, :, :] -= strength
    p = psi * factor
    p = np.maximum(p, 0)
    return p / p.sum()

COORDS = np.arange(N_AXIS) * DX
CENTER = (N_AXIS - 1) * DX / 2.0

def _gaussian_factor(cx, cy, cz, sigma_p):
    factor = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((COORDS[i]-cx)**2 + (COORDS[j]-cy)**2 + (COORDS[k]-cz)**2)
                factor[i,j,k] = np.exp(-0.5 * r2 / sigma_p**2)
    return factor

def _radial_mask(r_inner, r_outer):
    mask = np.zeros((N_AXIS, N_AXIS, N_AXIS))
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((COORDS[i]-CENTER)**2 + (COORDS[j]-CENTER)**2 + (COORDS[k]-CENTER)**2)
                if r_inner <= r <= r_outer:
                    mask[i,j,k] = 1.0
    return mask

def P_G_etroite_centree(psi, s):
    f = 1.0 + s * _gaussian_factor(CENTER, CENTER, CENTER, 0.5)
    return (psi * f) / (psi * f).sum()
def P_G_standard_centree(psi, s):
    f = 1.0 + s * _gaussian_factor(CENTER, CENTER, CENTER, 0.8)
    return (psi * f) / (psi * f).sum()
def P_G_large_centree(psi, s):
    f = 1.0 + s * _gaussian_factor(CENTER, CENTER, CENTER, 1.5)
    return (psi * f) / (psi * f).sum()
def P_G_decentree_x(psi, s):
    f = 1.0 + s * _gaussian_factor(CENTER + DX, CENTER, CENTER, 0.8)
    return (psi * f) / (psi * f).sum()
def P_G_decentree_diag(psi, s):
    f = 1.0 + s * _gaussian_factor(CENTER + 0.7*DX, CENTER + 0.7*DX, CENTER + 0.7*DX, 0.8)
    return (psi * f) / (psi * f).sum()
def P_A_anneau_interne(psi, s):
    f = 1.0 + s * _radial_mask(0.7, 1.3)
    return (psi * f) / (psi * f).sum()
def P_A_anneau_moyen(psi, s):
    f = 1.0 + s * _radial_mask(1.3, 1.9)
    return (psi * f) / (psi * f).sum()
def P_A_anneau_externe(psi, s):
    f = 1.0 + s * _radial_mask(1.9, 2.5)
    return (psi * f) / (psi * f).sum()
def P_A_shell_peripherique(psi, s):
    f = 1.0 + s * _radial_mask(2.5, 100.0)
    return (psi * f) / (psi * f).sum()

VARIANTS_REGISTRY = {
    "G_etroite_centree":     ("G", P_G_etroite_centree),
    "G_standard_centree":    ("G", P_G_standard_centree),
    "G_large_centree":       ("G", P_G_large_centree),
    "G_decentree_x":         ("G", P_G_decentree_x),
    "G_decentree_diag":      ("G", P_G_decentree_diag),
    "A_anneau_interne":      ("A", P_A_anneau_interne),
    "A_anneau_moyen":        ("A", P_A_anneau_moyen),
    "A_anneau_externe":      ("A", P_A_anneau_externe),
    "A_shell_peripherique":  ("A", P_A_shell_peripherique),
}

PAIRS_GA = [
    ("G_standard_centree", "A_anneau_moyen"),
    ("G_large_centree", "A_shell_peripherique"),
    ("G_decentree_x", "A_anneau_externe"),
    ("G_etroite_centree", "A_anneau_interne"),
]
PAIRS_GG = [
    ("G_standard_centree", "G_large_centree"),
    ("G_etroite_centree", "G_decentree_diag"),
]
PAIRS_AA = [
    ("A_anneau_moyen", "A_anneau_externe"),
    ("A_anneau_interne", "A_shell_peripherique"),
]
ALL_PAIRS = (
    [(p, "G/A") for p in PAIRS_GA]
    + [(p, "G/G") for p in PAIRS_GG]
    + [(p, "A/A") for p in PAIRS_AA]
)


def calibrate_strength(P_fn, psi_base, target_amp, s_bounds=(1e-6, 5.0)):
    def err(s):
        return float(np.linalg.norm(P_fn(psi_base, s) - psi_base)) - target_amp
    try:
        s_root = brentq(err, s_bounds[0], s_bounds[1], xtol=1e-8)
        amp_obt = float(np.linalg.norm(P_fn(psi_base, s_root) - psi_base))
        return s_root, amp_obt, "OK"
    except Exception as e:
        return None, None, f"FAILED({e})"


def run_sequence_trajectory(psi_t0, h_t0, P_first_fn, s_first, P_second_fn, s_second,
                              n_sep_steps, n_post_steps, D, beta, gamma, h0, dt):
    psi = P_first_fn(psi_t0.copy(), s_first)
    h = h_t0.copy()
    psis = [psi.copy()]
    hs = [h.copy()]
    for _ in range(n_sep_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        psis.append(psi.copy())
        hs.append(h.copy())
    psi = P_second_fn(psi, s_second)
    psis.append(psi.copy())
    hs.append(h.copy())
    for _ in range(n_post_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        psis.append(psi.copy())
        hs.append(h.copy())
    return np.array(psis), np.array(hs)


def K_metrics(psi_AB, h_AB, psi_BA, h_BA, psi_t0, h_t0, dt, Δt_sep):
    n = min(len(psi_AB), len(psi_BA))
    psi_AB = psi_AB[:n]; h_AB = h_AB[:n]
    psi_BA = psi_BA[:n]; h_BA = h_BA[:n]

    Δt_sep_steps = int(Δt_sep / dt)
    idx_pre_end = Δt_sep_steps
    idx_post_start = Δt_sep_steps + 1

    norm_diff_psi = np.array([np.linalg.norm(psi_AB[t] - psi_BA[t]) for t in range(n)])
    norm_diff_h = np.array([np.linalg.norm(h_AB[t] - h_BA[t]) for t in range(n)])
    norm_diff_ext = np.array([
        np.linalg.norm(np.concatenate([psi_AB[t].flatten() - psi_BA[t].flatten(),
                                         h_AB[t].flatten() - h_BA[t].flatten()]))
        for t in range(n)
    ])
    disp_AB_psi = np.array([np.linalg.norm(psi_AB[t] - psi_t0) for t in range(n)])
    disp_BA_psi = np.array([np.linalg.norm(psi_BA[t] - psi_t0) for t in range(n)])
    disp_AB_h = np.array([np.linalg.norm(h_AB[t] - h_t0) for t in range(n)])
    disp_BA_h = np.array([np.linalg.norm(h_BA[t] - h_t0) for t in range(n)])
    disp_AB_ext = np.array([
        np.linalg.norm(np.concatenate([psi_AB[t].flatten() - psi_t0.flatten(),
                                         h_AB[t].flatten() - h_t0.flatten()]))
        for t in range(n)
    ])
    disp_BA_ext = np.array([
        np.linalg.norm(np.concatenate([psi_BA[t].flatten() - psi_t0.flatten(),
                                         h_BA[t].flatten() - h_t0.flatten()]))
        for t in range(n)
    ])
    mean_disp_psi = 0.5 * (disp_AB_psi + disp_BA_psi)
    mean_disp_h = 0.5 * (disp_AB_h + disp_BA_h)
    mean_disp_ext = 0.5 * (disp_AB_ext + disp_BA_ext)

    def safe_div(a, b):
        return a / b if b > 1e-30 else 0.0

    # Post-composition (critère principal)
    auc_diff_post_psi = float(np.trapezoid(norm_diff_psi[idx_post_start:], dx=dt))
    auc_diff_post_h = float(np.trapezoid(norm_diff_h[idx_post_start:], dx=dt))
    auc_diff_post_ext = float(np.trapezoid(norm_diff_ext[idx_post_start:], dx=dt))
    auc_disp_post_psi = float(np.trapezoid(mean_disp_psi[idx_post_start:], dx=dt))
    auc_disp_post_h = float(np.trapezoid(mean_disp_h[idx_post_start:], dx=dt))
    auc_disp_post_ext = float(np.trapezoid(mean_disp_ext[idx_post_start:], dx=dt))

    # Total et pré (rapportés, hors verdict)
    auc_diff_ext_total = float(np.trapezoid(norm_diff_ext, dx=dt))
    auc_disp_ext_total = float(np.trapezoid(mean_disp_ext, dx=dt))
    if idx_pre_end > 0:
        auc_diff_pre_ext = float(np.trapezoid(norm_diff_ext[:idx_pre_end+1], dx=dt))
        auc_disp_pre_ext = float(np.trapezoid(mean_disp_ext[:idx_pre_end+1], dx=dt))
    else:
        auc_diff_pre_ext = 0.0
        auc_disp_pre_ext = 0.0

    # K instantané post
    K_inst = {}
    for dt_after in [25.0, 50.0, 100.0, 200.0]:
        idx_after = idx_post_start + int(dt_after / dt)
        if idx_after < n:
            K_inst[f"K_inst_psi_+{int(dt_after)}"] = safe_div(
                float(norm_diff_psi[idx_after]), float(mean_disp_psi[idx_after]))
            K_inst[f"K_inst_h_+{int(dt_after)}"] = safe_div(
                float(norm_diff_h[idx_after]), float(mean_disp_h[idx_after]))
            K_inst[f"K_inst_ext_+{int(dt_after)}"] = safe_div(
                float(norm_diff_ext[idx_after]), float(mean_disp_ext[idx_after]))
            K_inst[f"C_inst_raw_ext_+{int(dt_after)}"] = float(norm_diff_ext[idx_after])
        else:
            K_inst[f"K_inst_psi_+{int(dt_after)}"] = None
            K_inst[f"K_inst_h_+{int(dt_after)}"] = None
            K_inst[f"K_inst_ext_+{int(dt_after)}"] = None
            K_inst[f"C_inst_raw_ext_+{int(dt_after)}"] = None

    return {
        # Principal pour verdict
        "K_AUC_post_psi": safe_div(auc_diff_post_psi, auc_disp_post_psi),
        "K_AUC_post_h": safe_div(auc_diff_post_h, auc_disp_post_h),
        "K_AUC_post_ext": safe_div(auc_diff_post_ext, auc_disp_post_ext),
        "C_AUC_post_raw_ext": auc_diff_post_ext,
        # Instantané (principal)
        **K_inst,
        # Total et pré (rapportés)
        "K_AUC_total_ext": safe_div(auc_diff_ext_total, auc_disp_ext_total),
        "K_AUC_pre_ext": safe_div(auc_diff_pre_ext, auc_disp_pre_ext),
        # Final (information de réabsorption)
        "K_final_ext": safe_div(float(norm_diff_ext[-1]), float(mean_disp_ext[-1])),
        "C_final_raw_ext": float(norm_diff_ext[-1]),
    }


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    T_final = 800.0

    print(f"=== 6d-θ-1 : G/A vs intra-familles à Δt_sep ∈ {{25, 100}} ===\n")
    print(f"  dt = {dt:.5f}, T_final = {T_final}")

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    def P1prime_plateau(psi, strength=0.05):
        coords = np.arange(N_AXIS) * DX
        c = (N_AXIS - 1) * DX / 2.0
        factor = np.ones_like(psi)
        for i in range(N_AXIS):
            for j in range(N_AXIS):
                for k in range(N_AXIS):
                    r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                    if r <= 1.5: factor[i,j,k] += strength
        return (psi * factor) / (psi * factor).sum()
    amp_P1 = float(np.linalg.norm(P1prime_plateau(psi_base, 0.05) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    target_amp = 0.1 * amp_P6
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6, D, beta_lock, gamma_v, h0, dt, n_short)
    print(f"  ||P6|| = {amp_P6:.4e}, target_amp (10%) = {target_amp:.4e}\n")

    # Calibrer toutes les variantes utilisées
    print(f"--- Calibration des variantes ---")
    needed = set()
    for (a, b), _ in ALL_PAIRS:
        needed.add(a); needed.add(b)
    s_by_label = {}
    for label in sorted(needed):
        family, P_fn = VARIANTS_REGISTRY[label]
        s, amp, status = calibrate_strength(P_fn, psi_base, target_amp)
        s_by_label[label] = s
        print(f"  {label:<28} s={s:.6f}, amp={amp:.4e}, {status}")

    # Lancer pour Δt_sep ∈ {25, 100}, calculer pour chaque paire
    print(f"\n--- Test principal : 8 paires × 2 Δt_sep ---\n")
    results = {}
    Δt_sep_values = [25.0, 100.0]
    for Δt_sep in Δt_sep_values:
        n_sep = int(Δt_sep / dt)
        n_post = int((T_final - Δt_sep) / dt)
        print(f"\n  === Δt_sep = {Δt_sep} ===")
        print(f"  {'pair':<50} {'cat':<5} {'K_AUC_post':>12} "
              f"{'K_inst+50':>12} {'K_inst+100':>12} {'K_AUC_pre':>11} {'K_final':>11}")
        results[f"dt_sep_{Δt_sep}"] = {}
        for (A_label, B_label), category in ALL_PAIRS:
            A_fam, P_A = VARIANTS_REGISTRY[A_label]
            B_fam, P_B = VARIANTS_REGISTRY[B_label]
            s_A = s_by_label[A_label]
            s_B = s_by_label[B_label]
            # Séquence A puis B
            psi_AB, h_AB = run_sequence_trajectory(
                psi_tau0, h_tau0, P_A, s_A, P_B, s_B, n_sep, n_post,
                D, beta_lock, gamma_v, h0, dt)
            # Séquence B puis A
            psi_BA, h_BA = run_sequence_trajectory(
                psi_tau0, h_tau0, P_B, s_B, P_A, s_A, n_sep, n_post,
                D, beta_lock, gamma_v, h0, dt)
            m = K_metrics(psi_AB, h_AB, psi_BA, h_BA, psi_tau0, h_tau0, dt, Δt_sep)
            pair_label = f"{A_label}↔{B_label}"
            results[f"dt_sep_{Δt_sep}"][pair_label] = {
                "category": category,
                **m,
            }
            print(f"  {pair_label:<50} {category:<5} "
                  f"{m['K_AUC_post_ext']:>12.4e} "
                  f"{m.get('K_inst_ext_+50', 0):>12.4e} "
                  f"{m.get('K_inst_ext_+100', 0):>12.4e} "
                  f"{m['K_AUC_pre_ext']:>11.4e} "
                  f"{m['K_final_ext']:>11.4e}")

    # === Analyse comparée par catégorie ===
    print(f"\n--- Comparaison G/A vs G/G vs A/A ---\n")
    categories = ["G/A", "G/G", "A/A"]
    metrics_to_compare = ["K_AUC_post_ext", "K_inst_ext_+50", "K_inst_ext_+100",
                          "C_AUC_post_raw_ext"]
    summary = {}
    for Δt_sep in Δt_sep_values:
        key = f"dt_sep_{Δt_sep}"
        print(f"\n  === Δt_sep = {Δt_sep} ===")
        print(f"  {'metric':<25} {'<G/A>':>12} {'<G/G>':>12} {'<A/A>':>12} "
              f"{'ratio GA/intra':>15}")
        summary[key] = {}
        for metric in metrics_to_compare:
            means = {}
            for cat in categories:
                vals = [data[metric] for data in results[key].values()
                         if data["category"] == cat and data.get(metric) is not None]
                means[cat] = float(np.mean(vals)) if vals else None
            # Ratio G/A vs moyenne intra (G/G + A/A)
            intra_vals = []
            for cat in ["G/G", "A/A"]:
                if means[cat] is not None:
                    intra_vals.append(means[cat])
            intra_mean = float(np.mean(intra_vals)) if intra_vals else None
            if means["G/A"] is not None and intra_mean is not None and intra_mean > 1e-30:
                ratio = means["G/A"] / intra_mean
            else:
                ratio = None
            summary[key][metric] = {
                "mean_G_A": means["G/A"],
                "mean_G_G": means["G/G"],
                "mean_A_A": means["A/A"],
                "mean_intra": intra_mean,
                "ratio_GA_to_intra": ratio,
            }
            ratio_str = f"{ratio:>15.4f}" if ratio is not None else "n/a".rjust(15)
            ga_str = f"{means['G/A']:>12.4e}" if means["G/A"] is not None else "n/a".rjust(12)
            gg_str = f"{means['G/G']:>12.4e}" if means["G/G"] is not None else "n/a".rjust(12)
            aa_str = f"{means['A/A']:>12.4e}" if means["A/A"] is not None else "n/a".rjust(12)
            print(f"  {metric:<25} {ga_str} {gg_str} {aa_str} {ratio_str}")

    # === Verdict ===
    print(f"\n=== Verdict θ-1 ===\n")
    # Critère principal : ratio K_AUC_post(G/A) / K_AUC_post(intra) à Δt_sep = 100
    key_peak = f"dt_sep_100.0"
    key_ctrl = f"dt_sep_25.0"
    ratio_peak_post = summary[key_peak]["K_AUC_post_ext"]["ratio_GA_to_intra"]
    ratio_peak_inst100 = summary[key_peak]["K_inst_ext_+100"]["ratio_GA_to_intra"]
    ratio_ctrl_post = summary[key_ctrl]["K_AUC_post_ext"]["ratio_GA_to_intra"]
    ratio_ctrl_inst100 = summary[key_ctrl]["K_inst_ext_+100"]["ratio_GA_to_intra"]

    print(f"  À Δt_sep = 100 (peak) :")
    print(f"    ratio K_AUC_post G/A vs intra      = {ratio_peak_post}")
    print(f"    ratio K_inst+100 G/A vs intra      = {ratio_peak_inst100}")
    print(f"  À Δt_sep = 25 (control) :")
    print(f"    ratio K_AUC_post G/A vs intra      = {ratio_ctrl_post}")
    print(f"    ratio K_inst+100 G/A vs intra      = {ratio_ctrl_inst100}")

    # Vérification floor : si tous les K_AUC_post < 1e-10, FLOOR
    all_post_values = []
    for Δt_sep in Δt_sep_values:
        for data in results[f"dt_sep_{Δt_sep}"].values():
            if data.get("K_AUC_post_ext") is not None:
                all_post_values.append(data["K_AUC_post_ext"])
    max_post = max(all_post_values) if all_post_values else 0.0
    print(f"\n  Max K_AUC_post sur toutes paires/dt_sep : {max_post:.4e}")

    if max_post < 1e-10:
        verdict = "θ-FLOOR"
    elif ratio_peak_post is None or ratio_peak_inst100 is None:
        verdict = "θ-INDETERMINÉ"
    elif ratio_peak_post > 3.0 and ratio_peak_inst100 > 2.0:
        verdict = "θ-PASS fort"
    elif ratio_peak_post > 1.5 or ratio_peak_inst100 > 1.5:
        verdict = "θ-PASS faible"
    elif 0.5 < ratio_peak_post < 1.5 and 0.5 < ratio_peak_inst100 < 1.5:
        verdict = "θ-FAIL (non-linéarité générique du moteur)"
    else:
        verdict = "θ-INDETERMINÉ"
    print(f"\n  VERDICT : {verdict}")

    output = {
        "dt_simulation": float(dt),
        "T_final": T_final,
        "amp_P6": float(amp_P6),
        "target_amp": float(target_amp),
        "s_by_label": s_by_label,
        "Δt_sep_values": Δt_sep_values,
        "Δt_peak": 100.0,
        "Δt_control": 25.0,
        "all_pairs": [{"pair": list(p), "category": c} for p, c in ALL_PAIRS],
        "results": results,
        "summary": summary,
        "verdict": verdict,
        "verdict_inputs": {
            "ratio_peak_K_AUC_post_GA_vs_intra": ratio_peak_post,
            "ratio_peak_K_inst100_GA_vs_intra": ratio_peak_inst100,
            "ratio_control_K_AUC_post_GA_vs_intra": ratio_ctrl_post,
            "ratio_control_K_inst100_GA_vs_intra": ratio_ctrl_inst100,
            "max_K_AUC_post": max_post,
        },
    }
    with open("/home/claude/mcq_v4/6d_theta_1_intra_extra.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
