"""
6d-θ-0 — Balayage temporel Δt_sep sur une paire forte.

Objectif : repérer si le commutateur trajectoriel est monotone, nul,
ou en cloche en fonction de l'écart temporel entre les deux
perturbations composées.

Paire : G_standard_centree ↔ A_anneau_moyen
Δt_sep ∈ {0, 25, 50, 100, 200, 400}
T_final = 800 mesuré depuis t0

État de départ : (psi_tau0, h_tau0) = P6 relaxé pendant 10 unités
(même que η/η-bis).

Amplitude : 10% de ||P6||, calibrée pour chaque variante.

Pour chaque Δt_sep :
- séquence GA : G à t0, A à t0+Δt_sep, évolution jusqu'à T_final
- séquence AG : A à t0, G à t0+Δt_sep, évolution jusqu'à T_final

Mesures :
- K_AUC = ∫||X_GA - X_AG||dt / ∫mean(||X_GA - X_t0||, ||X_AG - X_t0||)dt
- K_final = ||X_GA(T) - X_AG(T)|| / mean(||X_GA(T)-X_t0||, ||X_AG(T)-X_t0||)
- Bruts (numérateur seul) gardés en parallèle
- Séparation ψ/h

Floor numérique : deux runs identiques GA vs GA pour mesurer le bruit.

Pas de Δ. Pas de 𝒢. Pas de RTS/RSR. Pas de MCQ.
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

def evolve_with_trajectory(psi, h, D, beta, gamma, h0, dt, n_steps):
    psis = np.zeros((n_steps + 1,) + psi.shape)
    hs = np.zeros((n_steps + 1,) + h.shape)
    psis[0] = psi
    hs[0] = h
    for n in range(n_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        psis[n + 1] = psi
        hs[n + 1] = h
    return psis, hs

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

def P_G_standard_centree(psi, strength):
    factor = 1.0 + strength * _gaussian_factor(CENTER, CENTER, CENTER, 0.8)
    return (psi * factor) / (psi * factor).sum()

def P_A_anneau_moyen(psi, strength):
    factor = 1.0 + strength * _radial_mask(1.3, 1.9)
    return (psi * factor) / (psi * factor).sum()


def calibrate_strength(P_fn, psi_base, target_amp, s_bounds=(1e-6, 5.0)):
    def err(s):
        return float(np.linalg.norm(P_fn(psi_base, s) - psi_base)) - target_amp
    s_min, s_max = s_bounds
    try:
        s_root = brentq(err, s_min, s_max, xtol=1e-8)
        amp_obt = float(np.linalg.norm(P_fn(psi_base, s_root) - psi_base))
        return s_root, amp_obt, "OK"
    except Exception as e:
        return None, None, f"FAILED ({e})"


def run_sequence(psi_t0, h_t0, P_first_fn, s_first, P_second_fn, s_second,
                  Δt_sep_steps, n_post_steps, D, beta, gamma, h0, dt):
    """Lance une séquence : (psi_t0, h_t0) -> first -> evolve Δt_sep -> second
    -> evolve n_post -> trajectoire complète depuis t0.
    Retourne (psi_traj, h_traj) où chaque traj est (n_total+1, 5, 5, 5).
    """
    # Appliquer la première perturbation à t0
    psi = P_first_fn(psi_t0.copy(), s_first)
    h = h_t0.copy()
    psis = [psi.copy()]
    hs = [h.copy()]
    # Évoluer pendant Δt_sep
    for _ in range(Δt_sep_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        psis.append(psi.copy())
        hs.append(h.copy())
    # Appliquer la seconde perturbation
    psi = P_second_fn(psi, s_second)
    psis.append(psi.copy())
    hs.append(h.copy())
    # Évoluer le reste
    for _ in range(n_post_steps):
        psi, h = step(psi, h, D, beta, gamma, h0, dt)
        psis.append(psi.copy())
        hs.append(h.copy())
    return np.array(psis), np.array(hs)


def K_metrics(psi_GA, h_GA, psi_AG, h_AG, psi_t0, h_t0, dt,
              Δt_sep, T_final):
    """Calcul K_AUC et K_final normalisés + bruts, sur ψ, h, et état étendu.

    Patch θ-0b : décomposition pré / post composition.
    - pré : intervalle [0, Δt_sep] (entre t0 et la seconde perturbation)
    - post : intervalle [Δt_sep, T_final] (après les deux perturbations)
    - K_inst_post à +25, +50, +100, +200 après la seconde perturbation
    """
    n = min(len(psi_GA), len(psi_AG))
    psi_GA = psi_GA[:n]; h_GA = h_GA[:n]
    psi_AG = psi_AG[:n]; h_AG = h_AG[:n]

    # Pour les indices : t_step k correspond au temps t = k*dt depuis t0
    # MAIS attention : la trajectoire contient deux frames consécutifs au
    # moment de la seconde perturbation (avant et après). C'est nécessaire
    # pour visualiser la perturbation.
    # Structure : index 0 = juste après G, puis Δt_sep_steps évolutions,
    # puis index Δt_sep_steps = juste avant A, puis index Δt_sep_steps+1
    # = juste après A, puis n_post_steps évolutions.
    # On a donc n_total = Δt_sep_steps + n_post_steps + 2 frames.

    # Pour la décomposition pré/post on identifie l'index de la transition.
    # Δt_sep_steps = int(Δt_sep / dt), avec la structure ci-dessus.
    Δt_sep_steps = int(Δt_sep / dt)
    # idx_pre_end = Δt_sep_steps : c'est l'index "juste avant la seconde perturbation"
    # idx_post_start = Δt_sep_steps + 1 : c'est l'index "juste après la seconde perturbation"
    idx_pre_end = Δt_sep_steps
    idx_post_start = Δt_sep_steps + 1

    # Normes diff par instant
    norm_diff_psi = np.array([np.linalg.norm(psi_GA[t] - psi_AG[t]) for t in range(n)])
    norm_diff_h = np.array([np.linalg.norm(h_GA[t] - h_AG[t]) for t in range(n)])
    norm_diff_ext = np.array([
        np.linalg.norm(np.concatenate([psi_GA[t].flatten() - psi_AG[t].flatten(),
                                        h_GA[t].flatten() - h_AG[t].flatten()]))
        for t in range(n)
    ])

    # Déplacements depuis t0
    disp_GA_psi = np.array([np.linalg.norm(psi_GA[t] - psi_t0) for t in range(n)])
    disp_AG_psi = np.array([np.linalg.norm(psi_AG[t] - psi_t0) for t in range(n)])
    disp_GA_h = np.array([np.linalg.norm(h_GA[t] - h_t0) for t in range(n)])
    disp_AG_h = np.array([np.linalg.norm(h_AG[t] - h_t0) for t in range(n)])
    disp_GA_ext = np.array([
        np.linalg.norm(np.concatenate([psi_GA[t].flatten() - psi_t0.flatten(),
                                        h_GA[t].flatten() - h_t0.flatten()]))
        for t in range(n)
    ])
    disp_AG_ext = np.array([
        np.linalg.norm(np.concatenate([psi_AG[t].flatten() - psi_t0.flatten(),
                                        h_AG[t].flatten() - h_t0.flatten()]))
        for t in range(n)
    ])

    mean_disp_psi = 0.5 * (disp_GA_psi + disp_AG_psi)
    mean_disp_h = 0.5 * (disp_GA_h + disp_AG_h)
    mean_disp_ext = 0.5 * (disp_GA_ext + disp_AG_ext)

    def safe_div(a, b):
        return a / b if b > 1e-30 else 0.0

    # AUC total
    auc_diff_psi = float(np.trapezoid(norm_diff_psi, dx=dt))
    auc_diff_h = float(np.trapezoid(norm_diff_h, dx=dt))
    auc_diff_ext = float(np.trapezoid(norm_diff_ext, dx=dt))
    auc_disp_psi = float(np.trapezoid(mean_disp_psi, dx=dt))
    auc_disp_h = float(np.trapezoid(mean_disp_h, dx=dt))
    auc_disp_ext = float(np.trapezoid(mean_disp_ext, dx=dt))

    # AUC pré-composition : [0, idx_pre_end]
    if idx_pre_end > 0:
        auc_diff_pre_psi = float(np.trapezoid(norm_diff_psi[:idx_pre_end+1], dx=dt))
        auc_diff_pre_h = float(np.trapezoid(norm_diff_h[:idx_pre_end+1], dx=dt))
        auc_diff_pre_ext = float(np.trapezoid(norm_diff_ext[:idx_pre_end+1], dx=dt))
        auc_disp_pre_psi = float(np.trapezoid(mean_disp_psi[:idx_pre_end+1], dx=dt))
        auc_disp_pre_h = float(np.trapezoid(mean_disp_h[:idx_pre_end+1], dx=dt))
        auc_disp_pre_ext = float(np.trapezoid(mean_disp_ext[:idx_pre_end+1], dx=dt))
    else:
        # Δt_sep = 0 : pas de fenêtre pré
        auc_diff_pre_psi = 0.0
        auc_diff_pre_h = 0.0
        auc_diff_pre_ext = 0.0
        auc_disp_pre_psi = 0.0
        auc_disp_pre_h = 0.0
        auc_disp_pre_ext = 0.0

    # AUC post-composition : [idx_post_start, n-1]
    auc_diff_post_psi = float(np.trapezoid(norm_diff_psi[idx_post_start:], dx=dt))
    auc_diff_post_h = float(np.trapezoid(norm_diff_h[idx_post_start:], dx=dt))
    auc_diff_post_ext = float(np.trapezoid(norm_diff_ext[idx_post_start:], dx=dt))
    auc_disp_post_psi = float(np.trapezoid(mean_disp_psi[idx_post_start:], dx=dt))
    auc_disp_post_h = float(np.trapezoid(mean_disp_h[idx_post_start:], dx=dt))
    auc_disp_post_ext = float(np.trapezoid(mean_disp_ext[idx_post_start:], dx=dt))

    # K instantané à différents temps après la seconde perturbation
    K_inst_post = {}
    for dt_after in [25.0, 50.0, 100.0, 200.0]:
        idx_after = idx_post_start + int(dt_after / dt)
        if idx_after < n:
            K_inst_post[f"K_inst_psi_+{int(dt_after)}"] = safe_div(
                float(norm_diff_psi[idx_after]),
                float(mean_disp_psi[idx_after]))
            K_inst_post[f"K_inst_h_+{int(dt_after)}"] = safe_div(
                float(norm_diff_h[idx_after]),
                float(mean_disp_h[idx_after]))
            K_inst_post[f"K_inst_ext_+{int(dt_after)}"] = safe_div(
                float(norm_diff_ext[idx_after]),
                float(mean_disp_ext[idx_after]))
            K_inst_post[f"C_inst_raw_ext_+{int(dt_after)}"] = float(norm_diff_ext[idx_after])
        else:
            K_inst_post[f"K_inst_psi_+{int(dt_after)}"] = None
            K_inst_post[f"K_inst_h_+{int(dt_after)}"] = None
            K_inst_post[f"K_inst_ext_+{int(dt_after)}"] = None
            K_inst_post[f"C_inst_raw_ext_+{int(dt_after)}"] = None

    return {
        # Total
        "K_AUC_psi": safe_div(auc_diff_psi, auc_disp_psi),
        "K_AUC_h": safe_div(auc_diff_h, auc_disp_h),
        "K_AUC_ext": safe_div(auc_diff_ext, auc_disp_ext),
        "C_AUC_raw_psi": auc_diff_psi,
        "C_AUC_raw_h": auc_diff_h,
        "C_AUC_raw_ext": auc_diff_ext,
        "AUC_disp_psi": auc_disp_psi,
        "AUC_disp_h": auc_disp_h,
        "AUC_disp_ext": auc_disp_ext,
        # Pré-composition
        "K_AUC_pre_psi": safe_div(auc_diff_pre_psi, auc_disp_pre_psi),
        "K_AUC_pre_h": safe_div(auc_diff_pre_h, auc_disp_pre_h),
        "K_AUC_pre_ext": safe_div(auc_diff_pre_ext, auc_disp_pre_ext),
        "C_AUC_pre_raw_ext": auc_diff_pre_ext,
        "AUC_disp_pre_ext": auc_disp_pre_ext,
        # Post-composition
        "K_AUC_post_psi": safe_div(auc_diff_post_psi, auc_disp_post_psi),
        "K_AUC_post_h": safe_div(auc_diff_post_h, auc_disp_post_h),
        "K_AUC_post_ext": safe_div(auc_diff_post_ext, auc_disp_post_ext),
        "C_AUC_post_raw_ext": auc_diff_post_ext,
        "AUC_disp_post_ext": auc_disp_post_ext,
        # K instantané post
        **K_inst_post,
        # Final
        "K_final_psi": safe_div(float(norm_diff_psi[-1]), float(mean_disp_psi[-1])),
        "K_final_h": safe_div(float(norm_diff_h[-1]), float(mean_disp_h[-1])),
        "K_final_ext": safe_div(float(norm_diff_ext[-1]), float(mean_disp_ext[-1])),
        "C_final_raw_ext": float(norm_diff_ext[-1]),
        "disp_GA_final_ext": float(disp_GA_ext[-1]),
        "disp_AG_final_ext": float(disp_AG_ext[-1]),
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

    print(f"=== 6d-θ-0b : décomposition pré / post composition ===\n")
    print(f"  dt = {dt:.5f}")
    print(f"  T_final = {T_final}, depuis t0 (état P6 relaxé pendant 10 unités)\n")

    # Préparer psi_base puis psi_tau0/h_tau0 (état P6 relaxé)
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
    amp_P1prime_std = float(np.linalg.norm(P1prime_plateau(psi_base, 0.05) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    target_amp = 0.1 * amp_P6
    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6,
                                D, beta_lock, gamma_v, h0, dt, n_short)
    print(f"  ||P6|| = {amp_P6:.4e}, target_amp (10%) = {target_amp:.4e}")

    # Calibrer G_std et A_moyen
    s_G, amp_G, st_G = calibrate_strength(P_G_standard_centree, psi_base, target_amp)
    s_A, amp_A, st_A = calibrate_strength(P_A_anneau_moyen, psi_base, target_amp)
    print(f"  G_standard_centree : s={s_G:.6f}, amp_obt={amp_G:.4e}, {st_G}")
    print(f"  A_anneau_moyen     : s={s_A:.6f}, amp_obt={amp_A:.4e}, {st_A}\n")

    # === Phase floor numérique ===
    print(f"--- Floor numérique : GA vs GA (deux runs identiques) ---")
    # Avec Δt_sep = 100 (valeur intermédiaire)
    Δt_floor = 100.0
    n_sep_floor = int(Δt_floor / dt)
    n_post_floor = int((T_final - Δt_floor) / dt)
    psi_GA1, h_GA1 = run_sequence(psi_tau0, h_tau0,
                                    P_G_standard_centree, s_G,
                                    P_A_anneau_moyen, s_A,
                                    n_sep_floor, n_post_floor,
                                    D, beta_lock, gamma_v, h0, dt)
    psi_GA2, h_GA2 = run_sequence(psi_tau0, h_tau0,
                                    P_G_standard_centree, s_G,
                                    P_A_anneau_moyen, s_A,
                                    n_sep_floor, n_post_floor,
                                    D, beta_lock, gamma_v, h0, dt)
    floor_diff = float(np.linalg.norm(psi_GA1[-1] - psi_GA2[-1]))
    floor_diff_h = float(np.linalg.norm(h_GA1[-1] - h_GA2[-1]))
    floor_ext_final = float(np.linalg.norm(
        np.concatenate([psi_GA1[-1].flatten() - psi_GA2[-1].flatten(),
                        h_GA1[-1].flatten() - h_GA2[-1].flatten()])))
    print(f"  ||ψ_GA1(T) - ψ_GA2(T)|| = {floor_diff:.4e}")
    print(f"  ||h_GA1(T) - h_GA2(T)|| = {floor_diff_h:.4e}")
    print(f"  ||X_GA1(T) - X_GA2(T)|| = {floor_ext_final:.4e}")
    print(f"  (devrait être au niveau du bruit numérique, autour de 1e-15)\n")

    # === Balayage Δt_sep ===
    Δt_sep_list = [0.0, 25.0, 50.0, 100.0, 200.0, 400.0]
    print(f"--- Balayage Δt_sep ---\n")
    print(f"  {'Δt_sep':>8} {'K_AUC_pre':>11} {'K_AUC_post':>12} "
          f"{'K_AUC_total':>11} {'K_inst+100':>11} {'K_final':>11}")
    results = {}
    for Δt_sep in Δt_sep_list:
        n_sep = int(Δt_sep / dt)
        n_post = int((T_final - Δt_sep) / dt)
        # GA
        psi_GA, h_GA = run_sequence(psi_tau0, h_tau0,
                                      P_G_standard_centree, s_G,
                                      P_A_anneau_moyen, s_A,
                                      n_sep, n_post,
                                      D, beta_lock, gamma_v, h0, dt)
        # AG
        psi_AG, h_AG = run_sequence(psi_tau0, h_tau0,
                                      P_A_anneau_moyen, s_A,
                                      P_G_standard_centree, s_G,
                                      n_sep, n_post,
                                      D, beta_lock, gamma_v, h0, dt)
        # Métriques
        m = K_metrics(psi_GA, h_GA, psi_AG, h_AG, psi_tau0, h_tau0, dt,
                       Δt_sep, T_final)
        results[f"dt_sep_{Δt_sep}"] = m
        # Affichage : focus sur la décomposition pré / post / total / final
        k_inst_100 = m.get("K_inst_ext_+100", None)
        k_inst_100_str = f"{k_inst_100:.4e}" if k_inst_100 is not None else "n/a"
        print(f"  {Δt_sep:>8.1f} "
              f"{m['K_AUC_pre_ext']:>11.4e} {m['K_AUC_post_ext']:>12.4e} "
              f"{m['K_AUC_ext']:>11.4e} {k_inst_100_str:>11} "
              f"{m['K_final_ext']:>11.4e}")

    # === Lecture du profil ===
    print(f"\n--- Lecture du profil ---\n")
    K_total_by_dt = {dt_s: results[f"dt_sep_{dt_s}"]["K_AUC_ext"] for dt_s in Δt_sep_list}
    K_post_by_dt = {dt_s: results[f"dt_sep_{dt_s}"]["K_AUC_post_ext"] for dt_s in Δt_sep_list}
    K_pre_by_dt = {dt_s: results[f"dt_sep_{dt_s}"]["K_AUC_pre_ext"] for dt_s in Δt_sep_list}
    K_inst100_by_dt = {dt_s: results[f"dt_sep_{dt_s}"].get("K_inst_ext_+100", 0)
                        for dt_s in Δt_sep_list}

    print(f"  Profil K_AUC total par Δt_sep :")
    for dt_s in Δt_sep_list:
        print(f"    {dt_s:>6.1f}: total={K_total_by_dt[dt_s]:.4e}  "
              f"pre={K_pre_by_dt[dt_s]:.4e}  post={K_post_by_dt[dt_s]:.4e}  "
              f"inst+100={K_inst100_by_dt[dt_s] if K_inst100_by_dt[dt_s] else 0:.4e}")

    # Pic basé sur K_AUC_post (et non K_AUC_total)
    Δt_peak_post = max(K_post_by_dt, key=lambda k: K_post_by_dt[k])
    K_peak_post = K_post_by_dt[Δt_peak_post]
    Δt_peak_total = max(K_total_by_dt, key=lambda k: K_total_by_dt[k])
    Δt_peak_inst = max(K_inst100_by_dt,
                        key=lambda k: K_inst100_by_dt[k] if K_inst100_by_dt[k] else 0)
    print(f"\n  Δt_peak (K_AUC_total) : {Δt_peak_total}")
    print(f"  Δt_peak (K_AUC_post)  : {Δt_peak_post}, valeur = {K_peak_post:.4e}")
    print(f"  Δt_peak (K_inst+100)  : {Δt_peak_inst}")

    # Lecture qualitative
    # Croissance K_AUC_post avec Δt_sep ?
    K_post_values = [K_post_by_dt[dt_s] for dt_s in Δt_sep_list]
    is_post_monotone = all(K_post_values[i] <= K_post_values[i+1] * 1.01
                            for i in range(len(K_post_values)-1))
    is_post_saturating = (K_post_values[-1] < K_post_values[-2] * 1.1
                           and K_post_values[-2] > K_post_values[-3]) if len(K_post_values) >= 3 else False
    print(f"\n  K_AUC_post est-il monotone croissant ? {is_post_monotone}")
    print(f"  Saturation/plateau possible ? {is_post_saturating}")

    Δt_control = 25.0 if Δt_peak_post > 100.0 else 400.0
    print(f"\n  Δt_control proposé : {Δt_control}")

    output = {
        "dt_simulation": float(dt),
        "T_final": T_final,
        "amp_P6": float(amp_P6),
        "target_amp": float(target_amp),
        "s_G_standard_centree": s_G,
        "s_A_anneau_moyen": s_A,
        "floor_numerique": {
            "diff_psi_final": floor_diff,
            "diff_h_final": floor_diff_h,
            "diff_ext_final": floor_ext_final,
        },
        "Δt_sep_list": Δt_sep_list,
        "results_by_dt_sep": results,
        "Δt_peak_K_AUC_total": Δt_peak_total,
        "Δt_peak_K_AUC_post": Δt_peak_post,
        "Δt_peak_K_inst100": Δt_peak_inst,
        "K_peak_post": K_peak_post,
        "Δt_control_proposed": Δt_control,
        "K_post_monotone": is_post_monotone,
        "K_post_saturating": is_post_saturating,
    }
    with open("/home/claude/mcq_v4/6d_theta_0b_decompose.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
