"""
6d-ε v2 — Contrôle L3 du corridor intermédiaire, patché.

Corrections par rapport à v1 :
1. Diagnostic explicite de τ_c : afficher C(lag), repérer ou non un
   crossing 1/e, donner τ_c par variante, et indiquer la valeur
   effectivement utilisée.
2. Quantile bilatéral : p_tail = min(p, 1-p), avec z-score signé pour
   distinguer queue basse / queue haute / centre.
3. Pas de contrainte de masse nulle sur h (la conservation de masse
   vaut pour ψ, pas pour h). Les contrôles C2/C3 sur h n'imposent
   plus de somme nulle, seulement la continuité temporelle pour C3.

Le reste est identique : N=200 tirages, 6 observables, comparaison
cos brut + cos centré.
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

def P_prime_gauss(psi, strength, sigma_p):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r2 = ((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                factor[i,j,k] += strength * np.exp(-0.5 * r2 / sigma_p**2)
    p = psi * factor
    return p / p.sum()

def P_prime_annular(psi, strength, r_inner=1.5, r_outer=2.5):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r_inner <= r <= r_outer:
                    factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()

def P1prime_std(psi, strength=0.05):
    coords = np.arange(N_AXIS) * DX
    c = (N_AXIS - 1) * DX / 2.0
    factor = np.ones_like(psi)
    for i in range(N_AXIS):
        for j in range(N_AXIS):
            for k in range(N_AXIS):
                r = np.sqrt((coords[i]-c)**2 + (coords[j]-c)**2 + (coords[k]-c)**2)
                if r <= 1.5: factor[i,j,k] += strength
    p = psi * factor
    return p / p.sum()


def compute_delta_full(psi_start, h_start, P_prime_fn, s_prime,
                       D, beta, gamma_v, h0, dt, n_dt):
    psis_ref, hs_ref = evolve_with_trajectory(
        psi_start.copy(), h_start.copy(),
        D, beta, gamma_v, h0, dt, n_dt)
    psi_pp = P_prime_fn(psi_start.copy(), s_prime)
    h_pp = h_start.copy()
    psis_with, hs_with = evolve_with_trajectory(
        psi_pp, h_pp, D, beta, gamma_v, h0, dt, n_dt)
    return psis_with - psis_ref, hs_with - hs_ref


def cos_pair(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-30 or n2 < 1e-30:
        return 0.0
    return float(np.dot(v1.flatten(), v2.flatten()) / (n1 * n2))


def cos_centered(v1, v2):
    v1c = v1.flatten() - v1.flatten().mean()
    v2c = v2.flatten() - v2.flatten().mean()
    n1 = np.linalg.norm(v1c)
    n2 = np.linalg.norm(v2c)
    if n1 < 1e-30 or n2 < 1e-30:
        return 0.0
    return float(np.dot(v1c, v2c) / (n1 * n2))


def compute_observables(r_psi, r_h):
    return {
        "psi_spatial_fluct": np.std(r_psi, axis=0),
        "h_spatial_fluct": np.std(r_h, axis=0),
        "psi_temp_mean_abs": np.mean(np.abs(r_psi), axis=(1,2,3)),
        "psi_temp_norm": np.sqrt(np.sum(r_psi**2, axis=(1,2,3))),
        "h_temp_mean_abs": np.mean(np.abs(r_h), axis=(1,2,3)),
        "h_temp_norm": np.sqrt(np.sum(r_h**2, axis=(1,2,3))),
    }


def measure_C_and_tau_c(r_psi, dt, max_lag_time=400.0):
    """Autocorrélation vectorielle ψ.
    Retourne (tau_c, C, lag_times) ; tau_c=None si pas de crossing 1/e."""
    n_t = r_psi.shape[0]
    r_flat = r_psi.reshape(n_t, -1)
    norms_sq = np.sum(r_flat**2, axis=1)
    if norms_sq.sum() < 1e-30:
        return None, None, None
    max_lag = min(n_t - 1, int(max_lag_time / dt))
    C = []
    for lag in range(max_lag):
        if lag == 0:
            C.append(1.0)
        else:
            prod = np.sum(r_flat[:n_t-lag] * r_flat[lag:n_t], axis=1)
            norm_window = np.sum(norms_sq[:n_t-lag])
            if norm_window > 1e-30:
                C.append(prod.sum() / norm_window)
            else:
                C.append(0.0)
    C = np.array(C)
    threshold = 1.0 / np.e
    crossing = np.where(C <= threshold)[0]
    lag_times = np.arange(max_lag) * dt
    if len(crossing) == 0:
        return None, C, lag_times
    tau_c_steps = crossing[0]
    tau_c = tau_c_steps * dt
    return tau_c, C, lag_times


def generate_psi_C1(rng, n_t, shape_spatial, amp):
    return rng.normal(0.0, amp, size=(n_t,) + shape_spatial)

def generate_psi_C2(rng, n_t, shape_spatial, amp):
    """C1 + masse nulle à chaque t (conservation pour ψ)."""
    r = generate_psi_C1(rng, n_t, shape_spatial, amp)
    return r - r.mean(axis=(1,2,3), keepdims=True)

def generate_psi_C3(rng, n_t, shape_spatial, amp, tau_c_steps):
    """C2 + continuité temporelle."""
    r = generate_psi_C2(rng, n_t, shape_spatial, amp)
    if tau_c_steps <= 0:
        return r
    alpha = np.exp(-1.0 / tau_c_steps)
    y = np.zeros_like(r)
    y[0] = r[0]
    for t in range(1, n_t):
        y[t] = alpha * y[t-1] + (1 - alpha) * r[t]
    target_amp = np.linalg.norm(r)
    cur_amp = np.linalg.norm(y)
    if cur_amp > 1e-30:
        y *= target_amp / cur_amp
    # Réimposer conservation
    return y - y.mean(axis=(1,2,3), keepdims=True)


def generate_h_C1(rng, n_t, shape_spatial, amp):
    """h sans contrainte de masse (h n'est pas conservé)."""
    return rng.normal(0.0, amp, size=(n_t,) + shape_spatial)

def generate_h_C2(rng, n_t, shape_spatial, amp):
    """Pour h, C2 = pas de contrainte (h n'est pas soumis à conservation).
    On garde le nom C2 pour cohérence mais c'est identique à C1."""
    return generate_h_C1(rng, n_t, shape_spatial, amp)

def generate_h_C3(rng, n_t, shape_spatial, amp, tau_c_steps):
    """h C3 = C1 + continuité temporelle, SANS conservation."""
    r = generate_h_C1(rng, n_t, shape_spatial, amp)
    if tau_c_steps <= 0:
        return r
    alpha = np.exp(-1.0 / tau_c_steps)
    y = np.zeros_like(r)
    y[0] = r[0]
    for t in range(1, n_t):
        y[t] = alpha * y[t-1] + (1 - alpha) * r[t]
    target_amp = np.linalg.norm(r)
    cur_amp = np.linalg.norm(y)
    if cur_amp > 1e-30:
        y *= target_amp / cur_amp
    return y  # PAS de soustraction de moyenne pour h


def all_pairwise_cos(values_list, centered=False):
    cos_fn = cos_centered if centered else cos_pair
    cos_list = []
    for i in range(len(values_list)):
        for j in range(i+1, len(values_list)):
            cos_list.append(cos_fn(values_list[i], values_list[j]))
    return cos_list


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_dt_long = int(800.0 / dt)

    print(f"=== 6d-ε v2 : contrôle L3 patché ===\n")
    print(f"  dt = {dt:.5f}")

    psi_base, h_base = evolve(psi0.copy(), h0f.copy(),
                              D, beta_lock, gamma_v, h0, dt, n_stab)

    amp_P1prime_std = float(np.linalg.norm(P1prime_std(psi_base) - psi_base))
    s_P6 = brentq(lambda s: float(np.linalg.norm(P6_face_dipole(psi_base, s) - psi_base))
                  - amp_P1prime_std, 1e-4, 0.99, xtol=1e-6)
    amp_P6 = float(np.linalg.norm(P6_face_dipole(psi_base, s_P6) - psi_base))
    target_amp = 0.1 * amp_P6

    variants = []
    for label, sigma_p in [("étroite", 0.5), ("standard", 0.8), ("large", 1.5)]:
        s_calib = brentq(lambda s: float(np.linalg.norm(P_prime_gauss(psi_base, s, sigma_p) - psi_base))
                          - target_amp, 1e-6, 1.0, xtol=1e-8)
        variants.append({
            "label": label, "s_calib": s_calib,
            "P_fn": (lambda psi, s, sp=sigma_p: P_prime_gauss(psi, s, sp)),
        })
    s_ann = brentq(lambda s: float(np.linalg.norm(P_prime_annular(psi_base, s) - psi_base))
                    - target_amp, 1e-6, 5.0, xtol=1e-8)
    variants.append({
        "label": "annulaire", "s_calib": s_ann,
        "P_fn": (lambda psi, s: P_prime_annular(psi, s)),
    })

    psi_P6 = P6_face_dipole(psi_base.copy(), s_P6)
    h_P6 = h_base.copy()
    psi_tau0, h_tau0 = evolve(psi_P6, h_P6,
                                D, beta_lock, gamma_v, h0, dt, n_short)
    n_3000 = int(3000.0 / dt)
    psi_tau3000, h_tau3000 = evolve(psi_tau0.copy(), h_tau0.copy(),
                                     D, beta_lock, gamma_v, h0, dt, n_3000)

    print(f"  Calcul des résidus réels...")
    residuals = {}
    for v in variants:
        d_psi_tau0, d_h_tau0 = compute_delta_full(
            psi_tau0, h_tau0, v["P_fn"], v["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_psi_ref, d_h_ref = compute_delta_full(
            psi_tau3000, h_tau3000, v["P_fn"], v["s_calib"],
            D, beta_lock, gamma_v, h0, dt, n_dt_long)
        d_flat = np.concatenate([d_psi_tau0.flatten(), d_h_tau0.flatten()])
        ref_flat = np.concatenate([d_psi_ref.flatten(), d_h_ref.flatten()])
        ref_norm_sq = float(np.dot(ref_flat, ref_flat))
        a_tau = float(np.dot(d_flat, ref_flat) / ref_norm_sq) if ref_norm_sq > 1e-30 else 0.0
        r_psi = d_psi_tau0 - a_tau * d_psi_ref
        r_h = d_h_tau0 - a_tau * d_h_ref
        residuals[v["label"]] = {"r_psi": r_psi, "r_h": r_h}

    # === Diagnostic τ_c ===
    print(f"\n=== Diagnostic τ_c (autocorrélation vectorielle ψ) ===\n")
    tau_c_by_variant = {}
    C_curves = {}
    for v in variants:
        r_psi_v = residuals[v["label"]]["r_psi"]
        tau_v, C, lag_times = measure_C_and_tau_c(r_psi_v, dt, max_lag_time=400.0)
        tau_c_by_variant[v["label"]] = tau_v
        C_curves[v["label"]] = C
        if tau_v is not None:
            print(f"  {v['label']:<12} τ_c = {tau_v:.3f}")
        else:
            print(f"  {v['label']:<12} PAS DE CROSSING 1/e")
            # Afficher C(lag) à quelques points pour comprendre
            print(f"    C(lag) :")
            sample_lags = [0, 1, 10, 50, 100, len(C)//2, len(C)-1]
            for sl in sample_lags:
                if sl < len(C):
                    print(f"      lag={sl} (t={sl*dt:.2f}) : C={C[sl]:+.4f}")
            # Min de C
            print(f"    min(C) = {C.min():+.4f} atteint à lag {C.argmin()} (t={C.argmin()*dt:.2f})")

    # Choisir τ_c : on prend la médiane des variantes qui ont un crossing
    valid_taus = [t for t in tau_c_by_variant.values() if t is not None]
    if valid_taus:
        tau_c_use = float(np.median(valid_taus))
        print(f"\n  τ_c utilisé (médiane des crossings détectés) : {tau_c_use:.3f}")
    else:
        # Pas de crossing : prendre le premier minimum de C comme proxy
        min_idx = C_curves["standard"].argmin()
        tau_c_use = min_idx * dt
        print(f"\n  AUCUN CROSSING détecté. Proxy = argmin(C) standard = {tau_c_use:.3f}")
    tau_c_steps_use = max(1, int(tau_c_use / dt))
    print(f"  En pas de temps : {tau_c_steps_use}")

    # Observables réelles + cos
    obs_names = ["psi_spatial_fluct", "h_spatial_fluct",
                 "psi_temp_mean_abs", "psi_temp_norm",
                 "h_temp_mean_abs", "h_temp_norm"]
    labels = ["étroite", "standard", "large", "annulaire"]
    real_observables = {}
    for label, r_dict in residuals.items():
        real_observables[label] = compute_observables(r_dict["r_psi"], r_dict["r_h"])

    real_min_cos_raw = {}
    real_min_cos_cent = {}
    print(f"\n=== Cos réels (rappel) ===")
    for name in obs_names:
        vals = [real_observables[lab][name] for lab in labels]
        coss = all_pairwise_cos(vals, centered=False)
        coss_c = all_pairwise_cos(vals, centered=True)
        real_min_cos_raw[name] = min(coss)
        real_min_cos_cent[name] = min(coss_c)
        print(f"  {name:<22} min brut = {min(coss):+.4f}, min centré = {min(coss_c):+.4f}")

    # === Génération contrôles ===
    print(f"\n=== Contrôles avec h libre (pas de conservation) ===\n")
    N_TIRAGES = 200
    rng = np.random.default_rng(seed=42)
    n_t = residuals["standard"]["r_psi"].shape[0]
    shape_spatial = residuals["standard"]["r_psi"].shape[1:]
    amp_psi = float(np.linalg.norm(residuals["standard"]["r_psi"])) / np.sqrt(n_t * 125)
    amp_h = float(np.linalg.norm(residuals["standard"]["r_h"])) / np.sqrt(n_t * 125)

    tau_c_values = [0.5 * tau_c_steps_use, 1.0 * tau_c_steps_use, 2.0 * tau_c_steps_use]
    control_levels = [
        ("C1", "C1", None),
        ("C2", "C2", None),  # ψ avec masse nulle, h sans contrainte
        ("C3_05tc", "C3", tau_c_values[0]),
        ("C3_1tc", "C3", tau_c_values[1]),
        ("C3_2tc", "C3", tau_c_values[2]),
    ]

    def generate_pair(level, tau_steps):
        if level == "C1":
            return (generate_psi_C1(rng, n_t, shape_spatial, amp_psi),
                    generate_h_C1(rng, n_t, shape_spatial, amp_h))
        elif level == "C2":
            # ψ : conservation, h : libre (= C1 pour h)
            return (generate_psi_C2(rng, n_t, shape_spatial, amp_psi),
                    generate_h_C2(rng, n_t, shape_spatial, amp_h))
        elif level == "C3":
            return (generate_psi_C3(rng, n_t, shape_spatial, amp_psi, int(tau_steps)),
                    generate_h_C3(rng, n_t, shape_spatial, amp_h, int(tau_steps)))

    print(f"  N_TIRAGES = {N_TIRAGES}, τ_c utilisé = {tau_c_use:.3f} (steps={tau_c_steps_use})")
    print(f"  Patch v3 : on stocke le MIN cos par tirage (pas les 6 cos individuels)")
    control_results = {}
    for level_name, level, tau_steps in control_levels:
        print(f"    {level_name}...")
        # Stocker une liste de min cos par tirage : 200 valeurs
        min_cos_raw = {n: [] for n in obs_names}
        min_cos_cent = {n: [] for n in obs_names}
        for trial in range(N_TIRAGES):
            realisations = [generate_pair(level, tau_steps) for _ in range(4)]
            obs_list = [compute_observables(rp, rh) for rp, rh in realisations]
            for name in obs_names:
                vals = [obs_list[i][name] for i in range(4)]
                coss_raw = all_pairwise_cos(vals, centered=False)
                coss_cent = all_pairwise_cos(vals, centered=True)
                # Patch v3 : on prend le MIN par tirage
                min_cos_raw[name].append(min(coss_raw))
                min_cos_cent[name].append(min(coss_cent))
        control_results[level_name] = {"raw": min_cos_raw, "cent": min_cos_cent}

    # === Synthèse avec quantile bilatéral et z-score (sur distribution des min par tirage) ===
    print(f"\n=== Synthèse : min cos réel vs distribution des min par tirage ===\n")
    print(f"  Pour chaque observable et niveau, on rapporte (N=200 tirages) :")
    print(f"  - moyenne et écart-type de la distribution des min cos hors diagonale par tirage")
    print(f"  - z-score signé = (min_cos_réel - mean) / std")
    print(f"  - p_tail = min(P(synth ≤ réel), P(synth ≥ réel))")
    print(f"  - verdict : 'central' si p_tail > 0.05, sinon queue\n")

    output = {
        "dt": float(dt),
        "tau_c_by_variant": {k: (float(v) if v else None) for k, v in tau_c_by_variant.items()},
        "tau_c_used": float(tau_c_use),
        "tau_c_steps_used": tau_c_steps_use,
        "N_TIRAGES": N_TIRAGES,
        "real_min_cos_raw": real_min_cos_raw,
        "real_min_cos_centered": real_min_cos_cent,
        "by_observable": {},
    }

    for name in obs_names:
        print(f"\n  --- {name} ---")
        print(f"    cos min réel : brut = {real_min_cos_raw[name]:+.4f}, "
              f"centré = {real_min_cos_cent[name]:+.4f}")
        obs_data = {"by_level": {}}
        print(f"    {'niveau':<10} {'mode':<6} {'mean':>10} {'std':>10} "
              f"{'z':>8} {'p_tail':>8} {'verdict':>14}")
        for level_name, _, _ in control_levels:
            for mode in ["raw", "cent"]:
                arr = np.array(control_results[level_name][mode][name])
                m = float(arr.mean())
                s = float(arr.std())
                real = real_min_cos_raw[name] if mode == "raw" else real_min_cos_cent[name]
                z = (real - m) / max(s, 1e-30)
                p_below = float((arr <= real).mean())
                p_above = float((arr >= real).mean())
                p_tail = min(p_below, p_above)
                if p_tail > 0.05:
                    verdict = "central"
                elif z > 0:
                    verdict = "queue haute"
                else:
                    verdict = "queue basse"
                print(f"    {level_name:<10} {mode:<6} {m:>+10.4f} {s:>10.4f} "
                      f"{z:>+8.2f} {p_tail:>8.4f} {verdict:>14}")
                obs_data["by_level"][f"{level_name}_{mode}"] = {
                    "mean": m, "std": s, "z": z, "p_tail": p_tail,
                    "verdict": verdict,
                }
        output["by_observable"][name] = obs_data

    # Lecture d'ensemble
    print(f"\n=== Lecture d'ensemble ===\n")
    print(f"  Pour chaque niveau et chaque mode, combien d'observables ont")
    print(f"  un cos réel CENTRAL (compatible avec la distribution synthétique) ?\n")
    print(f"  {'niveau':<12} {'central brut':>16} {'central centré':>18}")
    summary = {}
    for level_name, _, _ in control_levels:
        c_raw = 0
        c_cent = 0
        for name in obs_names:
            if output["by_observable"][name]["by_level"][f"{level_name}_raw"]["verdict"] == "central":
                c_raw += 1
            if output["by_observable"][name]["by_level"][f"{level_name}_cent"]["verdict"] == "central":
                c_cent += 1
        print(f"  {level_name:<12} {c_raw}/{len(obs_names):>14} {c_cent}/{len(obs_names):>16}")
        summary[level_name] = {"central_raw": c_raw, "central_cent": c_cent}
    output["summary"] = summary

    with open("/home/claude/mcq_v4/6d_epsilon_v3_min_per_trial.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
