"""
6d-ζ — Test L1 : décomposition partagée / spécifique sur observables
temporelles, avec contrôle leave-one-out.

Question : existe-t-il une décomposition robuste entre composante
partagée entre variantes et composante spécifique à P′, capable
d'expliquer le corridor intermédiaire observé sur les observables
temporelles ?

Protocole (Alex) :

Pour chaque observable temporelle O ∈ {ψ_temp_mean_abs, ψ_temp_norm,
h_temp_mean_abs, h_temp_norm} :

1. O_v par variante v ∈ {étroite, standard, large, annulaire}
2. Composante partagée Ō = mean_v O_v
3. Projection : a_v = <O_v, Ō> / ||Ō||²
   O_shared_v = a_v · Ō
   O_specific_v = O_v − O_shared_v
4. f_shared_v = ||O_shared_v||² / ||O_v||²
   f_specific_v = ||O_specific_v||² / ||O_v||²
5. cos(O_v, Ō)
6. Leave-one-out : Ō_{−v} = mean des 3 autres variantes
   a_v^LOO, O_shared_v^LOO, O_specific_v^LOO
7. Mesures de stabilité :
   - cos(Ō, Ō_{−v})
   - cos(O_specific_v, O_specific_v^LOO)
   - écart f_shared_v vs f_shared_v^LOO
8. Aussi en version centrée (O_v − mean(O_v)) comme diagnostic.

Pas de SVD. Pas de spatial. Pas de Δ. Pas de 𝒢.
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


def decompose_and_LOO(observables_by_variant, labels):
    """Pour un observable donné (dict label -> série temporelle),
    fait la décomposition partagée/spécifique avec leave-one-out.
    Retourne un dict de résultats.
    """
    # Empiler dans l'ordre des labels
    O = {lab: np.asarray(observables_by_variant[lab]).flatten() for lab in labels}
    # Composante partagée globale
    O_bar = np.mean([O[lab] for lab in labels], axis=0)
    O_bar_norm_sq = float(np.dot(O_bar, O_bar))

    results = {"O_bar_norm": float(np.linalg.norm(O_bar)), "per_variant": {}}

    for lab in labels:
        O_v = O[lab]
        norm_O_v_sq = float(np.dot(O_v, O_v))

        # Projection sur Ō
        if O_bar_norm_sq > 1e-30:
            a_v = float(np.dot(O_v, O_bar) / O_bar_norm_sq)
        else:
            a_v = 0.0
        O_shared_v = a_v * O_bar
        O_specific_v = O_v - O_shared_v
        norm_shared_sq = float(np.dot(O_shared_v, O_shared_v))
        norm_specific_sq = float(np.dot(O_specific_v, O_specific_v))
        f_shared = norm_shared_sq / max(norm_O_v_sq, 1e-30)
        f_specific = norm_specific_sq / max(norm_O_v_sq, 1e-30)
        cos_v_bar = cos_pair(O_v, O_bar)

        # Leave-one-out
        others = [O[l] for l in labels if l != lab]
        O_bar_LOO = np.mean(others, axis=0)
        O_bar_LOO_norm_sq = float(np.dot(O_bar_LOO, O_bar_LOO))
        if O_bar_LOO_norm_sq > 1e-30:
            a_v_LOO = float(np.dot(O_v, O_bar_LOO) / O_bar_LOO_norm_sq)
        else:
            a_v_LOO = 0.0
        O_shared_v_LOO = a_v_LOO * O_bar_LOO
        O_specific_v_LOO = O_v - O_shared_v_LOO
        norm_shared_LOO_sq = float(np.dot(O_shared_v_LOO, O_shared_v_LOO))
        norm_specific_LOO_sq = float(np.dot(O_specific_v_LOO, O_specific_v_LOO))
        f_shared_LOO = norm_shared_LOO_sq / max(norm_O_v_sq, 1e-30)
        f_specific_LOO = norm_specific_LOO_sq / max(norm_O_v_sq, 1e-30)
        cos_v_bar_LOO = cos_pair(O_v, O_bar_LOO)

        # Stabilité
        cos_Obar_OLOO = cos_pair(O_bar, O_bar_LOO)
        cos_specific = cos_pair(O_specific_v, O_specific_v_LOO)
        delta_f_shared = abs(f_shared - f_shared_LOO)

        results["per_variant"][lab] = {
            "a_v": a_v,
            "f_shared": f_shared,
            "f_specific": f_specific,
            "cos_v_Obar": cos_v_bar,
            "a_v_LOO": a_v_LOO,
            "f_shared_LOO": f_shared_LOO,
            "f_specific_LOO": f_specific_LOO,
            "cos_v_OLOO": cos_v_bar_LOO,
            "cos_Obar_OLOO": cos_Obar_OLOO,
            "cos_specific_stability": cos_specific,
            "delta_f_shared_LOO": delta_f_shared,
        }
    return results


def main():
    gamma_v, D, h0, beta_lock = 1.0, 0.1, 1.0, 60.0
    psi0 = make_psi_centered()
    h0f = np.full((N_AXIS, N_AXIS, N_AXIS), h0)
    psi_max = float(psi0.max())
    dt = 0.5 * min(cfl_dt_max(h0, D), 1.0 / (beta_lock * psi_max + gamma_v))
    n_stab = int(50.0 / dt)
    n_short = int(10.0 / dt)
    n_dt_long = int(800.0 / dt)

    print(f"=== 6d-ζ : Test L1 — décomposition partagée/spécifique ===\n")

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

    print(f"  Calcul des résidus...")
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

    labels = ["étroite", "standard", "large", "annulaire"]
    obs_names = ["psi_temp_mean_abs", "psi_temp_norm",
                 "h_temp_mean_abs", "h_temp_norm"]

    # Calculer les observables temporelles pour chaque variante
    obs_data = {}  # name -> {label -> série temporelle}
    obs_data_centered = {}
    for name in obs_names:
        obs_data[name] = {}
        obs_data_centered[name] = {}
        for lab in labels:
            r_psi = residuals[lab]["r_psi"]
            r_h = residuals[lab]["r_h"]
            if name == "psi_temp_mean_abs":
                serie = np.mean(np.abs(r_psi), axis=(1,2,3))
            elif name == "psi_temp_norm":
                serie = np.sqrt(np.sum(r_psi**2, axis=(1,2,3)))
            elif name == "h_temp_mean_abs":
                serie = np.mean(np.abs(r_h), axis=(1,2,3))
            elif name == "h_temp_norm":
                serie = np.sqrt(np.sum(r_h**2, axis=(1,2,3)))
            obs_data[name][lab] = serie
            obs_data_centered[name][lab] = serie - serie.mean()

    # === Décomposition + LOO pour chaque observable, brut et centré ===
    all_results = {}
    for name in obs_names:
        for mode, data_dict in [("brut", obs_data), ("centré", obs_data_centered)]:
            print(f"\n  === {name} ({mode}) ===")
            res = decompose_and_LOO(data_dict[name], labels)
            all_results[f"{name}_{mode}"] = res

            print(f"  ||Ō|| = {res['O_bar_norm']:.4e}")
            print(f"  {'variante':<12} {'a_v':>8} {'f_shared':>10} {'f_spec':>10} "
                  f"{'cos(v,Ō)':>10} {'a_v_LOO':>10} {'f_shared_LOO':>14} "
                  f"{'cos(Ō,Ō_LOO)':>15} {'cos_spec_stab':>14}")
            for lab in labels:
                r = res["per_variant"][lab]
                print(f"  {lab:<12} {r['a_v']:>8.4f} {r['f_shared']:>10.4f} "
                      f"{r['f_specific']:>10.4f} {r['cos_v_Obar']:>10.4f} "
                      f"{r['a_v_LOO']:>10.4f} {r['f_shared_LOO']:>14.4f} "
                      f"{r['cos_Obar_OLOO']:>15.4f} {r['cos_specific_stability']:>14.4f}")

    # === Synthèse globale ===
    print(f"\n=== Synthèse globale ===\n")
    print(f"  Question : la décomposition partagée/spécifique est-elle robuste ?")
    print(f"  Critères de robustesse :")
    print(f"  (a) f_shared > 0.5 sur les 4 variantes (composante partagée dominante)")
    print(f"  (b) |f_shared - f_shared_LOO| < 0.05 (stabilité au leave-one-out)")
    print(f"  (c) cos(Ō, Ō_LOO) > 0.95 (composante partagée stable)")
    print(f"  (d) cos_specific_stability > 0.90 (composante spécifique stable)\n")

    summary = {}
    for key, res in all_results.items():
        f_shared_min = min(r["f_shared"] for r in res["per_variant"].values())
        delta_f_max = max(r["delta_f_shared_LOO"] for r in res["per_variant"].values())
        cos_Obar_OLOO_min = min(r["cos_Obar_OLOO"] for r in res["per_variant"].values())
        cos_spec_min = min(r["cos_specific_stability"] for r in res["per_variant"].values())
        criteria = {
            "(a) f_shared_min > 0.5": f_shared_min > 0.5,
            "(b) delta_f_max < 0.05": delta_f_max < 0.05,
            "(c) cos_Obar_min > 0.95": cos_Obar_OLOO_min > 0.95,
            "(d) cos_spec_min > 0.90": cos_spec_min > 0.90,
        }
        all_pass = all(criteria.values())
        verdict = "ROBUSTE" if all_pass else "FRAGILE"
        summary[key] = {
            "f_shared_min": float(f_shared_min),
            "delta_f_max": float(delta_f_max),
            "cos_Obar_OLOO_min": float(cos_Obar_OLOO_min),
            "cos_specific_stab_min": float(cos_spec_min),
            "criteria": criteria,
            "verdict": verdict,
        }
        print(f"  {key:<30}")
        print(f"    f_shared min = {f_shared_min:.4f}  (>0.5 ? {criteria['(a) f_shared_min > 0.5']})")
        print(f"    delta_f max  = {delta_f_max:.4f}  (<0.05 ? {criteria['(b) delta_f_max < 0.05']})")
        print(f"    cos(Ō,Ō_LOO) min = {cos_Obar_OLOO_min:.4f}  (>0.95 ? {criteria['(c) cos_Obar_min > 0.95']})")
        print(f"    cos_spec_stab min = {cos_spec_min:.4f}  (>0.90 ? {criteria['(d) cos_spec_min > 0.90']})")
        print(f"    VERDICT : {verdict}\n")

    output = {
        "obs_names": obs_names,
        "labels": labels,
        "results": {},
        "summary": summary,
    }
    # Convertir all_results en JSON-serializable
    for key, res in all_results.items():
        output["results"][key] = {
            "O_bar_norm": res["O_bar_norm"],
            "per_variant": res["per_variant"],
        }
    with open("/home/claude/mcq_v4/6d_zeta_L1_decomposition.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON sauvegardé.")


if __name__ == "__main__":
    main()
