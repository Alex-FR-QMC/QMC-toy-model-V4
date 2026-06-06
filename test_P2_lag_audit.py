# -*- coding: utf-8 -*-
"""
6d - Audit P2-lag : corrélation croisée temporelle I_h(t) vs I_psi(t).

Source : trajectoires exportées par test_6d_iota_export_traj.py
(JSON : 6d_iota_export_traj.json).

Pour chaque paire (M, P'), on dispose de :
- a_h(t) = ||I_h(t)||_2  (norme spatiale du champ d'interaction sur h)
- a_psi(t) = ||I_psi(t)||_2  (idem sur psi)

On calcule la corrélation croisée normalisée entre les deux séries
temporelles, sur trois modes :
- brut : a_h(t), a_psi(t)
- centré : a(t) - mean(a)
- différences : Delta_a(t) = a(t+1) - a(t)

Verdict principal : sur différences.
Fenêtre principale : [-50, +50] unités de temps.
Fenêtre audit secondaire : [-200, +200] unités de temps.

Convention :
- lag > 0 : I_h précède I_psi
- lag < 0 : I_psi précède I_h
"""

import json
import numpy as np


def cross_corr_lag(x, y, max_lag_steps):
    """Corrélation croisée normalisée entre x et y.
    Retourne (lags_array, corrs_array).
    lag > 0 signifie x(t) corrélé avec y(t+lag), donc x précède y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-30 or ny < 1e-30:
        lags = np.arange(-max_lag_steps, max_lag_steps + 1)
        return lags, np.zeros(2 * max_lag_steps + 1)
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    corrs = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            # x[:n-lag] avec y[lag:n] ; on compare x(t) et y(t+lag)
            xs = x[:n-lag] if lag > 0 else x
            ys = y[lag:n] if lag > 0 else y
        else:
            # lag négatif : x(t-|lag|) avec y(t)
            l = -lag
            xs = x[l:n]
            ys = y[:n-l]
        if len(xs) < 10:
            corrs[i] = 0.0
            continue
        nxs = float(np.linalg.norm(xs))
        nys = float(np.linalg.norm(ys))
        if nxs < 1e-30 or nys < 1e-30:
            corrs[i] = 0.0
            continue
        corrs[i] = float(np.dot(xs, ys) / (nxs * nys))
    return lags, corrs


def analyze_pair(a_h, a_psi, dt, max_lag_time):
    """Pour une paire (a_h, a_psi), calcule les corrélations brutes,
    centrées et sur différences, dans la fenêtre [-max_lag_time, +max_lag_time]."""
    max_lag_steps = int(max_lag_time / dt)
    a_h = np.asarray(a_h, dtype=float)
    a_psi = np.asarray(a_psi, dtype=float)

    # Brut
    lags, corr_brut = cross_corr_lag(a_h, a_psi, max_lag_steps)
    # Centré
    a_h_c = a_h - a_h.mean()
    a_psi_c = a_psi - a_psi.mean()
    _, corr_cent = cross_corr_lag(a_h_c, a_psi_c, max_lag_steps)
    # Différences
    da_h = np.diff(a_h)
    da_psi = np.diff(a_psi)
    _, corr_diff = cross_corr_lag(da_h, da_psi, max_lag_steps)

    def extract(corrs):
        idx_zero = max_lag_steps  # index du lag 0
        corr_zero = float(corrs[idx_zero])
        idx_max = int(np.argmax(corrs))
        lag_max_steps = idx_max - max_lag_steps
        lag_max_time = lag_max_steps * dt
        corr_max = float(corrs[idx_max])
        gain = corr_max - corr_zero
        return {
            "lag_max_steps": lag_max_steps,
            "lag_max_time": lag_max_time,
            "corr_max": corr_max,
            "corr_zero": corr_zero,
            "gain_lag": gain,
            "sign_lag": int(np.sign(lag_max_steps)),
        }

    return {
        "brut": extract(corr_brut),
        "centre": extract(corr_cent),
        "diff": extract(corr_diff),
    }


def main():
    with open("/home/claude/mcq_v4/6d_iota_export_traj.json") as f:
        data = json.load(f)
    dt = data["dt_simulation"]
    print(f"=== 6d audit P2-lag — corrélation croisée I_h(t) vs I_psi(t) ===\n")
    print(f"  dt = {dt:.5f}")
    print(f"  Fenêtre principale : [-50, +50] unités de temps")
    print(f"  Fenêtre audit       : [-200, +200] unités de temps\n")

    # Pour chaque paire (M, P'), récupérer trajectoires et analyser
    results_short = {}   # [-50, +50]
    results_long = {}    # [-200, +200]
    for key, pair_data in data["results"].items():
        a_h = pair_data["trajectory_a_h"]
        a_psi = pair_data["trajectory_a_psi"]
        if (max(a_h) < 1e-30) or (max(a_psi) < 1e-30):
            continue
        results_short[key] = analyze_pair(a_h, a_psi, dt, max_lag_time=50.0)
        results_long[key] = analyze_pair(a_h, a_psi, dt, max_lag_time=200.0)

    # === Affichage : fenêtre principale [-50,+50] ===
    print(f"=== FENÊTRE PRINCIPALE [-50, +50] ===\n")
    print(f"  Verdict basé sur mode 'diff' (différences temporelles)\n")
    print(f"  {'pair':<48} {'mode':<7} {'lag(t)':>8} {'corr_max':>10} "
          f"{'corr_zero':>10} {'gain':>10} {'sign':>5}")
    for key in results_short:
        for mode in ["brut", "centre", "diff"]:
            r = results_short[key][mode]
            print(f"  {key:<48} {mode:<7} {r['lag_max_time']:>+8.2f} "
                  f"{r['corr_max']:>+10.4f} {r['corr_zero']:>+10.4f} "
                  f"{r['gain_lag']:>+10.4f} {r['sign_lag']:>+5d}")
        print()

    # === Synthèse mode diff ===
    print(f"=== SYNTHÈSE MODE DIFF (verdict principal) ===\n")
    print(f"  {'pair':<48} {'lag(t)':>8} {'gain':>10} {'sign':>5}")
    diff_lags = []
    diff_gains = []
    diff_signs = []
    for key, r in results_short.items():
        d = r["diff"]
        diff_lags.append(d["lag_max_time"])
        diff_gains.append(d["gain_lag"])
        diff_signs.append(d["sign_lag"])
        print(f"  {key:<48} {d['lag_max_time']:>+8.2f} "
              f"{d['gain_lag']:>+10.4f} {d['sign_lag']:>+5d}")
    n_pos = sum(1 for s in diff_signs if s > 0)
    n_neg = sum(1 for s in diff_signs if s < 0)
    n_zero = sum(1 for s in diff_signs if s == 0)
    n_total = len(diff_signs)
    print(f"\n  Signes des lags (mode diff) :")
    print(f"    positifs (I_h précède I_psi) : {n_pos}/{n_total}")
    print(f"    négatifs (I_psi précède I_h) : {n_neg}/{n_total}")
    print(f"    nuls : {n_zero}/{n_total}")
    print(f"  Gain moyen : {np.mean(diff_gains):.4f}")
    print(f"  Gain max  : {max(diff_gains):.4f}")
    print(f"  Gain min  : {min(diff_gains):.4f}")

    # === Audit fenêtre longue [-200,+200] ===
    print(f"\n=== AUDIT FENÊTRE LONGUE [-200, +200] (contrôle, non verdict) ===\n")
    print(f"  Mode diff seulement :")
    print(f"  {'pair':<48} {'lag(t)':>8} {'gain':>10} {'sign':>5}")
    n_outside_short = 0
    for key, r in results_long.items():
        d = r["diff"]
        outside = abs(d["lag_max_time"]) > 50.0
        if outside:
            n_outside_short += 1
        flag = "  *" if outside else ""
        print(f"  {key:<48} {d['lag_max_time']:>+8.2f} "
              f"{d['gain_lag']:>+10.4f} {d['sign_lag']:>+5d}{flag}")
    print(f"\n  Paires avec pic principal hors fenêtre courte : "
          f"{n_outside_short}/{len(results_long)}")
    print(f"  (si > 0, lag possiblement long, à interpréter avec prudence)")

    # === Verdict ===
    print(f"\n=== VERDICT P2-lag ===\n")

    # Critères :
    # - lag_max > 0 (positif) sur plusieurs paires
    # - gain_lag non trivial
    # - signe cohérent
    # - pas porté par une seule paire

    # Cohérence du signe : on accepte n_pos majoritaire ou n_neg majoritaire,
    # tant que les deux ne sont pas équilibrés
    sign_coherent = (n_pos >= 7 or n_neg >= 7) and (n_pos != n_neg)
    # Gain non trivial : gain max > 0.01, plusieurs paires avec gain > 0.001
    gain_nontrivial = max(diff_gains) > 0.01
    n_with_gain = sum(1 for g in diff_gains if g > 0.001)
    multi_paires = n_with_gain >= 3
    # Lag réellement non nul
    n_with_nonzero_lag = sum(1 for l in diff_lags if abs(l) > 0.1)
    multi_lag = n_with_nonzero_lag >= 5

    print(f"  Signe cohérent (n_pos ou n_neg >= 7/10) : {sign_coherent}")
    print(f"    n_pos = {n_pos}, n_neg = {n_neg}, n_zero = {n_zero}")
    print(f"  Gain non trivial (max > 0.01) : {gain_nontrivial}")
    print(f"    gain_max = {max(diff_gains):.4f}")
    print(f"  Plusieurs paires avec gain > 0.001 : {multi_paires}")
    print(f"    n_with_gain = {n_with_gain}/{n_total}")
    print(f"  Lag réellement non nul (> 0.1 unité, > 5 paires) : {multi_lag}")
    print(f"    n_with_nonzero_lag = {n_with_nonzero_lag}/{n_total}")

    # Décision
    if sign_coherent and gain_nontrivial and multi_paires and multi_lag:
        verdict = "P2-lag PASS faible"
    elif (not sign_coherent) and (not gain_nontrivial):
        verdict = "P2-lag FAIL"
    else:
        verdict = "P2-lag INDETERMINÉ"

    # Contrôle : si gain mode diff est négligeable vs gain mode brut, c'est tendance commune
    brut_gains = [results_short[key]["brut"]["gain_lag"] for key in results_short]
    if max(brut_gains) > 0.05 and max(diff_gains) < 0.005:
        verdict_qualifier = " (signal probablement tendance commune, pas latence)"
        verdict += verdict_qualifier

    print(f"\n  VERDICT : {verdict}")

    # Sauvegarder
    output = {
        "dt": dt,
        "max_lag_short": 50.0,
        "max_lag_long": 200.0,
        "convention": "lag > 0 : I_h précède I_psi",
        "results_short_window": results_short,
        "results_long_window": results_long,
        "summary_diff_mode_short": {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_zero": n_zero,
            "n_total": n_total,
            "gain_mean": float(np.mean(diff_gains)),
            "gain_max": float(max(diff_gains)),
            "gain_min": float(min(diff_gains)),
            "n_outside_short_window": n_outside_short,
        },
        "verdict": verdict,
        "verdict_criteria": {
            "sign_coherent": bool(sign_coherent),
            "gain_nontrivial": bool(gain_nontrivial),
            "multi_paires_with_gain": bool(multi_paires),
            "multi_paires_with_nonzero_lag": bool(multi_lag),
        },
    }
    with open("/home/claude/mcq_v4/6d_P2_lag_audit.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé.")


if __name__ == "__main__":
    main()
