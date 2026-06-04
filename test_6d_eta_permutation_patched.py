"""
6d-η — Audit permutation exacte des labels G/A.

Test α : sur les 9 variantes G+A (les 3 D sont laissées hors du test),
énumérer les C(9,5)=126 attributions possibles de 5 labels G_perm et
4 labels A_perm. Pour chaque permutation, calculer
A_GA_perm = mean(G_perm,G_perm) - mean(G_perm,A_perm) et comparer
A_GA_observé à la distribution exacte.

Effectué pour les 8 cas (4 observables × 2 modes), sur cos_ss_pure
ET séparément sur contrib_cos_ss.

Pas de relance du moteur. Aucune nouvelle géométrie.
Pas de Δ, pas de 𝒢, pas de lecture MCQ.

Diagnostic secondaire optionnel : clustering simple sur cos_ss_pure.
"""

import numpy as np
import json
from itertools import combinations


def main():
    with open("/home/claude/mcq_v4/6d_eta_confirmation.json") as f:
        eta_data = json.load(f)

    labels = eta_data["labels"]
    families = eta_data["families"]
    n = len(labels)
    print(f"=== 6d-η audit permutation labels G/A ===\n")
    print(f"  {n} variantes : {labels}")
    print(f"  Familles  : {families}\n")

    # Indices des variantes G et A
    idx_G = [i for i, f in enumerate(families) if f == "G"]
    idx_A = [i for i, f in enumerate(families) if f == "A"]
    idx_D = [i for i, f in enumerate(families) if f == "D"]
    n_G = len(idx_G)
    n_A = len(idx_A)
    n_GA = n_G + n_A

    idx_GA = idx_G + idx_A
    print(f"  Indices G observés : {idx_G} (n={n_G})")
    print(f"  Indices A observés : {idx_A} (n={n_A})")
    print(f"  Indices D (exclus) : {idx_D} (n={len(idx_D)})\n")

    # Énumérer les C(9,5)=126 attributions possibles de 5 G_perm parmi les 9 G+A
    permutations = list(combinations(idx_GA, n_G))
    n_perms = len(permutations)
    print(f"  Nombre de permutations énumérées : {n_perms}\n")

    def compute_A_GA(matrix, idx_G_set, idx_A_set):
        """A_GA = mean(G,G) - mean(G,A) sur paires distinctes."""
        GG_vals, GA_vals = [], []
        for i in idx_G_set:
            for j in idx_G_set:
                if i >= j: continue
                GG_vals.append(matrix[i][j])
            for j in idx_A_set:
                GA_vals.append(matrix[i][j])
        m_GG = float(np.mean(GG_vals)) if GG_vals else None
        m_GA = float(np.mean(GA_vals)) if GA_vals else None
        if m_GG is None or m_GA is None:
            return None
        return m_GG - m_GA

    audit_results = {}
    obs_modes = list(eta_data["by_observable"].keys())

    for key in obs_modes:
        data = eta_data["by_observable"][key]
        result = {}
        for matrix_name in ["cos_ss_pure_matrix", "contrib_cos_ss_matrix"]:
            matrix = data[matrix_name]
            # A_GA observé (sur les labels d'origine)
            A_GA_obs = compute_A_GA(matrix, idx_G, idx_A)
            # Distribution sous permutation
            A_GA_perm_dist = []
            for perm_G in permutations:
                perm_G_set = set(perm_G)
                perm_A_set = set(idx_GA) - perm_G_set
                a = compute_A_GA(matrix, list(perm_G_set), list(perm_A_set))
                if a is not None:
                    A_GA_perm_dist.append(a)
            A_GA_perm_dist = np.array(A_GA_perm_dist)

            # Quantile et p-value avec tolérance flottante
            # (A_GA_obs est l'une des 126 valeurs permutées, donc l'égalité
            # doit être traitée avec eps)
            eps = 1e-10
            n_extreme_high = int((A_GA_perm_dist >= A_GA_obs - eps).sum())
            n_extreme_low = int((A_GA_perm_dist <= A_GA_obs + eps).sum())
            # Plancher : la permutation correspondant aux labels observés
            # est nécessairement dans la distribution, donc p_high ≥ 1/n_perms
            n_perms_total = len(A_GA_perm_dist)
            p_one_sided_high = max(n_extreme_high, 1) / n_perms_total
            p_one_sided_low = max(n_extreme_low, 1) / n_perms_total
            # Rang : nombre de permutations strictement inférieures, +1.
            # Cappé à n_perms (et non n_perms+1).
            rank = min(int((A_GA_perm_dist < A_GA_obs - eps).sum()) + 1,
                       n_perms_total)
            quantile = rank / n_perms_total

            # Statistiques
            mean_perm = float(A_GA_perm_dist.mean())
            std_perm = float(A_GA_perm_dist.std())
            z_obs = (A_GA_obs - mean_perm) / max(std_perm, 1e-30)
            max_perm = float(A_GA_perm_dist.max())
            min_perm = float(A_GA_perm_dist.min())

            result[matrix_name] = {
                "A_GA_obs": A_GA_obs,
                "mean_perm": mean_perm,
                "std_perm": std_perm,
                "min_perm": min_perm,
                "max_perm": max_perm,
                "z_obs": z_obs,
                "rank": rank,
                "quantile": quantile,
                "p_high": p_one_sided_high,
                "p_low": p_one_sided_low,
            }
        audit_results[key] = result

    # Affichage
    print(f"=== Résultats par observable/mode ===\n")
    print(f"  Pour chaque cas : A_GA_observé vs distribution sous permutation")
    print(f"  rank = position dans la distribution (1 = plus bas, 126 = plus haut)")
    print(f"  p_high = P(A_GA_perm >= A_GA_obs) sur 126 permutations")
    print(f"  z = (A_GA_obs - mean_perm) / std_perm\n")

    for key in obs_modes:
        print(f"  --- {key} ---")
        for matrix_name in ["cos_ss_pure_matrix", "contrib_cos_ss_matrix"]:
            r = audit_results[key][matrix_name]
            short = "cos_ss_pure" if "pure" in matrix_name else "contrib_cos_ss"
            print(f"    {short:<18} : A_GA_obs = {r['A_GA_obs']:+.4f}, "
                  f"mean_perm = {r['mean_perm']:+.4f}, "
                  f"std_perm = {r['std_perm']:.4f}, "
                  f"z = {r['z_obs']:+.2f}, "
                  f"rank = {r['rank']}/{n_perms}, "
                  f"p_high = {r['p_high']:.4f}")
        print()

    # Synthèse globale
    print(f"=== Synthèse globale ===\n")
    print(f"  Combien de cas sur 8 ont A_GA observé dans le top X% ?\n")

    for matrix_name in ["cos_ss_pure_matrix", "contrib_cos_ss_matrix"]:
        short = "cos_ss_pure" if "pure" in matrix_name else "contrib_cos_ss"
        in_top_5 = sum(1 for key in obs_modes
                        if audit_results[key][matrix_name]["p_high"] < 0.05)
        in_top_1 = sum(1 for key in obs_modes
                        if audit_results[key][matrix_name]["p_high"] < 0.01)
        # Le 99e percentile correspond à rank >= 0.99 × 126 = 125
        in_top_q99 = sum(1 for key in obs_modes
                          if audit_results[key][matrix_name]["quantile"] >= 0.99)
        print(f"  {short:<18} : top 5% = {in_top_5}/8, "
              f"top 1% = {in_top_1}/8, "
              f"quantile ≥ 99% = {in_top_q99}/8")

    # Examen détaillé du cas fragile psi_temp_norm_centré
    print(f"\n=== Cas fragile : psi_temp_norm_centré ===\n")
    fragile_key = "psi_temp_norm_centré"
    if fragile_key in audit_results:
        for matrix_name in ["cos_ss_pure_matrix", "contrib_cos_ss_matrix"]:
            short = "cos_ss_pure" if "pure" in matrix_name else "contrib_cos_ss"
            r = audit_results[fragile_key][matrix_name]
            verdict_local = ("extrême" if r["p_high"] < 0.05
                             else "non extrême" if r["p_high"] < 0.5
                             else "banal voire en queue inverse")
            print(f"  {short:<18} : A_GA_obs = {r['A_GA_obs']:+.4f}, "
                  f"rank {r['rank']}/{n_perms}, "
                  f"p_high = {r['p_high']:.4f} → {verdict_local}")
    print()

    # Lecture finale
    print(f"=== Lecture finale ===\n")
    # Sur cos_ss_pure
    cos_ss_p_high_list = [audit_results[k]["cos_ss_pure_matrix"]["p_high"]
                          for k in obs_modes]
    contrib_cos_p_high_list = [audit_results[k]["contrib_cos_ss_matrix"]["p_high"]
                                for k in obs_modes]
    cos_ss_max_p = max(cos_ss_p_high_list)
    contrib_max_p = max(contrib_cos_p_high_list)
    print(f"  Sur cos_ss_pure : p_high max sur 8 cas = {cos_ss_max_p:.4f}")
    print(f"  Sur contrib_cos_ss : p_high max sur 8 cas = {contrib_max_p:.4f}\n")

    if cos_ss_max_p < 0.05 and contrib_max_p < 0.05:
        verdict_audit = "AUDIT CONFIRME : axe G/A statistiquement non banal sur tous les cas (cos_ss_pure et contrib_cos_ss)"
    elif cos_ss_max_p < 0.05:
        verdict_audit = "AUDIT PARTIEL : axe G/A extrême sur cos_ss_pure (direction spécifique), moins net sur contrib_cos_ss (contribution au cos)"
    elif contrib_max_p < 0.05:
        verdict_audit = "AUDIT PARTIEL : axe G/A extrême sur contrib_cos_ss mais pas sur cos_ss_pure (peu commun, à examiner)"
    else:
        verdict_audit = "AUDIT NE CONFIRME PAS : A_GA n'est extrême ni sur cos_ss_pure ni sur contrib_cos_ss sur tous les cas"

    print(f"  {verdict_audit}\n")

    # Diagnostic secondaire : clustering sur cos_ss_pure (annexe exploratoire)
    print(f"=== Annexe : clustering exploratoire sur cos_ss_pure ===\n")
    print(f"  Test : à partir de la matrice cos_ss_pure observée, peut-on")
    print(f"  retrouver les familles G/A/D sans les labels ?\n")

    # On utilise une approche simple : MDS 2D, puis k-means à 3 clusters
    try:
        from sklearn.manifold import MDS
        from sklearn.cluster import KMeans
        sklearn_available = True
    except ImportError:
        sklearn_available = False

    if sklearn_available:
        # On prend la moyenne des cos_ss_pure sur les 8 cas pour avoir une
        # matrice de similarité robuste
        all_cos_ss = np.mean([
            np.array(eta_data["by_observable"][k]["cos_ss_pure_matrix"])
            for k in obs_modes
        ], axis=0)
        # Convertir similarité en distance : d = 1 - cos
        dist_matrix = 1.0 - all_cos_ss
        # Forcer symétrie et diagonale nulle
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0.0)
        # MDS 2D
        try:
            mds = MDS(n_components=2, dissimilarity='precomputed',
                       random_state=42, normalized_stress='auto')
            embedding = mds.fit_transform(dist_matrix)
            # K-means à 3 clusters
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding)
            print(f"  Clustering à 3 clusters sur MDS-2D de la matrice cos_ss_pure moyennée :")
            print(f"  {'variante':<28} {'famille_vraie':>12} {'cluster':>10}")
            for i, lab in enumerate(labels):
                print(f"  {lab:<28} {families[i]:>12} {cluster_labels[i]:>10}")

            # Vérifier la correspondance cluster ↔ famille
            from collections import Counter
            family_in_cluster = {}
            for c in set(cluster_labels):
                in_cluster = [families[i] for i in range(n) if cluster_labels[i] == c]
                family_in_cluster[int(c)] = dict(Counter(in_cluster))
            print(f"\n  Composition des clusters :")
            for c, comp in family_in_cluster.items():
                print(f"    cluster {c} : {comp}")
            audit_results["_clustering"] = {
                "cluster_labels": [int(c) for c in cluster_labels],
                "family_in_cluster": family_in_cluster,
            }
        except Exception as e:
            print(f"  Clustering échoué : {e}")
            audit_results["_clustering"] = {"error": str(e)}
    else:
        print(f"  scikit-learn non disponible, clustering omis.")
        audit_results["_clustering"] = {"error": "sklearn unavailable"}

    # Sauvegarder
    output = {
        "n_variants_total": n,
        "n_G": n_G,
        "n_A": n_A,
        "n_D": len(idx_D),
        "n_permutations": n_perms,
        "idx_G_observed": idx_G,
        "idx_A_observed": idx_A,
        "idx_D_excluded": idx_D,
        "by_obs_mode": {k: v for k, v in audit_results.items() if k != "_clustering"},
        "clustering_diagnostic": audit_results.get("_clustering", None),
        "summary": {
            "cos_ss_pure_max_p_high": cos_ss_max_p,
            "contrib_cos_ss_max_p_high": contrib_max_p,
            "verdict_audit": verdict_audit,
        },
    }
    with open("/home/claude/mcq_v4/6d_eta_permutation_audit.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON sauvegardé : /home/claude/mcq_v4/6d_eta_permutation_audit.json")


if __name__ == "__main__":
    main()
