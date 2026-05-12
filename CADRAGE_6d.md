# Phase 6d — Cadrage initial

**Statut** : verrouillé après brainstorm session 12, post-clôture 6c-B.
Mis à jour avec amendements numerics-doc §0bis (terminologie
conformal-conservative) et C3 audit (non-commutativité 6d-γ).

**Principe directeur** : 6d ne prouve pas MCQ. 6d teste falsifiablement
les propriétés émergentes de Ch3 sous instrumentation
**conformal-conservative** (approximation Euclidienne de la dynamique
géométrique Ch3, pas Laplace-Beltrami strict ni géométrie riemannienne
complète — cf. numerics-doc §0bis), et produit une carte des
propriétés observables vs non-observables selon le niveau
d'instrumentation.

---

## 1. Objet de la phase

Phase 6d = **test de réalisabilité conformal-conservative de la master
equation Ch3 §3.1.2** sur N=1 instance.

L'objet n'est pas de "faire marcher MCQ". L'objet est de mesurer **où
l'approximation conformal-conservative atteint son plafond** : quelles
propriétés émergentes prédites par Ch3 §3.6.11 sont empiriquement
observables sous h(θ) scalaire plein, et quelles propriétés exigent
soit la tensorialisation H(θ) (dette Ch4) soit l'enrichissement
multi-instance (QMCᴺ, hors 6d).

**Note terminologique** : "conformal-fidèle" → "conformal-conservatif".
Ce changement précis (cf. numerics-doc §0bis + C2 audit) rappelle que
l'approximation ne réalise pas Laplace-Beltrami strict mais une
géométrie effective de flux où h(θ) module les coefficients d'un
opérateur conservatif Euclidien.

---

## 2. Décisions architecturales tranchées

### D1 — Géométrie : h(θ) scalaire plein sur Θ

`dᵢ^ℋ(θ) = hᵢ(θ)` avec hᵢ : Θᵢ → [h_min, h₀] plein sur la grille
discrète (5×5×5).

Pas de marginales h_T/h_M/h_I (déjà saturées en V4). Pas de
diag(h_T(θ), h_M(θ), h_I(θ)) — hypothèse directionnelle injustifiée par
Ch3 et préfigurant la tensorialisation au lieu de l'éprouver. Pas de
tenseur H(θ) plein — dette de Ch4, prématuré ici.

L'anisotropie observée en 6d émergera de la **variation spatiale** de
h(θ) (Ch3 §3.3.5.II), pas d'une décomposition par axe.

### D2 — Régime : N=1 instrumenté pleinement

Mono-instance, instrumentée jusqu'au bout de l'approximation
conformal-conservative.
Cohérent avec Ch3 §3.1.10 : le mono-instance est le point de départ
computationnel, structurellement plus pauvre mais viable.

QMCᴺ (cross-perception, asymmetric trust, cascade buffering, thermal
mosaic, interferential 𝒢 enrichment) reste explicitement hors 6d.

### D3 — Régulation : g_Ω minimal deux canaux

Pas les 7 sensibilités directionnelles de §3.1.5 dès 6d. Deux canaux
ciblés sur les KNV qui ont effectivement piégé V4 6c-B :

- **Canal 1 — anti-collapse distributionnel** : g_Ω augmente D_eff
  smoothly quand H → H_crit ou Var → Var_min (KNV 5, KNV 11). Attaque
  l'engine-driven fusion observée en T27' baseline.
- **Canal 2 — anti-pétrification morphologique** : g_Ω augmente
  D_eff^form (composante de fond, pas fluctuation) quand
  ‖∂_t dᵢ^ℋ‖ < ε_petr persiste au-delà de τ^form (KNV 7). Instancie le
  canal RR³ → δ𝕋* → δD_eff^form de Ch3 §3.6.3.III.

Φ_corr également minimal : la composante anti-collapse est suffisante
pour Ch3 §3.4.1 P₅ (Var → 0), avec le bémol que la régulation reste
continue, pas seuillée.

---

## 3. Programme de tests — 4 propriétés émergentes Ch3 + cross-talk métrique

### 3.1 Propriétés 1-4 : revendications Ch3 explicites

| # | Propriété émergente | Source Ch3 | Instrumentation requise | Critère falsifiable |
|---|---|---|---|---|
| 1 | Dual-timescale memory | §3.3 induced debt 12 | h(θ) plein + 𝔊^{ero} = γ·h·(1−h/h₀) non-linéaire | Single trace érode dans t_restore_max ; cycles répétés en même direction laissent trace persistante non érodée |
| 2 | Morphological latency | §3.3 cross-tension 4 | h(θ) plein + g_k(θ, h) explicite | Pendant contraction rapide à ω^pos, τ'_k reflète h pré-contraction avec lag mesurable au scale ω^form |
| 3 | Productive forgetting | §3.4 cross-tension 11 | h(θ) plein + g_k = θ_k/h(θ) + 𝔊^{ero} actif | Érosion d'une mémoire profonde → augmentation observable de contribution g_k de cette direction → cascade perturbative cross-modulaire |
| 4 | Self-opacity 4 niveaux (testable indirectement) | §3.5 cross-tension 18 | h(θ) plein + 𝓜_π identifiabilité finie 𝒫 | Reconstructibilité limitée : à partir de 𝒫(ψᵢ) = {Var, H, R, D_KL, Ξ_obs}, plusieurs ψᵢ admissibles produisent les mêmes 𝒫. Test de pseudo-inversion. |

### 3.2 Propriété 5 : cross-talk métrique (B2 audit — phénomène structurel émergent)

**Asymétrie de statut explicite** : les propriétés 1-4 sont des
**revendications Ch3** (propriétés émergentes que Ch3 prédit
explicitement). La propriété 5 est un **phénomène structurel émergent
induit par la métrique pleine h(θ)** — pas une revendication Ch3
canonique, mais une conséquence directe de la perte de séparabilité
(§1.8 numerics-doc) qui justifie à elle seule le passage 6c-B → 6d.

**Test direct de perte de séparabilité** : si h(θ) plein dégénère
empiriquement vers diagonal (ou marginal), la phase 6d ne révèle pas
plus que 6c-B. Tester explicitement cette non-dégénérescence est
donc essentiel.

| # | Propriété émergente | Source | Instrumentation requise | Critère falsifiable |
|---|---|---|---|---|
| 5 | **Cross-talk métrique** | numerics-doc §1.8 (conséquence h(θ) plein non séparable) | h(θ) plein + comparaison h plein vs h projeté sur marginales | g_T mesuré sous h(θ) plein **doit différer significativement** de g_T mesuré sous h artificiellement diagonal (reconstitué depuis les marginales h_T, h_M, h_I à chaque step). Si différence négligeable → h(θ) plein dégénère, 6d n'apporte rien de plus que 6c-B. Si différence significative → cross-talk métrique actif, perte de séparabilité empiriquement confirmée. |

**Statut théorique** : la propriété 5 n'a pas le statut canonique Ch3
des propriétés 1-4. Elle est testée parce qu'elle est **nécessaire
mais pas suffisante** : sans cross-talk observable, h(θ) plein
n'apporte rien de structurellement nouveau ; avec cross-talk
observable, on a au moins établi que l'instrumentation 6d révèle un
phénomène absent en 6c-B.

**Cinquième propriété Ch3 différée** (interferential 𝒢 enrichment
QMCᴺ, §3.5 cross-tension 19) : explicitement hors 6d (exige N≥2). À
ne pas confondre avec la propriété 5 cross-talk métrique qui, elle,
est testée en 6d-β.

---

## 4. Observables prioritaires à instrumenter

Pas dans V4. À ajouter en 6d :

- **dᵢ^ℋ(θ) plein** : tableau h ∈ ℝ^{5×5×5} par module, évoluant via
  𝔊^{sed} = −β·ψ·h et 𝔊^{ero} = +γ·h·(1−h/h₀) à chaque step.
- **Laplacien conformal-conservatif** ∂_a(h^{d-2}·D_eff·∂_a ψ) avec
  ∂_a au sens Euclidien (différences finies) : pas le Laplacien
  Euclidien avec coefficient h-modulé seulement (ce que V4 fait), mais
  forme conservative Euclidienne modulée par h(θ). Voir Ch3 §3.1.3.I
  pour la distinction structurelle, et numerics-doc §0bis + §1.2 pour
  l'implémentation. **N'est pas Laplace-Beltrami strict** — voir dette
  géométrique en §0bis.
- **Drift conformal-conservatif** ∂_a(h^{d-2}·ψ·∂_a Φ_eff) dans la
  même convention conservative Euclidienne (cf. numerics-doc §1.3).
- **g_k = θ_k/h(θ)** : coupling functions explicites, avec h(θ) plein.
- **𝒞_T(t) = d_𝓗(S(t), S(t−T))** : observable morphologique
  principale, remplace les proxies T*_proxy_h_M de 6c-B. Calculée
  comme distance L² dans la métrique conforme `g_{ab} = h²·δ_{ab}`
  présente entre l'état étendu courant et l'état étendu à t−T.
- **τ_meta** : functionnel des slow-scale observables détectant fausse
  stabilité (Ch3 §3.6.8.IV).
- **β_QMC = ⟨‖∇_{dᵢ^ℋ}Φ_eff‖⟩ / D_eff** : température morphodynamique
  comme observable, pas paramètre.

---

## 5. τ'_ref — règle stricte

Conforme à Ch3 §3.6.8.I et §3.6.12.II : τ'_ref est **non-instanciable**.

- **PAS un attracteur**
- **PAS une cible**
- **PAS une loss implicite**
- **PAS un score**

Lu **uniquement** via 𝒞_T (distance géométrique trajectoire-vs-passé)
et τ_meta (détection de stabilité fausse). Toute tentative de calculer
τ'_ref directement viole le non-closure et est interdite par
construction de l'expérimentation.

Si une mesure de "convergence" apparaît dans une analyse 6d, la première
question doit être : ai-je accidentellement implémenté τ'_ref comme cible ?

---

## 6. Garde-fous explicites

À inscrire dans tout document 6d et à relire avant chaque session :

1. **6d N=1 ne teste pas MCQᴺ.** Si un phénomène inter-instance semble
   absent, cela ne dit rien — il est structurellement hors champ.
2. **6d ne teste pas le tenseur H(θ) complet.** Si l'anisotropie reste
   plafonnée empiriquement, la conclusion est "exige tensorialisation
   Ch4", pas "MCQ invalide".
3. **6d teste la limite de l'approximation conformal-conservative.**
   Aucune phase antérieure ne l'a fait sérieusement. Voir numerics-doc
   §0bis pour la distinction conformal-conservatif vs Laplace-Beltrami
   strict, et §0quater pour la co-construction des garde-fous
   numériques.
4. **Si un phénomène §3.6.11 échoue empiriquement, la conclusion
   correcte est : *non observable sous h(θ) scalaire plein*. Jamais
   *MCQ invalide* ni *propriété émergente fausse*.**
5. **Aucun verdict sur Φ_extra à ce stade.** H1/H2/H3 héritées de 6c-B
   restent ouvertes.
6. **Multi-seed dès le début.** {42, 123, 2024} comme baseline ; règle
   d'escalade {7, 999} reste manuelle.
7. **Baseline no-coupling parallèle** : préservé comme protocole standard.

---

## 7. Critère de succès réel

Le succès de 6d **n'est pas** que les 4 propriétés émergentes
apparaissent toutes sous h(θ) plein.

Le succès est de produire la **carte de réalisabilité** :

| Propriété | Observée sous h marginal (6c-B) | Observée sous h(θ) plein (6d) | Exige tensoriel | Exige N>1 |
|---|---|---|---|---|
| 1. Dual-timescale memory | ? | À mesurer | À déterminer | Non |
| 2. Morphological latency | Indirecement (T25' projection découplée) | À mesurer | À déterminer | Non |
| 3. Productive forgetting | Non testée | À mesurer | À déterminer | Non |
| 4. Self-opacity (reconstructibilité limitée) | Non testée | À mesurer | À déterminer | Non |
| 5. Interferential 𝒢 enrichment | Hors champ | **Hors 6d** | À déterminer | **Oui** |

Cette carte est l'output de 6d. Elle informe Ch4 par le bas — empiriquement,
pas spéculativement.

---

## 7bis. Hiérarchie de sous-phases 6d-α / 6d-β / 6d-γ

Hiérarchie verrouillée :

| Sous-phase | Objet | Output | Prérequis |
|---|---|---|---|
| **6d-α** | Validation numérique du moteur conformal-conservatif | Schéma validé sur cas analytiques + semi-analytiques + test pivot ψ↔h passé | Spec 6d-α-numerics.md complet |
| **6d-β** | Tests falsifiables des 4 propriétés émergentes Ch3 §3.6.11 testables N=1 | Carte propriété × observabilité sous h(θ) plein | 6d-α clos PASS |
| **6d-γ** | Cartographie des limites scalaires + observables exploratoires de plafond | Carte des phénomènes nécessitant tensorialisation H(θ) ou QMCᴺ | 6d-β clos avec verdicts |
| **Ch4** | Tensorialisation motivée empiriquement | Théorie tensorielle H(θ) | 6d-γ produit indications structurelles |

**Aucun verdict sur tensorialisation Ch4 avant que 6d-γ ne soit
complet.**

### Observables exploratoires de plafond scalaire pour 6d-γ

Au-delà des 4 propriétés émergentes (§3) qui sont l'objet principal de
6d-β, 6d-γ ajoute des **observables exploratoires** pour déterminer si
h(θ) scalaire atteint son plafond structurel et signaler ce qui
exigerait tensorialisation.

**C3 — Non-commutativité morphologique (correction C3 audit, observable
exploratoire 6d-γ)** :

Test : appliquer deux contractions successives en directions différentes
(A puis B vs B puis A) et comparer les états finaux.

```
État_AB(t_final) := contraction_A(T_A) puis contraction_B(T_B)
État_BA(t_final) := contraction_B(T_B) puis contraction_A(T_A)
```

Mesurer : `‖État_AB - État_BA‖` dans la métrique conforme.

**Lecture** :
- Si `‖État_AB - État_BA‖ → 0` : transformations morphologiques
  commutent. Compatible avec h(θ) scalaire.
- Si `‖État_AB - État_BA‖` significativement non nul : non-commutativité
  morphologique présente. **Probablement non capturable par h(θ)
  scalaire** parce que la métrique conforme `g_{ab} = h²·δ_{ab}`
  reste conformément plate localement (pas de Christoffel non-trivial,
  pas de transport parallèle non-trivial, pas de courbure tangentielle).
  Suggère que le vrai plafond Ch4 est la non-commutativité, pas
  seulement l'anisotropie.

**Statut** : observable **candidate pour déterminer si le scalaire
h(θ) atteint son plafond**. PAS exigence de passage 6d-β. PAS bloquant
pour 6d-γ globalement — c'est un test supplémentaire qui enrichit la
carte de réalisabilité.

**Hypothèses sous-jacentes** :
- Dépendance de chemin (path-dependence)
- Hystérésis directionnelle
- Transport dépendant de l'ordre des contractions
- Rotation locale des directions principales

Ces hypothèses sont structurellement impossibles avec `g_{ab} = h²·δ_{ab}`
même hétérogène spatialement (h(θ) variable). Leur observation
empirique signalerait **directement la nécessité de tensorialisation**
H(θ) plutôt que simple raffinement scalaire.

**Conséquence pour la lecture Ch4** : si 6d-γ détecte de la
non-commutativité robuste, le discriminant Ch4 devient probablement
"non-commutativité" plutôt que "anisotropie forte". Ce n'est pas une
conclusion à tirer en 6d-γ — c'est une indication empirique à reporter
en Ch4.

### Lecture théorique secondaire (C audit — tension enregistrée)

**Hypothèse à enregistrer, pas à valider** : si la non-commutativité
s'avérait observable robustement sous h(θ) scalaire en 6d-γ, elle
suggérerait que Ch4 se reposerait comme **théorie d'une algèbre
locale de transformations morphologiques** plutôt que comme **théorie
d'anisotropie tensorielle**.

Différence de nature :
- Ch4 "géométrique tensorielle" : H(θ) matrice symétrique positive
  capturant l'anisotropie directionnelle au point θ. Approche
  géométrie riemannienne classique.
- Ch4 "algébrique" : structure de groupe ou groupoïde de
  transformations morphologiques non-commutatives. Approche
  algébrique des dynamiques avec mémoire de trajectoire.

Ces deux lectures **ne sont pas exclusives** — une théorie tensorielle
avec courbure non-triviale produit naturellement de la
non-commutativité (transport parallèle dépendant du chemin). Mais
elles **diffèrent de motivation** : la première formalise
l'hétérogénéité directionnelle ; la seconde formalise une algèbre
locale.

**Caveats avant toute promotion de cette lecture** :

La non-commutativité observée pourrait provenir de :
- hétérogénéité spatiale forte de h(θ)
- discrétisation (effets de stencil non commutatifs)
- non-linéarités de coupling
- garde-fous (D_min, ψ_floor, clipping h)
- fallback Backward Euler asymétrique
- effets de grille corner cells
- **pseudo-hystérésis discrète** (la dynamique discrète peut produire
  des pseudo-non-commutativités par chemin d'arrondi numérique)

Toutes ces sources peuvent produire une non-commutativité observable
**sans** que la dynamique sous-jacente Ch3 soit non-commutative au
sens algébrique. Discriminer ces sources demande des protocoles
spécifiques qui ne sont **pas** dans 6d-γ.

**Statut épistémique strict** :

| Statut | Position |
|---|---|
| Tension théorique enregistrée | **Oui** |
| Hypothèse pour discussion Ch4 | **Oui** |
| Relecture canonique de Ch4 | **Non** |
| Repositionnement de Ch4 | **Non** |

La non-commutativité reste **un observable exploratoire 6d-γ**, pas
une exigence 6d-β, pas une relecture canonique de Ch4. La frontière
est importante : 6d-β doit rester centré sur les propriétés
explicitement revendiquées par Ch3 (propriétés 1-4 + cross-talk
métrique §3.2 comme phénomène structurel induit). La
non-commutativité reste encore :
- une hypothèse émergente
- un possible plafond scalaire
- une tension interprétative
- **pas une propriété Ch3 verrouillée**

---

## 8. Distinctions structurelles à préserver de 6c-B

Hérités du mémo de transition, à maintenir actifs :

- **Modulation ≠ contraction** : si T25'-équivalent montre encore
  modulation/redistribution, la lecture est *modulation persiste sous
  h(θ) plein*, pas *contraction réelle*.
- **Redistribution interne ≠ projection τ'** : la morphological latency
  (test 2) est précisément la formalisation de cette tension — c'est
  une propriété **prédite** par Ch3, pas un défaut.
- **G_proxy ≠ 𝒢 plein** : G_proxy = ψ-roughness reste un proxy. Le
  vrai 𝒢 = ‖∂τ'/∂Γ_meta‖ devient mesurable en 6d **si** Γ_meta est
  tracé explicitement comme accumulation (∫𝔊^{sed}). À implémenter.
- **LOW_REST n'est pas un KNV** : Δ_centred bas en repos centré reste
  STR-compatible. Le canal anti-pétrification (g_Ω canal 2) doit
  distinguer LOW_REST de pétrification réelle (∂_t h bloqué).
- **Δ_shape vs Δ_centred** : tension préservée. Sous h(θ) plein, Δ_shape
  a-t-il un comportement différent de 6c-B ? À mesurer.

---

## 9. Ce qui ne doit plus être inventé en 6d

Tentation à éviter (rappel issu du brainstorm session 12) :

- ❌ Inventer un ETH/VCH minimal — Ch3 §3.1.1 + §3.5.2 + §3.4 le posent
  déjà.
- ❌ "Rendre Γ_meta active" — Ch3 §3.1.4 + §3.1.5 la définissent comme
  active.
- ❌ Inventer une famille de τ'_ref proxies — Ch3 §3.6.8 spécifie 𝒞_T
  et τ_meta comme proxies non-instanciables.
- ❌ Ajouter une couche D "RR/RR²/RR³ agissent" — Ch3 §3.6.10 dit
  explicitement qu'ils ne sont pas des opérateurs mais des lectures
  multi-échelles. Aucun agent.
- ❌ "Optimisation" sous quelque forme que ce soit. Pas de loss, pas de
  cible, pas de score. Le système maintient. Il ne converge pas.

---

## 10. Première étape concrète à décider

Le passage h marginal → h(θ) plein change l'engine substantiellement :

- Le tableau d'état devient `psi[5,5,5]` + `h[5,5,5]` par module au lieu
  de `psi[5,5,5]` + `h_T[5], h_M[5], h_I[5]`.
- Le Laplacien conformal-conservatif sur grille 5×5×5 demande un schéma
  numérique adapté (volumes finis avec flux conservatifs et facteurs
  `h^{d-2} = h¹` aux interfaces). Voir numerics-doc §1.2 pour la
  spécification complète.
- 𝔊^{sed} et 𝔊^{ero} deviennent locales en θ (pas marginales).

C'est essentiellement un **rewrite partiel de l'engine** plutôt qu'un
ajout incrémental. À planifier comme refactor 6d-α avant tout test.

---

*Cadrage 6d, à relire avant chaque session 6d. Les 4 tests émergents
sont le programme 6d-β. La carte de réalisabilité enrichie par les
observables exploratoires C3 (non-commutativité) est l'output 6d-γ.
Aucun verdict sur MCQ. Aucune cible. Conformal-conservatif ou rien.*
