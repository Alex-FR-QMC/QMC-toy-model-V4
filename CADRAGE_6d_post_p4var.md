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

| Propriété | Observée sous h(θ) plein (6d-β cycle 2) | Forme effective émergente |
|---|---|---|
| 1. Dual-timescale memory | Non, sous forme attendue (τ_h≫τ_ψ) | Verrouillage géométrique hérité |
| 2. Morphological latency | Non, sous forme attendue (retard ψ→h) | Transition de régime de couplage (t≈3-5) |
| 3. Productive forgetting | Non, sous forme attendue (effacement) | Fermeture progressive du cône de transformabilité |
| 4. Self-opacity | Non, sous forme attendue (opacité globale) | Opacité **anisotrope** de la reconstructibilité |
| 5. Interferential 𝒢 enrichment | **Hors 6d** | Exige N>1 |

Aucune des 4 propriétés testables N=1 n'apparaît sous sa forme Ch3
attendue. Les 4 produisent à la place un objet de la même famille :
**la dynamique d'accessibilité morphologique sous verrouillage**.

Cette carte est l'output de 6d. Elle informe Ch4 par le bas — empiriquement,
pas spéculativement. Détail complet : voir `6d-beta-validation-report.md`
et `6d-beta-cycle-2-mini-cadrage-transversal.md` (documents compagnons).

---

## 7ter. Contraintes émergentes du cycle 6d-β (provisoire, révisable)

**Statut** : inscription minimale des contraintes émergentes du cycle 2.
Provisoire et révisable. Le détail (protocoles, valeurs, chronologie,
caveats techniques) reste dans les documents compagnons et n'est pas
absorbé ici.

**1. Dissociation opératoire des objets**. Sous h(θ) scalaire plein,
cinq objets que Ch3 §3.6.11 laissait implicitement co-localisés se
dissocient empiriquement : stabilité, mémoire, transformabilité,
accessibilité, reconstructibilité historique. Leurs valeurs évoluent
indépendamment.

**2. Verrouillage morphologique comme pathologie de viabilité**. Le
système ne perd pas sa viabilité par instabilité mais par verrouillage
morphologique progressif : stabilité/persistance/mémoire ↑ pendant
qu'accessibilité/transformabilité/reconstructibilité ↓, sans signal au
niveau stationnaire. Formulation paradigmatique : *sous certaines
géométries opératoires, la persistance locale peut devenir
anti-corrélée à la réactivabilité globale*. Cette tension résonne avec
KNV, 𝒢(t), la réactivabilité minimale, RTS vs RSR, et la distinction
MCQ stabilité/viabilité. 6d-β ne l'a pas inventée — il l'a rendue
mesurable.

**3. Pression structurale vers H(θ) (asymétrie à préserver)**. 6d-β
**ne démontre pas** que H(θ) tensoriel résout la dissociation. 6d-β
montre que h(θ) scalaire produit une **pression structurale récurrente**
vers une structure plus riche. Ces deux énoncés ne sont pas symétriques
et ne doivent pas être collapsés : la pression est un fait empirique,
la résolution par le tensoriel est une hypothèse non testée.

**4. Résidu numérique dissous**. Le ratio Δh/Δψ ≈ 5.3666 (observé
stable sur plusieurs protocoles) est une coordonnée lisse du régime
structuré, linéaire en β (pente ≈ 0.0894/unité), indépendante du schéma
temporel. Pas un invariant universel ni un artefact.

**5. Cohérence transversale non présupposée**. Le cadre interprétatif
"dynamique d'accessibilité morphologique sous verrouillage" a émergé
par confrontation des propriétés, pas par programmation préalable. Ce
processus de co-production théorie↔opérativité fait partie du résultat
et ne doit pas être relu rétroactivement comme contenu dès l'origine.

**Garde-fou** : cette inscription ne doit pas devenir orthodoxie de
transition vers 6d-γ. Les lectures 6d-β peuvent se révéler partielles
ou localement vraies seulement. Le mini-cadrage reste une structure
réflexive révisable, pas un ensemble d'axiomes.

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

## 11. Amendements post-6d-α

**Statut** : section ajoutée à la clôture provisoire 6d-α (session 28).
**Principe** : préserver le cadrage initial (§1-10) comme historique
des hypothèses, et inscrire ici les tensions empiriques apparues que
le cadrage initial ne pouvait pas anticiper. La discipline retenue
est de **laisser les observations contraindre rétroactivement le
cadrage** plutôt que de forcer le moteur à confirmer la théorie a
priori.

Référence : `6d-alpha-validation-report.md` couches 2 (observé) et
4 (tensions ouvertes), `6d-alpha-numerics.md` §5.8 et §6.6.

### 11.1 Acquis empiriques 6d-α utiles pour 6d-β

Le programme 6d-α a produit des observables que le cadrage initial ne
listait pas. Elles doivent être intégrées au programme 6d-β :

| Observable empirique | Source | Statut |
|---|---|---|
| Mémoire morphologique étendue `corr(-log h, ∫ψdt)` | §4a-δ | observable trajectorielle non-tautologique |
| Feedback ψ↔h sur dispersion (COM dépendant du régime) | §4a-ε | observable de feedback dynamique |
| Stratification accès / réactivabilité / transformabilité | §4a-η | distinction structurelle |
| Bifurcation locale `β·ψ_local/γ` au moment de revisite | §4a-η | mécanisme de verrouillage |
| Multi-bassins empiriques (A/B1 même attracteur, B2 différent) | §5.7 | dépendance au chemin |
| Géométrie de convergence (AUC_Dh) | §5.7 | observable trajectorielle |
| Bassin absorbant empirique sous perturbation bornée | §6.2 | test de transformabilité dynamique |
| Distinction `ENGINE_GUARANTEE` vs `REGIME_VIABILITY` | §6 | discipline méthodologique |

### 11.2 Propriété 5bis — structure de bassin et dépendance au chemin

Le cadrage §3.1 listait 4 propriétés émergentes Ch3 (dual-timescale,
latency, productive forgetting, self-opacity). 6d-α a révélé une
**cinquième propriété structurelle empirique** qui n'était pas
anticipée explicitement par le cadrage initial :

| # | Propriété émergente | Source empirique | Instrumentation requise | Critère falsifiable |
|---|---|---|---|---|
| 5bis | **Structure de bassin et dépendance au chemin** | §5.7 (multi-bassins) + §6.2 (perturbation bornée) | Famille A + famille B avec initialisations différentes ; mesure de `D_h(t)` trajectoriel et `AUC_Dh` ; test de perturbation bornée sur bassins quasi-stationnaires | Différentes initialisations atteignent attracteurs distincts (multi-bassins) OU mêmes attracteurs par géométries de convergence mesurablement différentes (`AUC_Dh` non négligeable même quand `D_h_final → 0`) |

**Statut** : propriété structurelle empiriquement observée en 6d-α,
**à confirmer en 6d-β** sous conditions plus riches (h(θ) variable
spatialement, drift, couplage 𝒞^N éventuel).

Cette propriété n'est pas une revendication Ch3 canonique au sens
strict, mais elle est **mécaniquement reliée** à des concepts Ch3 :
- bifurcation locale ↔ mécanisme de KNV local
- dépendance au chemin ↔ trajectoire de Γ_meta
- bassin absorbant empirique ↔ candidat préliminaire pour modulation
  de transformabilité (𝒢 plein)

Le risque méthodologique principal est de **sur-ontologiser** ces
bassins. Voir §11.5.

### 11.3 Reformulation propriété 5 (cross-talk métrique)

Le cadrage §3.2 propose de tester le cross-talk métrique par
comparaison `g_T` sous `h(θ) plein` vs `h artificiellement diagonal`.

**Acquis empirique partiel** : §5.7 fournit une **préfiguration**
empirique du cross-talk métrique :
- mémoire morphologique de h(t) non-séparable (l'histoire de ψ
  modifie h, qui modifie la diffusion future)
- multi-bassins empiriques selon initialisation
- bassin absorbant local sous perturbation bornée

**Mais §5.7 n'épuise pas le test** parce que :
- h_init est uniforme dans tous les tests 6d-α (la variation spatiale
  de h émerge uniquement de la dynamique, pas de l'initialisation)
- pas de comparaison directe `h(θ) plein` vs `h projeté diagonal`
- pas de mesure de g_T sous métrique anisotrope explicite

**Reformulation** :

> §5.7 préfigure le cross-talk métrique sans encore l'épuiser. La
> non-séparabilité historique (via accumulation `h(t)`) est empiriquement
> validée, mais le couplage métrique plein au sens fort (h(θ)
> spatialement anisotrope dès l'initialisation, comparaison h plein
> vs h diagonal) reste à tester en 6d-β.

### 11.4 Distinction "observable trajectorielle" vs "observable stationnaire"

Le piège §5 (tautologie au point fixe) doit être systématiquement
évité pour toutes les propriétés 1-4 + 5bis.

Pour chaque propriété, le programme 6d-β doit **explicitement**
spécifier si la mesure est :
- **stationnaire** : mesure dans un état quasi-stationnaire couplé.
  Risque tautologique élevé si la propriété se ramène à une identité
  algébrique au point fixe.
- **trajectorielle** : mesure sur la dynamique de convergence ou
  transition. Risque tautologique faible.

**Table de classification à compléter en début de 6d-β** :

| Propriété | Mesure stationnaire ? | Mesure trajectorielle ? | Risque tautologique |
|---|---|---|---|
| 1. Dual-timescale memory | au point fixe, h_∞ = h₀·(1-β·ψ_∞/γ) trivial | `∫𝔊^{ero}·dt` accumulé sur cycles répétés | trajectorielle dominante → faible |
| 2. Morphological latency | non pertinent (latence = phénomène dynamique) | lag entre `τ'_k` et h(θ) pendant contraction | trajectorielle native → faible |
| 3. Productive forgetting | non pertinent | cascade post-érosion | trajectorielle native → faible |
| 4. Self-opacity | reconstructibilité depuis 𝒫 à t fixé | reconstructibilité depuis 𝒫(t) sur trajectoire | à clarifier |
| 5bis. Structure de bassin | bassins comme attracteurs | géométrie de convergence `AUC_Dh` | trajectorielle dominante → faible |

### 11.5 Anti-tautology guardrail (règle méthodologique de premier ordre)

> **Toute observable validée à stationnarité doit être accompagnée
> d'une mesure trajectorielle indépendante.**

Cette règle s'applique à **toutes** les futures observables 6d-β :
- `𝒞_T(t)` : mesure trajectorielle native, pas de risque immédiat,
  mais à comparer avec `𝒞_T(stationnaire)` qui pourrait être
  trivialement nul
- `τ_meta` : par construction trajectorielle (détection de stabilité
  fausse), mais à vérifier qu'elle ne se ramène pas à un compteur
  d'éloignement au point fixe
- `𝒢 = ‖∂τ'/∂Γ_meta‖` : à mesurer sur trajectoire, pas à l'équilibre
  où ∂τ'/∂Γ_meta peut être tautologiquement nul
- `RR`, `RR²` : déjà non-opérateurs (Ch3 §3.6.10), mais leur lecture
  multi-échelle doit être trajectorielle
- `β_QMC = ⟨‖∇_{dᵢ^ℋ}Φ_eff‖⟩ / D_eff` : à mesurer trajectoriellement,
  l'équilibre `Φ_eff = 0` rendrait cette observable triviale

**Procédure** : avant de coder le test d'une nouvelle observable,
répondre à trois questions :

1. Cette observable est-elle satisfaite par construction au point
   fixe couplé du système d'ODE ? Si oui, c'est tautologique.
2. Cette observable change-t-elle entre familles A/B/C
   d'initialisations différentes au même bassin ? Si non, elle
   n'apporte rien de plus que la cohérence locale.
3. Cette observable distingue-t-elle des trajectoires qui aboutissent
   au même état final ? Si oui, elle capture la géométrie de
   transformation.

Seule une réponse positive à (3) garantit un observable non-tautologique
informatif.

### 11.6 Intégration de la distinction accès / réactivabilité / transformabilité

L'instrumentation 𝒞_T plein doit intégrer la distinction structurelle
de §4a-η :

- **ACCÈS** : ψ_local atteint la zone (mesurable par `ψ_mass_local`)
- **RÉACTIVABILITÉ MORPHOLOGIQUE** : h peut remonter au-dessus du
  seuil de résolution (`h > H_RESOLUTION`)
- **TRANSFORMABILITÉ EFFECTIVE** : capacité de transport restaurée
  (`h > H_FUNCTIONAL` ET `β·ψ_local/γ < 1` ET capacité de transformation
  mesurable)

**Conséquence pour 𝒞_T plein** : un `𝒞_T(t)` qui ne distingue pas
ces trois niveaux peut mesurer une "stabilité morphologique" trompeuse.
Par exemple, une cellule à `h = 10⁻²⁰` est numériquement "active"
(au-dessus du float floor strict), mais sa contribution à `𝒞_T` au
sens géométrique conforme est négligeable.

`𝒞_T` doit donc être instrumentée avec **double mesure** :
- `𝒞_T_resolution` : distance dans la métrique conforme sur cellules
  `h ≥ H_RESOLUTION`
- `𝒞_T_functional` : distance dans la métrique conforme sur cellules
  `h ≥ H_FUNCTIONAL`

Si les deux diffèrent significativement, le système est dans un régime
de **léthargie morphologique partielle** (cf. §4a-ζ).

### 11.7 Shift conceptuel principal post-6d-α

Le vrai changement de perspective post-6d-α est :

> **Les objets pertinents ne sont plus principalement des états
> stationnaires, mais des géométries de transformation.**

Plus précisément :
- **bassins** d'attraction et leurs frontières
- **transitions** entre bassins (réactivation, fragmentation)
- **réactivabilité conditionnelle** (locale, sous perturbation, etc.)
- **AUC trajectorielles** mesurant le chemin de convergence
- **dépendance au chemin** entre initialisations différentes
- **stabilité sous perturbation** (test de transformabilité dynamique)

Ce shift rapproche le moteur de l'esprit MCQ (dynamique de
transformations sans cible) plutôt que d'un système relaxatif vers
équilibre. Mais **il faut éviter de sur-ontologiser** ce qu'on
observe :

- **multi-bassins empiriques** ≠ **multi-attracteurs MCQ
  structurels**
- **bassin absorbant empirique** ≠ **bassin absorbant structurel**
- **mémoire trajectorielle** ≠ **téléonomie émergente**

La règle reste : 6d teste la **réalisabilité conformal-conservative**
des propriétés émergentes Ch3, pas la validité de MCQ.

### 11.8 Programme 6d-β recadré

Conséquences pour le programme 6d-β :

1. **Avant tout nouveau code** : compléter la table §11.4 avec mesure
   stationnaire vs trajectorielle pour chaque propriété 1-4 + 5bis.

2. **Premiers tests** : observables dérivées de phénomènes déjà
   observés en 6d-α (transitions de bassin, réactivation conditionnelle,
   fermeture locale, mémoire trajectorielle, géométrie de convergence).
   Ne **pas** projeter immédiatement RR/RR², 𝒢, 𝕋(t) comme
   couches "fortes" — laisser émerger leur nécessité.

3. **Tests reportés à plus tard 6d-β ou 6d-γ** :
   - sweep ε_ψ pour caractériser seuil de réactivation
   - bruit stochastique comme mécanisme réactivateur indépendant
   - drift redistributif (h(θ) avec Φ ≠ 0)
   - couplage inter-modulaire 𝒞^N (hors 6d strict)

4. **Multi-seed dès le début** (§6.6 du cadrage initial préservé) :
   `{42, 123, 2024}`. Particulièrement important pour la propriété
   5bis (les bassins peuvent dépendre subtilement du seed dans les
   régimes proches de la frontière).

5. **Garde-fou anti-tautology** §11.5 appliqué à chaque nouvelle
   observable, sans exception.

6. **Préserver la discipline 6d-α** : mini-spec avant code, audit
   à chaque étape, distinction `ENGINE_GUARANTEE` vs `REGIME_VIABILITY`,
   caveat strict sur "empirique vs structurel".

---

*Fin de §11. Le cadrage 6d initial (§1-10) reste valide à 80%.
Les amendements §11 intègrent les tensions productives apparues en
6d-α. Aucun verdict sur MCQ. Aucune cible. Conformal-conservatif ou
rien. Les géométries de transformation sont désormais l'objet central.*

---

*Cadrage 6d, à relire avant chaque session 6d. Les 4 tests émergents
sont le programme 6d-β. La carte de réalisabilité enrichie par les
observables exploratoires C3 (non-commutativité) est l'output 6d-γ.
Aucun verdict sur MCQ. Aucune cible. Conformal-conservatif ou rien.*

---

## 12. Amendements post-6d-γ

**Statut** : inscription minimale des contraintes émergentes du cycle 3.
Provisoire et révisable. Le détail (protocoles, perturbations P1-P6,
chronologie des onze contacts, séquence des lectures successives,
caveats techniques) reste dans les documents compagnons et n'est pas
absorbé ici.

### 12.1 Chronologie réelle du cycle (à préserver)

Le cycle 6d-γ s'est ouvert sur la question du plafond scalaire et sur
le discriminant exploratoire entre anisotropie et non-commutativité
morphologique. Il n'a validé ni l'une ni l'autre de ces lectures.

En revanche, il a produit plusieurs contraintes empiriques qui semblent
davantage compatibles avec une dynamique où **les trajectoires portent
plus d'information que les attracteurs eux-mêmes**.

Ce déplacement n'était pas anticipé. Il a émergé progressivement
pendant l'exploration du plafond scalaire, à mesure que les pistes
"anisotropie forte stabilisée" et "nouvelles coordonnées conservées"
se sont successivement refermées. Préserver cette chronologie est
important : le caractère **inattendu** de ce qui a été trouvé donne
du poids à la contrainte obtenue, et empêche de relire rétrospectivement
le cycle comme une marche dirigée vers son résultat.

### 12.2 Contraintes empiriques verrouillées (Niveau 1)

Trois contraintes empiriques sont désormais établies pour le régime
verrouillé β=60 et devront rester respectées par toute future lecture
du plafond scalaire ou du discriminant de tensorialisation,
indépendamment du cadre théorique adopté :

**(a) Persister ≠ être conservé.** Un mode peut décroître lentement
(τ ~ 50 sur P6) sans être un invariant asymptotique. À t=200, un mode
géométrique dipolaire représentait ~2% du résidu de P6 ; à t=3000,
il est dissipé à 10⁻¹³ près. La distinction "persistance transitoire"
vs "conservation asymptotique" est empiriquement nécessaire pour
décrire le régime — elle ne se réduit pas à un garde-fou méthodologique.

**(b) La géométrie de la perturbation agit davantage sur les
trajectoires que sur l'attracteur.** Six perturbations qualitativement
distinctes (centrée, périphérique, anisotrope, bipolaire radiale,
voisines isolées, dipôle de faces) convergent asymptotiquement vers
un résidu essentiellement déterminé par une seule coordonnée scalaire
(Δψ_centre ou son équivalent linéaire). Les différences de trajectoire
sont en revanche fortes, structurées géométriquement, et persistantes
sur des échelles de temps qui peuvent excéder d'un facteur 10 celles
du mode dominant.

**(c) Attracteur réduit ≠ dynamique pauvre.** Le régime verrouillé
β=60 possède un attracteur fortement compressif (essentiellement
1 scalaire conservé sur les six perturbations testées), mais cette
compression ne préjuge pas de la richesse dynamique du régime. Cette
distinction est plus structurelle que (a) et (b) : elle dit quelque
chose sur **comment lire un régime** au-delà de ce cycle particulier.
Inférer la pauvreté dynamique à partir de la dimension de l'attracteur
est une erreur de lecture que ce cycle a rendue mesurable.

### 12.3 Lectures compatibles non démontrées (Niveau 2)

Les contraintes du Niveau 1 sont compatibles avec plusieurs lectures
théoriques. Aucune n'est démontrée par le cycle, mais aucune n'est
exclue. Ces lectures restent ouvertes pour des cycles ultérieurs :

- Possible manifestation d'une dynamique de **régulation réflexive**
  (RR/RR² au sens MCQ) : le système autoriserait temporairement certaines
  informations géométriques avant de les réabsorber. Cohérent avec
  l'écart persistance/conservation, mais une dissipation purement
  passive d'un mode propre lent du laplacien linéarisé rend également
  compte des données.

- Possible signature observable du **gradient transformable 𝒢** :
  le gradient transformable serait davantage corrélé à la richesse des
  trajectoires admissibles qu'à celle des attracteurs accessibles.
  Hypothèse devenue plus crédible à la lumière du cycle, sans avoir été
  testée comme telle.

- Possible expression de la **non-commutativité morphologique**
  évoquée dans le cadrage initial (§11.7) : si la mémoire est dans le
  chemin, l'ordre des transformations devrait modifier les
  trajectoires intermédiaires sans nécessairement modifier l'état
  asymptotique. Le rapprochement est plausible mais 6d-γ teste des
  perturbations uniques, pas des compositions.

### 12.4 Question ouverte explicite (Niveau 3)

Pour les cycles suivants :

> **Les modes lents observés sont-ils uniquement des propriétés
> spectrales passives du régime, ou constituent-ils les premières
> signatures observables d'une dynamique de transformabilité plus
> profonde ?**

Cette formulation préserve la tension entre les deux pôles sans la
refermer. Elle laisse au cycle suivant la liberté de tester l'un ou
l'autre, ou de découvrir que la dichotomie elle-même est mal posée.

### 12.5 Résultat méthodologique du cycle

Le cycle 6d-γ n'a pas principalement produit de nouveaux invariants.
Il a renforcé la capacité à **distinguer un mode lent transitoire
d'un invariant asymptotique**. Cette distinction est à la fois :

- un **résultat empirique** (12.2.a)
- une **discipline méthodologique** stabilisée pendant le cycle

La discipline ne se capitalise pas. Elle se reprend à chaque cycle.
Plusieurs fois pendant 6d-γ, des conclusions intermédiaires ont été
posées ("nouvelle coordonnée conservée", "fonctionnelle linéaire
indépendante") puis corrigées par un test ultérieur. Le cycle ne
peut être lu comme une marche directe vers son résultat — il est
fait des corrections successives qui l'ont stabilisé.

### 12.6 Cohérence transversale avec 6d-β

La dissociation §7ter.1 (stabilité, mémoire, transformabilité,
accessibilité, reconstructibilité) portait sur des **propriétés
d'états**. La distinction 12.2.c (attracteur ≠ dynamique) porte
sur la **relation entre états et trajectoires**. Les deux observations
sont du même type — distinctions empiriquement nécessaires que la
pensée habituelle tendrait à confondre — mais elles vivent dans des
registres différents et ne doivent pas être fusionnées en une liste
unique. Le rapprochement est légitime ; la fusion serait prématurée.

**Garde-fou** : cette inscription ne doit pas devenir orthodoxie de
transition vers les cycles suivants. Les lectures 6d-γ peuvent se
révéler partielles ou localement vraies seulement. Le cadrage reste
une structure réflexive révisable, pas un ensemble d'axiomes.

---

*Fin de §12. Le cadrage 6d initial (§1-10) reste valide à 80%. Les
amendements §11 intègrent les tensions productives apparues en 6d-α.
Les amendements §7ter intègrent les contraintes émergentes de 6d-β.
Les amendements §12 intègrent les contraintes émergentes de 6d-γ.
Aucun verdict sur MCQ. Aucune cible. Conformal-conservatif ou rien.
Les trajectoires de transformation sont désormais l'objet d'observation
distinct des attracteurs.*

---

## 13. Amendements post-6d-δ

**Statut** : inscription minimale du cycle 6d-δ (Fenêtre de
réactivabilité sous mode lent). Provisoire et révisable. Le détail
(cadrage 6d-δ, calibration P′, premier passage invalide,
patch χ_slow, décomposition amplitude/résidu, extension Δt,
test d'universalité, vérification de robustesse temporelle) reste
dans les documents et scripts compagnons et n'est pas absorbé ici.

### 13.1 Chronologie réelle du cycle

Le cycle 6d-δ s'est ouvert sur la tension résiduelle laissée par §12 :
les modes lents observés en 6d-γ sont-ils des propriétés spectrales
passives, ou portent-ils une transformabilité effective transitoire ?

Le cadrage 6d-δ a préinscrit trois hypothèses (H0 passif, HA1
réactivant, HA2 verrouillant) et une quantité opératoire χ(τ) destinée
à les discriminer.

Le premier passage avec χ référencé à E₀ s'est révélé invalide :
le contrôle interne χ(τ=3000) ≈ 0 supposait que P6(3000) ≈ E₀, ce
que §12.2.a ne disait pas (§12 disait que l'extra dissipe, pas que
P6 revient à E₀). Le patch χ_slow référencé à P6(3000) a corrigé
ce problème de définition.

La caractérisation s'est ensuite déplacée du verdict ternaire vers
une analyse fine de la structure de χ_slow, sortant progressivement
du périmètre initial du cadrage 6d-δ. Préserver cette chronologie
est important : ce qui a été appris post-verdict n'est pas la
réponse à la question préinscrite, mais une caractérisation
inattendue de l'objet mesuré.

### 13.2 Couche 1 — Résultat du cycle 6d-δ

**Verdict préinscrit prononcé** : INTERMÉDIAIRE. χ_slow non trivial
sur la fenêtre lente (τ ∈ {0, 50, 100, 200, 400}), mais les ratios
AUC réponse / AUC référence à τ=3000 sont tous compris entre 1.0000
et 1.0029 — sous le seuil HA1 (>1.10) et au-dessus du seuil HA2 (<0.90).

**Caractérisation post-verdict** :

- χ_slow décroît continûment sur la fenêtre 0–400, atteint le bruit
  numérique à τ=400 et est nul à τ ≥ 1500. Signature temporelle
  cohérente avec la dissipation du mode lent observée en 6d-γ.
- χ_slow n'est pas réductible à une modulation d'amplitude globale :
  la décomposition Δ_τ = a_τ · Δ_3000 + r_τ montre que la composante
  résiduelle vectorielle r_τ domine la composante d'amplitude à tous
  les τ de la fenêtre lente (ratio R/A ≥ 4 à τ=0, croissant ensuite).
- À Δt long (Δt ∈ {200, 400, 800}), la norme du résidu sature à
  ~6.0e-4 et sa direction converge (cos finals successifs : 0.993
  → 0.999 → 1.000). Le résidu a une structure géométrique stable.

**Conséquence** : la catégorie INTERMÉDIAIRE capture le résultat
du cadrage initial mais ne suffit pas à décrire la structure
effectivement observée. χ_slow détecte une composante vectorielle
résiduelle stable, dominée par une signature non-amplitudinale,
sans correspondre à aucune des trois hypothèses préinscrites.

### 13.3 Couche 2 — Résultat post-verdict

L'exploration de l'universalité du résidu (test avec quatre
variantes de P' : gaussienne étroite, standard, large, et annulaire,
toutes calibrées à 10% de ||P6||) a produit un résultat inattendu :

- Aucun secteur du résidu n'est robustement universel. La trace
  temporelle moyennée signée `mean(r_psi)` apparaissait universelle
  (cos > 0.995) mais cette universalité est un artefact d'annulation
  imposé par la conservation de la masse (mean_signed/mean_abs ≈ 10⁻⁸).
  Sur les observables non sensibles au signe (mean_abs, norm spatiale),
  l'universalité disparaît.

- Les observables robustes se regroupent dans une bande intermédiaire
  récurrente : cos minimaux entre variantes systématiquement dans
  l'intervalle 0.73–0.95 pour six observables indépendantes
  (ψ_spatial_fluct, h_spatial_fluct, ψ_temp_mean_abs, ψ_temp_norm,
  h_temp_mean_abs, h_temp_norm).

- Cette convergence n'est pas un cas isolé : plusieurs observables
  indépendantes tombent dans le même ordre de grandeur, sans qu'il y
  ait de raison a priori que ce soit le cas.

**Fait empirique stabilisé** : il existe un corridor intermédiaire
récurrent entre universalité et dépendance dans les réalisations
du résidu r_τ=0 à Δt long, observable sur plusieurs grandeurs
indépendantes mesurant la structure géométrique ou dynamique du
résidu.

### 13.4 Couche 3 — Question ouverte

L'origine du corridor intermédiaire récurrent reste à élucider.

Trois lectures sont compatibles avec les données actuelles :

- **(L1)** Propriété structurelle du résidu : il existerait une
  "quantité" partagée entre variantes et une "quantité" spécifique
  à P', dont le ratio fixerait le corridor.
- **(L2)** Effet du calibrage commun : la contrainte d'amplitude
  identique (10% de ||P6||) forcerait un alignement partiel
  indépendamment de la structure interne.
- **(L3)** Effet de l'espace de mesure : la taille de la grille
  (5×5×5) et la fenêtre Δt fixée imposeraient des cos minimaux
  ~0.7 indépendamment du contenu.

Ces trois lectures ne sont pas testées par 6d-δ. Aucune n'est
privilégiée. Tant que cette origine n'est pas identifiée, il serait
prématuré de caractériser la structure du corridor lui-même.

**Question ouverte pour un cycle ultérieur éventuel** :

> Le corridor intermédiaire récurrent observé est-il une propriété
> du contenu mesuré, du protocole de calibration, ou de l'espace de
> mesure ? Avant cette discrimination, sa caractérisation structurelle
> reste prématurée.

### 13.5 Résultat méthodologique

Le cycle 6d-δ a illustré, à plusieurs reprises, le pattern
méthodologique stabilisé en 6d-γ : une lecture intermédiaire est
posée, un contrôle simple invalide ou nuance, la formulation se
déplace. En particulier :

- Le premier passage avec χ référencé à E₀ a été posé comme test
  principal, puis invalidé par la lecture attentive de §12.2.a.
- La signature "directionnelle x" du résidu observée à Δt=100 a
  été rétro-invalidée par l'extension à Δt=800.
- L'"universalité temporelle" prononcée a été rétro-invalidée par
  le test de robustesse mean_signed vs mean_abs.

Ces trois corrections successives ne sont pas des défauts du cycle.
Elles sont constitutives de la méthode : poser une lecture, la
soumettre à un contrôle qui la teste, accepter le déplacement quand
le contrôle invalide. La discipline ne se capitalise pas.

**Garde-fou** : cette inscription ne doit pas devenir orthodoxie de
transition vers les cycles suivants. L'existence du corridor
intermédiaire est un fait empirique ; son interprétation reste ouverte.
Aucune identification directe à 𝒢, RR/RR², ou à toute structure MCQ
ne se déduit du présent cycle. Le cadrage reste une structure
réflexive révisable.

---

*Fin de §13. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11 intègrent 6d-α. Les amendements §7ter intègrent
6d-β. Les amendements §12 intègrent 6d-γ. Les amendements §13
intègrent 6d-δ. Aucun verdict sur MCQ. Aucune cible.
Conformal-conservatif ou rien. Les trajectoires de transformation
sont l'objet d'observation distinct des attracteurs (§12). L'existence
d'un corridor intermédiaire récurrent entre universalité et dépendance
des résidus est désormais une contrainte empirique à laquelle toute
lecture future devra rester compatible (§13).*

---

## 14. Amendements post-6d-ε

**Statut** : inscription minimale du cycle 6d-ε (Contrôle L3-simple
du corridor intermédiaire). Provisoire. Le détail (mesure τ_c,
trois versions du test avec corrections successives, distributions
de min cos par tirage) reste dans les documents et scripts
compagnons et n'est pas absorbé ici.

### 14.1 Objet du cycle

§13.4 laissait L1/L2/L3 comme lectures compatibles non testées de
l'origine du corridor intermédiaire récurrent observé en 6d-δ
(cos hors diagonale dans la bande 0.73-0.95 sur six observables
indépendantes).

Le cycle 6d-ε a testé L3 sous sa forme la plus simple : le corridor
est-il banal dans l'espace de mesure sous les contraintes de
conservation de masse (pour ψ) et de continuité temporelle ?

### 14.2 Protocole

Trois niveaux de contrôle synthétique, N=200 tirages chacun, mêmes
six observables que §13 :

- **C1** : bruit gaussien centré, aucune contrainte
- **C2** : C1 + conservation de masse à chaque t (pour ψ uniquement ;
  h non contraint)
- **C3** : C2 + continuité temporelle AR(1) à trois longueurs
  caractéristiques (0.5τ_c, τ_c, 2τ_c)

τ_c mesuré par autocorrélation vectorielle ψ. Quantiles bilatéraux
(p_tail = min(p_below, p_above)) et z-scores signés. Verdict par
observable : central / queue haute / queue basse.

Correction méthodologique appliquée en v3 : la statistique de
référence est la distribution des **min cos hors diagonale par
tirage** (200 valeurs), pas la distribution agrégée des cos
individuels (1200 valeurs), puisque l'objet réel comparé est
lui-même un min hors diagonale.

### 14.3 Résultat

**L3-simple rejeté.**

0/6 observables centrales pour aucun des 5 niveaux de contrôle,
dans aucun des deux modes (cos brut, cos centré). Tous z-scores
massifs, p_tail < 0.0001 pour toutes les paires (observable, niveau,
mode).

Le corridor 0.73-0.95 n'est compatible avec :
- bruit gaussien pur ;
- bruit + conservation de masse seule ;
- bruit + conservation + continuité temporelle AR(1) dans les
  ordres de longueur de corrélation testés.

### 14.4 Caveat méthodologique important

La mesure de τ_c sur les résidus réels a révélé un fait
diagnostique : seulement 2 variantes sur 4 (étroite, annulaire)
ont atteint le seuil 1/e dans la fenêtre 400 unités. Les variantes
standard et large restent à C(lag=400) ≈ 0.45-0.50.

Cela signifie que la structure temporelle du résidu réel n'est
pas représentable par un unique temps de corrélation exponentiel
simple. Les contrôles AR(1) testés ne couvrent donc pas toutes les
structures temporelles plausibles.

**Conséquence stricte** : 6d-ε rejette L3-simple (= modèles
homogènes AR(1) dans les échelles testées), pas toute explication
par l'espace de mesure. Une structure temporelle multi-échelle
ou non-exponentielle pourrait encore produire le corridor sans
qu'il soit propriété du contenu.

### 14.5 Conséquence sur les lectures ouvertes en §13.4

- **L3 (effet de l'espace de mesure)** : rejeté dans sa forme
  simple. Une variante multi-échelle reste théoriquement possible
  mais nécessiterait un test distinct.
- **L1 (composante partagée + composante spécifique)** : devient
  la piste empirique privilégiée pour un cycle ultérieur. Cibler
  en priorité les observables temporelles, où le rejet de L3 est
  le plus net (cos synthétiques tous > 0.99 alors que cos réels ~0.77).
- **L2 (effet du calibrage commun à 10% de ||P6||)** : reste non
  testée. Sa pertinence dépend de la suite.

### 14.6 Garde-fou

Le rejet de L3-simple ne constitue **aucun argument** en faveur d'une
interprétation MCQ du corridor. Il dit seulement que le corridor n'est
pas explicable par les contrôles instrumentaux les plus simples. Aucune
identification directe à 𝒢, RR/RR², ou à toute structure du paradigme
ne se déduit du présent cycle. Le lien éventuel avec Δ, 𝒢 ou d'autres
objets ne pourra venir qu'après avoir établi une structure empirique
partagée/spécifique robuste (L1), et après avoir éprouvé d'autres
voies instrumentales si elles deviennent pertinentes.

---

*Fin de §14. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11 intègrent 6d-α. Les amendements §7ter intègrent
6d-β. Les amendements §12 intègrent 6d-γ. Les amendements §13
intègrent 6d-δ. Les amendements §14 intègrent 6d-ε. Aucun verdict
sur MCQ. Aucune cible. Conformal-conservatif ou rien. Le corridor
intermédiaire 6d-δ est désormais non explicable par L3 dans sa
forme simple. L'investigation L1 ciblée sur les observables
temporelles devient l'horizon de question le plus immédiat, à
ouvrir seulement si pertinent.*

---

## 15. Amendements post-6d-ζ

**Statut** : inscription minimale du cycle 6d-ζ (Test L1 sur les
observables temporelles). Provisoire. Le détail (décomposition
partagée/spécifique, leave-one-out, décomposition pairwise des cos
en contributions SS et ss) reste dans les documents et scripts
compagnons et n'est pas absorbé ici.

### 15.1 Objet du cycle

§14 a rejeté L3-simple : le corridor intermédiaire observé en 6d-δ
n'est pas explicable par les contrôles synthétiques bruit + conservation
+ continuité AR(1) testés.

Le cycle 6d-ζ teste L1 : le corridor vient-il d'une décomposition
partagée/spécifique entre variantes de P′ ? Test ciblé sur les
observables temporelles, où le rejet de L3 était le plus net.

### 15.2 Protocole

Pour chaque observable temporelle (psi_temp_mean_abs, psi_temp_norm,
h_temp_mean_abs, h_temp_norm), en mode brut et centré :

- Composante partagée Ō = mean_v O_v (moyenne sur les 4 variantes)
- Projection orthogonale : O_v = Shared_v + Specific_v avec
  Shared_v = a_v · Ō, a_v = <O_v, Ō>/||Ō||²
- Leave-one-out : Ō_{-v} = moyenne sur les 3 autres, comparer
- Décomposition pairwise : <O_i, O_j> = <Shared_i, Shared_j> +
  <Specific_i, Specific_j> (termes croisés nuls par construction)
- Contributions au cos : contrib_cos_SS = <Shared_i, Shared_j>/(||O_i|| ||O_j||),
  idem pour ss

Moyenne comme composante partagée, pas SVD : choix transparent,
sans hypothèse de rang 1.

### 15.3 Résultat 1 — L1 fragile mais soutenu

**Critères de robustesse satisfaits sur les 8 cas** (4 observables × 2 modes) :

- (a) f_shared > 0.5 sur les 4 variantes (minimum observé : 0.68)
- (c) cos(Ō, Ō_{-v}) > 0.95 partout (minimum : 0.983)
- (d) cos(Specific_v, Specific_v^LOO) > 0.90 partout (minimum : 0.983)

**Critère qui échoue** :

- (b) |f_shared − f_shared_LOO| < 0.05 : échoue partout
  (delta_f max varie de 0.05 à 0.14)

L'échec de (b) vient principalement de l'annulaire et de l'étroite,
dont l'inclusion ou l'exclusion modifie significativement Ō. Avec
seulement 4 variantes, le LOO est intrinsèquement sensible. Cette
fragilité peut refléter la petite taille d'échantillon plutôt qu'une
instabilité structurelle.

### 15.4 Résultat 2 — mécanisme du corridor

Décomposition pairwise des cos : par construction et orthogonalité,

> cos_total = contrib_cos_SS + contrib_cos_ss

Valeurs mesurées (moyennes sur les 6 paires hors diagonale,
moyennées sur les 8 cas) :

- <contrib_cos_SS> ≈ +0.91 (composante partagée dominante)
- <contrib_cos_ss> ≈ −0.02 (composante spécifique : contribution
  faible et négative en moyenne)

Mais la moyenne cache une structure de variance forte :

- cos(Specific_i, Specific_j) varie de **−0.99 à +0.98** selon les paires
- Paires gaussiennes proches : cos(s, s) positif (alignement)
- Paires impliquant l'annulaire : cos(s, s) négatif (antagonisme,
  jusqu'à −0.97)

Le corridor 0.73–0.95 ne vient pas seulement d'une composante partagée
dominante. Il vient d'un partagé dominant **modulé par des spécifiques
géométriquement signés**. C'est la variance de la composante ss qui
fait descendre certaines paires (étroite-annulaire) à cos ≈ 0.75 et
en laisse d'autres (standard-large) à cos ≈ 0.99.

Charges spécifiques par variante (||Specific|| / ||O||) :

- large : 0.09–0.11 (le plus central)
- standard : 0.16–0.20
- annulaire : 0.22–0.35
- étroite : 0.38–0.56 (le plus périphérique)

### 15.5 Interprétation empirique

Sur l'échantillon de quatre variantes, le corridor intermédiaire est
produit par :

1. Un mode temporel partagé dominant, présent dans toutes les variantes
2. Une composante spécifique de poids variable selon la géométrie de P′
3. Une **orientation signée** des composantes spécifiques : l'annulaire
   est antagoniste (cos négatif) aux trois gaussiennes ; les gaussiennes
   entre elles sont partiellement alignées (cos positif)

L'annulaire révèle un axe antagoniste par rapport aux perturbations
gaussiennes — résultat non anticipé par le protocole initial, qui
visait seulement la part shared/specific.

### 15.6 Caveats

- L'échantillon ne contient que 4 variantes ; le LOO est intrinsèquement
  sensible et le critère (b) échoue. La fragilité observée pourrait
  être une propriété de la taille d'échantillon plutôt que de la
  structure du résidu.
- La composante partagée est définie par moyenne arithmétique, pas par
  un objet intrinsèque (mode propre, observable conservé). Une autre
  définition pourrait donner une décomposition différente.
- La structure antagoniste annulaire/gaussiennes est observée sur une
  seule géométrie non-gaussienne. Sa généralité reste à confirmer.
- **Aucune lecture Δ, 𝒢 ou MCQ directe** ne se déduit de ce cycle.
  Le résultat reste empirique : description quantitative du corridor,
  pas explication théorique.

### 15.7 Conséquence

La suite naturelle n'est pas l'interprétation MCQ. C'est un cycle de
confirmation géométrique : élargir l'espace des perturbations P′
(plusieurs anneaux à rayons différents, shell périphérique, dipôle
radial, dipôle de faces, perturbation coin/corner, gaussiennes
décentrées, double-lobe, anisotrope oblique) pour tester si l'axe
antagoniste annulaire/gaussiennes est stable ou contingent à
l'échantillon actuel.

Si la structure antagoniste persiste sur 8 à 12 variantes diverses,
alors L1 sera empiriquement verrouillé et la question de son lien
éventuel avec un objet MCQ deviendra légitime. Sinon, il faudra
reformuler L1.

L2 (effet du calibrage commun) reste non testé. Sa pertinence dépendra
des résultats du cycle de confirmation géométrique.

---

*Fin de §15. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12/§13/§14/§15 intègrent 6d-α/β/γ/δ/ε/ζ.
Aucun verdict sur MCQ. Aucune cible. Conformal-conservatif ou rien.
Le corridor 6d-δ est désormais mécanistiquement décomposé en
contributions cos_SS dominantes et cos_ss signées géométriquement.
La généralité de l'axe antagoniste annulaire/gaussiennes reste
l'objet du cycle suivant éventuel.*

---

## 16. Amendements post-6d-η

**Statut** : inscription minimale du cycle 6d-η (Confirmation
géométrique de l'axe antagoniste) et de son audit permutation.
Provisoire. Le détail (12 variantes calibrées, matrices 12×12,
test exact sur C(9,5)=126 attributions, clustering exploratoire)
reste dans les documents et scripts compagnons.

### 16.1 Objet du cycle

§15.7 a ouvert la question : l'axe antagoniste annulaire/gaussiennes
observé en 6d-ζ est-il une structure réelle de l'espace des
perturbations P′, ou seulement un effet contingent de l'échantillon
à 4 variantes ?

6d-η teste cette généralité géométrique en élargissant à 12 variantes
réparties en trois familles : gaussiennes (G, 5 variantes), anneaux
et shells (A, 4 variantes), dipolaires et signées (D, 3 variantes).
Moteur, P6, référence P6(3000), Δt long, résidu et observables
temporelles inchangés.

### 16.2 Protocole

- 12 variantes P′ calibrées à 0.1 × ||P6|| (statut OK partout, aucune
  SATURATED ni FAILED)
- Décomposition partagée/spécifique sur la moyenne des 12 (comme en
  6d-ζ), mêmes observables temporelles, brut + centré
- Matrices 12×12 : cos_total, cos_ss_pure, contrib_cos_ss
- Scores intra/inter famille sur cos_ss_pure ; score d'antagonisme
  A_GA = mean(G,G) − mean(G,A)
- Critères préinscrits :
  (1) cos_ss(G,G) globalement positif (>0.1)
  (2) cos_ss(G,A) globalement négatif (<-0.1)
  (3) A_GA moyen > 0.2
  (4) A_GA min > 0 (stable sur tous cas)
- Audit permutation exacte sur 9 variantes G+A (C(9,5)=126 attributions),
  matrices cos_ss_pure et contrib_cos_ss
- Clustering exploratoire en annexe (MDS+k-means sur cos_ss_pure moyennée)

### 16.3 Résultat principal

**Verdict critériel : η-PASS fort.**

Les 4 critères passent sur les 8 cas (4 observables × 2 modes) :

- cos_ss(G,G) > 0 sur 7/8 cas (moyenne +0.49)
- cos_ss(G,A) < 0 sur 8/8 cas (moyenne −0.59)
- A_GA moyen = +1.08
- A_GA min = +0.06 (juste au-dessus de zéro)

Le cas le plus fragile est psi_temp_norm centré : A_GA = +0.06,
avec mean(G,G) = −0.14 (déjà négatif). L'antagonisme G/A survit
formellement mais à la limite.

**Résultats internes notables** :
- mean(A,A) = +0.74 > mean(G,G) = +0.49 : les anneaux/shells sont
  plus cohérents entre eux que les gaussiennes entre elles.
- mean(D,D) = −0.21 : la famille dipolaire ne forme pas un cluster
  cohérent.

### 16.4 Audit permutation exacte

Sur 126 attributions possibles de 5 G_perm + 4 A_perm parmi les 9
variantes G+A (les 3 D laissées hors du test α) :

- **cos_ss_pure** : 7/8 cas avec p_high < 0.05, 6/8 avec p_high ≤ 0.0079
  (plancher à 1/126). 7/8 cas avec quantile ≥ 99% (rank ≥ 125/126).
- **contrib_cos_ss** : 6/8 cas avec p_high < 0.05, 6/8 avec p_high ≤ 0.0079.

**Cas faible identifié** : psi_temp_norm centré. Sur cos_ss_pure,
rank 88/126 (p_high = 0.31) ; sur contrib_cos_ss, rank 103/126
(p_high = 0.19). A_GA_obs est inférieur à la médiane de la
distribution sous permutation. Ce n'est pas un effet borderline :
A_GA est franchement banal sous permutation pour ce cas précis.

Le cas faible coïncide exactement avec le caveat préinscrit en
§16.3 (A_GA = +0.06 seulement). L'audit n'invalide pas l'axe G/A :
il confirme que ce cas spécifique est banal sous permutation,
cohérent avec sa fragilité déjà documentée.

### 16.5 Diagnostic secondaire — clustering

Clustering MDS+k-means à 3 clusters sur la matrice cos_ss_pure
moyennée sur les 8 cas :

- **Cluster 1** : 4 A + D_double_lobe → famille A retrouvée à 4/4
- **Cluster 0** : 3 G + D_dipole_y → 3 gaussiennes (étroite, standard, diag)
- **Cluster 2** : 2 G + D_dipole_radial → 2 gaussiennes (large, décentrée_x)

La famille A est entièrement retrouvée comme cluster cohérent **sans
labels imposés**, soutien indépendant du test de permutation. Les
gaussiennes se divisent en deux sous-groupes, cohérent avec mean(G,G)
inférieur à mean(A,A). Les D sont dispersés sur les trois clusters.

### 16.6 Lecture honnête

L'axe G/A est confirmé empiriquement comme axe géométrique
**centre/couronne** dans les composantes spécifiques des observables
temporelles. La confirmation est forte mais non universelle :

- 7/8 cas extrêmes sur cos_ss_pure
- 6/8 cas extrêmes sur contrib_cos_ss
- 1 cas (psi_temp_norm centré) banal sous permutation, déjà identifié
  comme fragile en 6d-η critères

La famille A présente une cohérence interne marquée (mean A,A = +0.74,
cluster retrouvé sans labels). Les G se divisent en sous-groupes
sans casser l'opposition globale à A. Les D ne forment pas une
famille au sens des composantes spécifiques.

### 16.7 Caveats

- L'axe identifié reste descriptif : il décrit où les composantes
  spécifiques se positionnent dans l'espace des perturbations P′,
  pas pourquoi.
- L'interprétation comme axe centre/couronne est plausible
  géométriquement (G = perturbations centrales, A = perturbations
  en couronne) mais reste une lecture, pas une démonstration.
- psi_temp_norm centré reste un cas faible et banal sous permutation.
  Toute lecture future de l'axe G/A devra rester compatible avec
  cette fragilité locale.
- η-bis (test d'amplitude à 5%/10%/20%) reste non lancé. Si l'axe
  devait être lu théoriquement plus loin, η-bis deviendrait nécessaire.
- L2 (effet du calibrage commun) reste non testé.
- **Aucune lecture Δ, 𝒢 ou MCQ directe** ne se déduit du présent cycle.

### 16.8 État du cadrage à ce point

Après 6d-α/β/γ/δ/ε/ζ/η :

- existence du corridor intermédiaire récurrent (§13)
- non explicable par bruit + conservation + AR(1) simple (§14)
- mécanistiquement décomposable en cos_SS dominant + cos_ss signé (§15)
- cos_ss signé organisable selon un axe géométrique centre/couronne
  confirmé sur 12 variantes (§16)

Cette chaîne empirique est entièrement descriptive. Aucun pas n'a
été franchi vers une interprétation MCQ. La question naturelle
suivante serait de tester si cet axe géométrique conserve sa
structure sous variations d'amplitude (η-bis) et de calibrage (L2).
Cela reste une option ouverte, non engagée.

---

*Fin de §16. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12/§13/§14/§15/§16 intègrent 6d-α à 6d-η.
Aucun verdict sur MCQ. Aucune cible. Conformal-conservatif ou rien.
L'axe géométrique centre/couronne dans les composantes spécifiques
est désormais une structure empirique confirmée sur 12 variantes,
avec caveat local sur psi_temp_norm centré.*

---

## 17. Amendements post-6d-η-bis

**Statut** : inscription minimale du contrôle L2 ciblé sur l'amplitude
commune (6d-η-bis). Provisoire. Le détail (calibration aux trois
amplitudes, diagnostic NOISY, contrôle de reproduction 10% vs η filtré,
audit permutation par amplitude) reste dans les documents et scripts
compagnons.

### 17.1 Objet

§16.7 signalait η-bis (test d'amplitude à 5%/10%/20%) comme dette
courte éventuelle, à lancer si une lecture théorique plus poussée
de l'axe G/A devait être tentée. §16.7 listait aussi L2 (effet du
calibrage commun) comme lecture restant non testée.

6d-η-bis répond directement à la forme la plus simple de L2 :
l'axe G/A confirmé en 6d-η dépend-il de la calibration commune à
10% de ||P6|| ?

### 17.2 Protocole

- 9 variantes G+A uniquement (les 3 D laissées hors du test, car
  la question est isolée à l'opposition G/A)
- Trois amplitudes : 5%, 10%, 20% de ||P6||
- Moteur, P6, référence P6(3000), Δt long, résidu et observables
  temporelles : **inchangés** par rapport à 6d-η
- Composante partagée Ō recalculée sur les 9 variantes G+A à chaque
  amplitude (option a)
- Diagnostic NOISY : seuil absolu (||r|| < 1e-9) + seuil relatif
  (norme d'observable < 1e-12)
- Audit permutation exact C(9,5)=126 par amplitude, sur cos_ss_pure
  et sur contrib_cos_ss
- Contrôle de reproduction séparé : sous-bloc G+A extrait de la
  matrice 12×12 de 6d-η (Ō sur 12) vs η-bis 10% (Ō sur 9)

### 17.3 Résultat

**Verdict critériel : η-bis PASS fort.**

Tableau de stabilité par amplitude :

- 5%  : 0 NOISY, A_GA_mean = +1.2769, A_GA_min = +0.2987,
        7/8 cas extrêmes cos_ss_pure, 7/8 contrib_cos_ss
- 10% : 0 NOISY, A_GA_mean = +1.2768, A_GA_min = +0.2976,
        7/8 cas extrêmes cos_ss_pure, 7/8 contrib_cos_ss
- 20% : 0 NOISY, A_GA_mean = +1.2766, A_GA_min = +0.2955,
        7/8 cas extrêmes cos_ss_pure, 7/8 contrib_cos_ss

Les A_GA varient à 10⁻⁴ près entre les trois amplitudes. La structure
mean(G,G) > 0 / mean(G,A) < 0 est préservée sur la majorité des cas.

Cas faible psi_temp_norm centré : A_GA reste positif (+0.30) aux
trois amplitudes mais ne devient pas extrême sous permutation
(p_high 0.13–0.14). Le caveat préinscrit en §16.6 persiste à
l'identique sous variation d'amplitude.

**Contrôle de reproduction** : 10% η-bis vs 6d-η filtré G+A.
A_GA différents d'environ +0.10 à +0.37 selon l'observable, η-bis
donnant systématiquement des valeurs plus élevées. Cohérent avec
le changement de définition de Ō (12 → 9 variantes, sans les D
qui adoucissaient le contraste G/A). Direction et structure
qualitatives identiques.

### 17.4 Lecture

L'axe géométrique centre/couronne identifié en §16 est strictement
invariant sous variation de l'amplitude commune dans la plage
testée (5% à 20% de ||P6||). L'opposition G/A n'est pas un artefact
du choix particulier de calibration à 10%.

### 17.5 Caveat — régime perturbatif linéaire

L'invariance quasi-parfaite des A_GA à 10⁻⁴ près sur un facteur 4
d'amplitude révèle que le test se situe dans un régime perturbatif
linéaire :

- les perturbations P′ sont petites devant ψ_base ;
- les résidus se scalent quasi linéairement avec l'amplitude ;
- les cos sont normalisés, donc indépendants du facteur d'échelle.

Dans ce régime, l'amplitude commune est une variable triviale au
sens où elle ne modifie pas la structure géométrique normalisée.
Le test discrimine donc précisément ce que tu lui demandes — l'axe
G/A ne dépend pas de l'amplitude commune dans cette plage — mais
il ne couvre pas :

- des amplitudes hors du régime perturbatif linéaire (très petites
  ou très fortes) qui activeraient une non-linéarité du moteur ;
- des calibrages différentiels par variante (chaque P′ à une
  amplitude propre).

### 17.6 Conséquence sur les lectures ouvertes en §13.4

- L3-simple : rejeté en §14
- L1 : reçu un support empirique précis en §15, géométriquement
  confirmé en §16 sur 12 variantes, statistiquement non banal sous
  permutation
- **L2 sous forme "amplitude commune" : rejeté en §17.** L'axe G/A
  ne dépend pas de la calibration commune dans la plage 5%–20%.
- L2 sous d'autres formes (amplitudes hors régime linéaire,
  calibrages différentiels) : non testé. Reste théoriquement possible.

### 17.7 Garde-fou

Aucune identification directe à 𝒢, Δ, RR/RR², ou à toute structure
MCQ ne se déduit du présent cycle. L'invariance observée est une
propriété empirique du résidu sous amplitude commune dans le régime
perturbatif linéaire ; rien d'autre. Le cadrage reste une structure
réflexive révisable.

### 17.8 État du cadrage à ce point

Après 6d-α/β/γ/δ/ε/ζ/η/η-bis :

- existence du corridor intermédiaire récurrent (§13)
- non explicable par bruit + conservation + AR(1) simple (§14)
- mécanistiquement décomposable en cos_SS dominant + cos_ss signé (§15)
- cos_ss signé organisable selon un axe géométrique centre/couronne
  confirmé sur 12 variantes (§16)
- axe invariant sous amplitude commune dans la plage 5%–20% (§17)

Cette chaîne empirique est entièrement descriptive. Six cycles
d'investigation depuis 6d-δ ont produit une caractérisation cohérente
du corridor, sans jamais introduire de lecture MCQ directe. La
sous-enquête δ → ε → ζ → η → η-audit → η-bis peut être considérée
comme un programme empirique autonome sur la structure géométrique
des résidus.

---

*Fin de §17. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12/§13/§14/§15/§16/§17 intègrent 6d-α à
6d-η-bis. Aucun verdict sur MCQ. Aucune cible.
Conformal-conservatif ou rien. L'axe géométrique centre/couronne
est désormais invariant sous amplitude commune dans la plage testée.
Les questions de non-linéarité hors régime linéaire et de calibrage
différentiel restent ouvertes mais non engagées.*

---

## 18. Amendements post-6d-θ

**Statut** : inscription minimale du cycle 6d-θ (Test de composition
G/A et dépendance à l'ordre). Provisoire. Le détail (balayage temporel
Δt_sep, décomposition pré/post composition, test principal avec
contrôles intra-famille) reste dans les documents et scripts compagnons.

### 18.1 Objet

§17 a verrouillé l'axe centre/couronne G/A comme structure empirique
robuste sur 9-12 variantes, invariante en amplitude commune dans le
régime perturbatif linéaire. Mais η/η-bis n'avaient mesuré qu'une
géométrie projective des résidus à une seule perturbation P′.

6d-θ pose une question différente : l'axe G/A confirmé en η/η-bis
est-il seulement une géométrie statique des réponses, ou produit-il
une asymétrie compositionnelle GA vs AG quand on applique deux
perturbations successives ?

C'est un test d'ordre 2 (interaction), pas d'ordre 1 (réponse).

### 18.2 θ-0 — premier balayage

Paire forte : G_standard_centree ↔ A_anneau_moyen.
Δt_sep ∈ {0, 25, 50, 100, 200, 400}, T_final = 800 depuis t0.

Premier passage : K_AUC_total croît monotone avec Δt_sep (3.4e-7 →
6.6e-3). Profil suggérant trois interprétations compatibles : pic
au-delà de 400, effet d'intégration, ou non-linéarité accumulée.

Floor numérique = 0 exactement (moteur strictement déterministe).
Tout signal mesuré est réel, pas du bruit.

### 18.3 θ-0b — décomposition pré/post

Patch méthodologique : K_AUC_total mélangeait trois objets distincts.

- K_AUC_pre : ∫||X_GA − X_AG||dt sur [t0, t0+Δt_sep] — mesure G-seul
  vs A-seul avant composition
- K_AUC_post : ∫||X_GA − X_AG||dt sur [t0+Δt_sep, T_final] — mesure
  la vraie trace post-composition
- K_inst à +25/+50/+100/+200 après la seconde perturbation —
  asymétrie instantanée

**Résultat décisif** : K_AUC_pre domine (×100) sur K_AUC_post à
Δt_sep = 25. La croissance monotone de K_AUC_total venait largement
de l'allongement de la fenêtre pré-composition dans l'intégrale.

K_AUC_post croît modérément (6.5e-4 → 1.3e-3 sur facteur 16 en
Δt_sep) — facteur 2 seulement. K_inst+100 a un pic réel à Δt_sep = 100
(1.31e-5), faible amplitude. K_final en plateau à 3.1e-6 dès
Δt_sep ≥ 100 — l'attracteur réabsorbe l'ordre.

Choix Δt_peak = 100 (pic instantané vrai), Δt_control = 25.

### 18.4 θ-1 — contrôles intra-famille

Test principal sur 8 paires à Δt_sep ∈ {25, 100} :
- 4 paires inter-famille G/A
- 2 paires intra-famille G/G
- 2 paires intra-famille A/A

Métriques principales : K_AUC_post_ext et K_inst+100.
K_AUC_total et K_AUC_pre rapportés mais exclus du verdict.

**Résultat** : G/A ne dépasse pas les contrôles intra-famille.

À Δt_sep = 100 :
- ratio K_AUC_post G/A vs intra : **0.943**
- ratio K_inst+100 G/A vs intra : **0.743**

À Δt_sep = 25 :
- ratio K_AUC_post G/A vs intra : **0.989**
- ratio K_inst+100 G/A vs intra : **0.745**

Tous les ratios sont **inférieurs** à 1. La paire produisant le plus
grand K_inst+100 est G/G (G_etroite ↔ G_decentree_diag = 6.0e-3 à
Δt_sep = 100), pas une paire G/A. La paire produisant le plus petit
K_inst+100 est A/A (A_anneau_moyen ↔ A_anneau_externe = ~5e-5).

### 18.5 Verdict

**θ-FAIL : dépendance d'ordre non spécifique à l'axe G/A.**

L'effet d'ordre existe au-dessus du floor (K_AUC_post jusqu'à 3.5e-3),
mais il est **générique** ou **pair-spécifique**, pas porté par l'axe
centre/couronne identifié en η/η-bis.

### 18.6 Lecture

- η/η-bis restent valides : l'axe G/A est une structure empirique
  robuste dans l'espace des réponses à une perturbation
- θ ajoute : cet axe ne porte pas, dans ce régime, une asymétrie
  d'ordre supérieure aux contrôles intra-famille
- Le corridor (§13-17) et l'axe G/A (§16-17) restent des **structures
  projectives de réponse**, pas des dynamiques de composition

### 18.7 Tension secondaire — piste non engagée

La magnitude du commutateur observée semble dépendre d'une géométrie
locale des supports et recouvrements des perturbations, plutôt que
de leur appartenance G/A. Plus précisément, les facteurs candidats
mentionnés sans test :

- support du masque de chaque P′
- recouvrement avec ψ_tau0
- recouvrement entre les deux perturbations
- localisation par rapport aux zones modifiées par P6
- diffusion et sédimentation h dans ces zones

Cette hypothèse n'est pas testée. Elle est enregistrée comme piste
secondaire éventuelle, à ne pas généraliser en "loi de distance"
simple.

### 18.8 Caveats

- Résultat valable pour ce moteur, cette amplitude 10%, cet état
  P6 relaxé, et Δt_sep ∈ {25, 100}
- Le régime testé est perturbatif linéaire (cf. §17.5) ; la
  composition aussi peut être dominée par cette linéarité
- Ne pas généraliser à toute composition possible
- En particulier : amplitudes hors régime linéaire, autres états
  de départ, autres séparations temporelles très longues, ou
  perturbations très localisées vs étendues — non testés

### 18.9 Garde-fou

Aucune identification directe à 𝒢, Δ, RR/RR², ou à toute structure
MCQ ne se déduit du présent cycle. Le résultat est entièrement
descriptif : il dit que l'axe G/A n'est pas compositionnel dans
le régime testé, rien d'autre.

### 18.10 État du cadrage à ce point

Après 6d-α/β/γ/δ/ε/ζ/η/η-bis/θ :

- existence du corridor intermédiaire récurrent (§13)
- non explicable par bruit + conservation + AR(1) simple (§14)
- mécanistiquement décomposable en cos_SS dominant + cos_ss signé (§15)
- cos_ss signé organisable selon un axe géométrique centre/couronne
  confirmé sur 12 variantes (§16)
- axe invariant sous amplitude commune dans la plage 5%–20% (§17)
- axe non compositionnel : pas d'asymétrie d'ordre spécifique
  par rapport aux contrôles intra-famille (§18)

Cette chaîne empirique est entièrement descriptive. Huit cycles
d'investigation depuis 6d-δ ont produit une caractérisation complète
de l'axe G/A : structure projective robuste, mais non compositionnelle.

---

*Fin de §18. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12/§13/§14/§15/§16/§17/§18 intègrent 6d-α
à 6d-θ. Aucun verdict sur MCQ. Aucune cible. Conformal-conservatif
ou rien. L'axe centre/couronne est désormais qualifié comme
structure projective de réponse, sans propriété compositionnelle
privilégiée dans le régime testé.*

---

## 19. Amendements post-6d-ι

**Statut** : inscription minimale du cycle 6d-ι (Sensibilité
morphologique locale, perturbation h-only). Provisoire. Le détail
(diagnostic clipping h_min, relance clean, métriques d'interaction
I_M et ratio R_I) reste dans les documents et scripts compagnons.

### 19.1 Objet

§18 a clos la branche compositionnelle ψ : l'axe G/A confirmé en
η/η-bis est projectif et non compositionnel. La bifurcation de
programme proposée a été de passer de "perturbations ψ → réponses
projectives" à "perturbations h → sensibilité morphologique".

6d-ι teste : h(θ) scalaire plein possède-t-il une expressivité
morphologique active ? Plus précisément : une perturbation contrôlée
de h conditionne-t-elle non trivialement la réponse dynamique à P′ ?

### 19.2 Méthode

État de départ : X_t0 = (psi_tau0, h_tau0), même état P6 relaxé que
η/θ.

Perturbation h-only :
- h_ε^M = max(h_min, h_tau0 · exp(ε · M_norm))
- M_norm centré (moyenne nulle), normalisé par max(|M_c|)
- pas de clip supérieur h0
- ε calibré pour ||δh||/||h_tau0|| ≈ 1%
- h_min = 1e-8 (réglage clean, après identification d'un h_tau0
  minimal à 1.22e-7 trop proche du seuil initial 1e-6)

Cinq masques :
- H_centre (gaussienne centrée σ=0.8)
- H_anneau_moyen (r ∈ [1.3, 1.9])
- H_shell (r ≥ 2.5)
- H_dipole_face (face j=0 vs face j=4)
- H_random_smooth (bruit gaussien lissé σ=0.8, seed fixé)

Deux P′ de lecture : G_standard_centree, A_anneau_moyen.

Quatre trajectoires par (M, P′) :
- T1 : (psi_tau0, h_tau0), sans P′
- T2 : (psi_tau0, h_ε^M), sans P′
- T3 : (P′(psi_tau0), h_tau0)
- T4 : (P′(psi_tau0), h_ε^M)

Objet principal : I_M(P′) = (T4 − T3) − (T2 − T1)

Ratio relatif : R_I = ||I_M(P′)|| / ||T4 − T3||

T_final = 800 depuis t0. Δt_sep entre h et P′ = 0 (P′ appliqué
immédiatement après h, sur le même état initial).

### 19.3 Résultat

Diagnostic clean : aucune cellule sous h_min = 1e-8, floor numérique
strictement = 0, valeurs principales quasi inchangées par rapport au
premier passage (variation ≤ 0.02% sur max, ~11% sur min).

Métriques principales sur les 10 paires (5 masques × 2 P′) :

- R_I_AUC_ext maximum : **2.247e-3** (H_anneau_moyen × A_anneau_moyen)
- R_I_AUC_ext minimum sur structurés : **4.43e-4** (H_dipole_face × G_standard)
- R_I_AUC_ext sur H_random_smooth : **8.65e-4 à 1.60e-3** (dans la même
  fenêtre que les masques structurés)
- Variabilité max/min entre masques structurés : **5.08**

Signal h-dominant : I_AUC_h ~ 50-150× I_AUC_psi sur toutes les paires.
L'interaction se manifeste principalement dans la trajectoire de h
elle-même, beaucoup moins dans celle de ψ.

### 19.4 Verdict

**ι-PASS modéré.**

Les critères préinscrits donnent ι-PASS fort par le verdict
automatique (R_I_max ≥ 1e-3, variabilité > 3.0). La lecture humaine
est plus prudente, parce que H_random_smooth produit un signal de
même ordre que les structurés.

### 19.5 Lecture

h(θ) scalaire plein possède une expressivité morphologique active
faible à modérée : une perturbation contrôlée de h conditionne la
réponse dynamique à P′, mais principalement via la trajectoire de h
elle-même, avec transmission faible vers ψ.

L'additivité stricte morphologie + P′ est rejetée :
(T4 − T3) ≠ (T2 − T1). L'interaction existe.

### 19.6 Limite

Le couplage h → ψ est faible. La perturbation h modifie l'évolution
future de h, mais sa transmission vers la dynamique de ψ est
atténuée (facteur 50-150× plus petit sur les AUC).

Donc ι ne démontre pas une transformabilité forte de ψ par la
morphologie h. Elle démontre seulement que h n'est pas purement
passif et qu'il conditionne sa propre trajectoire en présence de P′.

### 19.7 Caveat structurel

H_random_smooth produit un signal d'interaction comparable aux
masques structurés (R_I dans la fenêtre 0.87-1.60e-3, à comparer à
4.4e-4 - 2.25e-3 pour les structurés). La géométrie du masque module
l'interaction d'un facteur ~5, mais ne la sépare pas qualitativement
d'une perturbation h lisse générique.

Le maximum H_anneau_moyen × A_anneau_moyen (2.25e-3) suggère un
couplage géométrique spécifique (morphologie h en couronne conditionne
mieux la réponse à perturbation ψ en couronne), mais H_centre × tout
P′ donne 1.5-1.6e-3, presque aussi fort. Pas de spécificité claire.

### 19.8 Conséquence

ι fournit un **candidat faible** de sensibilité morphologique
effective, sans plus. Ce n'est **pas 𝒢_eff robuste** au sens
opératoire :

- 𝒢 plein supposerait une transformabilité forte de ψ par la
  morphologie ; ι montre une transmission atténuée
- 𝒢_eff structuré supposerait une spécificité géométrique
  marquée ; ι montre une dépendance au masque modeste avec
  random dans la même fenêtre

ι confirme cependant que h scalaire plein **n'est pas purement
passif**. C'est un résultat empirique non trivial.

### 19.9 Garde-fou

Aucune identification directe à 𝒢, Δ, RR/RR², ou à toute structure
MCQ ne se déduit du présent cycle. ι fournit un candidat empirique
faible d'expressivité morphologique, pas une identification théorique.
Le résultat reste descriptif.

### 19.10 État du cadrage à ce point

Après 6d-α/β/γ/δ/ε/ζ/η/η-bis/θ/ι :

- existence du corridor intermédiaire récurrent (§13)
- non explicable par bruit + conservation + AR(1) simple (§14)
- mécanistiquement décomposable en cos_SS dominant + cos_ss signé (§15)
- cos_ss signé organisable selon un axe géométrique centre/couronne
  confirmé sur 12 variantes (§16)
- axe invariant sous amplitude commune dans la plage 5%–20% (§17)
- axe non compositionnel (§18)
- expressivité morphologique h-only faible à modérée, dominée par
  l'évolution propre de h, couplage h → ψ atténué (§19)

Cette chaîne a produit deux résultats complémentaires :
- la branche **projective ψ** : corridor + axe G/A robustes, mais
  non compositionnels
- la branche **morphologique h** : interaction h × P′ mesurable
  mais faible, couplage h → ψ atténué

Aucune des deux branches n'a fourni à elle seule un candidat 𝒢
plein. Mais ensemble elles caractérisent le régime perturbatif
linéaire du toy model 6d.

---

*Fin de §19. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12/§13/§14/§15/§16/§17/§18/§19 intègrent
6d-α à 6d-ι. Aucun verdict sur MCQ. Aucune cible.
Conformal-conservatif ou rien. h(θ) scalaire plein est désormais
qualifié comme possédant une expressivité morphologique faible à
modérée, principalement réfléchie dans sa propre trajectoire et
peu transmise vers ψ.*

---

## 20. Relecture transversale κ de la carte de réalisabilité

### 20.1 Statut

§20 est l'inscription minimale d'une **relecture transversale**, pas
d'un cycle expérimental. Le document compagnon `6d_kappa_synthese.md`
contient le détail de la relecture des 6 propriétés du cadrage (P1 à
P5 + P5bis) à la lumière des cycles §13–§19. §20 acte seulement les
modifications de la carte de réalisabilité qui en découlent.

κ a été produit sans contrainte de nouveauté : pour chaque propriété,
l'étiquette **Confirme / Modifie / Ouvre une piste** a été déterminée
à partir du contenu, pas anticipée.

### 20.2 Résultat général

κ ne renverse pas la carte de réalisabilité produite par 6d-β et
amendée en §11. Les propriétés Ch3 émergentes restent non observées
sous leur forme attendue, et le plafond scalaire constaté reste
constaté.

Ce que κ ajoute : la non-observabilité est désormais **qualifiée par
des mécanismes empiriques**, pas seulement par des étiquettes "non".
Trois mécanismes nouveaux apparaissent :

- **A. P5 reçoit le support empirique le plus direct** :
  l'interaction h × P′ mesurée en §19 (ι) est une forme alternative
  de test du cross-talk métrique, faible et h-dominante, mais
  mesurable et non triviale.
- **B. P4 devient la lecture transversale la plus structurante** :
  la chaîne §13–§19 produit cinq canaux d'observation
  (corridor projectif des résidus, axe G/A centre/couronne,
  commutateur générique non spécifique, interaction h × P′,
  transmission h → ψ atténuée) qui ne se reconstruisent pas
  mutuellement. Ce n'est pas la self-opacity Ch3 stricte (le test
  de pseudo-inversion §3.5 cross-tension 18 reste non effectué),
  mais c'est une forme empirique de **non-coïncidence
  observationnelle**.

A et B sont **deux modifications de natures différentes** : A est
un support empirique direct sur une propriété, B est une lecture
structurelle qui qualifie l'ensemble de la carte. Les deux doivent
être préservées sans être collapsées l'une dans l'autre.

### 20.3 Modifications inscrites

- **P3 — Productive forgetting** : non confirmé sous forme Ch3.
  §19 ajoute une contrainte quantitative nouvelle : le couplage
  h → ψ est atténué (I_AUC_h ~ 50-150× I_AUC_psi), ce qui rend P3
  sous forme Ch3 strictement non observable dans le régime testé.

- **P4 — Self-opacity** : non testée sous forme Ch3 stricte
  (pseudo-inversion non effectuée). §13–§19 produisent en revanche
  une non-réductibilité empirique entre canaux d'observation, qui
  est la lecture transversale la plus structurante de κ.

- **P5 — Cross-talk métrique** : reçoit le support empirique le plus
  direct via §19. L'additivité stricte morphologie + P′ est rejetée :
  (T4 − T3) ≠ (T2 − T1). Mais le signal est faible (R_I jusqu'à
  2.25e-3) et principalement contenu dans l'évolution de h elle-même.

### 20.4 Piste ouverte

- **P2 — Morphological latency** : non confirmée sous forme Ch3.
  §19 fournit cependant un objet de mesure potentielle non exploité :
  corrélation croisée temporelle entre I_h(t) et I_ψ(t) sur les
  données ι déjà produites, sans nouveau moteur. Piste mesurable,
  non testée.

### 20.5 Confirmations

- **P1 — Dual-timescale memory** : confirmé comme mémoire morphologique
  faible / verrouillage hérité. §19 précise que la mémoire reste
  interne à h, sans modulation à deux échelles distinctes sur ψ.

- **P5bis — Structure de bassin** : identifiée comme **zone aveugle**
  des cycles §13–§19. Aucun cycle récent n'a varié les initialisations
  au sens §11.2 (familles A vs B distinctes). Le statut empirique de
  P5bis reste celui établi en 6d-α, non retesté.

### 20.6 Conséquence

κ justifie cette inscription §20 courte (modifications réelles sur
P3, P4, P5) sans devenir un §20 de transition (pas de question
structurante unique qui s'impose et engage la suite).

Deux options post-κ sont légitimes mais non engagées à ce stade :

- **Option minimale** : tester P2 sans nouveau moteur via
  corrélation croisée I_h(t) / I_ψ(t) sur les données ι existantes.
  Léger, directement issu de κ, soit enrichit la carte soit ferme
  une piste rapidement.

- **Option structurante** : ouvrir un cycle λ (cross-talk h → ψ
  ciblé) si P5 doit devenir l'axe expérimental privilégié de la
  suite. Demande nouveau code et nouvelle décision de programme.

La décision entre ces options dépasse §20.

### 20.7 Garde-fou

§20 est une inscription réflexive, pas une promotion théorique.

- Aucune identification à Δ, 𝒢 plein, RR/RR², ou structure MCQ
  ne se déduit de κ.
- La non-réductibilité entre canaux d'observation (B) reste une
  observation descriptive ; elle ne se confond pas avec la
  self-opacity Ch3 stricte, qui reste non testée.
- Aucun repositionnement de Ch4 ne se déduit de κ.
- La carte de réalisabilité produite par 6d-β et amendée en §11
  reste l'objet d'output 6d.

---

*Fin de §20. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12 à §19 intègrent 6d-α à 6d-ι. §20 inscrit
la relecture transversale κ. Aucun verdict sur MCQ. Aucune cible.
Conformal-conservatif ou rien. La carte de réalisabilité scalaire
est désormais qualifiée par cinq canaux d'observation non
mutuellement réductibles, dont l'un (P5 cross-talk métrique) reçoit
le support empirique direct le plus net.*

---

## 21. Audit P2-lag (compagnon de §20.4)

### 21.1 Statut

§21 est un **audit secondaire** issu de la piste ouverte en §20.4
(Morphological latency) par la relecture transversale κ. Ce n'est
ni un cycle expérimental nouveau (§13 à §19), ni une relecture
transversale (§20). C'est une mesure ciblée exploitant les
trajectoires de ι (§19), sans nouveau moteur, sans nouvelle
configuration, sans nouvelle géométrie.

Le détail (script `test_P2_lag_audit.py`, sortie
`6d_P2_lag_audit.json`) reste dans les fichiers compagnons.

### 21.2 Objet

§20.4 avait inscrit P2 (Morphological latency) avec l'étiquette
« Ouvre une piste » : §19 fournissait un objet de mesure potentielle
non exploité — la corrélation croisée temporelle entre I_h(t) et
I_ψ(t) sur les données ι déjà produites.

L'audit §21 répond à la question : existe-t-il un décalage temporel
mesurable entre l'interaction morphologique I_h(t) et l'interaction
ψ I_ψ(t) ?

### 21.3 Méthode

Réexport des trajectoires ι (script `test_6d_iota_export_traj.py`),
identique au run clean de §19 en tout point sauf l'export
supplémentaire des séries temporelles ||I_h(t)|| et ||I_ψ(t)||.

Pour chaque paire (M, P′) — 5 masques × 2 P′ = 10 paires :

- a_h(t) = ||I_h(t)||₂
- a_ψ(t) = ||I_ψ(t)||₂

Corrélation croisée normalisée calculée sur trois modes :
- **brut** : a_h(t), a_ψ(t)
- **centré** : a(t) − mean(a)
- **différences** : Δa(t) = a(t+dt) − a(t)

Convention : lag > 0 ⇔ I_h précède I_ψ ; lag < 0 ⇔ I_ψ précède I_h.

Verdict principal sur les différences. Fenêtre principale
[−50, +50] unités de temps. Fenêtre longue [−200, +200] en
contrôle uniquement (non verdict).

### 21.4 Résultat

Sur le mode différences, fenêtre principale :

- **10/10 paires** avec lag négatif
- Lag moyen ≈ **−2.6 unités de temps**
- Range : [−3.01, −2.00]
- Gain moyen (corr_max − corr_zero) ≈ +0.27
- Range gain : [+0.16, +0.34]
- Aucun pic principal hors fenêtre courte (audit fenêtre longue
  confirme l'absence de lag long parasite)

Variation modeste entre masques et entre P′ : la dispersion
inter-paires est faible.

### 21.5 Verdict

**P2 sous forme Ch3 rejetée.** L'attente Ch3 §3.3 cross-tension 4
(τ'_k reflète h pré-contraction avec lag mesurable, c'est-à-dire
ψ retardé par rapport à h) n'est pas observée.

**Latence inverse ψ→h documentée empiriquement.** Sur les
différences temporelles des interactions, I_ψ(t) précède I_h(t)
d'environ 2.6 unités de temps, sur les 10 paires sans exception.

### 21.6 Lecture

L'ordre temporel observé (variations de I_ψ précèdent les
variations de I_h) est **compatible avec un couplage ψ→h via la
structure du moteur** : P′ agit directement sur ψ à t = 0, puis h
suit via G_sed et G_ero. La diffusion de ψ dépend également de h
(couplage bidirectionnel via le flux D ∇·(h ∇ψ)).

Cette lecture est cohérente avec les données mais reste une **lecture
plausible**, pas une démonstration causale. Le moteur 6d a une
dynamique bidirectionnelle ; "ψ précède h temporellement" ne signifie
pas "ψ cause h" au sens strict.

La spécificité morphologique du lag est faible. La variation
inter-masques est modeste (~1 unité de temps sur ~2.6 mesurés). Le
lag observé est probablement **générique au moteur** dans ce régime,
pas une signature morphologique des masques spécifiques.

### 21.7 Conséquence sur la carte

P2 change de statut :

- **avant §21** : piste ouverte (κ étiquette « Ouvre une piste »).
- **après §21** : contrainte empirique documentée — la latence
  observée existe, mais de **signe opposé** à l'attente Ch3.

La piste est donc **fermée sous sa forme Ch3** et **convertie en
contrainte empirique** : dans le régime 6d testé, l'ordre temporel
observable des variations d'interaction est ψ→h, pas h→ψ.

C'est cette conversion (piste ouverte → contrainte empirique
documentée) qui justifie l'existence de §21, sans en faire un
cycle expérimental ni un §21 de transition.

### 21.8 Garde-fou

- Aucune validation de morphological latency Ch3 ne se déduit
  de §21. La forme Ch3 stricte (ψ retardé par rapport à h) est
  rejetée, pas confirmée.
- Le vocabulaire reste empirique : *documenté*, *compatible avec*.
  Aucun énoncé causal démonstratif n'est inscrit.
- Aucune ouverture de λ ne se déduit de §21. Si λ devait être
  ouvert ultérieurement, sa formulation devrait tenir compte de
  ce résultat : non pas amplification d'une latence h→ψ inexistante,
  mais éventuel test de lisibilité du canal h→ψ malgré la
  dominance temporelle ψ→h observée.
- Aucune lecture Δ, 𝒢, RR/RR², Ch4 ou MCQ ne se déduit du
  présent audit.

---

*Fin de §21. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12 à §19 intègrent 6d-α à 6d-ι, §20 inscrit
la relecture transversale κ, §21 acte la conversion de la piste
P2 en contrainte empirique de latence inverse ψ→h. Aucun verdict
sur MCQ. Aucune cible. Conformal-conservatif ou rien. La carte de
réalisabilité scalaire est désormais qualifiée par cinq canaux
d'observation non mutuellement réductibles plus une contrainte
empirique d'ordre temporel ψ→h.*

---

## 22. λ-A : séparabilité géométrique modérée et effet opératoire stratifié

### 22.1 Statut

§22 est l'inscription minimale post-λ-A. Pas un nouveau cycle long.
Spécification opératoire dans le document compagnon
`6d_lambda0_h_proj_specification.md`, exécution et résultats dans
`test_6d_lambda_A_v2.py` et `6d_lambda_A_v2.json`. Une version
préliminaire λ-A-v0 a été produite puis rejetée méthodologiquement
(métriques opératoires non robustes aux cellules h quasi-effondrées) ;
λ-A-v2 est la version retenue, avec ratios symétriques bornés,
strates de masquage et checkpoints temporels.

### 22.2 Objet

Le test direct canonique §3.2 du cadrage (comparer h(θ) plein à
h(θ) artificiellement reconstruit depuis les marginales) n'avait
été effectué dans aucun cycle 6d-α à 6d-ι. §20 (relecture κ) avait
identifié cette absence et signalé P5 comme la propriété recevant
le support empirique le plus direct de §13–§19 sous forme
alternative (via §19 ι), tout en notant que le test direct restait
non effectué.

λ-A-v2 effectue ce test direct sur une famille minimale d'états
représentatifs.

### 22.3 Méthode

- **Définition de h_proj** : projection produit-marginales canonique
  h_proj(i,j,k) = h_T(i) · h_M(j) · h_I(k) / S²
  où h_T/h_M/h_I sont les sommes 1D et S la masse totale. Propriétés
  vérifiées analytiquement et par contrôles synthétiques (C1–C5
  PASS) : positivité, conservation masse, conservation marginales,
  idempotence sur h constant et h séparable.

- **Famille d'états** : E0 (h_tau0 = P6 relaxé), E1 (après évolution
  avec G_standard), E2 (après évolution avec A_anneau). Checkpoints
  temporels t = 0, 10, 50, 100, 200, 400, 800 pour E1/E2.
  15 états testés au total.

- **Ratios symétriques** : R_sym = ||a−b|| / (||a|| + ||b|| + 1e-30),
  bornés dans [0, 1]. Seuils : 0.005 (faible), 0.05 (significatif),
  0.3 (dominant).

- **Strates de masquage** :
  - A — all interfaces
  - B — h_face_full > 1e-6 (morphologiquement actif)
  - C_psi — B ∩ ψ_face > 0.01·max(ψ_face) (présence active)
  - C_grad — B ∩ |grad ψ|_face > 0.01·max(|grad ψ|) (transport actif)

- **Verdict opératoire** : grille C_grad principal × C_psi
  cohérence, avec garde-fou de localité (fraction de faces retenues).

- **g_k** : secondaire, en quantiles log10(g_full/g_proj) sur
  masque actif, exclu du verdict.

### 22.4 Résultat

**Couche géométrique** :
- D_proj stable à **0.084–0.085** sur les 15 états
- Variation D_proj quasi nulle entre t = 0 et t = 800
- Identique pour E0, E1, E2

**Couche opératoire C_psi** (h-active ∩ ψ-active, ~98% des faces) :
- R_sym C_psi ≈ **0.012** sur les 15 états
- Stable, sous le seuil PASS 0.05 mais au-dessus du seuil faible 0.005

**Couche opératoire C_grad** (h-active ∩ gradient-active) :
- À t = 0 : R_sym C_grad ≈ 0.011, fraction ≈ 8.3% des faces
- À t ≥ 10 : **fraction = 0**, R_sym C_grad = 0
- La strate devient vide après t = 10

**Couche A** (toutes interfaces) et pondérations D_psi/D_grad :
- Écart full/proj massif aux interfaces h-effondrées
- Norme absolue ||J_full − J_proj|| dominée par les zones où h_full
  est microscopique (jusqu'à 10⁻⁸⁹ à t = 800)

### 22.5 Verdict

**GEOM-PASS modéré.** D_proj stable à ~0.084 sur les 15 états
qualifie une non-séparabilité géométrique modérée, héritée de l'état
P6 relaxé et non amplifiée par G_standard ou A_anneau.

**OPER-faible dans les zones h-actives.** R_sym C_psi ≈ 0.012
sur 98% des faces. Effet flux mesurable mais sous le seuil PASS
0.05. Stable, au-dessus du seuil faible 0.005.

**OPER-fort mais non verdictif dans les zones h-effondrées.** Aux
interfaces où h_full est quasi-nul, h_proj reconstruit des valeurs
non négligeables depuis les marginales, ce qui produit un écart
flux massif. Cette différence n'est pas une mesure de cross-talk
métrique : elle mesure ce que h_proj fait là où h_full bloque déjà
le transport. Non verdictif sur P5.

**COMBINÉ** : non-séparabilité géométrique stable, effet opératoire
faible dans le régime fonctionnel, projection très disruptive sur
les zones effondrées.

Forme courte : **GEOM-PASS modéré + OPER-faible/stratifié.**

### 22.6 Lecture

Le fait structurellement nouveau de λ-A-v2 n'est pas la valeur de
D_proj ou de R_sym C_psi (modérée et faible respectivement). Il
est la **dissociation spatiale** observée au cours de l'évolution :

La strate C_grad devient vide après t = 10 parce que **les faces
h-actives ne coïncident plus avec les faces gradient-actives selon
les seuils préinscrits**. Cela ne signifie pas que le gradient
disparaît ; cela signifie que l'intersection h-active ∩
gradient-active se vide.

Concrètement : les zones où h reste fonctionnel (centre, zones
où la sédimentation maintient h proche de sa valeur initiale) ne
sont plus celles où les gradients ψ sont les plus forts. La
dynamique 6d sépare spatialement ces deux conditions.

C'est un fait empirique sur le moteur dans ce régime, indépendant
de la projection. La projection le rend visible, mais ne le
produit pas.

### 22.7 Conséquence sur la carte

P5 reçoit son **test direct** via λ-A-v2 :

- Avant §22 (état du cadrage post-κ) : P5 supportée par §19 sous
  forme alternative (interaction h × P′ via ι, faible et h-dominante),
  test direct §3.2 non effectué.
- Après §22 : test direct effectué. **Support faible et stratifié**,
  pas un PASS fort. La non-séparabilité géométrique est stable
  mais modérée ; son effet opératoire dans les zones fonctionnelles
  est faible (R_sym ≈ 0.012) ; l'écart massif aux zones effondrées
  n'est pas verdictif sur P5.

Le statut de P5 sur la carte de réalisabilité passe de :

  *"support empirique indirect via ι, test direct non effectué"*

à :

  *"test direct effectué : non-séparabilité géométrique modérée
  stable, effet opératoire faible dans les zones fonctionnelles,
  dissociation spatiale observée au cours de l'évolution"*.

La chaîne §19 (indirect) + §22 (direct) qualifie P5 de manière
cohérente : présence d'un cross-talk métrique mesurable mais faible
sous h(θ) scalaire plein dans ce régime.

### 22.8 Garde-fou

- Aucune ouverture de λ-B. Un moteur PROJECTED dynamique serait
  dominé par la réouverture des zones h-effondrées par h_proj, ce
  qui ne testerait pas le cross-talk métrique mais l'effet
  destructeur d'une projection marginale sur les barrières
  morphologiques locales.
- Aucune identification à Δ, 𝒢, RR/RR², ou structure MCQ ne se
  déduit de §22. Le test direct est faible.
- Aucune lecture Ch4 ne se déduit.
- Caveat de portée : λ-A-v2 teste la non-séparabilité sur E0/E1/E2.
  Un résultat faible ne falsifie pas la non-séparabilité de h(θ)
  plein hors de cette famille.

---

*Fin de §22. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12 à §19 intègrent 6d-α à 6d-ι, §20 inscrit
la relecture transversale κ, §21 acte la conversion de la piste P2
en contrainte empirique de latence inverse ψ→h, §22 acte le test
direct de séparabilité P5. Aucun verdict sur MCQ. Aucune cible.
Conformal-conservatif ou rien. La carte de réalisabilité scalaire
est désormais qualifiée par cinq canaux d'observation non
mutuellement réductibles, une contrainte empirique d'ordre temporel
ψ→h, et un test direct de séparabilité montrant une non-séparabilité
géométrique stable, un effet opératoire faible dans les zones
fonctionnelles, et une dissociation spatiale entre zones h-actives
et zones gradient-actives.*

---

## 23. Clôture de branche λ/P5 et options post-λ

### 23.1 Statut

§23 est une **décision de programme post-§22**. Pas un nouveau
cycle expérimental. Pas un nouveau résultat empirique. §23 acte
la fermeture de la branche λ/P5 et nomme les options restantes
sans en engager aucune.

### 23.2 Clôture λ/P5

La branche λ/P5 est fermée à ce stade.

- Pas de λ-B.
- Pas de λ-B-0.
- Pas de nouveau code λ.

Raison structurelle (héritée de §22.8) : un moteur PROJECTED
dynamique serait dominé par la réouverture des zones h-effondrées
par h_proj. Il ne testerait plus proprement le cross-talk métrique,
mais l'effet destructeur d'une projection marginale sur les barrières
morphologiques locales.

### 23.3 Options post-λ

Trois options restent ouvertes, sans engagement immédiat.

**A — Consolidation / arrêt propre de la branche 6d scalaire.**
Le corpus 6d est désormais dense : propriétés Ch3 non observées
sous forme attendue, corridor projectif, axe G/A, axe non
compositionnel, expressivité h faible, latence inverse ψ→h, P5
direct faible/stratifié.

**B — P5bis-0.** Option recommandée si l'on poursuit. Objet :
retester bassins, dépendance au chemin, initialisations, géométries
de convergence, et lien avec la dissociation fonctionnelle
h-active / gradient-active observée en §22. P5bis est la zone
aveugle la plus proche des résultats actuels (cf. §20.5).

**C — P4 strict.** Pseudo-inversion / reconstructibilité limitée
au sens Ch3 §3.5 cross-tension 18. Important, mais plus conceptuel
et plus lourd. À garder en réserve après clarification des bassins.

### 23.4 Recommandation suspendue

Recommandation méthodologique : **B sous forme minimale**, c'est-à-dire
P5bis-0 comme mini-cadrage avant code éventuel.

Question directrice candidate : la dissociation h-active /
gradient-active observée en §22 (λ-A-v2) est-elle spécifique au
régime P6/G/A, ou se retrouve-t-elle dans les familles
d'initialisation et les structures de bassin déjà identifiées en
6d-α ?

Mais **aucun code n'est engagé par §23**. La décision est suspendue.

### 23.5 Signal méthodologique

La branche 6d scalaire approche de son point de saturation comme
programme expérimental autonome. Ce n'est pas un échec : c'est
l'accomplissement partiel de sa fonction d'origine — cartographier
les limites de h(θ) scalaire plein sous approximation
conformal-conservative.

**Limites structurelles de portée** (non démonstrations de nécessité) :

- grille 5×5×5 très coarse
- N = 1 (instance unique, hors MCQᴺ)
- approximation conformal-conservative, pas Laplace-Beltrami strict
- h(θ) scalaire conforme, pas H(θ) tensoriel
- phénomènes fins d'anisotropie, non-commutativité, latence locale,
  transport parallèle non testés proprement ou hors portée
  instrumentale

Ces limites sont **observées**, pas postulées, et elles encadrent
ce que les cycles §12–§22 ont pu produire.

### 23.6 Horizon non engagé

À plus long terme, trois directions restent possibles :

- clôture propre de la branche 6d scalaire ;
- formulation ultérieure d'un nouveau toy model motivé par les
  limites observées ;
- préparation d'un cadrage Ch4 empirico-théorique.

Aucune de ces directions n'est engagée par §23. Leur détail
appartiendrait à un document compagnon de transition ultérieur,
pas au cadrage 6d.

### 23.7 Garde-fou

§23 ferme λ/P5 et suspend la suite.

- Pas de λ-B.
- Pas de nouveau code.
- Pas de Δ.
- Pas de 𝒢.
- Pas de MCQ.
- Pas de Ch4 immédiat.
- Pas de nouveau toy model inscrit dans le cadrage.

La coda empirique du cadrage reste celle inscrite en fin de §22.
§23 ne modifie pas la carte de réalisabilité ; il acte que les
cycles expérimentaux 6d en l'état suffisent à la qualifier.

---

*Fin de §23. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12 à §22 intègrent les cycles 6d-α à 6d-λ
et leurs relectures. §23 acte la clôture de la branche λ/P5 et
suspend la décision sur la suite. Aucun verdict sur MCQ. Aucune
cible. Conformal-conservatif ou rien.*

---

## 24. P5bis-A — Bassins fonctionnels et dissociation non dégénérée spécifique à B2

### 24.1 Statut

P5bis-A est un test post-λ de la zone aveugle P5bis identifiée
en §20.5 et reprise en §23.3. Pas un test MCQ. Pas un λ-B. Pas
un P4 strict. La spécification opérationnelle est dans le document
compagnon `6d_p5bis0_specification.md` ; l'exécution dans
`test_6d_p5bis_A.py` et `6d_p5bis_A.json`.

§24 acte ce que P5bis-A a produit, sans engager P5bis-B ni nouveau
cycle.

### 24.2 Réplication 6d-α

Les familles A/B1/B2/B3 héritées de 6d-α §5.7 reproduisent la
structure de bassin à β = 60 :

- A↔B2 : Dh_final = 0.297 (rapport 6d-α §2.3 : 0.295). **B2 reste
  bassin empirique distinct.**
- A↔B1 : Dh_final = 4.0e-8. **Même attracteur, chemin non trivial**
  (AUC_Dh élevé en cours de trajectoire).
- A↔B3 : Dh_final = 5.4e-15. **Contrôle uniforme/minimal**, convergence
  vers même attracteur.

P5bis-A ne réinvente pas 6d-α §5.7. Il le réactive et ajoute la
couche fonctionnelle post-λ.

### 24.3 Résultat nouveau post-λ

La dissociation h-active / gradient-active mesurée en §22 (λ-A-v2)
est **bassin-spécifique**, pas générique au moteur. Plus précisément
**bassin-spécifique sous forme non dégénérée pour B2 seulement** :

| Famille | frac_h_active | frac_grad_active | intersection | Jaccard | DISSOC |
|---|---|---|---|---|---|
| A | 1.000 | 1.000 | 1.000 | 1.000 | h_grad_overlap_preserved |
| B1 | 1.000 | 0.333 | 0.333 | 0.333 | h_grad_overlap_preserved |
| B2 | 0.880 | 0.120 | **0.000** | **0.000** | **h_grad_dissociated** |
| B3 | 1.000 | 0.000 | 0.000 | 0.000 | GRAD_DEGENERATE |

B2 est **le seul bassin combinant** un gradient actif non trivial
(frac_grad_active ≈ 0.12, grad_max ≈ 0.028) et une **absence totale
d'intersection** avec h-active. Les autres ne présentent pas cette
configuration :

- A et B1 ne présentent pas la dissociation non dégénérée de B2.
- B3 est dégénéré par uniformité initiale (pas de gradient possible).

Verdict candidat : **DISSOC-BASIN-SPECIFIC, sous forme de dissociation
non dégénérée spécifique à B2**.

Contrôles β = 45 et β = 80 sur A vs B2 (et A vs B1) cohérents
avec 6d-α §2.3 : B2 reste bassin distinct à β = 45 et 80 ; B1
devient lui-même bassin distinct à β = 80.

### 24.4 Lien avec §22

La dissociation h-active / gradient-active observée en λ-A-v2 sur
le régime P6/G/A **n'est probablement pas une propriété générique
du moteur**. Elle correspond plutôt à un régime de type B2 : zones
h encore fonctionnelles et zones gradient-actives spatialement
séparées.

§22 reçoit donc une **qualification** : la dissociation
h-active / gradient-active est liée à certains bassins ou chemins,
pas à toute dynamique 6d.

§22.6 ("dissociation spatiale comme fait empirique du moteur,
indépendante de la projection") reste valide, mais le caractère
"fait empirique du moteur" doit être lu comme **bassin-dépendant**,
pas universel.

### 24.5 Underflow et bassin absorbant empirique

B2 atteint LONG_UNDERFLOW_DOMINATED à T = 3000 (h_min ≈ 5e-324,
fraction non négligeable de cellules sous floor numérique
double-précision).

Cela confirme le diagnostic 6d-α §2.5 (ABSORBING_BASIN_EMPIRICAL_B2)
sous le diagnostic underflow rigoureux introduit en P5bis-0 §10
C2.

Formulation : **ABSORBING_BASIN_EMPIRICAL_B2 renforcé, sous caveat
underflow explicite**. La tension §4.5 du rapport 6d-α (*"empirical
absorbing ≠ structural absorbing"*) reste préservée.

### 24.6 Réactivation

La réactivation fonctionnelle telle que codée en P5bis-A **n'est
pas verdictive**.

- A × G_standard / A_anneau : REACT-NONE lisible
- B2 × G_standard / A_anneau : REACT-NONE lisible
- B1 × G_standard / A_anneau : REACT-UNCLASSIFIED, ΔJaccard ≈
  +0.667 mais Dext ≈ 7.6e-10
- B3 × G_standard / A_anneau : REACT-UNCLASSIFIED, ΔJaccard ≈
  +1.000 mais Dext ≈ 7.6e-10

Les labels REACT-UNCLASSIFIED de B1 et B3 sont **probablement dus
à la sensibilité du seuil relatif** `0.01 × max(|grad ψ|)` : une
petite perturbation change le max global de |grad ψ|, ce qui décale
le seuil et modifie quelles faces sont au-dessus, sans qu'il y ait
réactivation fonctionnelle réelle.

Aucun REACT-FUNCTIONAL robuste n'est observé. La métrique de
réactivation fonctionnelle telle que codée doit être considérée
comme **non robuste** au seuil relatif. À raffiner si la question
de la réactivation devient centrale, mais pas requise pour répondre
à P5bis.

Inscription : **réactivation non concluante, métrique à raffiner
si nécessaire**. Pas de nouveau cycle ouvert sur cette base.

### 24.7 Conséquence sur la carte

P5bis passe de **"zone aveugle des cycles §13–§21"** (état post-κ)
à :

  *"testée post-λ : multi-bassins 6d-α répliqués ; dissociation
  fonctionnelle non dégénérée spécifique à B2 ; réactivation non
  verdictive."*

Cela **ferme P5bis comme dette immédiate** du programme. La case
P5bis de la carte de réalisabilité est désormais qualifiée.

### 24.8 Garde-fou

- Pas de λ-B.
- Pas de h_proj utilisé dynamiquement.
- Pas de P4 strict dans §24.
- Pas de Δ, 𝒢, RR/RR², MCQ, Ch4.
- Bassin empirique ≠ bassin structurel.
- Dissociation non dégénérée B2 ≠ loi générale du moteur.
- La réactivation n'est pas un résultat ; c'est un caveat
  méthodologique.

---

*Fin de §24. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12 à §22 intègrent les cycles 6d-α à 6d-λ
et leurs relectures, §23 ferme la branche λ/P5, §24 ferme la
zone aveugle P5bis avec un résultat post-λ qualifiant la
dissociation §22 comme bassin-spécifique non dégénérée sur B2.
Aucun verdict sur MCQ. Aucune cible. Conformal-conservatif ou
rien. La carte de réalisabilité scalaire est désormais qualifiée
par cinq canaux d'observation non mutuellement réductibles, une
contrainte empirique d'ordre temporel ψ→h, un test direct de
séparabilité montrant une non-séparabilité géométrique stable
avec dissociation spatiale au cours de l'évolution, et un test
de bassins post-λ montrant que cette dissociation est portée
sous forme non dégénérée par le bassin B2.*

---

## 25. P4-A — Reconstructibilité sous filtration observable : absence de signal structurant

### 25.1 Statut

P4-A teste la dernière dette P4 stricte (§3.5 cross-tension 18,
relue en §20 comme lecture transversale la plus structurante du
programme, jamais testée sous sa forme propre avant §25).

Pas un test MCQ. Pas une preuve de self-opacity. Pas une
pseudo-inversion ML. Pas un § de transition programme.

La spécification opérationnelle est dans le document compagnon
`6d_p4_strict_0_specification.md` ; l'exécution dans
`test_6d_p4_strict_A.py` et `6d_p4_strict_A.json`.

### 25.2 Méthode

- **Bibliothèque P5bis-A noyau** : 108 états (4 familles A/B1/B2/B3
  × 3 valeurs de β ∈ {45, 60, 80} × 9 checkpoints temporels).
  Total : 5778 paires distinctes.
- **Distances d'état séparées** : D_state_ψ et D_state_h, pas de
  distance étendue agrégée comme axe principal de verdict.
- **Filtration progressive** :
  𝒫_min ⊂ 𝒫_functional ⊂ 𝒫_trajectory_intrinsic ⊂ 𝒫_rich.
  Toutes les observables sont intrinsèques à X ou à sa propre
  trajectoire (pas d'AUC relatif à A).
- **Régressions log-log** D_obs vs D_state, par axe × niveau,
  avec filtrage ε_floor = 1e-12 sur D_state ET sur D_obs.
- **Top paires ambiguës** par z-score (rang-based, pas seuils
  pré-figés) en diagnostic secondaire.
- **Contrôle de reproduction P5bis-A** obligatoire sur 6
  agrégats clés avant tout calcul.

### 25.3 Résultat

**Reproduction P5bis-A : PASS strict** (6/6 agrégats reproduits
à mieux que 1e-6 : Dh A↔B2 = 0.2966, Dh A↔B1 = 4.0e-8, B2 frac
h-active = 0.880, B2 frac grad-active = 0.120, B2 intersection
= 0, B2 h_min T=3000 = 4.9e-324).

**α faibles sur tous niveaux** :
- axe ψ : α ∈ [+0.06, +0.11] (P_min plus discriminant que P_functional,
  contrintuitif)
- axe h : α ∈ [+0.16, +0.24]

**R² faibles sur tous niveaux** : 0.16 à 0.30.

**Underflow non structurant** : régressions avec et sans paires
underflow donnent des α quasi identiques (écart < 0.02).

**Pas d'anisotropie forte** : α_h ≈ 2 × α_ψ uniformément, mais
les deux restent faibles.

### 25.4 Lecture

Le verdict automatique heuristique `P4-OBS-SATURATION-candidate`
**n'est pas retenu**. La signature d'une saturation observable
propre serait α faible **avec R² élevé** ; ici α faible est
associé à R² également faible, ce qui caractérise un **nuage
dispersé**, pas une loi de saturation lisible.

Les top paires "OBS-CLOSE / STATE-FAR" identifiées par z-score
ne montrent **pas** une pseudo-inversion forte (pas de cas
D_obs ≈ 0 avec D_state grand). Elles correspondent
majoritairement à :
- des paires intra-trajectoire entre checkpoints proches
  (B1 β=80 ck6 vs ck7, B2 β=60 ck3 vs ck4)
- des cas de sur-discrimination locale où D_obs > D_state d'un
  facteur 5 à 10

C'est une conséquence de l'hétérogénéité de la bibliothèque, qui
mélange régimes intra-trajectoire et inter-bassin sans que la
régression log-log globale ne les sépare.

### 25.5 Verdict

**P4-NO-STRONG-SIGNAL** (label introduit ici, hors liste P4-0,
parce que les labels préinscrits ne couvraient pas exactement
le cas observé).

Lecture :

> P4-A ne fournit ni signal fort de non-injectivité empirique,
> ni saturation observable propre, ni discrimination proportionnelle
> simple, ni anisotropie franche. La relation D_obs / D_state est
> un nuage dispersé sans loi log-log lisible.

La self-opacity stricte Ch3 §3.5 n'est **ni validée ni falsifiée**.
Sous la bibliothèque P5bis-A et la filtration 𝒫_min → 𝒫_rich
définie ici, P4 ne modifie pas la carte de réalisabilité.

### 25.6 Conséquence

Dette P4 fermée à ce niveau.

Pas de P4-B.
Pas de nouveau code immédiat.
La décision de clôture ou de continuation de la branche 6d
scalaire relève d'une décision de programme séparée.

§25 ferme la dette P4 stricte sans modifier positivement la
carte de réalisabilité.

### 25.7 Garde-fou

- Pas de Δ, 𝒢, RR/RR², MCQ, Ch4.
- Pas de pseudo-inversion ML.
- Pas de h_proj.
- Pas de généralisation hors bibliothèque P5bis-A.
- Pas de généralisation hors filtration 𝒫_min → 𝒫_rich définie ici.
- **P4-NO-STRONG-SIGNAL ≠ absence d'opacité en général**. C'est
  seulement l'absence d'un signal structurant **sur la bibliothèque
  testée sous la filtration considérée**.

### 25.8 Audit P4-VAR — stratification de la dispersion

P4-VAR a été lancé après §25 comme micro-audit de second ordre
afin de tester la phrase de §25.4 : *"nuage dispersé sans loi
log-log lisible"*. L'objectif n'était pas de rouvrir P4, mais de
vérifier si la dispersion résiduelle du nuage était homogène ou
stratifiée par des variables déjà présentes dans la bibliothèque
P5bis-A. Le script et les résultats sont dans
`test_6d_p4_var_audit.py` et `6d_p4_var_audit.json`.

**Méthode** : résidus log-log r = log(D_obs) − (α log(D_state) + c),
calculés sur l'axe h et le niveau 𝒫_functional, puis comparaison
de la variance résiduelle par classes de paires. Bootstrap 1000
permutations pour la significativité ; taille d'effet via
variance_ratio et delta_median_abs_r.

**Résultat** : **P4VAR-STRATIFIED**. La dispersion n'est pas
homogène.

Faits principaux :
- intra_trajectory : variance_ratio ≈ 26.5 (variance résiduelle
  très élevée intra-trajectoire)
- B3_involving : variance_ratio ≈ 3.05 (variance élevée)
- B2_involving, B2_vs_nonB2, dissoc_involving, dissoc_vs_nondissoc :
  variance_ratio ∈ [0.086, 0.248] (variance **basse** dans la classe,
  donc régions plus cohérentes du nuage)
- underflow_involving : significatif (p = 0.047) mais variance
  basse (ratio ≈ 0.19), non source principale du bruit

**Lecture critique** : B2 et les états dissociés **ne portent pas
le bruit**. Ils forment au contraire des régions où D_obs et
D_state sont mieux corrélés, donc mieux ajustés par la tendance
log-log. La dispersion forte vient surtout des paires
intra-trajectoire (checkpoints successifs d'une même évolution)
et de B3 (dégénérescence fonctionnelle par gradient uniforme).

Cette inversion par rapport à l'intuition naïve ("B2 = source
de bruit") doit être préservée dans la formulation finale.

**Vérification croisée par régressions séparées** :
- intra_trajectory_only : α = +0.242, R² = 0.21
- inter_family_same_beta_only : α = +0.056, R² = 0.16
- no_B3_no_underflow : α = +0.120, R² = 0.19

Aucun sous-régime ne produit une loi log-log propre (tous les α
restent < 0.25 et R² < 0.25). La stratification existe sans
qu'aucun régime ne rétablisse une loi reconstructive.

**Conséquence** : P4-A reste P4-NO-STRONG-SIGNAL au sens de §25.5.
Aucune loi log-log propre, aucune saturation observable propre,
aucune pseudo-inversion forte. Mais la phrase *"nuage dispersé"*
doit être qualifiée : **le nuage n'est pas amorphe ; il est
stratifié par régimes**, principalement par l'opposition
intra-trajectoire / inter-bassin et par la dégénérescence B3.

**Formulation finale** : *P4 ne donne pas de loi moyenne
reconstructive, mais la dispersion de son nuage est structurée
par le mélange intra-trajectoire / inter-bassin et par la
dégénérescence B3.*

**Garde-fou** :
- P4VAR-STRATIFIED **ne valide pas** P4.
- P4VAR-STRATIFIED **ne rouvre pas** la self-opacity Ch3 §3.5.
- P4VAR-STRATIFIED **n'ajoute pas** de canal positif à la carte
  de réalisabilité.
- Il qualifie seulement la nature du no-signal de §25.

---

*Fin de §25. Le cadrage 6d initial (§1-10) reste valide. Les
amendements §11/§7ter/§12 à §24 intègrent les cycles 6d-α à
6d-λ, leurs relectures, la clôture de la branche λ/P5, et la
fermeture de la zone aveugle P5bis. §25 ferme la dette P4 stricte
sans signal structurant. Un audit de variance postérieur (§25.8)
qualifie ce no-signal : la dispersion du nuage P4 est stratifiée,
principalement par l'opposition intra-trajectoire / inter-bassin
et par B3, sans produire de loi reconstructive ni rouvrir P4.
La coda empirique de §22/§24 reste inchangée : §25 n'ajoute pas
de canal d'observation positif. Aucun verdict sur MCQ. Aucune
cible. Conformal-conservatif ou rien.*
