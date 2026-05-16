# 6d-α — Numerics specification (v2 amendée)

**Statut** : spec contraignante écrite avant tout refactor engine.
**Version** : v2 (amendement post-stress-test).
**Principe** : ce document précède le code et le contraint. Aucun choix
numérique n'est admis sans justification écrite ici. La spec ne se
rédige pas a posteriori.

**Changelog v2 vs v1** : 12 corrections / précisions issues du stress-test
auto-effectué :
- (a) §2.2 — solution analytique drift quadratique corrigée
- (b) §2.1 — σ_0 ≥ 1.5·dx pour respecter Nyquist
- (c) §2.1 — Var = 2D·t par axe explicite
- (d) §5 — test pivot reformulé en densité continue (arbitrage 1)
- (e) §1.4 — variant Crank-Nicolson semi-implicite explicite (arbitrage 2)
- (f) §1.4 + §10 — procédure fallback Backward Euler local positivité
- (g) §1.2 + §0bis — repositionnement épistémologique : "approximation
  conservative Euclidienne de l'opérateur conforme" (arbitrage 5)
- (h) §4.2 — latence reformulée en `1/(β·ψ_local)`
- (i) §6.4 — condition hors-équilibre uniquement
- (j) §1.0 — dx = 1.0 fixé explicitement (arbitrage 3)
- (k) §1.8 — perte de séparabilité actée formellement (arbitrage 4)
- (l) §5.6 — "conditionnel au régime numérique testé"

---

## 0. Discipline méthodologique

Chaque décision numérique de ce document doit comporter les six champs
suivants explicitement :

| Champ | Contenu |
|---|---|
| Hypothèse Ch3 concernée | Section précise (§3.X.Y) |
| Choix numérique retenu | Formulation discrète |
| Alternatives rejetées | Liste explicite |
| Raison du rejet | Argument structurel ou numérique |
| Artefacts attendus possibles | Risques connus |
| Observable de contrôle | Test diagnostique du choix |

Aucune décision n'est verrouillée sans ces six champs renseignés.
La rationalisation a posteriori est explicitement interdite — si une
décision est faite en cours d'implémentation, ce document doit être
amendé en première opération, avant le code.

### Vérification dimensionnelle obligatoire

Toute formule proposée dans ce document, ses annexes ou ses amendements
doit avoir ses unités vérifiées explicitement avant validation. Pour
chaque formule, lister les dimensions des termes individuels et
vérifier la cohérence avec la grandeur censée être produite (densité,
taux temporel, longueur, adimensionnel).

**Une formule sans vérification dimensionnelle explicite n'est pas
considérée comme verrouillée.** Cette règle s'applique rétroactivement :
les formules existantes non encore vérifiées sont à vérifier avant
implémentation.

Justification : le stress-test de la v1 a révélé 12 erreurs dont
plusieurs dimensionnelles (test pivot §5, A.1 récurrence, A.3 basin).
La vérification dimensionnelle est le garde-fou structurel le moins
coûteux contre ce pattern d'erreur.

---

## 0ter. Frontière ontologique numérique

6d-α valide **uniquement** :
- cohérence numérique interne du schéma
- compatibilité locale avec certaines signatures Ch3
- stabilité des observables émergentes dans le régime testé

6d-α ne valide **pas** :
- l'ontologie MCQ au sens Ch1
- τ' comme objet fondamental (cf. §5.6)
- RR³ comme réalité dynamique (cf. §6 — RR³ non mesurable directement
  sans g_Ω complet)
- les propriétés tensorielles Ch4
- les phénomènes QMCᴺ (N=1 imposé)

Tout résultat 6d-α s'interprète strictement dans le scope du premier
bloc. Toute conclusion qui chevauche le second bloc est hors champ.

### Avertissement : discret ≠ continuum

Les cas analytiques (§2) valident la **cohérence locale** du schéma
avec sa propre limite continue (convergence vers le bon continuum
quand dx→0). Ils **ne valident pas** l'équivalence dynamique entre
6d-α-sur-grille-5×5×5 et Ch3-au-continuum.

Sur grille 5×5×5, certains phénomènes peuvent être **purement
discrets** :
- effets de coin et bords amplifiés par la petite taille
- modes propres de l'opérateur discret différents du spectre continu
- résonances de grille (artefacts de stencil)
- cascade discrète à variance réduite

Conséquence : tout résultat 6d-β qui dépend de la résolution doit
être lu comme "valide sur grille 5×5×5", pas "valide en MCQ continu".

---

## 0quater. Co-construction — avertissement méta-méthodologique

Les garde-fous numériques de 6d-α (D_min > 0, ψ_floor, fallback
positivity Backward Euler, clipping h ∈ [h_min, h₀], bruit minimal
aux régions actives, exclusions des cellules saturées en §5)
**introduisent des conditions de stabilité non strictement dérivées
de Ch3** mais nécessaires à l'intégrabilité du schéma discret.

Conséquence : **6d ne teste pas l'émergence pure des propriétés MCQ**.
6d teste **l'exploration contrainte de leur réalisabilité numérique**
dans un schéma qui pré-suppose une viabilité minimale par construction.

La frontière entre :
- **émergence dynamique** : propriété qui apparaît malgré les
  contraintes minimales
- **stabilisation implémentée** : propriété qui dépend d'un garde-fou
  injecté

est **structurelle**, pas évitable. Chaque observation 6d-β doit être
passée au crible : *cette propriété émerge-t-elle malgré les garde-fous,
ou parce qu'un garde-fou a été nécessaire à son apparition ?*

### Statut épistémique de cette section

Cette section est **méta-méthodologique**, pas théorique. Elle décrit
la nature du dispositif 6d-α, pas une propriété MCQ validée.

**Note réflexive** : on peut observer que ce dispositif lui-même
exhibe une propriété structurellement analogue à ce que MCQ décrit —
la viabilité numérique du système dépend de garde-fous qui modifient
les conditions de viabilité observées. Cette analogie est
**méthodologiquement structurante** et mérite d'être notée. Elle
**peut être lue comme analogue** à une propriété MCQ, mais **elle ne
constitue pas une proposition théorique validée**. Toute promotion de
cette observation en théorème MCQ serait prématurée à ce stade.

### Alarme active — isomorphisme structurel partiel (B1 audit)

Au-delà de l'analogie ci-dessus, une **tension structurelle plus
précise** doit rester active comme alarme :

**Les contraintes de viabilité numérique tendent à converger vers des
structures analogues aux contraintes de viabilité théoriques.**

Correspondances observables :
- `D_min > 0` ↔ non-absorption diffusive (Ch3 §3.1.6)
- `ψ_floor` ↔ présence minimale (que Ch3 refuse mais que le discret
  impose)
- `fallback positivity BE` ↔ maintien de positivité (KNV 5 transposé)
- `clipping h ∈ [h_min, h₀]` ↔ corridor métrique (§3.4.1 prédit, pas
  posé)
- `exclusion saturation §5` ↔ KNV 7 (pétrification) délimité par
  construction
- `CFL adaptatif` ↔ adaptation temporelle multi-échelle (proto-RR³ ?)

Cette correspondance n'est pas accidentelle : pour qu'un schéma
discret soit viable, il doit reproduire approximativement les
conditions de viabilité du continuum. Mais cela crée une **alarme
structurelle** :

> Les stabilisateurs numériques **ne reproduisent pas MCQ** — ils
> sont **analogues** à certaines de ses structures de viabilité.
> Cette distinction est importante : *reproduisent* impliquerait
> identité ; *analogues* signale une tension structurelle sans
> collapse interprétatif.

**Lecture honnête de ce que 6d teste vraiment** :

> **6d ne teste probablement pas l'émergence de la viabilité
> elle-même.**
> 
> **6d teste quels phénomènes émergents Ch3 deviennent observables
> quand certaines conditions minimales de viabilité sont déjà
> préservées par le schéma.**

Ce qui reste extrêmement intéressant — mais ce n'est pas la même
chose qu'une validation MCQ par émergence pure.

---

## 0bis. Repositionnement épistémologique (correction g + C2 audit)

Le schéma 6d-α est une **approximation conformal-conservative de la
dynamique géométrique Ch3**. Il **ne réalise pas** :
- le Laplace-Beltrami strict
- une géométrie riemannienne complète (pas de Christoffel, pas de
  transport parallèle non-trivial, pas de divergence covariante stricte)
- une structure measure-preserving sur variété

Ce qu'il réalise est une **géométrie effective de flux** où h(θ)
module les coefficients d'un opérateur conservatif Euclidien :

- Au sens Laplace-Beltrami strict (non implémenté) :
  `∂_t ψ = (1/h^d)·∂_a(h^{d-2}·D·∂_a ψ)`,
  qui conserve `∫ψ·h^d dθ` (mesure géométrique) mais pas `∫ψ dθ`.
- Au sens conformal-conservatif Euclidien (retenu ici) :
  `∂_t ψ = ∂_a(h^{d-2}·D·∂_a ψ)`,
  qui conserve `∫ψ dθ` (mesure Euclidienne, cohérent avec Ch3 §3.1.8 :
  *"Conservation of presence: ∫_{Θᵢ} ψᵢ^α(t, θ) dθ = 1 for all t"*).

**Le lien avec Ch3 est préservé via h(θ) qui module les flux**, mais
la dynamique conformal-conservative **n'est pas géométriquement
équivalente** à la dynamique Ch3 stricte. C'est une approximation
explicite avec dette tensorielle (Ch4) et dette géométrique
(formulation riemannienne complète, hors 6d).

**Ce que 6d-γ produira est donc** : une carte des limites de
l'approximation **conformal-conservative** de la géométrie Ch3, à
métrique conforme `g_{ab} = h²·δ_{ab}`. Toute généralisation à Ch4
(tensorialisation H(θ)) doit être lue à travers ce filtre — la
"conformal approximation scalaire" telle que testée ici **n'est pas**
le Laplace-Beltrami strict, et n'est pas non plus une géométrie
riemannienne complète.

Le choix conservatif Euclidien est dicté par Ch3 §3.1.8 (conservation
Euclidienne demandée). La conservation géométrique aurait demandé une
renormalisation permanente avec dépendance forte au schéma
h-discrétisation.

---

## 1. Discrétisation et schéma

### 1.0 Grille et unités (correction j)

| Paramètre | Valeur fixe | Justification |
|---|---|---|
| Dimension d | 3 | Modèle factoriel T/M/I |
| Taille N par axe | 5 | Hérité 6c-B, suffisant pour tests 6d-α/β |
| dx (pas de grille) | 1.0 | Θ adimensionnel, espace cartésien unitaire |
| Volume Θ | 5×5×5 = 125 cellules | dx³ = 1, donc N_total = 125 |
| Densité ρ | ρ(θ) = ψ_ijk / dx³ = ψ_ijk | Coïncidence numérique en dx=1 |

**Note** : la coïncidence ψ_ijk ≡ ρ_ijk en dx=1 simplifie les expressions,
mais conceptuellement on travaille en densité ρ pour les comparaisons
théoriques (cf. test pivot §5).

### 1.1 Représentation d'état

| Hypothèse Ch3 | §3.1.1 — état étendu Ψᵢ^α = (ψᵢ^α, dᵢ^{ℋ,α}) |
|---|---|
| Choix retenu | `psi[5,5,5]` + `h[5,5,5]` par module (h scalaire plein sur Θ) |
| Alternatives rejetées | (i) marginales h_T[5], h_M[5], h_I[5] ; (ii) tenseur H[5,5,5,3,3] |
| Raison rejet (i) | Saturé en V4 6c-B, ne reproduit pas l'anisotropie spatiale Ch3 §3.3.5 |
| Raison rejet (ii) | Dette Ch4 explicite, prématuré ici |
| Artefacts attendus | Anisotropies sub-grille à h variant rapidement |
| Contrôle | Test isotropie discrète §3 |

### 1.2 Forme du Laplacien conformal-conservatif

| Hypothèse Ch3 | §3.1.3.I + §3.1.8 — opérateur intrinsèque à dᵢ^ℋ ; conservation Euclidienne |
|---|---|
| Choix retenu | Forme conservative volumes finis : `∂_t ψ = ∇·J` avec `J = h^{d-2}·D_eff·∇ψ`, d=3, où ∇ est l'opérateur différentiel Euclidien (différences finies sur grille θ) |
| Alternatives rejetées | (a) Laplace-Beltrami strict en forme non-conservative ; (b) Laplacien Euclidien avec coefficient h-modulé (ce que V4 fait) ; (c) random walk on graph weighted by h |
| Raison rejet (a) | Conserverait `∫ψ·h^d dθ`, pas `∫ψ dθ` — non-aligné Ch3 §3.1.8 |
| Raison rejet (b) | Viole §3.1.3.I — n'est pas l'opérateur conformément modulé |
| Raison rejet (c) | Biais aux coins de grille, pas de littérature standard de validation |
| Artefacts attendus | Anisotropies axiales sub-grille, dissipation numérique aux interfaces |
| Contrôle | Cas analytique 1 (diffusion isotrope) §2.1 + isotropie discrète §3 |

**Statut** : ce n'est **pas** un Laplacien géométrique au sens
riemannien strict. C'est une **approximation conformal-conservative**
qui module les flux Euclidiens par h(θ). Voir §0bis pour la dette.

**Forme discrète** :
```
J_{i+1/2,j,k} = -h_{i+1/2,j,k} · D_{eff,i+1/2,j,k} · (ψ_{i+1,j,k} - ψ_{i,j,k}) / dx
∂_t ψ_{i,j,k} = -(J_{i+1/2,j,k} - J_{i-1/2,j,k}) / dx + ... (j,k analogues)
```

avec `h_{i+1/2,j,k} = harmonic_mean(h_{i,j,k}, h_{i+1,j,k}) =
2·h_{i,j,k}·h_{i+1,j,k} / (h_{i,j,k} + h_{i+1,j,k})` (assure positivité du
flux et symétrie de l'opérateur).

**Rappel pédagogique critique pour implémentation** : le coefficient de
face en 3D est **`h_face` à la puissance 1**, pas `1/h` ni `1/h²`. Cela
provient de `h^{d-2} = h^{3-2} = h¹`. Si l'implémentation introduit
`1/h_face`, c'est un bug d'inversion à corriger immédiatement.

### 1.3 Forme du drift conformal-conservatif (correction B1 audit)

| Hypothèse Ch3 | §3.1.3.II — drift dans gradient de Φ_eff évalué dans dᵢ^ℋ |
|---|---|
| Choix retenu | `J_drift = h^{d-2}·ψ·(∂Φ_eff/∂θ)` avec `∂/∂θ` au sens **Euclidien** (différences finies sur grille θ), schéma upwind pour positivité |
| Alternatives rejetées | (a) Gradient géométrique `g^{ab}·∂_b Φ` = `(1/h²)·∂Φ` ; (b) Schéma central |
| Raison rejet (a) | Donnerait `J_drift = h^{d-2-2}·ψ·∂Φ = h^{-1}·ψ·∂Φ`, structure inversée par rapport à `J_diff = h¹·D·∂ψ`. Incohérence des coefficients diffusion/drift. |
| Raison rejet (b) | Peut produire ψ < 0 aux gradients raides, viole positivité |
| Artefacts attendus | Diffusion artificielle upwind d'ordre 1 |
| Contrôle | Cas analytique 2 (drift quadratique pur) §2.2 |

**Convention déclarée — conservative Euclidienne uniforme** :

Diffusion `J_diff = h^{d-2}·D·∇ψ` et drift `J_drift = h^{d-2}·ψ·∇Φ`
partagent la **même structure de flux** (coefficient `h¹` en 3D) et le
**même opérateur différentiel ∇ au sens Euclidien**. Cette uniformité
maintient la conservation `∫ψ dθ` Euclidienne pour les deux termes.

**Conséquence structurelle** : le système **n'est pas un gradient-flow
géométriquement cohérent** au sens Laplace-Beltrami strict. C'est une
**convention conservative Euclidienne** appliquée uniformément. Le
detailed balance au sens Wasserstein-2 géométrique n'est **pas
garanti** — ce qui peut produire, en théorie, des biais stationnaires
dans des régimes à h très hétérogène.

Cette absence de detailed balance géométrique est un **trade-off
explicite**, pas une omission. Elle est justifiée par la priorité
donnée à la conservation Euclidienne (Ch3 §3.1.8) sur la cohérence
géométrique stricte (qui aurait demandé une renormalisation
permanente).

**Schéma upwind discret** : `J_drift_{i+1/2} = h_face · ψ_upwind · ∂Φ_face`
avec ψ_upwind = ψ_i si ∂Φ_face > 0 (flux vers +x), ψ_{i+1} sinon.

### 1.4 Schéma temporel — semi-implicite Crank-Nicolson (corrections e+f)

| Hypothèse Ch3 | §3.1.5 — D_eff modulé par g_Ω, peut spike pendant anti-collapse |
|---|---|
| Choix retenu | **Semi-implicite Crank-Nicolson** : opérateur de diffusion linéarisé avec coefficients (h, D_eff) évalués à t_n, schéma de Crank-Nicolson appliqué à l'opérateur linéarisé. Drift et coupling : Euler explicite. |
| Alternatives rejetées | (a) Tout explicite avec dt très petit ; (b) Plein implicite Crank-Nicolson (itéré pour atteindre t_{n+1/2}) |
| Raison rejet (a) | CFL violée pendant g_Ω·D_eff spikes |
| Raison rejet (b) | Itération peut ne pas converger ou être très coûteuse |
| Conséquence semi-implicite | **Ordre 1 en temps** au lieu d'ordre 2 nominal Crank-Nicolson. Trade-off accepté : simplicité d'implémentation et robustesse. |
| Artefacts attendus | Amortissement Crank-Nicolson sur hautes fréquences temporelles ; perte de précision temporelle d'ordre 2 |
| Contrôle | Comparaison explicite vs implicite §9 (tolérance relâchée à **10%** vu l'ordre 1, pas 5%) |

**Schéma précis** :
```
A_n = I - (dt/2) · L(h_n, D_eff(ψ_n))    # linéarisé à t_n
B_n = I + (dt/2) · L(h_n, D_eff(ψ_n))
ψ^*_{n+1} = solve(A_n · ψ_{n+1} = B_n · ψ_n)            # diffusion semi-impl
ψ_{n+1} = ψ^*_{n+1} + dt · (drift_explicit + coupling_explicit + noise)
```

**Procédure fallback positivité** (correction f) :
1. Après chaque step CN, vérifier `min(ψ_{n+1}) ≥ -ε_positivity`
   avec ε_positivity = 1e-12.
2. Si violation, **rollback ce step** et re-exécuter avec **Backward
   Euler** (1er ordre, inconditionnellement positif) sur la diffusion :
   ```
   A_n^BE = I - dt · L(h_n, D_eff(ψ_n))
   ψ_{n+1} = solve(A_n^BE · ψ_{n+1} = ψ_n) + dt · (drift + coupling + noise)
   ```
3. Logger événement BE-fallback dans diagnostics §8, **avec
   localisation spatiale** (correction B3 audit) :
   - Carte `BE_count[i,j,k]` accumulée sur la simulation (incrémentée
     à chaque step où la cellule (i,j,k) déclenche le rollback)
   - Corrélation entre BE-fallback et régions de gradient fort
     ∇ψ ou de h faible
4. **Alarme concentration** : si BE-fallback **concentré** (>50% des
   occurrences dans <20% du volume Θ), pathologie locale même si la
   fréquence globale est sous le seuil 5%.
5. Si fréquence globale BE-fallback > 5% des steps sur run de
   validation, alarme schéma : dt probablement trop grand ou
   pathologie locale.

**Risque structurel à surveiller (B3 audit)** : Backward Euler est
extrêmement dissipatif. Si le fallback se déclenche systématiquement
aux régions à gradient fort ou aux épisodes anti-collapse, il peut
**filtrer dissipativement** précisément les phénomènes que 6d veut
mesurer (latence, mémoire résiduelle, cross-talk, émergence de
corridors). La concentration spatiale est donc un diagnostic plus
informatif que la fréquence globale.

**Solveur linéaire** : `scipy.linalg.solve` dense sur matrice 125×125.
Reconstruction de A_n à chaque step car h évolue. Pas de précalcul
d'inverse, pas de solveur itératif (overhead inutile à cette échelle).

### 1.5 Bruit multiplicatif conservatif

| Hypothèse Ch3 | §3.1.3.V — η = σ_η·D_eff·ψ·ξ, multiplicatif, structuré par ψ |
|---|---|
| Choix retenu | Bruit sur arêtes : à chaque arête (i,j,k)-(i+1,j,k), ξ_edge ~ N(0,1) iid, perturbation de flux `J_noise = σ_η·D_eff·sqrt(ψ_eff·dt)·ξ_edge` avec ψ_eff = max(ψ_face, ψ_floor) |
| Alternatives rejetées | (a) Bruit sur cellules avec correction de masse a posteriori ; (b) Bruit Itô-Stratonovich sur ψ directement |
| Raison rejet (a) | Correction de masse = opérateur global caché, peut introduire synchronisation implicite et homéostasie numérique |
| Raison rejet (b) | Difficile à conservativiser, dépendance forte au schéma de discrétisation |
| Artefacts attendus | Vortex de bruit aux échelles de grille, corrélations spatiales induites par stencil |
| Contrôle | Logging §8 (variance flux bruités, corrélations spatiales) |

Note : le facteur `sqrt(dt)` dans l'amplitude est essentiel pour que
la contribution de bruit converge dans la limite continue (Wiener
increment).

**ψ_floor = 10⁻⁵·ψ_typical** (correction B1 audit, où ψ_typical ≈ 1/N_total = 0.008 sur 5×5×5).

**Caveat ontologique explicite** : ψ_floor est un **artefact numérique
de discrétisation conservative**, pas une hypothèse MCQ.

La discrétisation `η ∝ √ψ` (volumes finis conservatifs) lie
l'irréductibilité du bruit à l'irréductibilité locale de présence —
c'est une **conséquence de la conservation de masse discrète**, pas
un postulat physique. Au continuum, η reste non-éliminable même quand
ψ = 0 (Ch3 §3.1.7) via la structure multiplicative
`σ_η·D·ψ·ξ` qui peut diverger en `√(temps)` tout en gardant ψ près de
0 en moyenne.

Sur grille discrète conservative, cette équivalence se brise. ψ_floor
est introduit pour **préserver la garantie §6.3** (bruit non
éliminable) **sans falsifier la dynamique normale** (10⁻⁵·ψ_typical
est suffisamment petit pour ne pas affecter les régions actives).

**Ne pas confondre** : ψ_floor introduit une **présence numérique
minimale**, pas une **présence MCQ minimale**. Ch3 ne pose **pas** de
présence minimale ontologique. Le glissement vers une lecture
"présence minimale fondamentale" serait une dérive ontologique à
éviter (cf. §0quater sur la co-construction).

**Addendum empirique post-implémentation (étapes 3a et 3a-bis)** :

Une fois le bruit conservatif sur arêtes implémenté et testé, trois
résultats empiriques sont à acter explicitement.

(i) **Trois couches d'invariants** identifiées et mesurées :
- (A) **Invariants structurels exacts** (conservation par
  construction antisymétrique flux ; antisymétrie). Vérifiés à
  machine precision (10⁻¹⁶) pour tous σ_η testés.
- (B) **Invariants statistiques** (moyenne d'ensemble cohérente
  avec déterministe au sens SEM ; scaling variance inter-runs ∝ σ_η).
  Mesurés via critère `diff/SEM < 3-4` au lieu d'un seuil relatif
  fixe (calibration statistique correcte).
- (C) **Invariants faibles/cassables** (positivité, stabilité locale).
  Cassables au-dessus d'un seuil en σ_η.

(ii) **Régime "conservatif mais non-positif"** identifié comme classe
comportementale explicite du schéma : à D=0.1, pour σ_η ≥ 0.25, ψ
devient massivement non-positif (100% des runs sur 20 répétitions
par configuration) tout en préservant la conservation à machine
precision. Min ψ atteint jusqu'à -3e-02.

(iii) **Stratification empirique de la transition** (cartographie
3a-bis à D=0.1 fixé) :
- σ_η = 0.05 → 0% violations sur 20 runs × 4 dt
- σ_η = 0.10 → 5-10% violations
- σ_η = 0.25 → 100% violations
- σ_η = 0.50 → 100% violations, magnitude ψ<0 plus forte

**Résultat établi** : à D fixé, σ_η domine dt dans la cartographie.
La transition est brutalement stratifiée par σ_η avec dépendance
faible en dt sur la fenêtre testée.

**Hypothèse candidate ouverte** : la frontière de positivité pourrait
dépendre d'un ratio bruit/diffusion de type σ_η²/D ou σ_η²·D selon
la normalisation effective du flux discret. 3a-bis ne peut pas
trancher parce que D reste constant dans le sweep actuel. Un
mini-sweep en D serait nécessaire pour identifier la forme
fonctionnelle exacte.

**Caveat conceptuel important** : ces résultats sont des propriétés
**du schéma stochastique discret** (Itô multiplicatif conservatif
Euclidien volumes finis avec ψ_floor sur grille 5×5×5 Neumann), pas
des propriétés de ψ en général au sens MCQ. Le pont éventuel entre
"schéma viole positivité au-dessus d'un seuil" et toute lecture
théorique (ψ champ tensionnel sous contraintes incompatibles partielles,
non-additivité de stabilisateurs) doit être traité comme **alarme
§0quater à surveiller**, pas comme théorème MCQ.

**Conséquence pratique pour 3b et au-delà** : restreindre les tests
combinés (OU + bruit, puis h dynamique + bruit) au **régime sûr**
σ_η ≤ 0.10 (à D=0.1) pour préserver la positivité. Le régime
non-positif reste disponible mais doit être explicitement marqué
comme test du schéma sous stress, pas comme test de la dynamique
nominale.

**Addendum post-3b — révision de régime productive** :

L'étape 3b (OU + bruit) a révélé que **le critère "régime sûr σ_η ≤ 0.1"
issu de 3a-bis n'est pas invariant sous l'ajout du drift**. Mesures
empiriques à σ_η = 0.05, D = 0.1 sur 50 runs :
- sans drift (3a-bis) : 0% runs avec ψ<0
- avec drift k=0.1 : 24/50 runs avec ψ<0 (48%)
- avec drift k=0.5 : 50/50 runs avec ψ<0 (100%)

**Lecture théorique** : le drift contracte la masse vers le minimum
de Φ, créant des **régions à ψ proche de zéro** loin du minimum
(typiquement près des bords). Dans ces régions, le bruit relatif
devient localement dominant — l'incrément stochastique
σ_η·D·√(ψ_face·dt)·ξ reste petit en absolu mais grand par rapport à
ψ_local. Donc franchissement ψ<0 facilité par concentration drift.

**Formulation prudente** : la robustesse de positivité dépend de
**l'interaction bruit × confinement × structure spatiale de ψ**.
Pas d'expression fonctionnelle σ_η_crit = f(k) à fixer prématurément.

**Distinction structurelle à conserver** :

> **Conservation est topologique** (tient par construction antisymétrique
> des flux, indépendante de l'état). **Positivité est dynamique**
> (dépend de l'état instantané ψ et de la géométrie effective).

Cette distinction est centrale pour la lecture des trois couches
d'invariants (§ci-dessus). Elle reste valide à 6d-α et le restera
quand h devient évolutif (3c et au-delà).

**Verdict 3b reformulé** : **PASS STRUCTUREL + RÉVISION DE RÉGIME**
- (A) Conservation à machine precision ✓
- (B) Moyennes d'ensemble cohérentes (diff/SEM < 2) ✓
- (C) Positivité non robuste sous confinement drift : MESURÉ et
  documenté comme propriété du schéma sous bruit multiplicatif Itô
  conservatif Euclidien, pas comme échec du modèle MCQ.

### 1.6 Conditions aux bords

| Hypothèse Ch3 | Aucune spécification explicite ; Θ supposé non-périodique |
|---|---|
| Choix retenu | Neumann zero-flux sur ψ et h (∂_n ψ = 0, ∂_n h = 0) |
| Alternatives rejetées | (a) Periodic ; (b) Dirichlet h = h₀ aux bords ; (c) Mixed |
| Raison rejet (a) | Topologie circulaire non dans Ch3, introduit cycles artificiels |
| Raison rejet (b) | Force la métrique aux bords, contraint la dynamique morphologique près du bord |
| Raison rejet (c) | Mélange de conditions sans justification structurelle |
| Artefacts attendus | Accumulation aux coins de grille, biais corner cells |
| Contrôle | Logging accumulation aux bords, test isotropie radiale §3 |

### 1.7 Pas de temps et CFL

| Hypothèse Ch3 | β ≪ D₀ (§3.2.5.IV) |
|---|---|
| Choix retenu | dt = min(0.5·dx²/(2·D_max·d), dx/‖∇Φ_max‖, 0.1/β, 0.1/γ) où D_max est la borne supérieure de g_Ω·D₀·f(H,R_prod) |
| Alternatives rejetées | dt fixe ad hoc |
| Raison rejet | Violation possible CFL pendant épisodes anti-collapse |
| Artefacts attendus | Adaptation dt peut introduire pas de temps variable, log diagnostic |
| Contrôle | Logger dt utilisé à chaque step, vérifier qu'il ne s'effondre pas |

### 1.8 Couplage non séparable — actage formel (correction k, arbitrage 4)

**Réalisation théorique fondamentale issue du stress-test :**

En 6c-B, h était marginal (h_T[5], h_M[5], h_I[5]). Donc les couplages
`g_k = θ_k/h_k(θ_k)` étaient **séparables par axe** : g_T ne dépendait
que de θ_T.

En 6d, h(θ) est plein sur Θ. Donc :
```
g_T(θ_T, θ_M, θ_I) = θ_T / h(θ_T, θ_M, θ_I)
```

**Conséquences structurelles non-anodines** :

1. **Perte de séparabilité** : la contribution de l'axe T au facteur
   partagé dépend non seulement de θ_T mais aussi de θ_M et θ_I via
   h(θ). Ce qui se passe sur l'axe M (sédimentation différentielle de h
   en fonction de θ_M) **affecte** la contribution g_T.

2. **Cross-talk métrique** : une contraction localisée à un point
   (θ_T_0, θ_M_0, θ_I_0) modifie h(θ) localement, ce qui modifie g_k(θ)
   pour tous k autour de ce point. Ce cross-talk est **absent** en 6c-B
   par construction.

3. **Lecture phénoménologique** : tout signal différentiel entre 6c-B
   et 6d sur les facteurs partagés peut être attribué soit à h(θ) plein
   en lui-même, soit à ce cross-talk métrique induit. Distinguer les
   deux nécessite des protocoles spécifiques.

**Cette perte de séparabilité est la signature même de la métrique
pleine et justifie à elle seule la phase 6d.** Elle doit être au centre
des lectures phénoménologiques 6d-β et 6d-γ.

**Test diagnostic associé** : protocole d'isolation cross-talk —
comparer g_k mesuré sous h(θ) plein vs h(θ) artificiellement diagonal
(reconstitué depuis les marginales). Si différence significative, le
cross-talk est actif et son amplitude est mesurée.

---

## 2. Cas tests analytiques purs

Trois cas où la solution est connue au continuum. **Tous doivent passer
avec tolérance documentée** avant de passer à la couche §4.

### 2.1 Cas 1 — Diffusion pure isotrope homogène (corrections b, c + amendement empirique étape 1)

Conditions : h(θ) = h₀ uniforme, V_𝒩 = 0, σ_η = 0, β = γ = 0
(co-production gelée), couplage absent.

**Initialisation** : ψ_0(θ) = gaussienne centrée θ_0, **σ_0 = 1.5·dx
minimum** (correction b — respect Nyquist sur grille 5×5×5).
Recommandation : σ_0 = 1.8·dx pour marge.

**Amendement empirique (étape 1 implémentation)** : la prédiction
continuum `Var(t) = σ_0² + 2·D·t` **n'est pas un critère PASS/FAIL
valide sur grille 5×5×5 avec Neumann**, parce que :

- La variance discrète d'une distribution centrée sur 5×5×5 est
  **bornée à Var_max = 2.0** (uniforme limite sur 5 points centrés
  [0..4]).
- Les conditions Neumann zero-flux **réfléchissent** la diffusion,
  donc dVar/dt **décroît continûment** vers 0 à mesure que ψ
  approche de l'uniforme stationnaire.
- Le spectre discret du Laplacien Neumann produit une **somme
  exponentielle de modes**, pas une croissance linéaire.

L'écart `continuum_gap = |dVar_premier_step - 2·D| / 2·D` peut être
significatif (~60% sur σ_0=1.5 en première mesure empirique) **sans
indiquer un bug du schéma**. C'est l'effet §0ter "discret ≠ continuum"
en action.

**Critère PASS amendé** : engine 3D doit coïncider avec un
**référentiel Neumann discret** construit explicitement, pas avec la
solution continuum infinie.

Référentiels Neumann discrets :
- **Euler discret 1D** : `(I + dt·D·L_N)^n · p_0` où L_N est le
  Laplacien discret 1D avec Neumann zero-flux, p_0 la marginale
  gaussienne initiale. Critère : engine 3D = Euler 1D à **machine
  precision** par séparabilité (gaussienne séparable + h uniforme
  + opérateur séparable).
- **Semi-discret exact** : `exp(D·L_N·t)·p_0`. Critère : engine 3D
  vs semi-discret à l'**erreur de discrétisation temporelle Euler
  près**, ≈ O(D²·dt²·t·‖L_N‖²) ≈ `dt·D·t · 10` comme seuil souple.

Solution analytique continuum (référence diagnostic non-bloquante) :
`ψ(θ, t) = (4πD_eff·t')^{-3/2}·exp(-‖θ-θ_0‖²/(4D_eff·t'))`
avec `t' = t + σ_0²/(2D_eff)`.

**Critères PASS de §2.1 (amendés)** :

| Critère | Mesure | Seuil |
|---|---|---|
| Conservation masse | `max |∑ψ - 1|` | 1e-10 |
| Positivité | `min ψ` au cours du temps | ≥ -1e-12 |
| Isotropie axes | `max |Var_a - Var_x|/Var_x` | 5% (10% si soft-fail §3) |
| Centre de masse stable | `max ‖COM(t) - COM(0)‖` | 1e-10 |
| **Engine 3D vs Euler 1D Neumann** | `max |ψ_engine_marg - ψ_euler1d|` | 1e-12 (machine precision) |
| **Engine 3D vs semi-discret 1D** | `max |ψ_engine_marg - exp(D·L·t)·p_0|` | `10·D·dt·t` (erreur Euler attendue ×10) |

**Diagnostic non-bloquant** :
- `continuum_gap_pct = |dVar_premier_step - 2D| / 2D · 100`
- `deviation_from_uniform = max |ψ - 1/N_total|`

Ces deux ne déclenchent **pas** un échec. Ils documentent l'écart à
la limite continuum infinie attendu sur petite grille bornée.

**Si échec d'un critère PASS** : bug schéma diffusion ou problème
d'interpolation initiale. Le critère "engine vs Euler 1D à machine
precision" est particulièrement fort : un écart > 1e-12 signale
quasi-certainement un bug du Laplacien discret ou de la divergence.

### 2.2 Cas 2 — Drift pur dans potentiel quadratique (correction a)

Conditions : h = h₀ uniforme, σ_η = 0, β = γ = 0, D_eff = 0,
Φ = ½·k·‖θ-θ_0‖², couplage absent.

Initialisation : ψ_0 gaussienne large (σ_0 ≥ 2·dx, centrée θ_0 ou
décentrée pour test transport).

**Solution analytique correcte** (correction a) :
- L'équation est `∂_t ψ = -∇·(ψ·∇Φ) = -k·d·ψ - k·(θ-θ_0)·∇ψ` (d=3)
- Méthode des caractéristiques : `ψ(θ, t) = ψ_0(θ_0 + (θ-θ_0)·e^{kt}) · e^{kdt}`
- **La variance autour de θ_0 décroît comme `Var(t) = Var_0·e^{-2kt}`**
- La masse totale est conservée par le facteur `e^{kdt}` qui compense
  la concentration géométrique.

**Test** :
- Mesurer Var(t) à plusieurs instants t_1, t_2, t_3.
- Ajuster une décroissance exponentielle : log(Var(t)) ≈ log(Var_0) - 2k·t.
- Tolérance : 10% sur le coefficient -2k mesuré vs théorique.

**Si échec** : bug schéma drift, problème upwind, ou conservation de masse.

### 2.3 Cas 3 — Ornstein-Uhlenbeck stationnaire

Conditions : h = h₀ uniforme, β = γ = 0, D_eff constant > 0,
Φ = ½·k·‖θ-θ_0‖², σ_η = 0.

Initialisation : ψ_0 gaussienne large centrée θ_0.

**Solution analytique** :
- ψ_stationnaire = gaussienne centrée θ_0, **variance par axe Var_∞ = D_eff/k**
- Convergence exponentielle au taux 2k vers cet équilibre.

**Test** :
- À t >> 1/(2k), mesurer Var_axis_obs.
- Comparer à Var_∞_axis = D_eff/k.
- Tolérance : 5%.
- Vérifier les 3 axes équivalents.

**Si échec** : équilibre diffusion/drift mal balancé numériquement.

---

## 3. Validation isotropie discrète

| Hypothèse Ch3 | §3.3.5 — anisotropie émergente, pas pré-câblée par schéma |
|---|---|
| Test | Diffusion radiale d'une gaussienne σ_0 = 1.8·dx centrée sur Θ_centre, h = h₀ uniforme, V_𝒩 = 0, σ_η = 0 |

**Critères de passage** :
- Var le long des 3 axes principaux (x, y, z) identique à **5%** près.
- Var le long des 4 diagonales corps (1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)
  identique aux axes à **10%** près.

**Soft-fail acceptable** (résolution intrinsèque 5×5×5) :
- Si test passe à 12-15% sur diagonales : documenter comme limite
  résolution intrinsèque, valider isotropie qualitative, passer.
- Si test échoue à >15% sur diagonales ou >5% sur axes : bug stencil,
  bloquant.

**Si échec dur** : le stencil discret introduit des directions privilégiées
qui contamineront toute mesure d'anisotropie en 6d-β.

**Si réussite** : toute anisotropie observée en 6d-β est interprétable
comme phénomène, pas comme artefact.

**Conséquence pour 6d-β (correction C3 audit)** : la tolérance
d'isotropie validée détermine le **seuil de lisibilité** des
anisotropies en 6d-β. Si l'isotropie passe à X% (par exemple 8% ou
13% en soft-fail), **toute anisotropie < X% observée en 6d-β est au
niveau du bruit numérique** et ne peut pas être lue comme phénomène
significatif.

En particulier : si soft-fail à 13%, les anisotropies < 13% en 6d-β
sont interprétativement inutilisables. La quantification fine des
faibles anisotropies est exclue par construction. Seuls des effets
clairement supérieurs au seuil d'isotropie validé sont lisibles.

---

## 4. Cas tests semi-analytiques co-production

Trois tests où la prédiction Ch3 est calculable mais où ψ et h
co-produisent activement. **Bloc obligatoire avant test pivot §5.**

### 4a-0 — Statut ontologique du régime h(t) (mini-spec préalable)

Avant les tests §4.1-§4.3, inscrire explicitement la rupture
conceptuelle introduite par `h(t)` évolutif. Cette mini-spec
précède tout code §4 et conditionne les critères de validation.

**Rupture méthodologique** : jusqu'à §3b inclus, ψ évoluait dans une
géométrie h donnée. À partir de §4, **ψ modifie la géométrie qui
modifie ψ**. La référence matricielle fixe `L_total(Φ, h, D)` utilisée
en §2.1-§2.3 et §3b **n'existe plus** comme objet unique : `L_total`
devient fonction du temps via `h(t)`.

**Rupture plus profonde — mémoire morphologique du support** : même
si `L_total(t)` reste instantanément linéaire en ψ, la dynamique
globale **n'est plus markovienne au sens simple en ψ seul**, parce
que `h` accumule une trace de l'histoire de ψ. Le support devient
**historique**. Cette propriété change la classe dynamique du
système, pas seulement la forme des références numériques.

**Audit des invariants à l'entrée du régime h(t)** :

| Invariant | Statut à h(t) | Justification |
|---|---|---|
| (A1) Σψ = 1 conservation | **SURVIT** | Antisymétrie flux ψ indépendante de h |
| (A2) Antisymétrie flux ψ | **SURVIT** | Topologique, indépendante de h(t) |
| (A3) h ∈ [h_min, h₀] postulé | **À VALIDER empiriquement** | Postulat MCQ-théorique D_min>0 |
| (B1) Moyenne ensemble vs réf déterministe | **SURVIT redéfini** | Réf = trajectoire co-évolutive σ_η=0 |
| (B2) Stationnaire ψ_∞ ∝ exp(-Φ/D) | **NE SURVIT PAS** | Remplacé par co-attracteurs `(ψ*, h*)` |
| (C1) Positivité ψ ≥ 0 | **DYNAMIQUE** | Dépendance attendue à trajectoire h(t) |
| (C2) Stabilité CFL | **REQUIERT révision** | Recalculée par step ou fixée sur h₀ initial |

**Nouveaux invariants émergents** :

(D1) **h comme variable d'état morphologique** : h ∈ [h_min, h₀]
postulé, à mesurer comme propriété émergente. Si h sort de cette
plage sans contrôle, signal de violation MCQ-admissibilité.

(D2) **Co-évolution (ψ(t), h(t))** : pas de référence matricielle
exacte. On peut comparer engine vs intégration RK4 indépendante
sur même `(ψ₀, h₀)` pour vérifier la cohérence numérique du schéma.

(D3) **Cohérence MCQ-admissibilité** : `d_𝓗(xᵢ, KNV) > ε` à
surveiller via proxies (h_min atteint, ψ_max atteint, accumulation
aux bords).

(D4) **Hiérarchie des temps caractéristiques** :
- τ_ψ ∼ dx²/D (diffusion) ou 1/k (drift)
- τ_noise ∼ 1/σ_η² (Wiener)
- τ_h ∼ 1/(β·ψ) (sédimentation) ou 1/γ (érosion)

Le régime qualitatif dépend des rapports `τ_h/τ_ψ` :
- `τ_h ≫ τ_ψ` : régime quasi-statique, h "voit" ψ moyenné
- `τ_h ∼ τ_ψ` : co-évolution forte, mémoire morphologique active
- `τ_h ≪ τ_ψ` : h équilibre instantanément, ψ voit géométrie effective

**Distinction structurelle à conserver** (héritée 3b) :
> **Conservation est topologique. Positivité est dynamique.**
À h fixe ou évolutif, cette distinction reste valide.

### 4a-1 — Décomposition en micro-étapes isolées

**Principe** : avant tout test §4.1-§4.3 (qui mobilisent
simultanément ψ-dynamique + h-dynamique), valider chaque générateur
h-dynamique isolément. La spec complète des étapes 4a sera **révisée
empiriquement** après les premiers tests, pas figée a priori.

**Micro-étape 4a-α** : `𝔊^sed = -β·ψ·h` seul, ψ fixe constante.
- ψ est paramètre (pas variable d'état) : h(t) doit décroître
  exponentiellement `h(t) = h₀·exp(-β·ψ·t)` si ψ uniforme constante.
- Critère : engine h vs solution analytique exponentielle exacte.
- Vérifier : positivité h (devrait tenir car exponentielle), pas de
  conservation à attendre (h n'est pas une probabilité).

**Micro-étape 4a-β** : `𝔊^ero = +γ·h·(1-h/h₀)` seul, ψ absent.
- h(t) suit logistique vers h₀.
- Si h₀_local < h₀_target, croissance ; si h₀_local > h₀_target
  (cas trivial avec coefficient négatif), décroissance.
- Critère : engine h vs solution logistique analytique exacte.

**Micro-étape 4a-γ** : combinaison `∂_t h = -β·ψ·h + γ·h·(1-h/h₀)`,
ψ fixe.
- Point fixe : h = 0 ou `h = h₀·(1 - β·ψ/γ)` (valide si β·ψ < γ).
- Critère : engine vs RK4 indépendant, observer point fixe atteint.

**Micro-étape 4a-δ** : couplage minimal ψ-h : diffusion ψ + `𝔊^sed`
+ `𝔊^ero`. Pas de drift, pas de bruit pour cette étape.
- Première vraie co-évolution ψ↔h.
- Critère : conservation Σψ exacte, h ∈ [h_min, h₀] mesuré.
- Observable principal : co-attracteur `(ψ*, h*)` éventuel.

**RÉVISION DE SPEC OBLIGATOIRE après 4a-δ** : avant d'ajouter drift,
bruit, ou §4.1, **arrêter et auditer ce qui a réellement été
observé**. Le code et la spec sont co-révisés à ce point. Les
sections §4a-ε et suivantes ne sont pas spécifiées maintenant ;
elles le seront après contact empirique avec 4a-α/β/γ/δ.

**Caveat MCQ** : aucun co-attracteur observé en §4a ne doit être
promu comme "structure MCQ" avant le test pivot §5. Les motifs
émergents en 4a sont des observables du schéma sous co-évolution
discrète, pas des structures théoriques attestées.

**Acquis empirique 4a-α (𝔊^sed seul, ψ fixe)** :

Test passé sur 3 configs (ψ uniforme modéré, ψ uniforme fort,
ψ inhomogène gaussien). Engine vs solution analytique exponentielle
cohérent à l'erreur Euler théorique près. Uniformité h préservée
à machine precision sous ψ uniforme.

**Distinction structurelle à conserver** :

> **Localité de l'opérateur ≠ localité de la mémoire émergente.**

`𝔊^sed = -β·ψ·h` est strictement **pointwise** (sans couplage
spatial entre cellules). Mais h accumule déjà `∫ ψ(x,s)·ds` (intégrale
temporelle locale de la présence de ψ), donc h n'est plus un
paramètre instantané — il devient un **dépôt historique local**.

Conséquence anticipée : dès que ψ deviendra mobile spatialement
(diffusion, drift), h héritera d'une **mémoire spatialement étendue**
même si 𝔊^sed reste pointwise. La trace en chaque cellule reflète
le passage de ψ à cet endroit au cours du temps.

**Ceci valide structurellement le diagnostic § 4a-0** : le support
devient historique. La rupture n'attend pas le couplage complet
ψ↔h ; elle est déjà présente dans la sédimentation seule.

**À noter** : 4a-α est un régime monotone (h décroît strictement vers
0 sous ψ constante positive). Le premier vrai changement qualitatif
(seuils, pseudo-attracteurs, hystérésis embryonnaire, dépendance
aux échelles de temps) est attendu en 4a-γ quand sédimentation et
restauration se concurrencent.

**Acquis empirique 4a-γ (𝔊^sed + 𝔊^ero combinés, ψ fixe uniforme)** :

Test passé sur 5 configs `βψ/γ ∈ {0.1, 0.5, 0.9, 1.0, 1.5}` (sous-critique,
proche critique, critique exact, sur-critique). Engine vs solution
analytique exacte de la TRAJECTOIRE (pas seulement point fixe)
cohérent à l'erreur Euler théorique.

**Bifurcation locale verrouillée empiriquement** : le statut numérique
du point fixe h=0 change selon βψ/γ :
- βψ/γ < 1 : instable (système quitte vers K = h₀·(1-βψ/γ) > 0)
- βψ/γ = 1 : critique (décroissance polynomiale h ~ h₀/(γt))
- βψ/γ > 1 : stable (système y converge exponentiellement)

Robustesse dynamique complète obtenue (auparavant en 4a-β h=0 était
seulement numériquement stable sans perturbation).

**Acquis empirique 4a-δ (co-évolution ψ↔h symétrique)** :

Premier vrai test du régime co-évolutif. Diffusion ψ + sédimentation
+ érosion. Sans drift, sans bruit. 3 configs A/B/C calibrés par
rapports d'échelles τ_ψ, τ_sed, τ_ero.

**Invariants validés** :
- (A1) Σψ conservation à machine precision (10⁻¹⁶) sur les 3 configs
- (A2) Symétrie x/y/z préservée à machine precision (10⁻¹⁶)
- (B1) Engine vs RK4 indépendant cohérent avec erreur Euler théorique
- Ordre temporel Euler validé empiriquement : ratios entre 1.86 et
  2.21 (attendu ≈ 2 pour Euler ordre 1)

**Découverte structurelle confirmée — mémoire morphologique étendue** :
`corr(-log(h_final/h₀), ∫ψ·dt) > 0.97` sur les 3 régimes. La métrique
finale h trace fidèlement l'intégrale temporelle du passage de ψ,
même quand les échelles τ_h et τ_ψ varient sur 2 ordres de grandeur.
**La distinction "localité de l'opérateur ≠ localité de la mémoire
émergente" est désormais empiriquement vérifiée sous co-évolution**.

**Découverte du régime quasi-collapse (cas C : τ_h ≪ τ_ψ)** :
dans le régime "h rapide / ψ lent", h_min atteint ~3.2e-48 — proche
de la dénormalisation flottante. Cela mérite une lecture stratifiée.

**Stratification à trois niveaux (à inscrire comme distinction
structurelle pour toute lecture future)** :

(α) **Collapse théorique de l'ODE locale** : `βψ > γ` implique
analytiquement `h → 0`. Comportement attendu, vérifié en 4a-γ.

(β) **Sortie d'admissibilité numérique** : `h < h_resolution_flottante`
(typiquement ~10⁻³⁰ pour float64). Cela signifie "on est sorti du
domaine numériquement résoluble", **PAS** que le moteur est buggé.

(γ) **Sortie d'admissibilité MCQ** : violation du postulat `h_min > 0`
au sens théorique. **Cela ne se déduit pas de (β)**. MCQ postule
h_min > 0, mais le moteur discret actuel n'a pas encore de mécanisme
interne empêchant dynamiquement le collapse — c'est différent de "le
moteur viole MCQ".

**Position retenue (option 2a) pour la suite** :

> Laisser h tendre librement vers 0 quand l'ODE locale l'impose.
> Interpréter `h < h_resolution` comme **sortie du domaine de
> résolution numérique**, pas encore comme violation MCQ.
> Logger explicitement la fraction de cellules sous seuil, leur
> connectivité et leur éventuelle réactivation.

**Rejets explicites** :

(a) **Pas de clipping `h = max(h, h_min)` exogène** : introduirait
une non-linéarité artificielle, un seuil exogène, une dissymétrie
engine/RK4, et surtout un faux stabilisateur qui masquerait
l'information.

(b) **Pas d'interdiction structurelle de `βψ > γ`** : ce serait
stériliser le moteur avant d'avoir observé ses transitions. Les
zones critiques, transitions locales, quasi-collapses et
réactivations éventuelles sont précisément là où les phénomènes
intéressants peuvent apparaître.

**Vraie question pour 4a-ε et au-delà** :

> Pas "h touche-t-il zéro ?" mais
> **"un support quasi-collapsé peut-il redevenir transformable ?"**

Cette question rapproche du test pivot §5 (KNV, réactivabilité,
mémoire morphologique, hystérésis, gradients transformables). C'est
l'observable à instrumenter à partir de 4a-ε.

**Instrumentation supplémentaire à introduire dès 4a-ε** :
- `fraction_cells_under_h_resolution(t)` : volume cellules sous seuil
- `connected_components_h_active(t)` : structure topologique du
  support actif
- `temps_de_reactivation` : si une cellule sous seuil remonte au-dessus

Diversité des régimes (A stable, B critique, C quasi-collapse)
maintenue comme **information dynamique**, pas comme erreur à filtrer.

**Acquis empirique 4a-ε (co-évolution ψ↔h avec ψ asymétrique)** :

ψ initial décentré en (1, 2, 2), h uniforme, mêmes 3 régimes A/B/C.

**Invariants validés** :
- Symétrie y↔z préservée à machine precision (10⁻¹⁶) sur les 3 régimes
  — quand la dynamique conserve une symétrie, l'engine la préserve.
- Σψ conservation et engine vs RK4 cohérents.

**Découverte structurelle — feedback ψ↔h sur la dispersion** :
COM_x final = 1.99 / 1.68 / 1.51 pour A/B/C. La convergence vers le
centre géométrique (2,2,2) est **ralentie quand h s'effondre**. La
sédimentation modifie la métrique qui ralentit la diffusion qui
ralentit la dispersion de ψ — premier feedback dynamique observable
qui n'est pas réductible aux opérateurs séparés.

**Cas C** : 24.8% des cellules sous seuil, 0 réactivation observée,
support encore connexe (n_components_resolved = 1). Phénomène de
**léthargie locale sans fragmentation globale**.

**Acquis empirique 4a-ζ (classification long terme du quasi-collapse)** :

Régime C à 3 durées (t = 50, 500, 5000) + témoin A à t=500.

**Classification raffinée à 5 labels** (post-audit Alex) :
NO_COLLAPSE, EXPANDING_COLLAPSE, STABILIZED_COLLAPSE, SHRINKING_COLLAPSE,
OSCILLATING_BOUNDARY. Avec règle d'ordre claire et tolérance relative
0.01 sur frac_under.

| Config | trend_resolution | trend_functional |
|---|---|---|
| A témoin t=500 | NO_COLLAPSE | NO_COLLAPSE |
| C t=50 | EXPANDING | EXPANDING |
| C t=500 | STABILIZED | STABILIZED |
| C t=5000 | STABILIZED | STABILIZED |

**Trois résultats à acter** :

(1) **Le quasi-collapse se stabilise rapidement** : entre t=100 et
t=5000 (50× plus long), `frac_under_functional` reste à 0.28 — pas
de progression observée. Le régime EXPANDING dure environ 100 unités,
puis STABILIZED.

(2) **Aucune réactivation observée sur t=5000** (35/125 cellules
passées sous seuil, 0 réactivées, max_n_crossings = 1). À formuler
prudemment : **persistance dans la fenêtre simulée**, pas
irréversibilité structurelle.

(3) **Support topologiquement préservé** : n_components_resolved = 1
à tous les temps. **Léthargie morphologique sans fragmentation
topologique** — distinction structurelle à conserver.

**Acquis empirique 4a-η (re-injection unique de ψ)** :

Protocole : Phase 1 (t=0→100) ψ centré en (1,2,2) → quasi-collapse.
Phase 2 (t=100→500) : re-injection ψ en (3,2,2), pas de modification
de h. Mesures spécifiques sur la zone ancienne collapsée.

**Classification : STRATIFIED_REACTIVATION** (raffinée post-audit Alex) :
- ACCESS_CONFIRMED : ψ revisite massivement la zone
- REACTIVATED_SHELL : 17/35 cellules récupèrent (h_post ≥ h_resolution)
- DYNAMICALLY_LOCKED_DEEP : 18/35 cellules restent piégées dans la
  fenêtre t=400 post-réinjection
  - NUMERICAL_FLOOR_CORE : sous-catégorie (1/35 cellule à h_pre < 1e-100)
    où le mécanisme inclut le verrouillage numérique strict

**Mécanisme bifurcationnel local identifié** :

Dans la zone collapsée pendant Phase 2 :
```
dh/dt ≈ h · (γ - β·ψ_local)
```

Le facteur (γ - β·ψ_local) détermine le destin local :

| Régime local | β·⟨ψ⟩/γ | Comportement |
|---|---|---|
| Sous-critique | ~0.47–0.57 | Réactivation possible (REACTIVATED_SHELL) |
| Sur-critique | ~1.56 | Verrouillage maintenu (DYNAMICALLY_LOCKED_DEEP) |

**Mesures empiriques (audit Alex sur 4a-η)** :
- Cellules réactivées : β·⟨ψ⟩/γ ∈ [0.47, 0.57] (sous-critique local)
- Cellules non réactivées : β·⟨ψ⟩/γ médiane ≈ 1.56 (sur-critique local)

Donc le verrouillage n'est PAS principalement numérique (`h ≈ 0`) mais
**bifurcationnel local** : la sédimentation locale persiste au-dessus
du seuil critique, maintenant ou approfondissant le verrouillage.

**Trois facteurs déterminant la réactivabilité locale** :

(a) Profondeur résiduelle h_pre : si trop faible, le temps logistique
    pour remonter dépasse la fenêtre simulée.
(b) Régime local γ - β·ψ_local pendant la revisite : si sous-critique,
    réactivation possible ; si sur-critique, verrouillage maintenu.
(c) Plancher numérique strict : 1 cellule sur 35 — phénomène marginal
    dans cette configuration mais à acter pour des régimes plus profonds.

**Formulation pivot post-4a-η (validée Alex)** :

> Un support quasi-collapsé peut redevenir transformable partiellement.
> La réactivabilité dépend de deux variables locales :
> (1) profondeur résiduelle h_pre,
> (2) régime local γ - β·ψ_local pendant la revisite.
>
> Sans bruit, sans h_min explicite, sans réinjection morphologique :
> - les zones à h résiduel suffisant ET régime sous-critique réactivent
> - les zones trop profondes OU localement sur-critiques restent
>   dynamiquement verrouillées dans la fenêtre simulée
>
> C'est la première preuve empirique d'une **réactivabilité stratifiée
> par profondeur morphologique et bifurcation locale**.

**Distinction conceptuelle à conserver pour la suite (mémo Alex)** :

Trois notions à ne pas confondre :
- **ACCÈS DE ψ** : ψ_local atteint la zone (mesurable par ψ_mass)
- **RÉACTIVABILITÉ MORPHOLOGIQUE** : h peut remonter au-dessus du seuil
  de résolution
- **TRANSFORMABILITÉ EFFECTIVE** : la métrique restaurée permet à
  nouveau des transformations significatives (concept à préciser, lien
  attendu avec 𝒢, Δ, RR/RR² et τ′_ref)

L'accès n'implique pas la réactivabilité. La réactivabilité n'implique
pas la transformabilité (h peut remonter sans atteindre h₀, donc avec
capacité de transport réduite). Cette stratification anticipe le test
pivot §5 sans préempter sa conclusion.

### 4.1 Profondeur de trace après N cycles

Protocole : appliquer N = 10 cycles de contraction-restoration dans la
même direction θ_dir. À chaque cycle : forcer ψ concentrée à θ_dir
pendant T_contract = 5·t_form, puis laisser relaxer pendant
T_restore = 10·t_form.

Prédiction Ch3 §3.3 induced debt 12 : trace résiduelle accumule, h
converge vers une valeur < h₀ - δ_min.

**Test** : à t_final, mesurer h(θ_dir). Comparer à prédiction
analytique de l'accumulation résiduelle (formule à dériver depuis
𝔊^{ero}, voir Annexe A.1). Tolérance : 20%.

### 4.2 Latence métrique mesurée (correction h)

Protocole : système en STR stationnaire avec ψ_local (densité moyenne
autour du point de mesure) bien définie. À t_0, perturbation
exogène brève (Γ_pert = δ-impulse) sur ψ qui amène ψ_local à ψ_pert.
Mesurer le délai entre réponse de ψ (au scale ω^pos) et réponse de h
(au scale ω^form).

**Prédiction Ch3 §3.3 cross-tension 4** : la latence est **inversement
proportionnelle à la densité présente** :
```
τ_latency ≈ 1 / (β · ψ_local)
```
Pas `1/β` simple. Si ψ_local << 1/β, la latence est très longue ;
si ψ_local >> 1/β, latence courte.

**Test** : délai_obs entre Δψ et Δh ∈ [0.5/(β·ψ_pert), 2/(β·ψ_pert)].
Tolérance : facteur 2.

**Note** : ce test exige de mesurer ψ_local explicitement avant
calcul du critère. Pas de critère universel `1/β` indépendant de ψ.

**Convention de mesure ψ_local pour 6d-α (correction C2 audit)** :
```
ψ_local = ψ(θ_dir)
```
valeur cellulaire directe au point de mesure.

**Caveat de stabilité** : cette convention est minimale et **n'est
probablement pas stable sous raffinement de grille** (sur 10×10×10
ou plus, la définition cellulaire de ψ donnerait une valeur
différente d'un facteur O(1) selon le pavage du voisinage). La
latence mesurée en 6d-α porte donc sur la **convention cellulaire
de la grille 5×5×5**, pas sur une "latence MCQ canonique".

**Test de robustesse différé à 6d-γ** : comparer
`ψ_local_cell = ψ(θ_dir)` à
`ψ_local_kernel = ∑_{voisinage} w(d)·ψ(θ)` (kernel Gaussien radius
1 cellule). Si latence varie d'un facteur > 2 entre les deux
définitions, la latence n'est **pas un observable stable** — elle
est une propriété du voxel choisi.

### 4.3 Stationnarité couplée — précurseur du test pivot §5

Protocole : initialisation ψ Gaussienne large (σ_0 = 1.8·dx),
h = h₀ uniforme, β/γ adapté pour ψ_∞ ≠ 0 partout (cf. note
calibration ci-dessous), 𝔊^{sed} et 𝔊^{ero} actifs, pas de
perturbation, pas de couplage inter-modulaire.

**Calibration β/γ** : pour que ψ_∞ ne soit ni saturée (β·ψ ≥ γ partout)
ni triviale (β·ψ ≪ γ partout), choisir β/γ tel que `β·ψ_typical/γ ≈ 0.5`
au régime stationnaire. Avec ψ_typical ≈ 1/125 ≈ 0.008 sur grille
5×5×5, β/γ ≈ 0.5/0.008 ≈ 60. À calibrer empiriquement par sweep.

Intégrer 5·t_form. Vérifier convergence vers régime stationnaire :
‖∂_t ψ‖ et ‖∂_t h‖ < ε_stat partout.

**Test préliminaire** : convergence atteinte ? Si oui, pré-condition
pour §5 satisfaite.

**Si échec** : pas de régime stationnaire couplé atteint, le test pivot
§5 n'est pas exécutable. Identifier pourquoi (oscillation
persistante ? divergence ?) avant de continuer.

---

## 5. Test bloquant — stationnarité couplée ψ ⇆ h (correction d)

**Test pivot de 6d-α. Échec ici = pas de passage 6d-β.**

| Hypothèse Ch3 | §3.3.1.II — `h_∞(θ) = h₀·(1 - β·ψ_∞(θ)/γ)` quand β·ψ_∞ < γ |
|---|---|
| Source de la formule | Stationnarité de la co-production §3.1.4 |
| Validité | Cellules où β·ψ_obs(θ)/γ < 1 (régions non saturées à h_min) |

### 5.1 Reformulation en densité continue (arbitrage 1)

**Le test compare des densités, pas des masses cellulaires.**

Définitions explicites :
- **ρ_obs(θ)** : densité observée à la cellule (i,j,k), définie comme
  `ρ_obs(θ_ijk) = ψ_ijk / dx³`. En dx=1 : `ρ_obs ≡ ψ_ijk` numériquement,
  mais conceptuellement ρ_obs est une densité.
- **ρ_pred(θ)** : densité prédite par la formule Ch3 :
  `ρ_pred(θ_ijk) = (γ/β) · (1 - h_obs(θ_ijk)/h₀)`
  avec γ et β unités cohérentes pour que γ/β ait dimension [densité]
  = [1/volume] = [1/dx³].

**Cohérence dimensionnelle** : γ en [1/temps], β en [volume/temps] →
γ/β en [1/volume] = [densité]. ✓

**Calibration empirique** : par construction du protocole §4.3, β/γ ≈ 60
sur grille 5×5×5 dx=1. Donc γ/β ≈ 1/60 ≈ 0.017. ρ_typical attendue
≈ 0.017·(1-h_typical/h₀). Avec h_typical ≈ 0.5·h₀, ρ_typical ≈ 0.008.
Cohérent avec ψ_obs ≈ 1/125 ≈ 0.008. ✓

### 5.2 Protocole

1. Conditions : système §4.3 réussi, régime stationnaire atteint.
2. Pour chaque cellule (i,j,k), calculer :
   - `ρ_obs(θ_ijk) = ψ_ijk / dx³`
   - `h_obs(θ_ijk)`
   - `ρ_pred(θ_ijk) = (γ/β) · (1 - h_obs(θ_ijk)/h₀)`
   - Résidu : `R(θ_ijk) = ρ_obs(θ_ijk) - ρ_pred(θ_ijk)`
3. Restreindre l'analyse aux cellules **non saturées** :
   `S = {θ : h_obs(θ) > h_min + ε_sat}` avec ε_sat = 0.05·(h₀-h_min).
4. Calculer trois métriques sur S :
   - `L2_rel = ‖R‖_{L²(S)} / ‖ρ_obs‖_{L²(S)}`
   - `corr = corrélation(ρ_obs[S], ρ_pred[S])`
   - `Linf_rel = max_{θ∈S} |R(θ)| / max_{θ∈S} ρ_obs(θ)`

### 5.3 Critères de passage

**PASS** ssi les **trois** sont satisfaits :
- L2_rel < 0.10
- corr > 0.95
- Linf_rel < 0.25

L'asymétrie est délibérée :
- L2 capture la cohérence globale (10% serré).
- corr capture la structure (0.95 strict).
- L∞ capture les pathologies locales mais avec tolérance plus large
  (25%) parce qu'un voxel frontière ne doit pas invalider 6d.

### 5.4 Documentation des cellules saturées

Logger explicitement : pourcentage de cellules saturées (h_obs ≤ h_min+ε_sat).
Si > 30% du volume Θ saturé, alarme : β/γ trop fort, régime non
représentatif.

### 5.5 Conséquences d'un échec

Si **un seul** des trois critères échoue :

1. Ne pas passer à 6d-β.
2. Identifier la cause :
   - Échec L2 → schéma global défaillant
   - Échec corr → structure ψ-h mal capturée (anisotropie schéma ?)
   - Échec L∞ → pathologie locale (probablement frontière, voxel singulier)
3. Décider : fix du schéma OU relâchement de la tolérance avec
   justification écrite (cas L∞ frontière).

**Pas de relâchement de L2 ou corr sans amendement structurel du
document avec justification théorique.**

### 5.6 Tension à conserver verbatim (correction l)

> Même si ce test passe : **τ' reste une observable émergente
> conditionnelle au régime numérique testé, pas une variable primitive
> garantie.** "τ' mesurable dans ce régime" ≠ "τ' ontologiquement
> restauré". La validation ψ↔h ne valide pas l'ontologie de τ' au sens
> Ch1 ; elle valide seulement que les observables τ', 𝒞_T, β_QMC
> reposent sur une co-production effective dans **le régime numérique
> spécifique testé** : grille 5×5×5, dx=1, h(θ) scalaire, paramètres
> β/γ calibrés, conditions de bord Neumann. Hors de ce régime, la
> validité doit être re-testée.

Cette phrase doit apparaître en clair dans le rapport de validation
6d-α.

### 5.7 Test de robustesse hors-distribution (B3 audit — verrou anti-loss-implicite)

**Motif** : sans test hors-distribution, §5 risque de devenir une
**loss implicite par calibration**. Le mécanisme : si β/γ est calibré
par sweep (A.5) pour optimiser `corr(ρ_obs, ρ_pred)`, on sélectionne
les paramètres qui font passer §5. Donc §5 ne révèle plus la
co-production — il **l'extrait par sélection**.

Ce serait exactement le type de dérive que MCQ refuse explicitement
(τ'_ref ne doit pas devenir loss implicite, cf. §5.6).

**Protocole en deux familles** :

**Famille A — calibration** :
- Initialisations gaussiennes centrées (§4.3 standard)
- Conditions §4.3 nominales
- β/γ déterminé par sweep A.5 sur cette famille
- Vérification §5 (L2/corr/L∞) sur cette famille

**Famille B — validation, jamais vue pendant calibration** :
- Initialisations différentes :
  - Gaussiennes décentrées (centre déplacé de Δ ≥ 1.5·dx du centre de Θ)
  - Distributions bimodales (deux peaks)
  - Distributions non-gaussiennes (par exemple uniforme bornée
    perturbée stochastiquement)
- β/γ figé à la valeur retenue de famille A — **pas de re-sweep**
- Vérification §5 (L2/corr/L∞) sur cette famille avec les paramètres
  calibrés sur A

**Critère strict** :

| Résultat | Verdict |
|---|---|
| PASS famille A **et** PASS famille B | **PASS §5 validé** — co-production structurelle |
| PASS famille A **et** FAIL famille B | **REVISION §5** — la co-production observée en A est probablement une propriété du protocole A, pas du système |
| FAIL famille A | BLOCKED §5 — schéma défaillant en régime nominal |
| FAIL famille A **et** PASS famille B | Pathologie inattendue, diagnostic spécifique requis |

**Lecture en cas d'échec famille B** : la calibration β/γ sur famille A
a produit un fit local plutôt qu'une révélation structurelle de la
co-production. Ne pas re-calibrer β/γ sur famille B — cela
re-introduirait la loss implicite à un niveau plus profond. Soit le
schéma a un biais structurel non détecté en famille A, soit la
définition de la co-production est moins universelle que Ch3
l'affirme.

**Note importante** : ce test ne valide la co-production que dans la
**superposition des deux familles**. Tout résultat 6d-β reposant sur
la fiabilité de τ', 𝒞_T, β_QMC doit citer "co-production validée
hors-distribution" et non "co-production validée".

---

## 6. Vérification des 4 garanties de non-clôture discrètement

| Hypothèse Ch3 | §3.1.6 + §3.1.7 — non-closure structurelle |
|---|---|
| Principe | Les 4 garanties au continuum doivent être préservées par le schéma |

### 6.1 Garantie D_min > 0 strict

Test : à chaque step, logger `min_{i,j,k} D_eff(i,j,k,t)`. Vérifier
qu'il reste > D_min·(1 - ε_machine·100) pour tous t.

**Échec** : g_Ω atteint 0 par troncature flottante ou implémentation
défaillante.

### 6.2 Garantie Φ_eff sans bassin absorbant

Test : à intervalles réguliers (chaque 100 steps), calculer la profondeur
maximale des minima locaux de Φ_eff (par scan de la grille). Vérifier
que cette profondeur ne dépasse pas D_min · dx² · constant_seuil
(constant_seuil = 10 par défaut, à valider théoriquement, voir Annexe A.3).

**Échec** : Φ_corr produit un puits assez profond pour piéger ψ malgré
la diffusion.

### 6.3 Garantie bruit multiplicatif non éliminable (caveat ψ_floor)

Test : dans les régions où ψ est faible (peak distant), mesurer
amplitude RMS du bruit injecté vs amplitude des gradients déterministes.
Vérifier que le ratio reste > σ_η · 0.1 (le bruit n'est pas écrasé).

**Caveat ontologique** : cette garantie est **préservée par
construction via ψ_floor** (cf. §1.5). Elle se lit donc :

- *Sens MCQ originel* (Ch3 §3.1.7) : η non éliminable indépendamment
  de ψ.
- *Sens 6d-α* : η non éliminable **dans les régions actives**
  (ψ_face > ψ_floor) et **bornée par σ_η·D·√(ψ_floor·dt)** dans les
  régions quasi-vides.

Cette divergence sémantique entre Ch3-continuum et 6d-α-discret est
**explicite** et n'invalide pas le test §6.3 — elle précise son scope.

**Échec** : bruit conservatif sur arêtes neutralise excessivement les
zones critiques en dépit de ψ_floor (par exemple si ψ_floor est mal
calibré).

### 6.4 Garantie 𝔊^{ero} actif (correction i)

**Reformulation** : la condition à équilibre étant trivialement
∂_t h = 0, on teste **hors équilibre uniquement**.

Protocole : initialisation où ψ ≈ 0 partout sauf une région compacte
(donc 𝔊^{sed} ≈ 0 partout sauf cette région). Hors région : 𝔊^{ero}
est le seul terme actif sur h.

Test : dans les cellules **hors région de sédimentation** où h initial
< h₀ - 0.1·(h₀-h_min) (donc érosion attendue active), mesurer ‖∂_t h‖
au step 1 et comparer à valeur théorique :
```
∂_t h_attendu = γ · h · (1 - h/h₀)
```

Vérifier : `‖∂_t h_obs‖ / ‖∂_t h_attendu‖ ∈ [0.5, 1.5]` (facteur 2 max
d'écart en raison du schéma temporel semi-implicite).

**Échec** : le solveur amortit excessivement l'érosion, ou implémentation
défaillante de 𝔊^{ero}.

### 6.5 Verdict global

Les 4 garanties doivent toutes passer pour que le schéma préserve la
non-closure. Si **une seule** échoue, l'homéostasie cachée est possible.

**Caveat de validation (auto-stress-test)** : §6 est **partiellement
tautologique**. §6.1 (D_min > 0) est garanti par construction
(g_Ω borné inférieurement). §6.3 (bruit non éliminable) est garanti
par construction via ψ_floor (§1.5). §6.4 (𝔊^ero actif) est garanti
si Backward Euler ne sur-amortit pas (testé en §9).

**§6.2 (Φ_eff sans bassin absorbant) reste un test informatif
indépendant** — il vérifie que Φ_corr ne produit pas accidentellement
un bassin trop profond malgré g_Ω. Pas garanti par construction.

Conséquence : §6 doit être lu comme **vérification d'intégrité du
schéma** (les garde-fous fonctionnent comme prévu) plus que comme
**test indépendant de la non-closure MCQ**. La non-closure MCQ stricte
n'est pas validée par §6 — elle est partiellement injectée par
construction (cf. §0quater sur la co-construction).

---

## 7. Stress-test KNV (diagnostic complémentaire, protocole (a))

| Hypothèse Ch3 | §3.6.12 — viabilité non garantie analytiquement, doit être maintenue dynamiquement |
|---|---|
| Statut | **Diagnostic complémentaire**, pas blocking. Échec ici n'arrête pas 6d-α. |

### 7.1 Protocole

Initialisations extrêmes admissibles (positives, normalisées, h ∈
[h_min, h₀]) :

- ψ_0 = δ régularisée (σ_0 = 1.5·dx) à un coin de Θ
- ψ_0 = uniforme (entropie maximale)
- h_0 = h_min uniforme (métrique pré-saturée)
- h_0 = mosaïque h_min/h₀ alternée (haute hétérogénéité initiale)

Intégrer 10·t_form. Mesurer l'état final.

### 7.2 Lectures

| Résultat | Lecture |
|---|---|
| Toutes initialisations → même corridor stationnaire | Suspect : homéostasie cachée |
| Existence de bassins KNV irréversibles | Sain : non-closure préservée |
| Dépendance forte aux conditions initiales | Attendu sous co-production non-linéaire |
| Échec partiel de régulation observable | Nécessaire pour validité Ch3 |

### 7.3 Que les protocoles (b) perturbation transitoire et (c) perturbation persistante soient **explicitement déférés**

Ces deux protocoles testent restoration et adaptation, qui sont des
phénomènes émergents 6d-β. Ils ne sont **pas** dans le champ de 6d-α et
ne doivent pas être implémentés ici.

---

## 8. Logging anti-artefact

Observables à logger à chaque pas (ou échantillonné selon coût) :

| Quantité | But |
|---|---|
| min/max/mean D_eff | Détecter spikes anti-collapse |
| min/max/mean h | Vérifier bornes [h_min, h₀] |
| ∑ψ - 1 | Conservation masse à machine precision |
| min ψ | Positivité (alarme si < -ε_machine) |
| **Fréquence BE-fallback** (global) | Détecter dt trop grand ou pathologie |
| **`BE_count[i,j,k]` spatial** (correction B3) | Détecter concentration spatiale BE-fallback |
| **Corrélation BE-fallback ↔ ∇ψ fort** (correction B3) | Détecter filtrage dissipatif des régions critiques |
| **Corrélation BE-fallback ↔ h faible** (correction B3) | Détecter filtrage dissipatif des régions contractées |
| variance locale flux bruités | Détecter sur-amplification stochastique |
| corrélations spatiales bruit | Détecter pseudo-structures induites par stencil |
| fréquence inversions de flux | Détecter chaos numérique artificiel |
| entropy production rate | Détecter bruit dissipatif caché |
| dt utilisé | Détecter pathologies CFL |
| profondeur basins Φ_eff | Détecter risque homéostasie |
| **g_T mesuré sous h plein vs h diagonal** (correction k) | Quantifier cross-talk métrique |

**Aucune compression** des logs en 6d-α. Tout est conservé pour
analyse a posteriori.

---

## 9. Comparaison explicite vs implicite

Pour chaque cas test §2 et §4, exécuter en parallèle :
- Schéma semi-implicite Crank-Nicolson + drift explicite (le schéma 6d nominal)
- Schéma Euler explicite avec dt très petit (référence non-amortie)

Observables comparés :
- Énergie spectrale haute fréquence (mesure dissipation Crank-Nicolson)
- Trajectoire pointwise (différence absolue)
- Temps de convergence (peut être faussement accéléré par implicite)

**Critère** : différence relative < **10%** sur observables principaux
(correction e — relâché de 5% à 10% vu l'ordre 1 du semi-implicite).
Si > 10%, soit l'implicite amortit excessivement (à documenter), soit
l'explicite n'a pas convergé (dt encore trop grand).

---

## 10. Verdict 6d-α

### 10.1 Conditions de clôture 6d-α

6d-α est clos avec verdict **PASS** ssi :

1. ✅ Tous les cas analytiques purs (§2.1, §2.2, §2.3) passent à 5%
   (10% pour 2.2 sur taux exponentiel).
2. ✅ Validation isotropie discrète (§3) passe à 5% axes / 10% diagonales,
   ou 12-15% diagonales avec soft-fail documenté.
3. ✅ Tous les cas semi-analytiques co-production (§4.1, §4.2, §4.3) passent à leurs tolérances respectives.
4. ✅ **Test pivot §5 passe** (L2 < 10%, corr > 0.95, L∞ < 25% sur cellules non saturées en densité ρ).
5. ✅ Les 4 garanties de non-clôture (§6) sont préservées discrètement.
6. ✅ Logging anti-artefact (§8) ne révèle pas de pathologie majeure non documentée.
7. ✅ Comparaison explicite vs implicite (§9) montre amortissement borné < 10%.
8. ✅ Fréquence BE-fallback (§1.4) < 5% des steps sur tests §2-§4.

### 10.2 Conditions de blocage

**BLOCKED** ssi :
- §5 échoue (test pivot)
- §2 échoue (cas analytiques purs)
- Plus d'une garantie §6 échoue
- Fréquence BE-fallback > 20% (signal grave)

**REVISION** ssi :
- §3, §4 échouent à tolérance
- Une garantie §6 échoue
- §9 montre amortissement > 10%
- Fréquence BE-fallback ∈ [5%, 20%]

En REVISION, fix du schéma puis re-run complet 6d-α. Pas de skip.

### 10.3 Typologie d'échec — lecture

À documenter explicitement dans le rapport de validation :

| Type d'échec | Localisation | Lecture |
|---|---|---|
| Échec analytique pur | §2 | Bug schéma |
| Échec semi-analytique co-production | §4 | Défaut dynamique ψ↔h |
| Échec test pivot densité | §5 | Co-production absente ou trop défaillante pour observables émergentes |
| Échec garantie non-closure | §6 | Schéma viole structurellement Ch3 §3.1.6/7 |
| Échec stress KNV | §7 | Probable homéostasie cachée (diagnostic) |
| Frequency BE-fallback excessive | §1.4 | dt trop grand ou pathologie schéma diffusion |
| **Cross-talk métrique anormalement faible (§1.8)** | §8 logging | Suggère que h(θ) plein dégénère vers diagonal — h ne varie pas assez spatialement, β/γ mal calibré, ou bug stencil |
| Succès universel (rien n'échoue jamais) | global | **Suspicion homéostasie cachée** : 6d-α n'a pas testé assez fort, ou la régulation est trop active |
| Succès avec échec partiel localisé | global | **Sain** : non-closure préservée, 6d-β autorisé |

Le succès universel est aussi suspect que l'échec systématique.

### 10.4 Rapport de validation 6d-α

Document `6d-alpha-validation-report.md` produit en clôture, contenant :
- Résultats numériques pour chaque test §2-§9
- Tolérances atteintes vs ciblées
- Cellules saturées (§5.4)
- Pathologies détectées (§8)
- **Cross-talk métrique mesuré (§1.8)**
- Verdict PASS / REVISION / BLOCKED
- Si PASS : passage explicite à 6d-β autorisé.
- Tension §5.6 reproduite verbatim.

---

## Annexes — dérivations sourcées au format 7-champs

**Format obligatoire pour chaque annexe** :

| Champ | Contenu |
|---|---|
| 1. Équation source | Référence Ch3 §X.Y.Z (ou dérivation explicite depuis master equation) |
| 2. Hypothèses de régime | Conditions sous lesquelles la dérivation tient |
| 3. Unités / dimensions | Vérification explicite, terme par terme |
| 4. Formule ou procédure | La dérivation elle-même |
| 5. Domaine de validité | Limites du régime |
| 6. Critère de passage | Tolérances et observables |
| 7. Lecture en cas d'échec | Interprétation du non-passage |

Ces annexes doivent être complétées **avant** que le refactor engine
ne commence. Deux d'entre elles (A.1, A.3) sont marquées **CORRECTION
BLOQUANTE** : elles remplacent des formulations v1 incorrectes
(dimensionnellement fausses ou approximations cachées).

---

### A.5 — Calibration β/γ pour test §5

*(À traiter en premier car A.1 et A.4 en dépendent.)*

**1. Équation source**
Stationnarité de la co-production §3.1.4 :
`h_∞(θ) = h₀·(1 - β·ψ_∞(θ)/γ)` quand `β·ψ_∞/γ < 1`.

**2. Hypothèses de régime**
- Régime stationnaire couplé atteint (test §4.3 passé)
- Pas de couplage inter-modulaire (N=1, pas de 𝒞^N)
- ψ_∞ non triviale (ni nulle ni concentrée à un point)

**3. Unités / dimensions**
- `β` : [volume·temps⁻¹] (pour que `β·ψ·h` ait dim [h/temps] = [h·temps⁻¹])
- `γ` : [temps⁻¹]
- `β/γ` : [volume]
- `β·ψ/γ` : `[volume]·[densité=1/volume] = adimensionnel` ✓
- `β/γ ≈ 60` en dx=1 : `[volume] = [dx³] = 1` numériquement, donc
  β/γ = 60 signifie que `1/(β/γ)` ≈ 0.017 a dimension densité ✓

**4. Formule ou procédure**
Sweep paramétrique :
```
β/γ ∈ {30, 45, 60, 80, 100}
```
Pour chaque valeur :
1. Lancer §4.3 (stationnarité couplée) avec ce ratio.
2. Mesurer en stationnarité : `fraction_saturée`, `mean(h/h₀)`,
   `L2_rel`, `corr(ρ_obs, ρ_pred)`, `L∞_rel`.

**Logging requis** (tous tracés) :
- fraction_saturée
- mean(h/h₀)
- L2_rel (vs prédiction §5)
- corr (vs prédiction §5)
- L∞_rel (vs prédiction §5)

**5. Domaine de validité**
La heuristique β/γ ≈ 60 dérive de `ψ_typical ≈ 1/N_total = 1/125`
(grille 5×5×5). Si la grille change, refaire le sweep.

**6. Critère de passage**

**Filtres durs** (à satisfaire d'abord) :
- `fraction_saturée < 30%`
- `0.2 < mean(h/h₀) < 0.8`

**Optimisation** (à appliquer parmi les valeurs satisfaisant les
filtres durs) :
- `corr(ρ_obs, ρ_pred)` maximale
- `L2_rel(ρ_obs vs ρ_pred)` minimale (départage si corr ex-aequo)

**Valeur retenue** : la valeur β/γ qui satisfait les deux filtres
durs et optimise corr/L2 parmi les survivants.

**Important** : la valeur retenue n'est pas celle qui "marche le
mieux" mais celle qui place le système hors saturation, hors
trivialité, hors explosion 1/h, **puis** optimise la qualité de
co-production.

**7. Lecture en cas d'échec**
- Aucune valeur du sweep ne satisfait les critères de régime informatif
  → la grille 5×5×5 ne supporte pas un régime stationnaire couplé non
  trivial. Bloquant : repenser soit la grille, soit la dynamique
  𝔊^{sed}/𝔊^{ero}.
- Sweep produit plusieurs valeurs valides
  → choisir celle de plus grande corr, documenter le choix.

---

### A.4 — Calibration ε_sat pour cellules saturées (§5.4)

**1. Équation source**
La formule §5 `ρ_pred = (γ/β)·(1 - h/h₀)` n'est valide que là où
`β·ψ_∞/γ < 1` (Ch3 §3.3.1). Dans la zone saturée (h → h_min),
la formule plafonne et n'est plus comparable à ρ_obs. ε_sat délimite
la zone saturée à exclure du test pivot.

**2. Hypothèses de régime**
- Test §5 exécuté à β/γ retenu par A.5
- Distribution de h_obs hétérogène (ni constante h₀ ni constante h_min)

**3. Unités / dimensions**
- ε_sat : `[longueur] = [h]` (dimensionnellement homogène à h)
- ε_sat / (h₀ - h_min) : adimensionnel ∈ [0, 1] ✓

**4. Formule ou procédure**
Définition initiale : `ε_sat = 0.05·(h₀ - h_min)` (5% du range métrique).

**Définition explicite de la fraction saturée à seuil x** :
```
frac_sat(x) = #{θ : h_obs(θ) ≤ h_min + x·(h₀ - h_min)} / N_total
```
où x ∈ [0, 1] est le seuil relatif au range [h_min, h₀].

Sweep de sensibilité :
```
ε_sat ∈ {0.02, 0.05, 0.10}·(h₀ - h_min)
```

Pour chaque valeur, mesurer :
- `frac_sat(0.02)`, `frac_sat(0.05)`, `frac_sat(0.10)`
- Verdict §5 (PASS / FAIL) pour chaque ε_sat

**Logging requis** : les trois fractions, et le verdict §5 par ε_sat.

**5. Domaine de validité**
Tant que `frac_sat < 30%` pour la valeur retenue. Sinon le régime
n'est plus représentatif (A.5 doit reprendre la main).

**6. Critère de passage (robustesse)**
**PASS §5 considéré robuste seulement si invariant sur deux valeurs
adjacentes de ε_sat**.

Les paires adjacentes sont : **{0.02, 0.05}** ou **{0.05, 0.10}**.
La paire {0.02, 0.10} **n'est pas adjacente** (saut de deux crans).

Si verdict change entre 0.02 et 0.05 (ou entre 0.05 et 0.10) :
- frontière saturée trop influente sur le test
- pathologie classique des tests à seuil

**7. Lecture en cas d'échec**
- Verdict varie entre les trois valeurs → test pivot dépend de la
  définition arbitraire de "saturée", non robuste. Bloquant.
- Verdict invariant sur deux valeurs adjacentes (mais varie sur la
  troisième) → robustesse partielle, documenter et passer avec
  réserve.
- Verdict invariant sur les trois → robustesse pleine, ε_sat = 0.05
  retenu.

---

### A.1 — Profondeur de trace après N cycles (§4.1) — CORRECTION BLOQUANTE

**Motif** : la formulation v1 (récurrence approximative) assumait
`β·ψ_high >> γ` sans le déclarer, ce qui faisait tester l'approximation
plutôt que la dynamique. Remplacée par intégration RK4 1D de l'ODE
exacte Ch3 §3.1.4.

**1. Équation source**
Master equation Ch3 §3.1.4 restreinte au sous-espace 1D θ = θ_dir,
en supposant que la dynamique locale en θ_dir est dominée par
𝔊^{sed} et 𝔊^{ero} (couplage spatial à θ ≠ θ_dir négligeable
pendant les épisodes de contraction-restauration courts) :

```
Pendant contraction (forcing ψ_local élevée à θ_dir) :
  ∂_t h = -β·ψ_local(t)·h + γ·h·(1 - h/h₀)

Pendant restauration (ψ_local ≈ 0 à θ_dir) :
  ∂_t h = γ·h·(1 - h/h₀)
```

**2. Hypothèses de régime**
- ψ_local(t) mesurée **dans la simulation 3D** au point θ_dir
  pendant les phases de contraction (pas une constante théorique)
- **Identification ρ = ψ en dx=1** : ψ_obs_local(t) = ψ(θ_dir, t)
  est la valeur cellulaire directe à la cellule θ_dir. En dx=1,
  ρ_local = ψ_cell numériquement (cf. §1.0).
- Couplage spatial **gelé explicitement** dans l'ODE 1D (approximation
  déclarée, pas omission silencieuse). Le flux diffusif `∇²ψ` est
  remplacé par zéro dans la dérivation 1D — cette gel est une
  approximation explicite, à comparer à la simulation 3D complète.
- Régime non saturé (h(θ_dir) > h_min + ε_sat)

**3. Unités / dimensions**
- `β·ψ·h` : `[volume·temps⁻¹]·[1/volume]·[longueur] = [longueur/temps]` ✓
- `γ·h·(1-h/h₀)` : `[temps⁻¹]·[longueur]·[adimensionnel] = [longueur/temps]` ✓
- `∂_t h` : `[longueur/temps]` ✓

Cohérence vérifiée. Tous les termes sont en `[longueur/temps]`.

**4. Formule ou procédure**
Pas de formule fermée. Procédure :
1. Lancer simulation 3D selon protocole §4.1 (N=10 cycles, T_contract=5·t_form, T_restore=10·t_form).
2. Enregistrer la trajectoire `ψ_obs_local(t) = ψ(θ_dir, t)` à chaque
   step de la simulation 3D.
3. Intégrer numériquement par RK4 l'ODE 1D :
   - Phases contraction : `dh/dt = -β·ψ_obs_local(t)·h + γ·h·(1-h/h₀)`
   - Phases restauration : `dh/dt = γ·h·(1-h/h₀)`
   avec h(t=0) = h₀ et ψ_obs_local(t) la trajectoire enregistrée en (2).
4. Comparer `h_pred_RK4(t_final)` à `h_obs(θ_dir, t_final)` dans la
   simulation 3D.

**Pas d'intégration RK4** : `dt_RK4 = dt_simulation` (un step RK4 par
step de simulation 3D). Si tolérance §6 non atteinte, **subdivision**
optionnelle : `dt_RK4 = dt_simulation / 4` avec interpolation linéaire
de ψ_obs_local(t) entre deux steps de la simulation. La subdivision
n'est utilisée qu'en cas d'échec, pas par défaut.

**5. Domaine de validité**
- Pendant que h(θ_dir) > h_min + ε_sat (sinon clipping change la dynamique)
- Pendant que ψ ne s'étale pas excessivement (couplage spatial reste
  négligeable)

Si h atteint h_min pendant la simulation, l'ODE 1D devient borne
supérieure de la prédiction, pas prédiction exacte.

**6. Critère de passage**
`|h_obs(θ_dir, t_final) - h_pred_RK4(N=10)| / h₀ < 0.20`

en régime non saturé. Tolérance 20% (l'ODE 1D néglige le couplage
spatial qui existe dans la simulation 3D).

**7. Lecture en cas d'échec**

Trois interprétations possibles d'un écart > 20%, à distinguer par
diagnostic supplémentaire (auto-stress-test) :

- **(a) Cross-talk métrique fort (signal §1.8 — informatif)** :
  l'écart provient du couplage spatial actif que l'ODE 1D a gelé.
  C'est un **vrai signal phénoménologique** de la perte de
  séparabilité §1.8. Diagnostic : mesurer Δh(θ ≠ θ_dir) pendant les
  cycles — si h varie significativement hors de θ_dir, le cross-talk
  est actif et l'approximation 1D est inadéquate **par signal vrai**,
  pas par bug.

- **(b) Bug structurel schéma 3D** : le schéma 3D ne reproduit pas
  fidèlement 𝔊^{sed}/𝔊^{ero}. Diagnostic : comparer les premiers
  steps du cycle à la dérivée analytique de l'ODE 1D. Si désaccord
  dès le step 1, c'est un bug.

- **(c) Régime saturé atteint** : h(θ_dir) → h_min pendant la
  simulation, l'ODE 1D devient borne supérieure. Diagnostic :
  vérifier la trajectoire h(θ_dir, t) — si elle a touché h_min,
  réduire ψ_high ou raccourcir T_contract.

Seuils opérationnels :
- Écart > 20% en régime non saturé : déterminer (a)/(b)/(c) avant
  conclusion.
- Écart > 50% : (b) ou (c) probables, bloquant en attente diagnostic.
- (a) confirmé : pas un échec, c'est une mesure de cross-talk.
  Documenter et passer à 6d-β (cross-talk devient observable
  phénoménologique).

---

### A.2 — Calibration ε_petr (§6.2)

**1. Équation source**
Pétrification = `‖∂_t h‖ ≈ 0` persistant dans une cellule active
(hors saturation et hors équilibre normal). Ch3 §3.6.3.III définit le
canal RR³ → δ𝕋* → δD_eff^form qui doit réveiller la cellule
pétrifiée. ε_petr est le seuil sous lequel `‖∂_t h‖` est considéré
comme pétrifié.

**2. Hypothèses de régime**
- Calibration mesurée pendant **phase transitoire active**, pas à
  équilibre
- Cellules actives uniquement (h > h_min + ε_sat)
- Test §4.3 exécuté avec β/γ retenu par A.5

**3. Unités / dimensions**
- `∂_t h` : `[longueur/temps]`
- `ε_petr` : `[longueur/temps]` ✓
- `median(‖∂_t h‖)` : `[longueur/temps]` ✓
- `τ_petr` : `[temps]` ✓
- `τ_petr / t_form` : adimensionnel ✓

**4. Formule ou procédure**
```
ε_petr = 0.1 · median(‖∂_t h‖) sur :
  - cellules actives : h > h_min + ε_sat
  - fenêtre temporelle : steps [10, 50] de la simulation §4.3
    (transitoire post-warmup, avant convergence)
```

**Échelle morphologique caractéristique t_form** :
```
t_form = 1/γ  (valeur par défaut)
```
γ est le coefficient d'érosion (Ch3 §3.1.4, 𝔊^{ero} = γ·h·(1-h/h₀)).
1/γ est l'échelle de temps caractéristique de relaxation logistique
de h vers h₀ en absence de sédimentation.

**Recalibration empirique optionnelle** : si la trajectoire h en
relaxation pure (§4.3 fin transitoire) montre un temps de demi-vie
significativement différent de 1/γ (facteur > 2), retenir la valeur
empirique mesurée et documenter l'écart.

**Durée seuil τ_petr** :
```
τ_petr = 5 · t_form = 5/γ
```

**Conversion en steps** (paramètre d'implémentation effectif) :
```
n_steps_τ_petr = τ_petr / dt = 5 / (γ · dt)
```
où dt est le pas de temps de la simulation (§1.7).

**5. Domaine de validité**
- Mesure transitoire valide tant que le système n'est pas encore
  convergé. Si convergence atteinte avant step 50, raccourcir la
  fenêtre [10, n_convergence].
- τ_petr = 5·t_form et t_form = 1/γ sont des **conventions
  initiales**, pas des constantes théoriques. À réajuster si les
  diagnostics §6.2 montrent trop/pas-assez de fausses alarmes
  (typiquement : si plus de 10% des cellules actives sont en
  "pétrification détectée" pendant un régime stationnaire normal,
  τ_petr trop court ou ε_petr trop élevé).

**6. Critère de passage**
Détection de pétrification : `‖∂_t h(θ)‖ < ε_petr` pour
`n_steps_τ_petr` steps consécutifs.

Logger :
- nombre de cellules en pétrification détectée
- durée moyenne des épisodes de pétrification (en steps et en
  unités t_form)
- réactivation par g_Ω (canal 2) si elle survient
- valeur effective de t_form si recalibrée empiriquement

**7. Lecture en cas d'échec**
- Trop de cellules en pétrification détectée pendant §4.3 → ε_petr
  trop élevé (capture des fluctuations normales)
- Pas de détection alors que h reste figé visuellement → ε_petr trop
  faible (passe sous le bruit)
- Détection mais pas de réactivation → canal 2 g_Ω inactif ou mal
  calibré

---

### A.3 — Profondeur basin Φ_eff (§6.2) — CORRECTION BLOQUANTE

**Motif** : la formulation v1 (`B_depth = ΔΦ/(D·dx²)`) n'était **pas
adimensionnelle** (résidu en `1/longueur²`). Remplacée par nombre de
Péclet local, qui est la grandeur adimensionnelle standard comparant
drift et diffusion.

**1. Équation source**
La compétition entre drift (qui crée des bassins) et diffusion (qui
les remplit) est gouvernée par le nombre de Péclet local. Ch3 §3.1.6
exige que Φ_eff n'ait pas de bassin absorbant, ce qui se traduit par
Pe borné.

**2. Hypothèses de régime**
- Φ_eff lisse (pas de discontinuités numériques)
- Bassin local identifiable (existence d'un minimum local de Φ_eff)
- D_min > 0 strict (garantie §6.1 passée)

**3. Unités / dimensions**
- `∇Φ_eff` : `[Φ/longueur] = [longueur/temps]` (puisque [Φ] =
  [longueur²/temps] pour Fokker-Planck)
- `L_basin` : `[longueur]`
- `D_min` : `[longueur²/temps]`
- `Pe = ‖∇Φ‖·L / D` : `[longueur/temps]·[longueur]/[longueur²/temps]
  = adimensionnel` ✓

Vérification croisée par interprétation physique : Pe = (taux de
transport par drift) / (taux de mixing par diffusion). Bien
adimensionnel ✓.

**4. Formule ou procédure**
```
Pe_basin = max_local(‖∇Φ_eff‖) · L_basin / D_min
```

où :
- `max_local(‖∇Φ_eff‖)` : maximum du gradient de Φ_eff dans un
  **voisinage de 3×3×3 cellules** centré sur le minimum local du
  bassin (27 cellules au total)
- `L_basin` : **largeur à mi-hauteur (FWHM, Full Width at Half
  Maximum)** du minimum local de Φ_eff, mesurée comme distance entre
  les deux points où `Φ_eff = Φ_min + (Φ_local_max - Φ_min)/2` le
  long de l'axe principal du bassin
- `D_min` : valeur minimale de D_eff dans le voisinage 3×3×3 du
  bassin

**Procédure de mesure** :
1. À intervalles réguliers (chaque 100 steps), scanner la grille pour
   identifier les minima locaux de Φ_eff (cellules où Φ < Φ des 26
   voisins).
2. Pour chaque minimum local identifié :
   - Définir le voisinage 3×3×3 autour du minimum
   - Calculer L_basin (FWHM) dans ce voisinage
   - Calculer `‖∇Φ‖_max` dans ce voisinage (max sur les 27 cellules
     du module du gradient discret)
   - Récupérer D_min dans ce voisinage
3. Calculer Pe_basin pour ce minimum.
4. Logger `max(Pe_basin)` sur tous les bassins détectés à ce step.
5. **Logger `Pe_basin(t)` comme observable temporelle** (correction
   C1 audit) : la valeur Pe varie dans le temps même à Φ_eff fixe
   parce que D_eff (modulé par g_Ω) et L_basin (dépendant de D)
   évoluent. Tracer Pe_basin(t) permet de détecter des **épisodes
   transitoires de quasi-absorption** invisibles dans une moyenne
   globale.

**5. Domaine de validité**
- Bassins isolés (deux bassins proches faussent L_basin)
- Φ_eff non-trivial (cas Φ = 0 partout : pas de bassin, Pe non
  défini, test passé trivialement)
- Résolution L_basin sur grille 5×5×5 nécessairement grossière :
  FWHM mesurable à ±1 cellule près, donc Pe à ±50% près
  intrinsèquement.

**6. Critère de passage — statut heuristique d'alarme**

**Statut épistémique** (correction B3 audit) : sur grille 5×5×5, la
mesure de Pe est **fragile**. L_basin (FWHM) est résolue par 2-3
cellules au plus, `‖∇Φ‖_max` est sensible à la position du gradient
dans le stencil, l'interpolation aux interfaces introduit du bruit.

Pe est donc une **heuristique d'alarme**, pas un seuil physique
robuste. Les seuils doivent être lus comme :

| Pe_basin | Lecture |
|---|---|
| < 5 | **Probablement** non-absorbant |
| [5, 10] | **Probablement** ralentissement significatif, à investiguer |
| > 10 | **Probablement** absorbant, **chercher confirmation indépendante** (ex: temps de séjour ψ dans le bassin) |

Une mesure isolée de Pe > 10 ne suffit pas à conclure à une violation
de §6.2 : confirmation par observable secondaire requise.

**Validation calibration sur OU stationnaire** : avec `Φ = ½k·‖θ‖²`,
`‖∇Φ‖_max ≈ k·σ_OU` où `σ_OU = √(D/k)`, donc
`‖∇Φ‖_max ≈ k·√(D/k) = √(k·D)`.
`L_basin ≈ σ_OU = √(D/k)` (à des facteurs FWHM/σ près).
`Pe_OU ≈ √(k·D) · √(D/k) / D = 1`.

**OU stationnaire fournit un régime de référence naturellement
d'ordre `Pe ~ O(1)`** (pas exactement 1 — dépend des conventions FWHM
vs σ, dimension, norme du gradient), cohérent avec un équilibre non
absorbant. Le seuil warning Pe=5 laisse une marge confortable
au-dessus de OU, le seuil failure Pe=10 marque le régime drift-dominé
absorbant.

**7. Lecture en cas d'échec**
- `Pe_basin ∈ [5, 10]` : warning, Φ_eff a un bassin assez profond
  pour ralentir significativement la diffusion. À documenter et
  surveiller via Pe_basin(t).
- `Pe_basin > 10` : failure, bassin absorbant probable. Garantie
  §6.2 violée. Φ_corr à reformuler (gradient max plafonné, ou
  L_basin contrôlé par lissage).
- `Pe_basin ≈ 1` partout : régime conforme à OU, sain.

---

Ces annexes doivent être complétées avec les valeurs numériques
obtenues par sweep avant qu'on tranche les paramètres en code.

---

*Spec contraignante v2 amendée. Aucun refactor engine ne commence avant
que ce document soit complet, signé, et que les amendements pendants
(annexes) soient résolus. La rationalisation a posteriori est interdite
— toute décision numérique faite en cours d'implémentation amende ce
document en première opération.*
