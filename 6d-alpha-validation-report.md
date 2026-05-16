# 6d-α — Rapport de validation

**Statut** : Clôture provisoire honnête du programme 6d-α.
**Date** : session 27.
**Référence** : `6d-alpha-numerics.md` (spec amendée) et JSON sessions 1-27.

---

## Synthèse exécutive

Le programme 6d-α a été conduit comme **co-construction empirique
contrainte** entre la spec théorique (Ch3 §3.1.6/7) et le moteur
numérique factoriel 6d. Le verdict est :

```
ENGINE_GUARANTEE_STATUS : PASS
REGIME_VIABILITY_STATUS : OPEN_WITH_LOCAL_KNV
PIVOT_STATUS_§5         : TAUTOLOGIQUE_AU_POINT_FIXE
PIVOT_STATUS_§5.7       : NON_TRIVIAL_PATH_DEPENDENCE
```

**Acquis principal** :

> 6d-α révèle l'apparition d'une **géométrie morphodynamique
> irréductible au point fixe local**. Pas une "preuve MCQ", mais
> l'identification empirique de structures (bassins, mémoire
> trajectorielle, stratified reactivation) qui ne se réduisent pas
> à la cohérence algébrique locale du système d'ODE.

**Discipline méthodologique acquise** : la distinction
*validité moteur vs viabilité régime* est désormais formalisée et
opérationnelle.

---

## Couche 1 — Garanti par construction

Cette couche concerne les propriétés inscrites **structurellement dans
le schéma numérique**. Elles sont par construction garanties, donc
leur observation expérimentale n'est pas une "découverte" mais une
vérification d'intégrité du moteur.

### 1.1 Schéma de discrétisation

| Propriété | Statut | Source |
|---|---|---|
| Volumes finis sur grille 5×5×5 Neumann | ✓ construit | §1.1-1.3 spec |
| Conservation Σψ par antisymétrie des flux | ✓ exact 10⁻¹⁶ | §2.1, §2.2, §2.3, §3a, §3b, §4a-δ |
| Antisymétrie flux ψ aux arêtes | ✓ topologique | §2.1 PASS |
| Plancher ψ_floor dans bruit multiplicatif | ✓ par construction | §1.5 |
| `g_Ω > 0` par construction (D > 0, h dyn) | ✓ §6.1 PASS | §6.1 |

### 1.2 Schémas temporels et CFL

| Propriété | Statut | Source |
|---|---|---|
| CFL diffusion `dt < dx²/(2D·DIM·h_max)` | ✓ respecté | §2.1 cas h variable |
| CFL drift `dt < dx/(DIM·h_max·∇Φ_max)` | ✓ respecté | §2.2 |
| **CFL combinée stricte** pour OU | ✓ découvert empiriquement | §2.3 |
| Stabilité Euler explicite vérifiée | ✓ ordre 1 vérifié | §4a-δ ratios 1.86-2.21 |
| CFL stochastique sous Itô multiplicatif | ✓ caractérisé | §3a, §3a-bis |

### 1.3 Tautologies locales identifiées

| Identité | Statut | Conséquence |
|---|---|---|
| `ψ_∞ = (γ/β)(1-h_∞/h₀)` au point fixe couplé | **TAUTOLOGIQUE** | §5 n'est pas un pivot MCQ fort |
| `dh/dt → 0` exact à h = 0 numérique | ✓ propriété logistique | Verrouillage local 4a-η |
| `D_eff = D·h` strictement positif partout où h > 0 | ✓ par construction | §6.1 ENGINE PASS |

**Conséquence méta-méthodologique** : §6 est partiellement
tautologique. §6.1, §6.3, §6.4 sont garantis par construction.
Seul §6.2 (test perturbation bornée) reste un test informatif
indépendant.

### 1.4 Critères statistiques calibrés

| Critère | Convention | Source |
|---|---|---|
| `diff_observée / SEM < 3-4` pour ensemble | empirique 3a | §3a corrigé |
| `frac_under_resolution = N_under / N_total` | construction | §4a-ζ, §6 |
| `H_RESOLUTION = 1e-6` vs `H_FUNCTIONAL = 1e-3` | double seuil | §4a-ζ, §6.2 |
| Tolérance relative 0.01 pour classifier tendance | 1% de N_total | §4a-ζ classify_trend |

---

## Couche 2 — Observé empiriquement

Cette couche concerne les **régularités émergentes** observées dans
les simulations, qui ne sont pas garanties par construction mais
révélées par l'expérience numérique.

### 2.1 Co-évolution ψ↔h

**Mémoire morphologique étendue (§4a-δ, §4a-ε)** :

Sur 3 régimes A/B/C calibrés par rapports d'échelles (`τ_ψ`,
`τ_sed`, `τ_ero`) :

```
corr(-log(h_final/h₀), ∫ψ·dt) ∈ [0.978, 0.999]
```

La métrique finale h trace fidèlement l'intégrale temporelle du
passage de ψ. **La distinction "localité de l'opérateur ≠ localité
de la mémoire émergente"** est empiriquement vérifiée sous co-évolution.

**Feedback ψ↔h sur la dispersion (§4a-ε)** :

Pour ψ initial décentré en (1, 2, 2), la convergence du centre de
masse vers (2, 2, 2) est **ralentie selon le régime** :

| Régime | COM_x final |
|---|---|
| A (h lent / ψ rapide) | 1.99 |
| B (échelles comparables) | 1.68 |
| C (h rapide / ψ lent) | 1.51 |

Premier observable empirique d'un effet **non réductible aux
opérateurs séparés** : la sédimentation modifie h, qui ralentit
la diffusion, qui canalise la dispersion future de ψ.

### 2.2 Stratification accès / réactivabilité / transformabilité

**Bifurcation locale verrouillée (§4a-γ)** :

5 régimes `β·ψ/γ ∈ {0.1, 0.5, 0.9, 1.0, 1.5}` testés ; statut
numérique du point fixe h=0 change selon le ratio :
- sous-critique → instable (système quitte vers K > 0)
- critique exact → décroissance polynomiale
- sur-critique → stable (collapse exponentiel)

**STRATIFIED_REACTIVATION (§4a-η)** :

Re-injection unique de ψ après quasi-collapse. Sur 35/125 cellules
collapsées :
- 17 réactivent (REACTIVATED_SHELL) — régime local sous-critique
  `β·⟨ψ⟩/γ ∈ [0.576, 0.581]`
- 18 restent verrouillées (DYNAMICALLY_LOCKED_DEEP) — régime local
  sur-critique `β·⟨ψ⟩/γ médiane = 1.566`
- 1 cellule au plancher numérique strict (NUMERICAL_FLOOR_CORE)

**Trois notions distinctes inscrites** :
- **ACCÈS DE ψ** : ψ_local atteint la zone
- **RÉACTIVABILITÉ MORPHOLOGIQUE** : h peut remonter au-dessus du
  seuil de résolution
- **TRANSFORMABILITÉ EFFECTIVE** : capacité de transport restaurée
  (concept à préciser en 6d-β)

L'accès n'implique pas la réactivabilité. La réactivabilité
n'implique pas la transformabilité.

### 2.3 Multi-bassins empiriques (§5.7)

Sur β ∈ {45, 60, 80}, 4 familles d'initialisations testées :

| β | D_h_final(A↔B1) | D_h_final(A↔B2) | D_h_final(A↔B3) |
|---|---|---|---|
| 45 | 9.5e-07 ✓ même | **0.163 ≠** | 4.8e-15 ✓ |
| 60 | 1.6e-05 ✓ même | **0.295 ≠** | 1.2e-14 ✓ |
| 80 | **0.233 ≠** | **0.729 ≠** | 0.090 limite |

**B2 (bimodale étroite) atteint un attracteur structurellement
différent de A**. Multi-bassins empiriques confirmés.

**Mécanisme séparateur identifié** : `β·ψ_max_init/γ` combiné à la
capacité de diffusion à uniformiser ψ avant collapse local.

### 2.4 Géométrie de convergence non-triviale

Même quand A et B1 convergent vers le **même** attracteur (D_h_final
≈ 10⁻⁵ à 10⁻⁷), **les trajectoires sont différentes** :

| β | AUC_Dh max sur comparaisons |
|---|---|
| 45 | 88.4 |
| 60 | 146.9 |
| 80 | 352.4 |

La mémoire morphologique de la trajectoire **n'est pas effacée par
la convergence finale**. La mémoire MCQ pertinente n'est pas dans
l'équation locale (tautologique) mais dans **le chemin d'accès au
bassin**.

### 2.5 Fermeture locale du transport sous régime collapse (§6.1, §6.2)

Dans le bassin B2 stationnaire (β=60) :
- 7/125 cellules collapsées (h ≈ 10⁻³²⁴)
- D_eff = D·h ≈ 10⁻³²⁵ dans ces cellules
- Test perturbation bornée ε_ψ = 0.01 sur cellule la plus profonde
- Résultat : 0/7 cellules remontent au-dessus de h_resolution sur
  t=100 post-perturbation
- ψ_mass dans zone : 0.24 (ACCESS_CONFIRMED), mais β·⟨ψ⟩/γ = 2.08
  (sur-critique local) maintient le verrouillage

**ABSORBING_BASIN_EMPIRICAL_B2** : bassin **empiriquement absorbant**
sous perturbation bornée 1% dans la fenêtre simulée.

### 2.6 Distinctions structurelles inscrites

| Distinction | Source | Statut |
|---|---|---|
| Conservation topologique ≠ Positivité dynamique | §3b | acquis |
| Stratification α/β/γ : collapse ODE ≠ sortie num ≠ violation MCQ | §4a-0 | acquis |
| Localité opérateur ≠ localité mémoire émergente | §4a-α | acquis |
| Accès ψ ≠ Réactivabilité ≠ Transformabilité | §4a-η | acquis |
| DYNAMICALLY_LOCKED_DEEP ≠ NUMERICAL_FLOOR_CORE | §4a-η | acquis |
| ENGINE_GUARANTEE ≠ REGIME_VIABILITY | §6 | acquis |
| Tautologie locale §5 ≠ structure globale §5.7 | §5.8 | acquis |
| Bassin empirique ≠ bassin structurel | §6.6 | acquis |

---

## Couche 3 — Explicitement NON validé

Cette couche concerne **ce qui doit être testé par 6d-β ou plus
tard**, sur lequel 6d-α ne se prononce pas.

### 3.1 Observables MCQ-théoriques non testées

| Observable | Statut 6d-α | Lieu de test |
|---|---|---|
| `τ'` comme variable primitive | NON validé | 6d-β |
| `𝒞_T` (distance morphologique pleine) | proxy L2 testé en §4a-ζ ; pleine NON validée | 6d-β |
| `RR / RR² / RR³` (régimes réflexifs) | NON instrumentés | 6d-β |
| `STR / RSR` | NON instrumentés | 6d-β |
| `MI / MV` (méta-invariant / méta-variation) | NON instrumentés | 6d-β |
| `β_QMC` | NON validé | 6d-β |
| `𝒢` plein (gradient transformable) | proxy `G_proxy` non testé | 6d-β |
| `𝕋(t)` (téléonomie) | NON instrumenté | 6d-β ou plus tard |
| `MCQᴺ` (couplage inter-modulaire 𝒞^N) | NON activé | 6d-β |

### 3.2 Conjectures MCQ non démontrées

| Conjecture | Statut | Raison |
|---|---|---|
| Attracteurs structurellement absorbants au sens MCQ | NON démontré | "Empirique" ≠ "structurel" |
| Co-production non-réductible à ODE+PDE classique | NON démontré | Formule locale tautologique |
| Multi-attracteurs garantis MCQ | NON démontré | Multi-bassins empiriques observés mais pas structurellement caractérisés |
| Téléonomie émergente | NON testée | Pas d'observable orienté `cible → action` |
| Présence MCQ minimale h_min strict | postulée, NON émergente | Le moteur permet h → 0 numériquement |

### 3.3 Tests structurels reportés à 6d-β

| Test | Motivation | Priorité |
|---|---|---|
| Sweep ε_ψ pour caractériser seuil de réactivation | Caractériser frontière de bassin | Moyenne |
| Bruit stochastique comme mécanisme réactivateur | Sortir B2 du bassin sans dépendance à ψ_local | Haute |
| Couplage inter-modulaire `𝒞^N` | Mécanisme non-local | Haute |
| Drift redistributif Φ ≠ 0 | Bassins multi-puits potentiels | Haute |
| `RR / RR²` instrumenté | Test régimes réflexifs | Haute |
| `𝒢` plein vs `G_proxy` | Capacité transformable | Moyenne |
| Robustesse hors-distribution §5.7 sur 6d-β | Anti-loss implicite | Critique |

---

## Couche 4 — Tensions ouvertes à préserver

Cette couche concerne les **tensions productives** que le programme
6d-α a révélées et qu'il ne faut **pas résoudre prématurément** par
ajout de mécanismes ou par réinterprétation simplificatrice.

### 4.1 Cohérence locale triviale vs structure globale non triviale

> Au point fixe couplé, `ψ = (γ/β)(1-h/h₀)` est satisfaite par
> construction algébrique. Mais §5.7 famille B montre que différents
> bassins existent, atteints depuis différentes initialisations.

La relation locale est tautologique. La structure des bassins ne
l'est pas. Tension à conserver : ne pas conclure "co-production
validée" depuis §5 ; ne pas conclure "système trivial" depuis la
tautologie locale.

### 4.2 Ouverture du moteur vs fermeture locale du régime

> Le moteur préserve ses 4 garanties par construction (§6 ENGINE PASS).
> Mais le régime B2 atteint produit un bassin localement absorbant
> sous perturbation bornée 1% (REGIME_VIABILITY = ABSORBING_EMPIRICAL).

**Validité moteur ≠ viabilité régime.** Cette divergence productive
prépare 6d-β : les observables émergentes sont à interpréter dans le
bassin spécifique, pas dans une supposée universalité du moteur.

### 4.3 Mémoire trajectorielle sans téléonomie

> AUC_Dh élevés (88-352) entre A et B1 même quand `D_h_final → 0`.
> Le système a une mémoire morphologique de la trajectoire de
> convergence, mais sans orientation téléonomique observable.

La mémoire ne suffit pas à constituer téléonomie. Tension à conserver
pour 6d-β quand `𝕋(t)` sera instrumenté.

### 4.4 Réactivation partielle sans transformabilité garantie

> §4a-η : 17/35 cellules réactivent (h > h_resolution), mais aucune
> mesure n'a été faite que ces cellules sont **transformablement
> équivalentes** à des cellules non-collapsées.

La distinction `RESOLUTION_REACTIVATION_ONLY` vs `FUNCTIONAL_REACTIVATION`
est inscrite mais le seuil `H_FUNCTIONAL = 1e-3` est heuristique. La
transformabilité effective requiert observables MCQ non-locales,
reportées à 6d-β.

### 4.5 Bassin empirique vs bassin structurel

> §6.2 ABSORBING_BASIN_EMPIRICAL : 0 réactivation sous ε_ψ=0.01 à
> t=100. Mais une perturbation différente, du bruit, ou un couplage
> non-local pourraient en principe permettre la réactivation.

Trois mécanismes non testés pourraient encore réactiver :
(i) perturbation diluée qui réduit `β·ψ_local/γ` sous 1,
(ii) bruit stochastique,
(iii) couplage `𝒞^N`.

**Empirical absorbing ≠ structural absorbing.** Tension à conserver
jusqu'à test 6d-β.

### 4.6 Tautologie algébrique au point fixe vs propriété émergente

> Le système d'ODE local impose `ψ = (γ/β)(1-h/h₀)` au point fixe
> couplé par algèbre. Donc §5 ne mesure pas une "co-production"
> émergente — il mesure la cohérence du moteur Euler avec son point
> fixe analytique.

Mais le **chemin** vers ce point fixe, **le bassin** atteint, et la
**géométrie de convergence** ne sont pas tautologiques. C'est là que
6d-β doit chercher les vraies observables émergentes.

---

## Conclusion provisoire

### Ce que 6d-α a accompli

1. **Moteur numérique 6d intègre** : 4 garanties préservées par
   construction, conservation Σψ machine precision, ordre Euler
   vérifié empiriquement.

2. **Acquis empiriques productifs identifiés** :
   - mémoire morphologique étendue (corr > 0.97 sur 3 régimes)
   - feedback ψ↔h sur la dispersion (COM_x dépend du régime)
   - STRATIFIED_REACTIVATION (mécanisme bifurcationnel local)
   - multi-bassins empiriques (§5.7 famille B)
   - bassin absorbant empirique sous perturbation 1% (§6.2)

3. **Distinctions structurelles inscrites** dans la spec :
   - validité moteur ≠ viabilité régime
   - tautologie locale ≠ structure globale
   - bassin empirique ≠ bassin structurel
   - accès ≠ réactivabilité ≠ transformabilité

4. **Méthodologie consolidée** :
   - critère statistique correct (diff/SEM)
   - stratification engine_guarantee / regime_viability
   - test de perturbation bornée comme outil empirique
   - protocoles famille A / famille B (§5.7 anti-loss-implicite)

5. **Cadre conceptuel préparé pour 6d-β** :
   - observables à instrumenter clarifiées (Couche 3)
   - tensions à préserver explicitées (Couche 4)
   - tests reportés listés

### Ce que 6d-α n'a pas accompli

1. **Pas de pivot MCQ fort validé** : §5 est tautologique au point
   fixe ; §5.7 révèle multi-bassins empiriques mais pas
   structurels au sens MCQ.

2. **Pas d'observable émergente non triviale instrumentée** :
   `RR/RR²`, `𝒢` plein, `τ'`, `𝒞_T` plein, `𝕋(t)` restent à
   construire en 6d-β.

3. **Pas de test stochastique en co-évolution** : le bruit a été
   testé sur diffusion seule (§3a) et diffusion+drift (§3b), pas
   en couplage ψ↔h.

4. **Pas de test couplage inter-modulaire** : `𝒞^N` non activé.

### Caveat ontologique strict (verbatim §5.6 / §10.5)

> τ' reste une observable émergente conditionnelle au régime
> numérique testé, pas une variable primitive garantie.

> "Multi-bassins empiriques" ≠ "multi-attracteurs structurellement
> garantis MCQ".

> "Bassin empiriquement absorbant" ≠ "bassin structurellement
> absorbant".

> Le programme 6d-α valide la **cohérence numérique du moteur** et
> révèle des **régularités empiriques productives**. Il ne valide
> **pas** une co-production MCQ structurelle au sens Ch3 strict.

### Passage à 6d-β

**Autorisé avec statut conditionnel** :

- Les observables 6d-β doivent être interprétées **dans le régime
  du bassin atteint**.
- Robustesse hors-distribution (§5.7 famille B) à tester sur chaque
  observable 6d-β.
- Toute conclusion MCQ forte 6d-β doit citer "co-production validée
  hors-distribution dans bassin spécifié".

**Discipline acquise à conserver** :
- mini-spec avant code (§4a-0 a démontré son utilité)
- audit Alex à chaque étape (a permis de corriger mes erreurs
  d'interprétation à plusieurs reprises : ordre Euler, bifurcation
  locale 4a-η, tautologie §5)
- distinction structurelle ENGINE / REGIME
- caveat strict sur "empirique vs structurel"

---

## Annexe — Index des sessions

| Session | Contenu principal | Acquis |
|---|---|---|
| 1-6 | §2.1 diffusion + §2.2 drift | conservation col_sums, CFL drift |
| 7-11 | §2.3 OU + mini-validation h fixe | CFL combinée stricte |
| 12-14 | §3a bruit + §3a-bis cartographie | conservation topologique ≠ positivité dynamique |
| 15-17 | §3b OU + bruit | drift déplace frontière positivité |
| 18-19 | §4a-α/β/γ | bifurcation locale verrouillée |
| 20-21 | §4a-δ/ε co-évolution + ζ classification | mémoire morphologique, léthargie sans fragmentation |
| 22-23 | §4a-η re-injection + raffinement | STRATIFIED_REACTIVATION, mécanisme bifurcationnel local |
| 24 | §A.5 sweep β | tautologie locale identifiée |
| 25 | §5.7 famille B | multi-bassins empiriques |
| 26 | spec §5.8 inscrite | distinction tautologie/structure |
| 27 | §6 + §10.5 + ce rapport | clôture provisoire 6d-α |

---

**Fin du rapport 6d-α.**

Le programme est clôturé. 6d-β peut maintenant être préparé
conceptuellement sur cette base.
