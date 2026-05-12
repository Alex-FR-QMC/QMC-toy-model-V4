"""
6d-α minimal state.

Contraintes strictes (validation Alex pour étape 1) :
1. état minimal : ψ[5,5,5] + h[5,5,5]
2. pas de g_Ω
3. pas de bruit
4. pas de coupling
5. pas de drift
6. h = h₀ uniforme (au démarrage)
7. diffusion pure conformal-conservative
8. test analytique unique : Var_a(t) ≈ Var_a(0) + 2·D·t par axe

Voir 6d-alpha-numerics.md §1.0, §1.1.

Module séparé de factorial/ : pas de modification de l'engine 6a/6b/6c
existant, qui sert encore de baseline historique.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# Constantes de grille (numerics-doc §1.0)
N_AXIS = 5  # taille par axe
DX = 1.0  # pas de grille (Θ adimensionnel)
DIM = 3  # dimension spatiale

# Bornes métriques par défaut (numerics-doc §1.0)
H0_DEFAULT = 1.0
H_MIN_DEFAULT = 0.1


@dataclass
class State6dMinimal:
    """
    État minimal 6d-α étape 1.

    Attributs :
    - psi : np.ndarray shape (5, 5, 5), distribution de présence normalisée.
            Σ psi = 1, psi ≥ 0 partout.
    - h : np.ndarray shape (5, 5, 5), facteur conforme métrique.
          h ∈ [h_min, h₀] partout.

    Pas de marginales h_T, h_M, h_I (ce serait factorial/ 6a-6c).
    Pas de Γ_meta accumulé séparément (h porte la mémoire métrique).
    Pas de V_𝒩 explicite (réduit à h pour 6d-α étape 1).
    """

    psi: np.ndarray
    h: np.ndarray
    h0: float = H0_DEFAULT
    h_min: float = H_MIN_DEFAULT

    def __post_init__(self) -> None:
        expected_shape = (N_AXIS, N_AXIS, N_AXIS)
        if self.psi.shape != expected_shape:
            raise ValueError(
                f"psi shape {self.psi.shape} ≠ attendu {expected_shape}"
            )
        if self.h.shape != expected_shape:
            raise ValueError(
                f"h shape {self.h.shape} ≠ attendu {expected_shape}"
            )

    # --- Invariants à vérifier en logging (numerics-doc §8) ---

    def total_mass(self) -> float:
        """Σ ψ (sans dx³ car dx=1). Doit rester 1.0 à machine precision."""
        return float(np.sum(self.psi))

    def min_psi(self) -> float:
        """min(ψ). Doit rester ≥ -ε_machine."""
        return float(np.min(self.psi))

    def max_psi(self) -> float:
        return float(np.max(self.psi))

    def min_h(self) -> float:
        return float(np.min(self.h))

    def max_h(self) -> float:
        return float(np.max(self.h))

    # --- Observables géométriques pour test diffusion §2.1 ---

    def variance_per_axis(self) -> tuple[float, float, float]:
        """
        Variance de la distribution ψ le long de chaque axe.

        Var_a = Σ_θ ψ(θ) · (θ_a - <θ_a>)²

        où <θ_a> est la moyenne pondérée selon ψ.

        Sur grille 5×5×5, les coordonnées d'axe vont de 0 à 4 (indices),
        avec dx=1. Le centre géométrique est à 2.

        Retourne (Var_x, Var_y, Var_z).
        """
        # Coordonnées 1D par axe : 0, 1, 2, 3, 4 (en unités dx=1)
        coords = np.arange(N_AXIS) * DX

        # Pour chaque axe, marginaliser sur les deux autres
        # psi_x[i] = Σ_{j,k} psi[i,j,k]
        psi_x = self.psi.sum(axis=(1, 2))
        psi_y = self.psi.sum(axis=(0, 2))
        psi_z = self.psi.sum(axis=(0, 1))

        def variance_1d(psi_axis: np.ndarray) -> float:
            total = psi_axis.sum()
            if total <= 0:
                return 0.0
            mean = (coords * psi_axis).sum() / total
            var = ((coords - mean) ** 2 * psi_axis).sum() / total
            return float(var)

        return (
            variance_1d(psi_x),
            variance_1d(psi_y),
            variance_1d(psi_z),
        )

    def variance_total(self) -> float:
        """Var totale = Var_x + Var_y + Var_z."""
        vx, vy, vz = self.variance_per_axis()
        return vx + vy + vz

    def center_of_mass(self) -> tuple[float, float, float]:
        """<θ> pondéré par ψ, par axe."""
        coords = np.arange(N_AXIS) * DX
        total = self.psi.sum()
        if total <= 0:
            return (2.0, 2.0, 2.0)  # centre géométrique par défaut

        psi_x = self.psi.sum(axis=(1, 2))
        psi_y = self.psi.sum(axis=(0, 2))
        psi_z = self.psi.sum(axis=(0, 1))

        return (
            float((coords * psi_x).sum() / total),
            float((coords * psi_y).sum() / total),
            float((coords * psi_z).sum() / total),
        )


# --- Constructeurs d'état initial ---


def make_gaussian_state(
    sigma_0: float,
    center: tuple[float, float, float] = (2.0, 2.0, 2.0),
    h_uniform: float = H0_DEFAULT,
    h_min: float = H_MIN_DEFAULT,
) -> State6dMinimal:
    """
    Initialise une gaussienne 3D centrée + h uniforme.

    Paramètres :
    - sigma_0 : écart-type de la gaussienne par axe.
                **MUST be ≥ 1.5·dx** pour respecter Nyquist (numerics §2.1).
                Recommandé : 1.8·dx (correction b auto-stress-test).
    - center : position du centre, par défaut (2,2,2) = centre géométrique
               de la grille 5×5×5.
    - h_uniform : valeur de h appliquée uniformément (=h₀ par défaut).
    - h_min : borne inférieure (utilisée pour validation, pas d'effet
              ici puisque h est uniforme à h_uniform).

    Retourne un État normalisé tel que Σ ψ = 1.
    """
    if sigma_0 < 1.5 * DX:
        raise ValueError(
            f"sigma_0={sigma_0} < 1.5·dx={1.5*DX} viole Nyquist (numerics §2.1)"
        )

    coords = np.arange(N_AXIS) * DX
    cx, cy, cz = center

    # Gaussienne 3D séparable (isotrope)
    gx = np.exp(-0.5 * ((coords - cx) / sigma_0) ** 2)
    gy = np.exp(-0.5 * ((coords - cy) / sigma_0) ** 2)
    gz = np.exp(-0.5 * ((coords - cz) / sigma_0) ** 2)

    psi = np.einsum("i,j,k->ijk", gx, gy, gz)
    psi = psi / psi.sum()  # normalisation Σψ = 1

    h = np.full((N_AXIS, N_AXIS, N_AXIS), h_uniform, dtype=float)

    return State6dMinimal(psi=psi, h=h, h0=h_uniform, h_min=h_min)
