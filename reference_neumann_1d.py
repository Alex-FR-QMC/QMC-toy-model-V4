"""
Référence discrète Neumann 1D pour validation du moteur diffusion 6d-α.

Construit le générateur discret L_N (matrice N×N) du Laplacien 1D avec
conditions Neumann zero-flux, identique à ce que produit le moteur 3D
sur chaque axe (h uniforme, séparabilité de l'état initial gaussien).

Deux références fournies :
1. Euler explicite discret : (I + dt·D·L)^n · p₀
2. Semi-discret exact : exp(D·L·t) · p₀

Critère §2.1 amendé :
- PASS : engine 3D coïncide avec (I + dt·D·L)^n · p₀ à 1e-12 près
  (Euler discret exact = ce que fait le moteur)
- PASS : engine 3D coïncide avec exp(D·L·t)·p₀ à l'erreur d'Euler près
  (équivaut à mesurer la consistance temporelle)
- Continuum Var = Var_0 + 2D·t : diagnostic seulement, pas critère.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm


def neumann_laplacian_1d(n: int, dx: float = 1.0) -> np.ndarray:
    """
    Génère le générateur discret 1D du Laplacien avec Neumann zero-flux.

    Schéma volumes finis identique au moteur 3D quand h=1 partout :
    L_ij = (1/dx²) · matrice tridiagonale [1, -2, 1] aux nœuds intérieurs,
    avec aux frontières : pas de flux sortant → coefficients ajustés.

    En volumes finis 1D Neumann :
    - cellule 0 : ∂_t ψ_0 = (1/dx²)·(ψ_1 - ψ_0)        (pas de flux à gauche)
    - cellule i intérieure : (1/dx²)·(ψ_{i-1} - 2·ψ_i + ψ_{i+1})
    - cellule N-1 : (1/dx²)·(ψ_{N-2} - ψ_{N-1})        (pas de flux à droite)

    Donc L est la matrice tridiagonale avec :
    - L[0,0] = -1, L[0,1] = +1
    - L[i,i-1] = +1, L[i,i] = -2, L[i,i+1] = +1 (i ∈ [1, N-2])
    - L[N-1,N-2] = +1, L[N-1,N-1] = -1
    - L *= 1/dx²
    """
    L = np.zeros((n, n), dtype=float)

    # Intérieur
    for i in range(1, n - 1):
        L[i, i - 1] = 1.0
        L[i, i] = -2.0
        L[i, i + 1] = 1.0

    # Bord gauche
    L[0, 0] = -1.0
    L[0, 1] = 1.0

    # Bord droit
    L[n - 1, n - 2] = 1.0
    L[n - 1, n - 1] = -1.0

    L /= dx * dx
    return L


def euler_discrete_reference(
    p0: np.ndarray, D: float, dt: float, n_steps: int
) -> np.ndarray:
    """
    Référence Euler explicite discrète : (I + dt·D·L)^n · p₀.

    Identique numériquement à ce que le moteur fait sur 1D (par
    séparabilité), à condition que L soit construit de la même façon.

    Retourne p à l'instant t = n_steps · dt.
    """
    n = len(p0)
    L = neumann_laplacian_1d(n)
    M = np.eye(n) + dt * D * L
    p = p0.copy()
    for _ in range(n_steps):
        p = M @ p
    return p


def semi_discrete_reference(p0: np.ndarray, D: float, t: float) -> np.ndarray:
    """
    Référence semi-discrète exacte : exp(D·L·t) · p₀.

    C'est la solution exacte de l'EDP discrétisée en espace mais
    continue en temps (générateur infinitésimal). L'écart avec
    euler_discrete_reference quantifie l'erreur de discrétisation
    temporelle Euler.
    """
    n = len(p0)
    L = neumann_laplacian_1d(n)
    return expm(D * L * t) @ p0


def variance_1d(p: np.ndarray, dx: float = 1.0) -> tuple[float, float]:
    """Retourne (mean, var) de la distribution 1D p."""
    coords = np.arange(len(p)) * dx
    total = p.sum()
    if total <= 0:
        return 0.0, 0.0
    mean = (coords * p).sum() / total
    var = ((coords - mean) ** 2 * p).sum() / total
    return float(mean), float(var)


def make_gaussian_1d(sigma_0: float, center: float = 2.0, n: int = 5, dx: float = 1.0) -> np.ndarray:
    """Initialisation gaussienne 1D non normalisée (la séparabilité 3D
    redonnera la normalisation correcte si appliquée par axe)."""
    coords = np.arange(n) * dx
    return np.exp(-0.5 * ((coords - center) / sigma_0) ** 2)


def marginal_x(psi_3d: np.ndarray) -> np.ndarray:
    """Extrait la marginale 1D sur l'axe x : ψ_x[i] = Σ_{j,k} ψ[i,j,k]."""
    return psi_3d.sum(axis=(1, 2))
