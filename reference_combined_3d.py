"""
Référence matricielle 3D combinée pour §2.3 Ornstein-Uhlenbeck.

Construit L_total = D · L_diffusion(h) + L_drift(Φ, h)
tel que ∂_t ψ_flat = L_total · ψ_flat.

Linéarité de la combinaison :
- L_diffusion(h) est symétrique, row_sums = col_sums = 0.
- L_drift(Φ, h) est asymétrique, col_sums = 0 (row_sums ≠ 0 en général).
- L_total est asymétrique en général, col_sums = 0 (linéarité).

Propriétés attendues de L_total :
- Asymétrique (héritée de L_drift)
- col_sums = 0 (conservation de masse stricte)
- off-diag ≥ 0 (heritée des deux composantes M-matrix)
- diag ≤ 0
- Valeur propre 0 simple → distribution stationnaire = mode propre droit
  de cette valeur propre, à comparer à Boltzmann ψ_∞ ∝ exp(-Φ/D)
  comme diagnostic non-bloquant.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm

from .state import N_AXIS
from .reference_neumann_3d import build_L3d_neumann
from .reference_drift_3d import build_L_drift_3d


def build_L_total(
    Phi: np.ndarray, h: np.ndarray, D: float
) -> np.ndarray:
    """
    Construit L_total = D · L_diffusion(h) + L_drift(Φ, h).

    Note : build_L3d_neumann retourne L sans le facteur D
    (cf. son docstring : ∂_t ψ_flat = D · L_3D · ψ_flat dans le moteur
    diffusion). Donc on multiplie ici par D explicitement.

    build_L_drift_3d retourne L directement (sans facteur supplémentaire).
    """
    L_diff = build_L3d_neumann(h)
    L_drift = build_L_drift_3d(Phi, h)
    return D * L_diff + L_drift


def euler_combined_3d(
    psi_0: np.ndarray,
    Phi: np.ndarray,
    h: np.ndarray,
    D: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """
    Référence Euler explicite combinée : (I + dt·L_total)^n · ψ_0_flat.
    """
    N = psi_0.shape[0]
    L = build_L_total(Phi, h, D)
    psi_flat = psi_0.ravel(order="C").copy()
    M = np.eye(N**3) + dt * L
    for _ in range(n_steps):
        psi_flat = M @ psi_flat
    return psi_flat.reshape((N, N, N))


def semi_discrete_combined_3d(
    psi_0: np.ndarray,
    Phi: np.ndarray,
    h: np.ndarray,
    D: float,
    t: float,
) -> np.ndarray:
    """
    Référence semi-discrète exacte : exp(L_total · t) · ψ_0_flat.
    """
    N = psi_0.shape[0]
    L = build_L_total(Phi, h, D)
    psi_flat = psi_0.ravel(order="C").copy()
    M_exp = expm(L * t)
    return (M_exp @ psi_flat).reshape((N, N, N))


def stationary_distribution(
    Phi: np.ndarray, h: np.ndarray, D: float, tol: float = 1e-12
) -> tuple[np.ndarray, float]:
    """
    Calcule la distribution stationnaire discrète : vecteur propre droit
    de L_total associé à la valeur propre nulle.

    Retourne (ψ_∞ shape (N,N,N) normalisé Σψ_∞=1, eigenvalue_residual).
    """
    N = Phi.shape[0]
    L = build_L_total(Phi, h, D)
    eigenvalues, eigenvectors_right = np.linalg.eig(L)
    # Trouver la vp la plus proche de 0 (en partie réelle)
    idx_zero = int(np.argmin(np.abs(eigenvalues)))
    lambda_zero = eigenvalues[idx_zero]
    vec = eigenvectors_right[:, idx_zero].real
    # Normaliser pour Σ = 1 (et signe positif)
    if vec.sum() < 0:
        vec = -vec
    vec = vec / vec.sum()
    return vec.reshape((N, N, N)), float(abs(lambda_zero))


def boltzmann_distribution(Phi: np.ndarray, D: float) -> np.ndarray:
    """
    Distribution de Boltzmann continuum : ψ_∞ ∝ exp(-Φ/D), normalisée.

    Référence DIAGNOSTIC pour comparaison avec la vraie stationnaire
    discrète. Sur grille bornée Neumann, peut différer significativement.
    """
    N = Phi.shape[0]
    psi_boltz = np.exp(-Phi / D)
    return psi_boltz / psi_boltz.sum()


def verify_L_total_properties(L: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Vérifie propriétés structurelles de L_total :
    1. Non symétrique (en général)
    2. col_sums = 0 (conservation)
    3. off-diag ≥ 0
    4. diag ≤ 0
    5. Au moins une vp = 0 (mode invariant)
    """
    max_asymmetry = float(np.max(np.abs(L - L.T)))
    is_symmetric = max_asymmetry < tol

    col_sums = L.sum(axis=0)
    max_col_sum = float(np.max(np.abs(col_sums)))
    cols_sum_to_zero = max_col_sum < tol

    row_sums = L.sum(axis=1)
    max_row_sum = float(np.max(np.abs(row_sums)))

    off_diag = L - np.diag(np.diag(L))
    min_off_diag = float(off_diag.min())
    off_diag_nonneg = min_off_diag >= -tol

    diag = np.diag(L)
    max_diag = float(diag.max())
    diag_nonpos = max_diag <= tol

    eigenvalues = np.linalg.eigvals(L)
    real_max = float(np.max(eigenvalues.real))
    real_min = float(np.min(eigenvalues.real))
    has_zero_eig = abs(real_max) < 1e-10

    return {
        "is_symmetric": bool(is_symmetric),
        "max_asymmetry": max_asymmetry,
        "cols_sum_to_zero": bool(cols_sum_to_zero),
        "max_col_sum": max_col_sum,
        "max_row_sum_info_only": max_row_sum,
        "off_diag_nonneg": bool(off_diag_nonneg),
        "min_off_diag": min_off_diag,
        "diag_nonpos": bool(diag_nonpos),
        "max_diag": max_diag,
        "has_zero_eig": bool(has_zero_eig),
        "real_max_eigenvalue": real_max,
        "real_min_eigenvalue": real_min,
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
    }
