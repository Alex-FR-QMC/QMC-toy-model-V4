"""
Référence matricielle 3D du drift Fokker-Planck upwind.

Construit L_drift_3D(Φ, h) ∈ ℝ^{N³×N³} tel que :

    ∂_t ψ_flat = L_drift_3D · ψ_flat

soit exactement le schéma upwind du moteur factorial_6d/drift.py pour
un potentiel Φ et une métrique h donnés.

Convention de signe (Option 1, validée Alex) :
    J_drift = -h_face · ψ_upwind · ∇Φ
    ∂_t ψ = -∇·J_drift

Propriétés attendues de L_drift_3D :
- Non symétrique (upwind brise la symétrie, contrairement au Laplacien)
- Somme de lignes = 0 (conservation de masse)
- Coefficients hors-diagonale ≥ 0 (M-matrix property : positivité
  sous Euler explicite si CFL respecté)
- Coefficients diagonaux ≤ 0
- Valeur propre 0 = mode propre (mais distribution non-uniforme :
  le mode invariant du drift est concentré sur le puits θ_0)
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm

from .state import N_AXIS, DX
from .engine import harmonic_mean


def build_L_drift_3d(Phi: np.ndarray, h: np.ndarray, dx: float = DX) -> np.ndarray:
    """
    Construit la matrice L_drift_3D ∈ ℝ^{N³×N³}.

    L_drift_3D · ψ_flat reproduit exactement la divergence drift
    calculée par le moteur drift.py.

    Aplatissement : numpy ravel order='C' :
        index_flat(i,j,k) = i*N² + j*N + k

    Pour chaque interface (i, i+1) en x :
    - flux à l'interface = -h_face · ψ_upwind · grad_Phi_x
    - grad_Phi_x = (Phi[i+1] - Phi[i])/dx
    - ψ_upwind = ψ[i+1] si grad_Phi >= 0 sinon ψ[i]
      (cohérent avec drift.py)
    - cellule i perd flux/dx, cellule i+1 gagne flux/dx
      (cohérent avec divergence_drift)

    Note : "flux" signé selon convention. Si grad_Phi >= 0, flux < 0,
    donc cellule (i+1) PERD masse (vers i), cellule i GAGNE masse.
    """
    N = h.shape[0]
    assert h.shape == (N, N, N)
    assert Phi.shape == (N, N, N)
    assert N == N_AXIS

    Ntot = N**3
    L = np.zeros((Ntot, Ntot), dtype=float)
    inv_dx = 1.0 / dx

    def idx(i: int, j: int, k: int) -> int:
        return i * N * N + j * N + k

    # Faces normales à x
    for i in range(N - 1):
        for j in range(N):
            for k in range(N):
                a = idx(i, j, k)        # cellule gauche
                b = idx(i + 1, j, k)    # cellule droite

                h_left = h[i, j, k]
                h_right = h[i + 1, j, k]
                h_face = 2.0 * h_left * h_right / (h_left + h_right + 1e-30)

                grad_Phi_x = (Phi[i + 1, j, k] - Phi[i, j, k]) * inv_dx

                # Source upwind (mêmes règles que drift.py)
                if grad_Phi_x >= 0.0:
                    # ψ_upwind = ψ_{i+1} (source = b)
                    coeff_face = -h_face * grad_Phi_x  # = J/ψ_b
                    # flux signé = coeff_face · ψ_b
                    # cellule a (i) perd -J/dx (mais J négatif → gain pour a)
                    # ∂t ψ_a += -J/dx = -coeff_face·ψ_b/dx
                    # ∂t ψ_b += +J/dx = +coeff_face·ψ_b/dx
                    # Note : coeff_face est négatif ici (grad_Phi positif × -h_face),
                    # donc L[a,b] = -coeff_face/dx = +h_face·grad/dx > 0 (entrée pour a)
                    # L[b,b] = +coeff_face/dx = -h_face·grad/dx < 0 (perte pour b)
                    L[a, b] += -coeff_face * inv_dx  # >= 0
                    L[b, b] += coeff_face * inv_dx   # <= 0
                else:
                    # grad_Phi_x < 0 → ψ_upwind = ψ_i (source = a)
                    # flux = -h_face·ψ_a·grad_Phi (positif car grad négatif × -)
                    # cellule a perd flux/dx, cellule b gagne flux/dx
                    coeff_face = -h_face * grad_Phi_x  # > 0 (grad négatif × -)
                    L[a, a] += -coeff_face * inv_dx   # <= 0 (perte pour a)
                    L[b, a] += coeff_face * inv_dx    # >= 0 (entrée pour b)

    # Faces normales à y
    for i in range(N):
        for j in range(N - 1):
            for k in range(N):
                a = idx(i, j, k)
                b = idx(i, j + 1, k)

                h_left = h[i, j, k]
                h_right = h[i, j + 1, k]
                h_face = 2.0 * h_left * h_right / (h_left + h_right + 1e-30)

                grad_Phi_y = (Phi[i, j + 1, k] - Phi[i, j, k]) * inv_dx

                if grad_Phi_y >= 0.0:
                    coeff_face = -h_face * grad_Phi_y
                    L[a, b] += -coeff_face * inv_dx
                    L[b, b] += coeff_face * inv_dx
                else:
                    coeff_face = -h_face * grad_Phi_y
                    L[a, a] += -coeff_face * inv_dx
                    L[b, a] += coeff_face * inv_dx

    # Faces normales à z
    for i in range(N):
        for j in range(N):
            for k in range(N - 1):
                a = idx(i, j, k)
                b = idx(i, j, k + 1)

                h_left = h[i, j, k]
                h_right = h[i, j, k + 1]
                h_face = 2.0 * h_left * h_right / (h_left + h_right + 1e-30)

                grad_Phi_z = (Phi[i, j, k + 1] - Phi[i, j, k]) * inv_dx

                if grad_Phi_z >= 0.0:
                    coeff_face = -h_face * grad_Phi_z
                    L[a, b] += -coeff_face * inv_dx
                    L[b, b] += coeff_face * inv_dx
                else:
                    coeff_face = -h_face * grad_Phi_z
                    L[a, a] += -coeff_face * inv_dx
                    L[b, a] += coeff_face * inv_dx

    return L


def euler_drift_3d(
    psi_0: np.ndarray, Phi: np.ndarray, h: np.ndarray, dt: float, n_steps: int
) -> np.ndarray:
    """
    Référence Euler explicite 3D pour drift pur :
    (I + dt·L_drift_3D)^n · ψ_0_flat.
    """
    N = psi_0.shape[0]
    L = build_L_drift_3d(Phi, h)
    psi_flat = psi_0.ravel(order="C").copy()
    M = np.eye(N**3) + dt * L
    for _ in range(n_steps):
        psi_flat = M @ psi_flat
    return psi_flat.reshape((N, N, N))


def semi_discrete_drift_3d(
    psi_0: np.ndarray, Phi: np.ndarray, h: np.ndarray, t: float
) -> np.ndarray:
    """
    Référence semi-discrète exacte pour drift pur :
    exp(L_drift_3D · t) · ψ_0_flat.
    """
    N = psi_0.shape[0]
    L = build_L_drift_3d(Phi, h)
    psi_flat = psi_0.ravel(order="C").copy()
    M_exp = expm(L * t)
    return (M_exp @ psi_flat).reshape((N, N, N))


def verify_L_drift_properties(L: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Vérifie les propriétés structurelles de L_drift :
    1. Asymétrique en général (upwind brise la symétrie)
    2. **Sommes de COLONNES = 0** (conservation de masse pour une matrice
       non symétrique). Détails : ∂_t (Σψ) = Σ_a Σ_b L[a,b]·ψ_b
       = Σ_b ψ_b · (Σ_a L[a,b]), donc il faut Σ_a L[a,b] = 0 ∀ b,
       c'est-à-dire col_sums = 0.
       (Pour Laplacien symétrique, L = L.T donc col_sums = row_sums.
       Pour L_drift asymétrique, ces deux quantités diffèrent.)
    3. Coefficients hors-diagonale ≥ 0 (M-matrix, positivité)
    4. Coefficients diagonaux ≤ 0
    5. Au moins une valeur propre = 0 (mode invariant)
    """
    Ntot = L.shape[0]

    # Symétrie (attendue : asymétrique sauf cas trivial Φ=0)
    max_asymmetry = float(np.max(np.abs(L - L.T)))
    is_symmetric = max_asymmetry < tol

    # Conservation : Σ COLONNES = 0
    col_sums = L.sum(axis=0)
    max_col_sum = float(np.max(np.abs(col_sums)))
    cols_sum_to_zero = max_col_sum < tol

    # Aussi calculer row_sums pour info (mais ce n'est PAS le critère
    # de conservation pour matrice non symétrique)
    row_sums = L.sum(axis=1)
    max_row_sum = float(np.max(np.abs(row_sums)))

    # M-matrix : off-diag ≥ 0
    off_diag = L - np.diag(np.diag(L))
    min_off_diag = float(off_diag.min())
    off_diag_nonneg = min_off_diag >= -tol

    # Diagonal ≤ 0
    diag = np.diag(L)
    max_diag = float(diag.max())
    diag_nonpos = max_diag <= tol

    # Valeurs propres
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
