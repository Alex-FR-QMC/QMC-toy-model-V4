"""
Référence matricielle 3D du Laplacien Neumann conformal-conservatif
discret pour validation étape 2a.

Construit L_3D(h) ∈ ℝ^{125×125} tel que :

    ∂_t ψ = D · L_3D(h) · ψ

soit exactement le schéma discret du moteur factorial_6d/engine.py pour
une métrique conforme h(θ) ∈ ℝ^{5×5×5} donnée fixe.

Référence Euler exacte : (I + dt·D·L_3D)^n · ψ_0_flat
Référence semi-discrète : exp(D·L_3D·t) · ψ_0_flat

Conventions :
- Aplatissement avec numpy `.ravel(order='C')` :
    index_flat = k + N·j + N²·i pour le tableau psi[i,j,k] ?
  En réalité ravel('C') donne : pour psi[i,j,k] de shape (Ni,Nj,Nk),
    index = i*Nj*Nk + j*Nk + k
  C'est l'ordre C (i varie le plus lentement).
- Moyenne harmonique aux interfaces (identique à engine.py).
- Neumann zero-flux : pas de flux aux interfaces externes.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm

from .state import N_AXIS, DX
from .engine import harmonic_mean


def build_L3d_neumann(h: np.ndarray, dx: float = DX) -> np.ndarray:
    """
    Construit le générateur Neumann discret 3D L_3D ∈ ℝ^{N³×N³}.

    Tel que `∂_t ψ_flat = D · L_3D · ψ_flat` soit identique à
    l'évolution sans le facteur D, c'est-à-dire qu'on retient
    le générateur sans coefficient D dehors.

    En entrée :
    - h : np.ndarray shape (N, N, N), facteur conforme.

    Sortie :
    - L_3D : np.ndarray shape (N³, N³), opérateur linéaire tel que
      l'engine 3D vérifie psi_new = psi + dt·D·(L_3D @ psi_flat).

    Conventions d'aplatissement : numpy ravel order='C' :
        index_flat(i,j,k) = i·N² + j·N + k
    """
    N = h.shape[0]
    assert h.shape == (N, N, N), f"h doit être (N,N,N), reçu {h.shape}"
    assert N == N_AXIS, f"N={N} ≠ N_AXIS={N_AXIS}"

    Ntot = N**3

    def idx(i: int, j: int, k: int) -> int:
        """Index flat numpy C-order."""
        return i * N * N + j * N + k

    L = np.zeros((Ntot, Ntot), dtype=float)
    inv_dx2 = 1.0 / (dx * dx)

    # On parcourt toutes les paires (cell, voisin) en construisant
    # h_face = moyenne harmonique entre les deux cellules.
    # Le coefficient de couplage est h_face · (1/dx²).

    # Faces normales à x (entre (i,j,k) et (i+1,j,k))
    for i in range(N - 1):
        for j in range(N):
            for k in range(N):
                h_left = h[i, j, k]
                h_right = h[i + 1, j, k]
                # harmonic_mean scalar
                h_face = 2.0 * h_left * h_right / (h_left + h_right + 1e-30)
                coupling = h_face * inv_dx2
                a = idx(i, j, k)
                b = idx(i + 1, j, k)
                # ψ_a perd coupling·(ψ_a - ψ_b), ψ_b gagne la même
                # ∂t ψ_a += coupling·(ψ_b - ψ_a)
                # ∂t ψ_b += coupling·(ψ_a - ψ_b)
                L[a, a] -= coupling
                L[a, b] += coupling
                L[b, b] -= coupling
                L[b, a] += coupling

    # Faces normales à y
    for i in range(N):
        for j in range(N - 1):
            for k in range(N):
                h_left = h[i, j, k]
                h_right = h[i, j + 1, k]
                h_face = 2.0 * h_left * h_right / (h_left + h_right + 1e-30)
                coupling = h_face * inv_dx2
                a = idx(i, j, k)
                b = idx(i, j + 1, k)
                L[a, a] -= coupling
                L[a, b] += coupling
                L[b, b] -= coupling
                L[b, a] += coupling

    # Faces normales à z
    for i in range(N):
        for j in range(N):
            for k in range(N - 1):
                h_left = h[i, j, k]
                h_right = h[i, j, k + 1]
                h_face = 2.0 * h_left * h_right / (h_left + h_right + 1e-30)
                coupling = h_face * inv_dx2
                a = idx(i, j, k)
                b = idx(i, j, k + 1)
                L[a, a] -= coupling
                L[a, b] += coupling
                L[b, b] -= coupling
                L[b, a] += coupling

    return L


def euler_discrete_3d(
    psi_0: np.ndarray, h: np.ndarray, D: float, dt: float, n_steps: int
) -> np.ndarray:
    """
    Référence Euler explicite 3D : (I + dt·D·L_3D)^n · ψ_0_flat.

    psi_0 : np.ndarray (N,N,N)
    h : np.ndarray (N,N,N), métrique conforme
    Retourne psi_final shape (N,N,N).
    """
    N = psi_0.shape[0]
    L = build_L3d_neumann(h)
    psi_flat = psi_0.ravel(order="C").copy()
    M = np.eye(N**3) + dt * D * L
    for _ in range(n_steps):
        psi_flat = M @ psi_flat
    return psi_flat.reshape((N, N, N))


def semi_discrete_3d(
    psi_0: np.ndarray, h: np.ndarray, D: float, t: float
) -> np.ndarray:
    """
    Référence semi-discrète exacte : exp(D·L_3D·t) · ψ_0_flat.

    Coûteux pour N³=125 mais reste tractable (expm sur matrice 125×125).
    """
    N = psi_0.shape[0]
    L = build_L3d_neumann(h)
    psi_flat = psi_0.ravel(order="C").copy()
    M_exp = expm(D * L * t)
    psi_flat_final = M_exp @ psi_flat
    return psi_flat_final.reshape((N, N, N))


def verify_L3d_properties(L: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Vérifie les propriétés structurelles de L_3D :
    1. Symétrie : L = L^T (la moyenne harmonique est symétrique)
    2. Sommes de lignes = 0 (conservation de masse : ∑ ∂t ψ_a = 0
       pour toute cellule a, parce que les flux entrants et sortants
       s'équilibrent au niveau global ; au niveau d'une ligne, la
       somme est la sortie nette de cette cellule, qui est zéro pour
       Neumann zero-flux quand on additionne sur toutes les cellules
       voisines)
    3. Coefficients hors-diagonale ≥ 0 (forme M-matrice : positivité
       sous Euler explicite si CFL respecté)
    4. Coefficients diagonaux ≤ 0
    5. Spectre : 0 valeur propre simple (mode constant) + autres
       toutes ≤ 0

    Retourne un dict de diagnostic.
    """
    Ntot = L.shape[0]

    is_symmetric = np.max(np.abs(L - L.T)) < tol
    max_asymmetry = float(np.max(np.abs(L - L.T)))

    row_sums = L.sum(axis=1)
    max_row_sum = float(np.max(np.abs(row_sums)))
    rows_sum_to_zero = max_row_sum < tol

    # Coefficients hors-diagonale
    off_diag = L - np.diag(np.diag(L))
    min_off_diag = float(off_diag.min())
    off_diag_nonneg = min_off_diag >= -tol

    # Coefficients diagonaux
    diag = np.diag(L)
    max_diag = float(diag.max())
    diag_nonpos = max_diag <= tol

    # Spectre (valeurs propres)
    eigenvalues = np.linalg.eigvalsh(L)
    max_eig = float(eigenvalues.max())
    min_eig = float(eigenvalues.min())
    # Devrait avoir une valeur propre = 0 (mode constant)
    has_zero_eig = abs(max_eig) < 1e-10

    return {
        "is_symmetric": bool(is_symmetric),
        "max_asymmetry": max_asymmetry,
        "rows_sum_to_zero": bool(rows_sum_to_zero),
        "max_row_sum": max_row_sum,
        "off_diag_nonneg": bool(off_diag_nonneg),
        "min_off_diag": min_off_diag,
        "diag_nonpos": bool(diag_nonpos),
        "max_diag": max_diag,
        "has_zero_eig": bool(has_zero_eig),
        "max_eigenvalue": max_eig,
        "min_eigenvalue": min_eig,
        "spectral_radius": max(abs(max_eig), abs(min_eig)),
    }
