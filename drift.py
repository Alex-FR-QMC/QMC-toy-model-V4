"""
6d-α drift pur — schéma Fokker-Planck conservatif avec convention
"descente dans le potentiel" (Option 1, validation Alex).

Convention de signe (Option 1) :

    flux advectif Fokker-Planck : J_drift = -h^{d-2} · ψ · ∇Φ
    ∂_t ψ = -∇·J_drift = +∇·(h^{d-2}·ψ·∇Φ)

Pour Φ = ½k‖θ-θ_0‖² (puits centré à θ_0), ∇Φ pointe vers l'extérieur
du centre, donc J_drift pointe vers l'intérieur → la masse descend vers
θ_0. C'est le comportement physique attendu pour un drift attractif.

Schéma upwind :
- À l'interface (i, i+1) normale à x : flux signé en fonction de
  ∇Φ aux interfaces.
- Si J_drift_x < 0 (flux vers la gauche), source = ψ_{i+1} :
  upwind = ψ_{i+1}.
- Si J_drift_x > 0 (flux vers la droite), source = ψ_i :
  upwind = ψ_i.

CFL drift : dt < dx / max(|h · ∇Φ|).

Boundary conditions : Neumann zero-flux (consistant avec diffusion).
Aux faces de bord, J_drift = 0 par construction (pas d'ajout de flux
hors grille). Pour Φ centré dans la grille (puits attractif), pas de
problème ; pour Φ décentré qui pousserait vers l'extérieur, masse
peut s'accumuler au bord.
"""

from __future__ import annotations
import numpy as np
from .state import State6dMinimal, N_AXIS, DX
from .engine import harmonic_mean


def compute_grad_Phi(Phi: np.ndarray, dx: float = DX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule le gradient de Φ aux interfaces de cellules par différence
    centrée vers l'avant.

    grad_Phi_x[i,j,k] = (Phi[i+1,j,k] - Phi[i,j,k]) / dx
    Shape : (N-1, N, N)

    Idem pour y, z.
    """
    grad_x = (Phi[1:, :, :] - Phi[:-1, :, :]) / dx  # (N-1, N, N)
    grad_y = (Phi[:, 1:, :] - Phi[:, :-1, :]) / dx  # (N, N-1, N)
    grad_z = (Phi[:, :, 1:] - Phi[:, :, :-1]) / dx  # (N, N, N-1)
    return grad_x, grad_y, grad_z


def compute_drift_flux(
    psi: np.ndarray,
    h: np.ndarray,
    Phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les flux d'advection Fokker-Planck aux interfaces avec
    convention Option 1 : J = -h_face · ψ_upwind · ∇Φ.

    Schéma upwind :
    - À l'interface entre i et i+1 (normale à x), grad_Phi_x positif
      signifie Φ_{i+1} > Φ_i, donc J = -h·ψ_upwind·grad_Phi_x < 0
      (flux vers la gauche, source = ψ_{i+1}). Donc upwind = ψ_{i+1}.
    - grad_Phi_x négatif : flux positif (vers la droite),
      source = ψ_i, upwind = ψ_i.

    Retourne (Jx_face, Jy_face, Jz_face) signés.
    """
    grad_x, grad_y, grad_z = compute_grad_Phi(Phi)

    h_face_x = harmonic_mean(h[:-1, :, :], h[1:, :, :])
    h_face_y = harmonic_mean(h[:, :-1, :], h[:, 1:, :])
    h_face_z = harmonic_mean(h[:, :, :-1], h[:, :, 1:])

    # Upwind ψ selon le sens du flux
    # J négatif (grad_Phi positif) → upwind = ψ_droite (i+1)
    # J positif (grad_Phi négatif) → upwind = ψ_gauche (i)
    psi_upwind_x = np.where(grad_x >= 0.0, psi[1:, :, :], psi[:-1, :, :])
    psi_upwind_y = np.where(grad_y >= 0.0, psi[:, 1:, :], psi[:, :-1, :])
    psi_upwind_z = np.where(grad_z >= 0.0, psi[:, :, 1:], psi[:, :, :-1])

    Jx = -h_face_x * psi_upwind_x * grad_x
    Jy = -h_face_y * psi_upwind_y * grad_y
    Jz = -h_face_z * psi_upwind_z * grad_z

    return Jx, Jy, Jz


def divergence_drift(
    Jx_face: np.ndarray,
    Jy_face: np.ndarray,
    Jz_face: np.ndarray,
) -> np.ndarray:
    """
    Divergence discrète des flux drift (identique en structure à la
    divergence diffusion, parce que le bilan de masse est le même
    quel que soit le flux).

    ∂_t ψ_{ijk} = -(J_{i+1/2} - J_{i-1/2})/dx + (idem y, z)
    """
    dpsi_dt = np.zeros((N_AXIS, N_AXIS, N_AXIS), dtype=float)

    dpsi_dt[:-1, :, :] -= Jx_face / DX
    dpsi_dt[1:, :, :] += Jx_face / DX

    dpsi_dt[:, :-1, :] -= Jy_face / DX
    dpsi_dt[:, 1:, :] += Jy_face / DX

    dpsi_dt[:, :, :-1] -= Jz_face / DX
    dpsi_dt[:, :, 1:] += Jz_face / DX

    return dpsi_dt


def cfl_dt_drift(h_max: float, grad_Phi_max: float) -> float:
    """
    CFL pour drift upwind explicite : dt < dx / max(|h · ∇Φ|).
    Facteur 0.5 marge.
    """
    if grad_Phi_max <= 0.0:
        return float("inf")
    return 0.5 * DX / (h_max * grad_Phi_max)


def step_drift_explicit(
    state: State6dMinimal,
    Phi: np.ndarray,
    dt: float,
) -> State6dMinimal:
    """
    Step drift pur (Euler explicite). Φ fixe, pas de diffusion.
    Pas de coupling, pas de bruit, pas de sédimentation.
    """
    Jx, Jy, Jz = compute_drift_flux(state.psi, state.h, Phi)
    dpsi_dt = divergence_drift(Jx, Jy, Jz)
    psi_new = state.psi + dt * dpsi_dt

    return State6dMinimal(
        psi=psi_new,
        h=state.h.copy(),
        h0=state.h0,
        h_min=state.h_min,
    )


def simulate_drift(
    state_init: State6dMinimal,
    Phi: np.ndarray,
    n_steps: int,
    dt: float,
    log_every: int = 1,
) -> tuple[State6dMinimal, dict]:
    """
    Simule n_steps pas de drift pur dans le potentiel Φ.
    """
    logs = {
        "step": [], "t": [],
        "total_mass": [], "min_psi": [], "max_psi": [],
        "min_h": [], "max_h": [],
        "var_x": [], "var_y": [], "var_z": [],
        "com_x": [], "com_y": [], "com_z": [],
        "dt_used": dt,
        "n_steps": n_steps,
    }

    def log_state(step: int, t: float, s: State6dMinimal) -> None:
        vx, vy, vz = s.variance_per_axis()
        cx, cy, cz = s.center_of_mass()
        logs["step"].append(step)
        logs["t"].append(t)
        logs["total_mass"].append(s.total_mass())
        logs["min_psi"].append(s.min_psi())
        logs["max_psi"].append(s.max_psi())
        logs["min_h"].append(s.min_h())
        logs["max_h"].append(s.max_h())
        logs["var_x"].append(vx)
        logs["var_y"].append(vy)
        logs["var_z"].append(vz)
        logs["com_x"].append(cx)
        logs["com_y"].append(cy)
        logs["com_z"].append(cz)

    state = state_init
    log_state(0, 0.0, state)

    for step in range(1, n_steps + 1):
        state = step_drift_explicit(state, Phi, dt)
        if step % log_every == 0:
            log_state(step, step * dt, state)

    return state, logs


# Utilitaires pour potentiels
def make_quadratic_potential(
    k: float,
    center: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> np.ndarray:
    """
    Φ(θ) = ½·k·‖θ - θ_0‖²

    Retourne Φ shape (N, N, N).
    """
    N = N_AXIS
    cx, cy, cz = center
    coords = np.arange(N) * DX
    Phi = np.zeros((N, N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            for k_idx in range(N):
                r2 = (coords[i] - cx) ** 2 + (coords[j] - cy) ** 2 + (coords[k_idx] - cz) ** 2
                Phi[i, j, k_idx] = 0.5 * k * r2
    return Phi


def grad_Phi_max(Phi: np.ndarray, dx: float = DX) -> float:
    """Max |∇Φ| aux interfaces de la grille."""
    gx, gy, gz = compute_grad_Phi(Phi, dx)
    return float(max(np.max(np.abs(gx)), np.max(np.abs(gy)), np.max(np.abs(gz))))
