"""
6d-α minimal engine - étape 1 : diffusion pure conformal-conservative.

Implémentation strictement minimale :
- diffusion volumes finis avec flux conservatifs
- moyenne harmonique de h aux interfaces de cellules
- coefficient de face h^{d-2} = h^1 en 3D (numerics-doc §1.2)
- Neumann zero-flux aux bords (numerics-doc §1.6)
- schéma temporel EXPLICITE Euler (CFL respecté)

Pas de Crank-Nicolson en étape 1 : on garde le schéma le plus simple
possible pour identifier les bugs sans ambiguïté. Crank-Nicolson sera
ajouté seulement si nécessaire (cas g_Ω modulant D fortement).

Pas de drift, pas de bruit, pas de coupling, pas de g_Ω.

Conformité numerics-doc :
- §1.0 dx=1.0, d=3
- §1.2 Laplacien conformal-conservatif J = h^{d-2}·D·∇ψ
- §1.6 Neumann zero-flux
- §1.7 CFL : dt < 0.5·dx²/(2·D·d)
"""

from __future__ import annotations
import numpy as np
from .state import State6dMinimal, N_AXIS, DX, DIM


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Moyenne harmonique aux interfaces : 2·a·b/(a+b).

    Garantit positivité et symétrie de l'opérateur (numerics-doc §1.2).
    Pour h > 0 partout, la moyenne harmonique reste > 0 et minorée
    par min(a,b).
    """
    # éviter division par zéro même si h > 0 par construction
    eps = 1e-30
    return 2.0 * a * b / (a + b + eps)


def compute_diffusion_flux(
    psi: np.ndarray,
    h: np.ndarray,
    D_eff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les flux diffusifs conservatifs aux interfaces.

    J^a_face = -h_face · D · (ψ_{i+1} - ψ_i) / dx

    avec h_face = harmonic_mean(h_i, h_{i+1}).

    Retourne (Jx_face, Jy_face, Jz_face) où :
    - Jx_face shape (N-1, N, N) : flux aux interfaces normales à x
    - Jy_face shape (N, N-1, N) : flux aux interfaces normales à y
    - Jz_face shape (N, N, N-1) : flux aux interfaces normales à z

    Convention de signe : J > 0 = flux vers +axe.
    """
    # Flux à l'interface entre (i, j, k) et (i+1, j, k) (normal à x)
    h_face_x = harmonic_mean(h[:-1, :, :], h[1:, :, :])
    grad_psi_x = (psi[1:, :, :] - psi[:-1, :, :]) / DX
    Jx_face = -h_face_x * D_eff * grad_psi_x  # shape (N-1, N, N)

    h_face_y = harmonic_mean(h[:, :-1, :], h[:, 1:, :])
    grad_psi_y = (psi[:, 1:, :] - psi[:, :-1, :]) / DX
    Jy_face = -h_face_y * D_eff * grad_psi_y  # shape (N, N-1, N)

    h_face_z = harmonic_mean(h[:, :, :-1], h[:, :, 1:])
    grad_psi_z = (psi[:, :, 1:] - psi[:, :, :-1]) / DX
    Jz_face = -h_face_z * D_eff * grad_psi_z  # shape (N, N, N-1)

    return Jx_face, Jy_face, Jz_face


def compute_divergence(
    Jx_face: np.ndarray,
    Jy_face: np.ndarray,
    Jz_face: np.ndarray,
) -> np.ndarray:
    """
    Calcule la divergence discrète à partir des flux aux interfaces.

    Bilan de masse pour la cellule (i,j,k) :
    ∂_t ψ_{ijk} = -(J^x_{i+1/2,j,k} - J^x_{i-1/2,j,k})/dx
                  -(J^y_{i,j+1/2,k} - J^y_{i,j-1/2,k})/dx
                  -(J^z_{i,j,k+1/2} - J^z_{i,j,k-1/2})/dx

    Conditions de bord Neumann zero-flux (numerics-doc §1.6) :
    les flux à travers les frontières externes sont nuls par construction
    (on n'ajoute pas de J_face à l'extérieur de la grille).

    Retourne dpsi_dt shape (N, N, N).
    """
    dpsi_dt = np.zeros((N_AXIS, N_AXIS, N_AXIS), dtype=float)

    # Flux entrants/sortants en x
    # Pour cellule i : flux sortant = J^x_face[i,:,:] (entre i et i+1)
    #                  flux entrant = J^x_face[i-1,:,:] (entre i-1 et i)
    # Cellule 0 : pas de flux entrant (Neumann)
    # Cellule N-1 : pas de flux sortant (Neumann)

    # Contribution sortante (cellules 0 à N-2 ont une face droite)
    dpsi_dt[:-1, :, :] -= Jx_face / DX
    # Contribution entrante (cellules 1 à N-1 ont une face gauche)
    dpsi_dt[1:, :, :] += Jx_face / DX

    # Idem pour y
    dpsi_dt[:, :-1, :] -= Jy_face / DX
    dpsi_dt[:, 1:, :] += Jy_face / DX

    # Idem pour z
    dpsi_dt[:, :, :-1] -= Jz_face / DX
    dpsi_dt[:, :, 1:] += Jz_face / DX

    return dpsi_dt


def cfl_dt_max(h_max: float, D_eff: float) -> float:
    """
    Pas de temps maximal pour stabilité explicite (numerics-doc §1.7).

    CFL diffusion 3D explicite :
        dt < 0.5 · dx² / (2 · D_max · d)

    où D_max = max(h) · D_eff puisque le flux est h·D·∇ψ.
    Facteur 0.5 = marge de sécurité.
    """
    D_max_effectif = h_max * D_eff
    return 0.5 * DX * DX / (2.0 * D_max_effectif * DIM)


def step_diffusion_explicit(
    state: State6dMinimal,
    D_eff: float,
    dt: float,
) -> State6dMinimal:
    """
    Effectue un pas de temps de diffusion pure (Euler explicite).

    Pas de g_Ω, pas de bruit, pas de drift, pas de coupling, pas
    d'évolution de h. C'est l'étape 1 minimale de 6d-α.

    Vérifie CFL avant exécution.

    Retourne un nouvel État (pas de mutation in-place).
    """
    h_max = float(state.h.max())
    dt_cfl = cfl_dt_max(h_max, D_eff)
    if dt > dt_cfl:
        raise ValueError(
            f"dt={dt} viole CFL (dt_max={dt_cfl:.4e} pour h_max={h_max}, D={D_eff})"
        )

    Jx, Jy, Jz = compute_diffusion_flux(state.psi, state.h, D_eff)
    dpsi_dt = compute_divergence(Jx, Jy, Jz)

    psi_new = state.psi + dt * dpsi_dt

    return State6dMinimal(
        psi=psi_new,
        h=state.h.copy(),  # h ne change pas (pas de 𝔊^sed/𝔊^ero)
        h0=state.h0,
        h_min=state.h_min,
    )


def simulate(
    state_init: State6dMinimal,
    D_eff: float,
    n_steps: int,
    dt: float | None = None,
    log_every: int = 1,
) -> tuple[State6dMinimal, dict]:
    """
    Simule n_steps pas de diffusion pure.

    Logue à chaque step (ou tous les log_every steps) :
    - total_mass (conservation, doit rester 1.0)
    - min_psi, max_psi (positivité)
    - min_h, max_h
    - variance par axe et totale
    - center of mass

    Retourne (état final, dict de logs).
    """
    if dt is None:
        dt = cfl_dt_max(state_init.h.max(), D_eff)
        # Marge supplémentaire
        dt = dt * 0.9

    logs = {
        "step": [],
        "t": [],
        "total_mass": [],
        "min_psi": [],
        "max_psi": [],
        "min_h": [],
        "max_h": [],
        "var_x": [],
        "var_y": [],
        "var_z": [],
        "var_total": [],
        "com_x": [],
        "com_y": [],
        "com_z": [],
        "dt_used": dt,
        "D_eff": D_eff,
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
        logs["var_total"].append(vx + vy + vz)
        logs["com_x"].append(cx)
        logs["com_y"].append(cy)
        logs["com_z"].append(cz)

    state = state_init
    log_state(0, 0.0, state)

    for step in range(1, n_steps + 1):
        state = step_diffusion_explicit(state, D_eff, dt)
        if step % log_every == 0:
            log_state(step, step * dt, state)

    return state, logs
