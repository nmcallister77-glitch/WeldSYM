"""Numba-accelerated explicit conduction step for the 3D thermal solver.

Importing this module requires ``numba`` to be installed. If it is absent,
``solver3d.py`` falls back to the pure NumPy implementation.
"""

from __future__ import annotations

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    # Stubs so the module can be imported even without numba.
    def njit(*_a, **_k):
        def _decorator(f):
            return f

        return _decorator

    def prange(*args):
        return range(*args)


@njit(cache=True, parallel=False, fastmath=False)
def numba_step(
    T,
    T_new,
    T_peak,
    Q,
    dwell_above,
    t_cross_hi,
    t_cross_lo,
    cooling_rate,
    extra_dwell_arrs,
    extra_thresholds,
    nx,
    ny,
    nz,
    dt,
    t,
    k,
    rho,
    cp,
    inv_dx2,
    inv_dy2,
    inv_dz2,
    T0,
    dz,
    convection,
    emissivity,
    sigma_sb,
    use_phase,
    solidus,
    liquidus,
    latent_heat,
    boiling,
    dwell_threshold,
    t85_upper,
    t85_lower,
):
    """One explicit time step with mirrored z-faces and Dirichlet x/y sides."""
    n_extra = extra_dwell_arrs.shape[0]

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for kk in range(nz):
                t0 = T[i, j, kk]

                # x and y neighbours are fixed at T0 on the plate edges.
                tm1x = T[i - 1, j, kk]
                tp1x = T[i + 1, j, kk]
                tm1y = T[i, j - 1, kk]
                tp1y = T[i, j + 1, kk]

                # z neighbours: mirror boundary at top and bottom faces.
                if kk > 0:
                    tm1z = T[i, j, kk - 1]
                else:
                    tm1z = T[i, j, 1]
                if kk < nz - 1:
                    tp1z = T[i, j, kk + 1]
                else:
                    tp1z = T[i, j, nz - 2]

                lap = (
                    (tm1x + tp1x - 2.0 * t0) * inv_dx2
                    + (tm1y + tp1y - 2.0 * t0) * inv_dy2
                    + (tm1z + tp1z - 2.0 * t0) * inv_dz2
                ) * k

                # Surface loss on the half-cell top/bottom control volumes.
                loss = 0.0
                if kk == 0 or kk == nz - 1:
                    flux = convection * (t0 - T0)
                    if emissivity > 0.0:
                        flux += emissivity * sigma_sb * (t0**4 - T0**4)
                    loss = 2.0 * flux / dz

                # Effective specific heat with latent-heat smearing.
                cp_eff = cp
                if use_phase:
                    if latent_heat > 0.0 and t0 > solidus and t0 < liquidus:
                        cp_eff += latent_heat / (liquidus - solidus)
                rho_cp = rho * cp_eff

                t_new = t0 + dt * (lap + Q[i, j, kk] - loss) / rho_cp

                if use_phase and boiling > 0.0 and t_new > boiling:
                    t_new = boiling

                T_new[i, j, kk] = t_new

                # Peak tracking and dwell.
                if t_new > T_peak[i, j, kk]:
                    T_peak[i, j, kk] = t_new

                if t0 > dwell_threshold and t_new > dwell_threshold:
                    dwell_above[i, j, kk] += dt

                for e in range(n_extra):
                    thr = extra_thresholds[e]
                    if t0 > thr and t_new > thr:
                        extra_dwell_arrs[e, i, j, kk] += dt

                # t8/5 crossings.
                if (
                    t0 >= t85_upper
                    and t_new < t85_upper
                    and t_cross_hi[i, j, kk] != t_cross_hi[i, j, kk]
                ):
                    span = t0 - t_new
                    frac = (t0 - t85_upper) / (span + 1e-15)
                    t_cross_hi[i, j, kk] = t + frac * dt
                    cooling_rate[i, j, kk] = span / dt

                if (
                    t0 >= t85_lower
                    and t_new < t85_lower
                    and t_cross_lo[i, j, kk] != t_cross_lo[i, j, kk]
                ):
                    span = t0 - t_new
                    frac = (t0 - t85_lower) / (span + 1e-15)
                    t_cross_lo[i, j, kk] = t + frac * dt
