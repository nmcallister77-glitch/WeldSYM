"""Fresnel optics and geometric keyhole absorption estimates."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

_ELEMENTARY_CHARGE = 1.602176634e-19
_VACUUM_PERMITTIVITY = 8.8541878188e-12
_ELECTRON_MASS = 9.1093837139e-31
_SPEED_OF_LIGHT = 299792458.0


def drude_refractive_index(
    wavelength: float, electron_density: float, resistivity: float
) -> tuple[float, float]:
    """Return the Drude-model complex refractive index ``(n, k)``."""
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if electron_density < 0.0 or resistivity < 0.0:
        raise ValueError("electron_density and resistivity must be non-negative")

    plasma_frequency_squared = (
        electron_density * _ELEMENTARY_CHARGE**2 / (_VACUUM_PERMITTIVITY * _ELECTRON_MASS)
    )
    angular_frequency = 2.0 * math.pi * _SPEED_OF_LIGHT / wavelength
    collision_frequency = plasma_frequency_squared * _VACUUM_PERMITTIVITY * resistivity
    denominator = angular_frequency**2 + collision_frequency**2
    epsilon_real = 1.0 - plasma_frequency_squared / denominator
    epsilon_imag = collision_frequency / angular_frequency * plasma_frequency_squared / denominator
    epsilon_magnitude = math.hypot(epsilon_real, epsilon_imag)
    n = math.sqrt(max(0.0, (epsilon_magnitude + epsilon_real) / 2.0))
    k = math.sqrt(max(0.0, (epsilon_magnitude - epsilon_real) / 2.0))
    return n, k


def fresnel_absorptivity(cos_theta: float, n: float, k: float) -> float:
    """Return unpolarised Fresnel absorptivity from vacuum into ``n + i k``."""
    if n < 0.0 or k < 0.0:
        raise ValueError("refractive-index components must be non-negative")

    cosine = min(1.0, max(0.0, float(cos_theta)))
    if cosine == 0.0:
        return 0.0

    sine_squared = 1.0 - cosine**2
    index = complex(n, k)
    index_squared = index**2
    transmitted_normal = cmath.sqrt(index_squared - sine_squared)

    rs = (cosine - transmitted_normal) / (cosine + transmitted_normal)
    rp = (index_squared * cosine - transmitted_normal) / (
        index_squared * cosine + transmitted_normal
    )
    reflectance = (abs(rs) ** 2 + abs(rp) ** 2) / 2.0
    return min(1.0, max(0.0, 1.0 - reflectance))


@dataclass(frozen=True)
class KeyholeAbsorptance:
    """Beam-weighted absorption from a flat surface and conical keyhole trace."""

    absorptance: float
    flat_absorptivity: float
    mean_reflections: float


def _cone_intersection(
    position: tuple[float, float, float],
    direction: tuple[float, float, float],
    mouth_radius: float,
    depth: float,
    minimum_distance: float,
) -> float | None:
    """Return the next positive ray distance to the conical wall, if any."""
    x, y, z = position
    dx, dy, dz = direction
    slope = mouth_radius / depth
    local_radius = mouth_radius * (1.0 - z / depth)
    a = dx**2 + dy**2 - (slope * dz) ** 2
    b = 2.0 * (x * dx + y * dy + slope * local_radius * dz)
    c = x**2 + y**2 - local_radius**2

    if abs(a) < 1e-14:
        roots = () if abs(b) < 1e-30 else (-c / b,)
    else:
        discriminant = b**2 - 4.0 * a * c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        roots = ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))

    for distance in sorted(roots):
        hit_depth = z + dz * distance
        if distance > minimum_distance and 0.0 < hit_depth < depth:
            return distance
    return None


def keyhole_absorptance(
    n: float,
    k: float,
    depth: float,
    mouth_radius: float,
    beam_radius: float | None = None,
    n_radial: int = 40,
    n_angular: int = 72,
    max_reflections: int = 50,
) -> KeyholeAbsorptance:
    """Trace stratified Gaussian rays through an axisymmetric conical keyhole.

    Ray power is integrated over ``0 <= r <= 2*beam_radius`` and normalised
    over the sampled beam. The reported mean reflection count is weighted by
    incident ray power over the whole beam.
    """
    flat = fresnel_absorptivity(1.0, n, k)
    if depth <= 0.0:
        return KeyholeAbsorptance(flat, flat, 0.0)
    if mouth_radius <= 0.0:
        raise ValueError("mouth_radius must be positive")
    if beam_radius is None:
        beam_radius = mouth_radius
    if beam_radius <= 0.0:
        raise ValueError("beam_radius must be positive")
    if n_radial <= 0 or n_angular <= 0:
        raise ValueError("ray counts must be positive")
    if max_reflections < 0:
        raise ValueError("max_reflections must be non-negative")

    radial_step = 2.0 * beam_radius / n_radial
    ring_weights = []
    ring_radii = []
    for ring in range(n_radial):
        radius = (ring + 0.5) * radial_step
        ring_radii.append(radius)
        ring_weights.append(
            math.exp(-2.0 * radius**2 / beam_radius**2) * radius * radial_step * 2.0 * math.pi
        )
    total_weight = math.fsum(ring_weights)
    angular_step = 2.0 * math.pi / n_angular
    absorbed = 0.0
    weighted_reflections = 0.0
    scale = max(depth, mouth_radius, beam_radius, 1e-12)
    minimum_distance = 1e-9 * scale

    for ring, radius in enumerate(ring_radii):
        ray_weight = ring_weights[ring] / total_weight / n_angular
        for angular_index in range(n_angular):
            angle = (angular_index + 0.5) * angular_step
            if radius >= mouth_radius:
                absorbed += ray_weight * flat
                continue

            position = (radius * math.cos(angle), radius * math.sin(angle), 0.0)
            direction = (0.0, 0.0, 1.0)
            remaining_power = ray_weight
            reflections = 0

            for _ in range(max_reflections):
                if position[2] <= 0.0 and direction[2] < 0.0:
                    break
                distance = _cone_intersection(
                    position, direction, mouth_radius, depth, minimum_distance
                )
                if direction[2] < 0.0:
                    exit_distance = -position[2] / direction[2]
                    if distance is None or exit_distance <= distance:
                        break
                if distance is None:
                    break

                hit = tuple(position[axis] + distance * direction[axis] for axis in range(3))
                hit_radius = math.hypot(hit[0], hit[1])
                if hit_radius <= minimum_distance:
                    break
                radial_normal = (hit[0] / hit_radius, hit[1] / hit_radius)
                normal_scale = 1.0 / math.sqrt(1.0 + (mouth_radius / depth) ** 2)
                normal = (
                    radial_normal[0] * normal_scale,
                    radial_normal[1] * normal_scale,
                    (mouth_radius / depth) * normal_scale,
                )
                incidence_cosine = abs(
                    direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2]
                )
                local_absorptivity = fresnel_absorptivity(incidence_cosine, n, k)
                absorbed += remaining_power * local_absorptivity
                remaining_power *= 1.0 - local_absorptivity
                reflections += 1
                if remaining_power < 1e-6:
                    break

                reflected = tuple(
                    direction[axis]
                    - 2.0
                    * (
                        direction[0] * normal[0]
                        + direction[1] * normal[1]
                        + direction[2] * normal[2]
                    )
                    * normal[axis]
                    for axis in range(3)
                )
                position = tuple(
                    hit[axis] + minimum_distance * reflected[axis] for axis in range(3)
                )
                direction = reflected

            weighted_reflections += ray_weight * reflections

    return KeyholeAbsorptance(
        absorptance=min(1.0, max(0.0, absorbed)),
        flat_absorptivity=flat,
        mean_reflections=weighted_reflections,
    )


__all__ = [
    "KeyholeAbsorptance",
    "drude_refractive_index",
    "fresnel_absorptivity",
    "keyhole_absorptance",
]
