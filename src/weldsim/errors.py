"""Exception types raised by Weld Sim."""

from __future__ import annotations


class WeldSimError(Exception):
    """Base class for all Weld Sim errors."""


class ValidationError(WeldSimError, ValueError):
    """A simulation input is missing, out of range or physically meaningless."""


class StabilityError(WeldSimError, ValueError):
    """The requested grid and time step violate the explicit-scheme CFL limit."""


class AbortError(WeldSimError):
    """The user cancelled the simulation before it finished."""


__all__ = ["WeldSimError", "ValidationError", "StabilityError", "AbortError"]
