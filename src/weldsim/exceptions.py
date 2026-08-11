"""Exception hierarchy for Weld Sim."""

from __future__ import annotations


class WeldSimError(Exception):
    """Base class for all errors raised by weldsim."""


class ConfigurationError(WeldSimError, ValueError):
    """Invalid simulation input (bad geometry, material or process parameters)."""


class SolverError(WeldSimError, RuntimeError):
    """The solver could not produce a usable solution."""


class OutputError(WeldSimError, OSError):
    """Simulation results could not be written to disk."""


class InputDataError(WeldSimError, ValueError):
    """Result data read back from disk is missing, malformed or inconsistent."""
