"""User-facing exceptions for FaceCloak workflows."""

from __future__ import annotations


class FaceCloakError(RuntimeError):
    """A readable error that can be surfaced directly in the UI."""
