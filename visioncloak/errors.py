"""User-facing exceptions for VisionCloak workflows."""

from __future__ import annotations


class VisionCloakError(RuntimeError):
    """Base exception for recoverable VisionCloak errors."""


# Backward-compatible alias for legacy imports.
UACloakError = VisionCloakError

