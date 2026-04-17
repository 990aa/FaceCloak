"""User-facing exceptions for UACloak workflows."""

from __future__ import annotations


class UACloakError(RuntimeError):
    """A readable error that can be surfaced directly in the UI."""
