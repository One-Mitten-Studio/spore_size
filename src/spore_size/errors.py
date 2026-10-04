"""Exceptions raised when a photo is unsuitable for measurement."""

from __future__ import annotations


class PhotoRejected(RuntimeError):
    """The photo is not a usable spore micrograph (reported as SKIPPED, not FAILED)."""
