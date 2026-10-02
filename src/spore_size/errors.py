"""Exceptions raised when a photo is unsuitable for measurement."""


class PhotoRejected(RuntimeError):
    """The photo is not a usable spore micrograph (reported as SKIPPED, not FAILED)."""
