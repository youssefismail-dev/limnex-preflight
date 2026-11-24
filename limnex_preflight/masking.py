"""Masking helpers. The scanner never stores or prints a full secret or personal value."""

from __future__ import annotations

import re

_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def clean(text: str, limit: int = 80) -> str:
    """Make arbitrary text safe for one-line display."""
    text = _CTRL.sub("?", text)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def mask_secret(value: str, prefix_len: int = 0) -> str:
    """Keep a public provider prefix plus at most two leading and two trailing secret characters.

    ``hf_`` + 34 chars  ->  ``hf_ab...9Z``. Values of 8 characters or fewer are fully hidden.
    """
    value = clean(value, 4096)
    n = len(value)
    if n <= 8:
        return "*" * n
    budget = n // 3  # never reveal more than a third of the value
    tail = min(2, budget // 2)
    head = min(prefix_len + 2, budget - tail)
    return f"{value[:head]}...{value[-tail:]}" if tail else f"{value[:head]}..."


def mask_word(value: str, _prefix: int = 0) -> str:
    """Mask a short personal token (user name): first character visible, rest hidden."""
    value = clean(value, 64)
    if len(value) <= 1:
        return "*"
    return value[0] + "*" * min(len(value) - 1, 8)


def mask_email(value: str, _prefix: int = 0) -> str:
    local, _, domain = value.rpartition("@")
    if "." in domain:
        host, _, tld = domain.rpartition(".")
        domain_m = (host[:1] + "***" if host else "***") + "." + tld
    else:
        domain_m = domain[:1] + "***"
    return f"{local[:1]}***@{domain_m}"


def mask_ip(value: str, _prefix: int = 0) -> str:
    parts = value.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.*.*"
    return mask_secret(value)


def mask_host(value: str, _prefix: int = 0) -> str:
    labels = value.split(".")
    if len(labels) == 1:
        return mask_word(value)
    return mask_word(labels[0]) + "." + ".".join(labels[1:])
