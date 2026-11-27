"""Limnex Preflight: static scanner for secrets, personal data and unsafe pickles in ML files."""

from .models import Finding, Severity
from .scanner import ScanOptions, ScanResult, scan_paths

__version__ = "0.1.1"

__all__ = ["Finding", "Severity", "ScanOptions", "ScanResult", "scan_paths", "__version__"]
