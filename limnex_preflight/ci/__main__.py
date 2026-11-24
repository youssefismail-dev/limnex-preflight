import sys

from .action import main

try:   # a safety net: what the action prints holds no lone surrogate (sanitize.no_surrogates), but a crash must not hide
    sys.stdout.reconfigure(errors="backslashreplace")
except (AttributeError, ValueError):
    pass
sys.exit(main())
