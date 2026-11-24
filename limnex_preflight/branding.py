"""Every name, URL and file name that identifies this tool, in ONE place.

Code reads the names from here; the places that cannot (pyproject.toml, action.yml, workflows, docs) carry them as text.
"""

TOOL_NAME = "limnex-preflight"  # CLI command, SARIF tool name, log prefix
DISPLAY_NAME = "Limnex Preflight"
REPO_URL = "https://github.com/youssefismail-dev/limnex-preflight"
IGNORE_FILE_NAME = ".preflightignore"  # fingerprint ignore file
SARIF_FINGERPRINT_KEY = "limnexFingerprint/v1"
ACTION_ENV_PREFIX = "LIMNEX_PREFLIGHT_"  # environment variables that carry the GitHub Action inputs
