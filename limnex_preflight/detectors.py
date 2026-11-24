"""Text detectors: the rule catalogue that every format handler feeds strings into.

Every rule is a compiled regex plus optional validation. Nothing here touches the network or
executes scanned content. Thresholds that matter for false positives are module constants so
they are documented in one place (see README, "Detection rules").
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Pattern, Tuple, Union

from .masking import clean, mask_email, mask_host, mask_ip, mask_secret, mask_word
from .models import CATEGORY_PII, CATEGORY_SCAN, CATEGORY_SECRET, CATEGORY_UNSAFE, Severity

# --------------------------------------------------------------------------------------
# Documented thresholds
# --------------------------------------------------------------------------------------
ENTROPY_MIN_LEN = 24  # standalone high-entropy candidates: 24..128 characters
ENTROPY_MAX_LEN = 128
ENTROPY_BASE64_THRESHOLD = 4.5  # bits per character, base64/base64url alphabet
ENTROPY_HEX_THRESHOLD = 3.0  # bits per character, pure-hex strings (max possible is 4.0)
CRED_VALUE_MIN_LEN = 8  # value of a secret-looking key (token, secret, api_key ...)
CRED_VALUE_MIN_ENTROPY = 3.0
PASSWORD_VALUE_MIN_LEN = 6  # value of a password-looking key
PASSWORD_VALUE_MIN_ENTROPY = 2.0
MIN_SCAN_LEN = 6  # strings shorter than this cannot hold anything we look for
HASH_HEX_LENGTHS = frozenset({32, 40, 56, 64, 96, 128})  # md5, sha1, sha224/256/384/512

Check = Callable[[str, "re.Match[str]", str], Union[bool, Severity]]
Masker = Callable[[str, int], str]


def shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    severity: Severity
    title: str
    tier: str  # provider | generic | pii | entropy | key | structural
    pattern: Optional[Pattern[str]] = None
    group: int = 0
    prefix_len: int = 0
    hints: Tuple[str, ...] = ()
    ci: bool = False  # hints are matched against the lower-cased text
    check: Optional[Check] = None
    masker: Masker = mask_secret


@dataclass
class Hit:
    rule: Rule
    value: str
    start: int
    end: int
    masked: str
    severity: Severity


# --------------------------------------------------------------------------------------
# Placeholder handling (used by generic and PII rules only; provider patterns are specific enough)
# --------------------------------------------------------------------------------------
_PLACEHOLDER_WORDS = {
    "", "none", "null", "nil", "true", "false", "changeme", "change_me", "change-me", "password",
    "passwd", "secret", "token", "apikey", "api_key", "api-key", "your_token", "your-token",
    "your_api_key", "your-api-key", "yourpassword", "your_password", "string", "str", "redacted",
    "undefined", "empty", "default", "admin", "test", "todo", "tbd", "n/a", "na",
}
_PLACEHOLDER_RE = re.compile(
    r"""^(?:
        x{3,}|\*{3,}|\.{3,}|-{3,}|_{3,}|0+ |
        [<{\[$%(].*|          # <TOKEN>, {token}, $TOKEN, %(token)s
        .*(?:your|example|placeholder|redacted|dummy|sample|fake|insert|replace)[-_a-z]*.*
    )$""",
    re.I | re.X,
)


def is_placeholder(value: str) -> bool:
    v = value.strip().strip("'\"`")
    if v.lower() in _PLACEHOLDER_WORDS:
        return True
    return bool(_PLACEHOLDER_RE.match(v))


# --------------------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------------------
def _check_jwt(value: str, m, text: str) -> bool:
    header = value.split(".", 1)[0]
    try:
        raw = base64.urlsafe_b64decode(header + "=" * (-len(header) % 4))
        obj = json.loads(raw)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return False
    return isinstance(obj, dict) and "alg" in obj


_URL_PW_PLACEHOLDERS = {"password", "pass", "passwd", "pwd", "secret", "token", "xxxx", "xxx", "****", "***"}


def _check_url_password(value: str, m, text: str) -> bool:
    pw = value
    if pw.lower() in _URL_PW_PLACEHOLDERS:
        return False
    if re.fullmatch(r"[<{\[$%(].*", pw) or re.fullmatch(r"\*+|x+", pw, re.I):
        return False
    return True


def _check_stripe(value: str, m, text: str) -> Union[bool, Severity]:
    return Severity.MEDIUM if "_test_" in value else True


def _check_azure_sas(value: str, m, text: str) -> bool:
    return True


def _is_word_like(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+([-_ ][A-Za-z]+)*", value))


_CODE_REF = re.compile(r"""[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\)?|\[)?[)\]}\\"',;]*""")
# a call whose first argument is a string literal: re.compile(r"...", load_pem(b"...") (the value stops at the quote)
_CALL_WITH_LITERAL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\([rRbBuUfF]{0,2}")
# a value that starts with a call left open, make_key(getattr(a, ... (the value stops at the space): code, not a value
_CALL_START = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\(")


# Between a credential's name and its value: "KEY = v", "key: v", and the two ways Python code sets an environment
# variable, os.environ['KEY'] = 'v' (a quote and a bracket before the "=") and os.environ.setdefault("KEY", "v") /
# os.putenv("KEY", "v") (a quoted name, a comma, a quoted value). The user review before launch found every rule blind to
# the last two, the most common way a notebook sets WANDB_API_KEY or KAGGLE_KEY.
_ASSIGN = r"""(?:["']?\]?\s*[:=]\s*|["']\s*,\s*)["']?"""


def _names_a_file(key: str) -> bool:
    """'scripts/mint_api_key.py' (a manifest of file hashes) names a file, not a variable."""
    return "/" in key or "\\" in key or ("." in key and key.rsplit(".", 1)[1].lower() in _FILE_EXT_TLDS)


def _check_generic_cred(value: str, m, text: str) -> bool:
    key_raw = m.groupdict().get("key") or ""
    if _names_a_file(key_raw):
        return False
    key = key_raw.lower()
    v = value.strip().strip("'\"")
    is_pw = any(p in key for p in ("passw", "pwd"))
    min_len = PASSWORD_VALUE_MIN_LEN if is_pw else CRED_VALUE_MIN_LEN
    min_ent = PASSWORD_VALUE_MIN_ENTROPY if is_pw else CRED_VALUE_MIN_ENTROPY
    if len(v) < min_len or is_placeholder(v):
        return False
    if v.lower() in ("true", "false", "none", "null") or re.fullmatch(r"-?\d+(\.\d+)?", v):
        return False
    if re.search(r"[\s<>{}]", v) or v.startswith(("/", "./", "~/")) or "://" in v:
        return False
    # a reference in code, not a value: os.environ[ ... , self.api_key, config.get( ...
    if _CODE_REF.fullmatch(v) and any(c in v for c in ".(["):
        return False
    if _CALL_WITH_LITERAL.fullmatch(v) or (_CALL_START.match(v) and v.count("(") > v.count(")")):
        return False
    if not is_pw and _is_word_like(v):  # "tokenizer-name", "access_token_type" and friends
        return False
    return shannon_entropy(v) >= min_ent


# --------------------------------------------------------------------------------------
# Regex helpers
# --------------------------------------------------------------------------------------
def _rx(pattern: str, flags: int = 0) -> Pattern[str]:
    return re.compile(pattern, flags)


_NB = r"(?<![A-Za-z0-9_])"  # not preceded by a word char
_NA = r"(?![A-Za-z0-9_])"  # not followed by a word char
_NB2 = r"(?<![A-Za-z0-9])"
_NA2 = r"(?![A-Za-z0-9])"


def _secret(
    rid: str,
    title: str,
    pattern: str,
    severity: Severity = Severity.HIGH,
    *,
    group: int = 1,
    prefix_len: int = 0,
    hints: Tuple[str, ...] = (),
    ci: bool = False,
    flags: int = 0,
    check: Optional[Check] = None,
    tier: str = "provider",
) -> Rule:
    return Rule(rid, CATEGORY_SECRET, severity, title, tier, _rx(pattern, flags), group, prefix_len, hints, ci, check)


PROVIDER_RULES: List[Rule] = [
    # --- cloud providers -------------------------------------------------------------
    _secret("aws-access-key-id", "AWS access key ID",
            _NB2 + r"((?:AKIA|ASIA|ABIA|ACCA|A3T[A-Z0-9])[A-Z0-9]{16})" + _NA2,
            prefix_len=4, hints=("AKIA", "ASIA", "ABIA", "ACCA", "A3T")),
    _secret("aws-secret-access-key", "AWS secret access key (assigned to an aws_secret... name)",
            r"aws[_\-. ]?secret[_\-. ]?(?:access[_\-. ]?)?key[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})" + r"(?![A-Za-z0-9/+=])",
            hints=("aws",), ci=True, flags=re.I),
    _secret("aws-session-token", "AWS session token (assigned to an aws_session_token name)",
            r"aws[_\-. ]?session[_\-. ]?token[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{100,})",
            hints=("aws",), ci=True, flags=re.I),
    _secret("gcp-api-key", "Google API key", r"(?<![A-Za-z0-9_\-])(AIza[0-9A-Za-z_\-]{35})(?![0-9A-Za-z_\-])",
            prefix_len=4, hints=("AIza",)),
    _secret("gcp-oauth-client-secret", "Google OAuth client secret", r"(GOCSPX-[A-Za-z0-9_\-]{28})(?![A-Za-z0-9_\-])",
            prefix_len=7, hints=("GOCSPX-",)),
    _secret("gcp-service-account-key-id", "Google service-account private_key_id",
            r"[\"']private_key_id[\"']\s*:\s*[\"']([0-9a-f]{40})[\"']", hints=("private_key_id",)),
    _secret("azure-storage-account-key", "Azure storage AccountKey",
            r"AccountKey\s*=\s*([A-Za-z0-9+/]{86}==)", hints=("accountkey",), ci=True, flags=re.I),
    _secret("azure-shared-access-key", "Azure SharedAccessKey",
            r"SharedAccessKey\s*=\s*([A-Za-z0-9+/]{43}=)", hints=("sharedaccesskey",), ci=True, flags=re.I),
    _secret("azure-sas-token", "Azure shared-access-signature token",
            r"\bsv=\d{4}-\d{2}-\d{2}&[^\s\"']{0,400}?\bsig=([A-Za-z0-9%+/=]{40,})",
            hints=("sig=",), ci=True, flags=re.I),
    _secret("azure-ad-client-secret", "Azure AD client secret",
            r"(?<![A-Za-z0-9_~.\-])([A-Za-z0-9_~.\-]{3}8Q~[A-Za-z0-9_~.\-]{31,34})(?![A-Za-z0-9_~.\-])",
            prefix_len=0, hints=("8Q~",)),
    # --- source hosts ----------------------------------------------------------------
    _secret("github-token", "GitHub token", _NB + r"(gh[pousr]_[A-Za-z0-9]{36,255})" + _NA,
            prefix_len=4, hints=("ghp_", "gho_", "ghu_", "ghs_", "ghr_")),
    _secret("github-fine-grained-pat", "GitHub fine-grained personal access token",
            _NB + r"(github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59})" + _NA, prefix_len=11, hints=("github_pat_",)),
    _secret("gitlab-token", "GitLab token",
            r"(?<![A-Za-z0-9_\-])(gl(?:pat|ptt|dt|rt|cbt|imt|soat|ft|agent|oas)-[A-Za-z0-9_\-]{20,})(?![A-Za-z0-9_\-])",
            prefix_len=6, hints=("glpat-", "glptt-", "gldt-", "glrt-", "glcbt-", "glimt-", "glsoat-", "glft-", "glagent-", "gloas-")),
    # --- ML / AI providers -----------------------------------------------------------
    _secret("huggingface-token", "Hugging Face access token", _NB + r"(hf_[A-Za-z0-9]{34,40})" + _NA,
            prefix_len=3, hints=("hf_",)),
    _secret("huggingface-org-token", "Hugging Face organization API token", _NB + r"(api_org_[A-Za-z0-9]{34})" + _NA,
            prefix_len=8, hints=("api_org_",)),
    _secret("openai-api-key", "OpenAI API key",
            r"(?<![A-Za-z0-9_\-])(sk-(?:(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{40,}|[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}|[A-Za-z0-9]{48}))(?![A-Za-z0-9_\-])",
            prefix_len=3, hints=("sk-",)),
    _secret("anthropic-api-key", "Anthropic API key",
            r"(?<![A-Za-z0-9_\-])(sk-ant-[A-Za-z0-9]{2,12}-[A-Za-z0-9_\-]{40,})(?![A-Za-z0-9_\-])",
            prefix_len=13, hints=("sk-ant-",)),
    _secret("kaggle-api-key", "Kaggle API key (kaggle.json style or KAGGLE_KEY)",
            r"(?:KAGGLE_KEY" + _ASSIGN + r"|[\"']key[\"']\s*:\s*[\"'])([0-9a-f]{32})(?![0-9a-zA-Z])",
            hints=("kaggle_key", '"key"', "'key'"), ci=True, flags=re.I,
            check=lambda v, m, t: ("kaggle_key" in m.group(0).lower()) or bool(re.search(r"kaggle|\"username\"", t, re.I))),
    _secret("kaggle-api-token", "Kaggle API token (KGAT_)", _NB + r"(KGAT_[0-9a-f]{32})" + _NA,
            prefix_len=5, hints=("KGAT_",)),
    _secret("wandb-api-key", "Weights & Biases API key",
            r"(?:wandb[_\-. ]?api[_\-. ]?key" + _ASSIGN
            + r"|wandb\.login\([^)\n]{0,80}?\bkey\s*=\s*[\"']"           # wandb.login(key="...")
            + r"|machine\s+api\.wandb\.ai\s+(?:login\s+\S+\s+)?password\s+"  # the entry wandb login writes to ~/.netrc
            + r")([0-9a-f]{40})(?![0-9a-zA-Z])",
            hints=("wandb",), ci=True, flags=re.I),
    _secret("replicate-api-token", "Replicate API token", _NB + r"(r8_[A-Za-z0-9]{37})" + _NA, prefix_len=3, hints=("r8_",)),
    _secret("groq-api-key", "Groq API key", _NB + r"(gsk_[A-Za-z0-9]{52})" + _NA, prefix_len=4, hints=("gsk_",)),
    _secret("databricks-token", "Databricks personal access token", _NB + r"(dapi[0-9a-f]{32})" + _NA,
            prefix_len=4, hints=("dapi",)),
    # --- messaging / payments / misc SaaS --------------------------------------------
    _secret("slack-token", "Slack token", r"(?<![A-Za-z0-9\-])(xox[abposr]-[A-Za-z0-9\-]{10,72}|xapp-\d-[A-Z0-9]+-\d+-[a-z0-9]+)",
            prefix_len=5, hints=("xox", "xapp-")),
    _secret("slack-webhook", "Slack incoming-webhook URL",
            r"(https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{20,})",
            prefix_len=33, hints=("hooks.slack.com",)),
    _secret("stripe-secret-key", "Stripe secret/restricted API key",
            _NB + r"((?:sk|rk)_(?:live|test)_[A-Za-z0-9]{24,99})" + _NA, prefix_len=8, hints=("sk_live_", "sk_test_", "rk_live_", "rk_test_"),
            check=_check_stripe),
    _secret("stripe-webhook-secret", "Stripe webhook signing secret", _NB + r"(whsec_[A-Za-z0-9]{32,})" + _NA,
            prefix_len=6, hints=("whsec_",)),
    _secret("twilio-api-key", "Twilio API key SID", r"(?<![A-Za-z0-9])(SK[0-9a-fA-F]{32})(?![A-Za-z0-9])", prefix_len=2, hints=("SK",),
            check=lambda v, m, t: bool(re.search(r"[0-9]", v)) and bool(re.search(r"[a-fA-F]", v[2:]))),
    _secret("twilio-auth-token", "Twilio auth token (assigned to a twilio... name)",
            r"twilio[^\n]{0,40}?token[\"']?\s*[:=]\s*[\"']?([0-9a-f]{32})(?![0-9a-zA-Z])", hints=("twilio",), ci=True, flags=re.I),
    _secret("sendgrid-api-key", "SendGrid API key", _NB + r"(SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43})" + _NA, prefix_len=3, hints=("SG.",)),
    _secret("npm-token", "npm access token", _NB + r"(npm_[A-Za-z0-9]{36})" + _NA, prefix_len=4, hints=("npm_",)),
    _secret("pypi-token", "PyPI API token", r"(pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,})", prefix_len=5, hints=("pypi-",)),
    _secret("digitalocean-token", "DigitalOcean token", _NB + r"(dop_v1_[a-f0-9]{64})" + _NA, prefix_len=7, hints=("dop_v1_",)),
    _secret("telegram-bot-token", "Telegram bot token", r"(?<![0-9A-Za-z])(\d{8,10}:AA[A-Za-z0-9_\-]{33})(?![A-Za-z0-9_\-])", prefix_len=0, hints=(":AA",)),
    # --- structural secrets ----------------------------------------------------------
    Rule("private-key-pem", CATEGORY_SECRET, Severity.CRITICAL, "PEM/OpenSSH/PGP private key block", "provider",
         _rx(r"-----BEGIN (?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED|PGP) )?PRIVATE KEY(?: BLOCK)?-----"), 0, 0,
         ("PRIVATE KEY",), False, lambda v, m, t: bool(_PEM_BODY.match(t, m.end())),
         lambda v, p: clean(v, 60) + " [body redacted]"),
    _secret("jwt", "JSON Web Token",
            r"(?<![A-Za-z0-9_\-])(eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})(?![A-Za-z0-9_\-])",
            prefix_len=3, hints=("eyJ",), check=_check_jwt),
    _secret("password-in-url", "Password embedded in a URL",
            r"\b[a-zA-Z][a-zA-Z0-9+.\-]{1,15}://[^\s:/@\"'<>]{1,64}:([^\s@/\"'<>]{3,128})@[^\s/\"'<>]+",
            prefix_len=0, hints=("://",), check=_check_url_password),
    _secret("bearer-token-header", "Authorization header with a credential",
            r"authorization[\"']?\s*[:=]\s*[\"']?(?:bearer|basic|token)\s+([A-Za-z0-9._~+/=\-]{16,})",
            hints=("authorization",), ci=True, flags=re.I,
            check=lambda v, m, t: not is_placeholder(v) and shannon_entropy(v) >= 3.0),
]

# A key block's body follows its header: whitespace, escaped newlines or JSON-list punctuation, optional RFC 1421 / PGP
# header lines, then a line of 40+ base64 characters. A header alone (a string compared in code, a sentence about PEM
# files, a MIME table) is not a key (pre-release test 5, T8: CRITICAL on such text; the masked value, which ends in the
# header, then matched again and read "[body redacted] [body redacted]").
_PEM_SEP = r"(?:\s|\\[nr]|[\"',\[\]])*"
_PEM_BODY = re.compile(_PEM_SEP + r"(?:(?:Proc-Type|DEK-Info|Comment|Version|Hash|Charset):[^\n\\]*" + _PEM_SEP + r")*"
                       r"[A-Za-z0-9+/=]{40}")

# Generic: credential-looking assignments. Medium severity, documented value checks.
_CRED_KEY = (
    r"(?P<key>[A-Za-z0-9_.\-]*(?:password|passwd|pwd|secret|api[_\-]?key|apikey|access[_\-]?key|"
    r"auth[_\-]?token|access[_\-]?token|private[_\-]?key|client[_\-]?secret|credential)[A-Za-z0-9_.\-]*)"
)
GENERIC_RULES: List[Rule] = [
    Rule("generic-credential", CATEGORY_SECRET, Severity.MEDIUM,
         "Value assigned to a password/secret/token/api-key style name", "generic",
         _rx(r"(?<![A-Za-z0-9])" + _CRED_KEY + _ASSIGN + r"(?P<val>[^\s\"',;&]{4,200})", re.I),
         0, 0, ("passw", "pwd", "secret", "key", "token", "credential"), True, _check_generic_cred),
    Rule("cli-secret-argument", CATEGORY_SECRET, Severity.MEDIUM,
         "Secret passed as a command-line argument (saved command line / argv)", "generic",
         _rx(r"(?<![A-Za-z0-9])--?[a-z0-9\-_]*?(?:password|passwd|secret|token|api[-_]?key|apikey|access[-_]?key)(?![a-z0-9])(?:=|[ \t]+)[\"']?(?P<val>[^\s\"']{8,200})", re.I),
         0, 0, ("password", "passwd", "secret", "token", "key"), True,
         lambda v, m, t: _check_generic_cred(v, m, t)),
]

# For these two generic rules the value is a named group ``val``; the raw match starts at the key.

ENTROPY_RULE = Rule("high-entropy-string", CATEGORY_SECRET, Severity.LOW,
                    "High-entropy string that may be a secret", "entropy", None, 0, 0, (), False, None)

# --------------------------------------------------------------------------------------
# Personal-information rules
# --------------------------------------------------------------------------------------
GENERIC_USERNAMES = {
    "public", "default", "default user", "all users", "shared", "guest", "user", "users", "username", "usr",
    "ubuntu", "runner", "runneradmin", "docker", "vscode", "jovyan", "ec2-user", "ubuntu-user", "root", "admin",
    "administrator", "colab", "me", "name", "yourname", "your_name", "your-name", "you", "someone", "foo", "bar",
    "home", "app", "ray", "circleci", "travis", "jenkins", "builder", "build", "nobody", "azureuser", "opc",
    "centos", "debian", "fedora", "ci", "git", "gitlab-runner", "worker", "workspace", "dev", "developer",
    "test", "tester", "myuser", "my_user", "my-user", "mnt", "sagemaker-user", "studio-lab-user", "paperspace",
    "ec2_user", "ubuntu_user", "unknown", "xxx", "xxxx", "vagrant", "pi", "oracle", "hadoop", "spark", "airflow",
    "cloudera", "ml", "model", "models", "data", "code", "src", "tmp", "temp",
}


def _bad_username(name: str) -> bool:
    n = name.strip().lower()
    return n in GENERIC_USERNAMES or n.startswith(("<", "{", "$", "%", "~")) or n.endswith((">", "}")) or n.startswith("$")


_USER = r"[^\\/\s\"'<>|:*?,;(){}\[\]$%`]{1,64}"


def _check_username(value: str, m, text: str) -> bool:
    return not _bad_username(value)


def _check_platform_path(value: str, m, text: str) -> Union[bool, Severity]:
    return True


PII_RULES: List[Rule] = [
    Rule("username-in-path", CATEGORY_PII, Severity.MEDIUM, "User name inside a Windows home path (C:\\Users\\<name>)", "pii",
         _rx(r"(?:[A-Za-z]:|/mnt/[a-z]|/cygdrive/[a-z])[\\/]+Users[\\/]+(" + _USER + r")", re.I), 1, 0,
         ("users",), True, _check_username, mask_word),
    Rule("username-in-path", CATEGORY_PII, Severity.MEDIUM, "User name inside a Linux home path (/home/<name>)", "pii",
         _rx(r"(?<![A-Za-z0-9_.\-])/home/([A-Za-z0-9][A-Za-z0-9._\-]{0,63})(?![A-Za-z0-9._\-])"), 1, 0,
         ("/home/",), False, _check_username, mask_word),
    Rule("username-in-path", CATEGORY_PII, Severity.MEDIUM, "User name inside a macOS home path (/Users/<name>)", "pii",
         _rx(r"(?<![A-Za-z0-9_.\-])/Users/([A-Za-z0-9][A-Za-z0-9._\-]{0,63})(?![A-Za-z0-9._\-])"), 1, 0,
         ("/Users/",), False, _check_username, mask_word),
    Rule("username-in-path", CATEGORY_PII, Severity.MEDIUM, "Owner name inside a Kaggle models/datasets/notebooks path", "pii",
         _rx(r"(?<![A-Za-z0-9_.\-])/kaggle/input/(?:datasets|models|notebooks|kernels)/([A-Za-z0-9][A-Za-z0-9._\-]{0,63})(?=/)"),
         1, 0, ("/kaggle/input/",), False, _check_username, mask_word),
    Rule("platform-path", CATEGORY_PII, Severity.MEDIUM,
         "Path under a hosted-notebook mount (/kaggle/..., /content/drive/...) naming a dataset, notebook or folder", "pii",
         _rx(r"(?<![A-Za-z0-9_.\-])(/kaggle/(?:input|usr/lib|usr/src)/[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\- ]+)*|/content/drive/(?:My ?Drive|MyDrive|Shareddrives)/[^\s\"'<>|*?]+)"),
         1, 0, ("/kaggle/", "/content/drive/"), False,
         lambda v, m, t: not re.match(r"/kaggle/input/(?:datasets|models|notebooks|kernels)(?:/|$)", v), mask_secret),
    Rule("platform-path-generic", CATEGORY_PII, Severity.LOW,
         "Standard hosted-notebook directory (the Kaggle working directory, the Colab Drive root): reveals where the file was produced", "pii",
         _rx(r"(?<![A-Za-z0-9_.\-])(/kaggle/(?:working|temp|tmp|lib|input|config|usr)(?:/[A-Za-z0-9._\-]+)?|/content/drive/(?:My ?Drive|MyDrive))(?![A-Za-z0-9._\-/])"),
         1, 0, ("/kaggle/", "/content/drive"), False, None, lambda v, p: clean(v, 60)),
]

_EMAIL_PLACEHOLDER_DOMAINS = {
    "example.com", "example.org", "example.net", "example.edu", "test.com", "domain.com", "email.com", "yourdomain.com",
    "mydomain.com", "company.com", "sample.com", "mail.com", "foo.com", "bar.com", "site.com", "website.com",
    "localhost." + "localdomain", "your.domain", "domain.tld", "acme.com", "host.com",
}
_EMAIL_NOT_PERSONAL_LOCAL = {
    "noreply", "no-reply", "donotreply", "do-not-reply", "git", "ssh", "hg", "user", "username", "name", "you", "me",
    "foo", "bar", "someone", "email", "mail", "your.email", "youremail", "your_email", "john.doe", "jane.doe",
    "johndoe", "janedoe", "first.last", "firstname.lastname", "root", "postmaster", "webmaster", "hostmaster", "abuse",
    "x", "a", "test", "example", "name.surname", "yourname", "your-name", "email_address",
}
_EMAIL_ROLE_LOCAL = {
    "info", "admin", "support", "contact", "hello", "sales", "team", "help", "security", "press", "office",
    "privacy", "legal", "licensing", "licence", "license", "enquiries", "inquiries", "hr", "jobs", "careers",
}
_FILE_EXT_TLDS = {
    "png", "jpg", "jpeg", "gif", "svg", "webp", "bmp", "tif", "tiff", "ico", "json", "yaml", "yml", "txt", "md", "py",
    "js", "ts", "html", "css", "csv", "tsv", "pkl", "pt", "pth", "bin", "onnx", "h5", "so", "dll", "whl", "zip", "gz",
    "tar", "xml", "cfg", "ini", "toml", "log", "npy", "npz", "ckpt", "pb", "tflite", "mp3", "mp4", "wav", "pdf", "exe",
    "c", "h", "cc", "cpp", "hpp", "rs", "go", "java", "rb", "sh", "bat", "ipynb", "lock", "safetensors", "gguf",
}


_VCS_BLOCK = re.compile(r"Author:\s*[^<\n]{0,80}<[^>\n]+>[\s\S]{0,400}?(?:Summary:|commit_hash:|Date:)")


def _in_vcs_metadata(value: str, m, text: str) -> bool:
    """Libraries such as CatBoost embed a `git log`-style block (Author: Name <mail> ... commit_hash:) in every file
    they write. That is the library maintainer's identity, not the publisher's."""
    start = max(0, m.start(1) - 120)
    return bool(_VCS_BLOCK.search(text[start: m.end(1) + 500])) and f"<{value}>" in text[start: m.end(1) + 5]


_URL_AFTER_HOST = re.compile(r"/|:\d")
_TOKEN_START = " \t\r\n\"'`<>(=,;["


def _is_url_authority(m, text: str) -> bool:
    """user@host inside a URL is its user-info and host, not an e-mail address: it follows '//'
    (http://trusted.com@evil.com), or it begins a scheme-less URL whose host is followed by a path or a port
    (abc12@bank-login.com/verify). An address inside a URL path (/users/jane@mail.com/settings) is still one."""
    start, end = m.start(1), m.end(1)
    if text[max(0, start - 2): start] == "//":
        return True
    begins_token = start == 0 or text[start - 1] in _TOKEN_START
    return begins_token and bool(_URL_AFTER_HOST.match(text, end))


def _check_email(value: str, m, text: str) -> Union[bool, Severity]:
    if _in_vcs_metadata(value, m, text) or _is_url_authority(m, text):
        return False
    local, _, domain = value.rpartition("@")
    d = domain.lower()
    tld = d.rpartition(".")[2]
    if tld in _FILE_EXT_TLDS or d in _EMAIL_PLACEHOLDER_DOMAINS:
        return False
    if d.endswith(".example") or d.endswith(".invalid") or d.endswith(".test"):
        return False
    ll = local.lower()
    if ll in _EMAIL_NOT_PERSONAL_LOCAL:
        return False
    if re.fullmatch(r"[0-9]+x?", ll) or ll.startswith("@"):
        return False
    if d.endswith("noreply.github.com"):
        return True  # real identity of a GitHub account
    if ll in _EMAIL_ROLE_LOCAL:
        return Severity.LOW
    return True


_IP_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
# Version words must start a word: "ver" at the end of "server"/"driver" and "v" at the end of "dev" hid private addresses,
# and so did the "@" of an ssh user@address line (pre-release test 3, R3). "@" is host context, not version context.
_VERSION_CTX = re.compile(r"(?:\b(?:version|ver|v|rev|build|release|opset)|==|>=|<=|~=|~|\^)\s*[=:]?\s*$", re.I)


_HOSTISH_CTX = re.compile(
    r"(?:(?:\b(?:ip|host|hostname|server|addr|address|node|endpoint|remote|master|worker|proxy|gateway|url|ssh)\w*[\"']?\s*[=:@]?\s*[\"']?)"
    r"|://|@)$", re.I)


def _is_private_ipv4(parts) -> bool:
    a, b = parts[0], parts[1]
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168) or (a == 100 and 64 <= b <= 127)


def _check_ipv4(value: str, m, text: str) -> Union[bool, Severity]:
    parts = [int(p) for p in value.split(".")]
    a, b = parts[0], parts[1]
    if a in (0, 127) or a >= 224:
        return False  # unspecified, loopback, multicast/reserved/broadcast
    if (a, b) == (169, 254):
        return False  # link-local, includes the well-known cloud metadata address
    if (a, b, parts[2]) in ((192, 0, 2), (198, 51, 100), (203, 0, 113)):
        return False  # RFC 5737 documentation ranges can never be real
    ctx = text[max(0, m.start(1) - 12): m.start(1)]
    if _VERSION_CTX.search(ctx):
        return False
    if _is_private_ipv4(parts):
        return True  # internal addressing always reveals something about the author's network
    # public addresses in prose are overwhelmingly CDNs, DNS servers and version numbers: require host-like context
    return bool(_HOSTISH_CTX.search(text[max(0, m.start(1) - 30): m.start(1)]))


def _mask_ip_value(v: str, p: int) -> str:
    return mask_ip(v)


_HOST_KEY = r"(?:host|hostname|host_name|node|node_name|nodename|machine|machine_name|computer_name|computername|server|server_name)"

HOST_RULES: List[Rule] = [
    Rule("email-address", CATEGORY_PII, Severity.MEDIUM, "E-mail address", "pii",
         _rx(r"(?<![A-Za-z0-9._%+\-])([A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}@[A-Za-z0-9][A-Za-z0-9.\-]{0,252}\.[A-Za-z]{2,24})(?![A-Za-z0-9\-])"),
         1, 0, ("@",), False, _check_email, mask_email),
    Rule("ip-address", CATEGORY_PII, Severity.LOW, "IPv4 address", "pii",
         _rx(r"(?<![\d.])(" + _IP_OCT + r"(?:\." + _IP_OCT + r"){3})(?![\d]|\.\d)"),
         1, 0, (".",), False, _check_ipv4, _mask_ip_value),
    Rule("ip-address", CATEGORY_PII, Severity.LOW, "IPv6 address (full form)", "pii",
         _rx(r"(?<![0-9A-Fa-f:])((?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4})(?![0-9A-Fa-f:])"),
         1, 0, (":",), False,
         lambda v, m, t: not v.lower().startswith(("fe80", "0:0:0:0:0:0:0:1", "0000:0000:0000:0000:0000:0000:0000:0001")),
         lambda v, p: clean(v[:9]) + ":*:*:*"),
    Rule("hostname", CATEGORY_PII, Severity.LOW, "Internal or machine host name (.internal/.corp/.lan/.local, cloud VM names)", "pii",
         _rx(r"(?<![A-Za-z0-9.\-_])((?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+(?:internal|corp|lan|intranet|localdomain|local)|"
             r"ec2-\d+-\d+-\d+-\d+\.(?:[a-z0-9\-]+\.)?compute(?:-\d)?\.amazonaws\.com|ip-\d+-\d+-\d+-\d+(?:\.[a-z0-9.\-]+)?|"
             r"[a-z0-9\-]+\.cloudapp\.azure\.com)(?![A-Za-z0-9\-_])", re.I),
         1, 0, (".internal", ".corp", ".lan", ".intranet", ".localdomain", ".local", "ec2-", "ip-", ".cloudapp."), True,
         lambda v, m, t: not re.fullmatch(r"(?:[a-z_]+\.)+local", v) or "-" in v or v.count(".") >= 2 or v[0].isupper(),
         mask_host),
    # "/" or a Windows "\" (written "\\" in JSON text). Retest of release candidate 8: a person's name in a Windows machine
    # name was missed in out\runs\<date>_<host>.
    Rule("hostname", CATEGORY_PII, Severity.LOW, "Machine name inside a Trainer/TensorBoard run directory (runs/<date>_<hostname>)", "pii",
         _rx(r"\bruns(?:/|\\{1,2})[A-Z][a-z]{2}\d{2}_\d{2}-\d{2}-\d{2}_([A-Za-z0-9][A-Za-z0-9.\-]{0,62})(?![A-Za-z0-9.\-])"),
         1, 0, ("runs/", "runs\\"), False, lambda v, m, t: v.lower() not in {"localhost", "host", "hostname"}, mask_host),
    Rule("hostname", CATEGORY_PII, Severity.LOW, "URL pointing at a single-label internal host", "pii",
         _rx(r"\b(?:https?|ftp|sftp|ssh|rsync|redis|postgres(?:ql)?|mysql|mongodb|amqp)://(?:[^/@\s\"']+@)?([A-Za-z0-9][A-Za-z0-9\-_]{1,62})(?::\d+)?(?=[/\s\"'?#]|$)", re.I),
         1, 0, ("://",), False,
         lambda v, m, t: v.lower() not in {"localhost", "host", "hostname", "server", "example", "domain", "yourhost", "your-host", "ip", "address", "x", "xxx", "mlflow", "tensorboard"} and not v.lower().startswith(("host", "your")),
         mask_host),
]

# Key-context rules (the key name of a JSON/HDF5/ONNX-metadata entry decides)
_USERNAME_KEYS = re.compile(r"^(?:.*[_\-.])?(?:username|user_name|userid|user_id|login|kaggle_username|hf_username|github_username)$", re.I)
# `pkg.torch.onnx.*` are the PyTorch exporter's own graph labels (original_node_name = a node of the graph, not a machine)
_HOSTNAME_KEYS = re.compile(r"^(?!pkg\.torch\.onnx\.)(?:.*[_\-.])?" + _HOST_KEY + r"$", re.I)
USERNAME_FIELD_RULE = Rule("username-field", CATEGORY_PII, Severity.MEDIUM, "Value of a user-name style field", "key", None, 0, 0, (), False, None, mask_word)
HOSTNAME_FIELD_RULE = Rule("hostname-field", CATEGORY_PII, Severity.LOW, "Value of a host-name style field", "key", None, 0, 0, (), False, None, mask_host)
CRED_FIELD_RULE = Rule("generic-credential", CATEGORY_SECRET, Severity.MEDIUM,
                       "Value assigned to a password/secret/token/api-key style name", "generic", None, 0, 0, (), False, None)

_CRED_KEY_RE = re.compile(
    r"(?:^|[_\-. ])(?:password|passwd|pwd|secret|api[_\-]?key|apikey|access[_\-]?key|auth[_\-]?token|access[_\-]?token|"
    r"private[_\-]?key|client[_\-]?secret|credentials?|token)$|^(?:.*[_\-. ])?(?:password|passwd|secret|api[_\-]?key|apikey|"
    r"access[_\-]?key|auth[_\-]?token|access[_\-]?token|private[_\-]?key|client[_\-]?secret)(?:[_\-. ].*)?$",
    re.I,
)
# Tokenizer vocabulary entries are called "*_token" everywhere; they are not credentials.
_TOKENIZER_KEY_RE = re.compile(
    r"(?:^|[_\-. ])(?:bos|eos|pad|unk|cls|sep|mask|start|end|additional_special|special|sentinel|decoder_start|"
    r"prefix|suffix|middle|fim|image|audio|video|eot|eom|bot|boi|eoi|mask_token|cls_token|pad_token|unk_token|"
    r"sep_token|bos_token|eos_token|max|num|n|vocab|max_new|min|new|added|stop|unused)[_\-. ]*"
    r"(?:tokens?|token_id|token_ids)$|(?:token|tokens)_(?:type|id|ids|count|length|embedding|embeddings|to_|per|num)",
    re.I,
)

ALL_REGEX_RULES: List[Rule] = PROVIDER_RULES + GENERIC_RULES + PII_RULES + HOST_RULES

# Non-regex rules (pickle analysis, scan diagnostics) registered here for --list-rules and SARIF.
EXTRA_RULES: List[Rule] = [
    ENTROPY_RULE,
    USERNAME_FIELD_RULE,
    HOSTNAME_FIELD_RULE,
    Rule("kaggle-credentials-json", CATEGORY_SECRET, Severity.HIGH, "kaggle.json style object (username + 32-hex key)", "structural"),
    Rule("unsafe-pickle-exec", CATEGORY_UNSAFE, Severity.CRITICAL,
         "Pickle imports a callable that runs code or commands (os, subprocess, builtins.eval/exec, ...)", "structural"),
    Rule("unsafe-pickle-io", CATEGORY_UNSAFE, Severity.HIGH,
         "Pickle imports a callable for network, file-system or indirect attribute access", "structural"),
    Rule("unsafe-pickle-nested", CATEGORY_UNSAFE, Severity.HIGH,
         "Pickle calls a loader (torch.load, pickle.loads, ...) on data that could not be analysed", "structural"),
    Rule("pickle-getattr-review", CATEGORY_UNSAFE, Severity.MEDIUM,
         "Pickle uses getattr with an attribute name that is neither known-benign nor known-dangerous (needs a human look)", "structural"),
    Rule("pickle-computed-callable", CATEGORY_UNSAFE, Severity.MEDIUM,
         "Pickle calls an object produced by another call (a resolver or factory), which a static scan cannot identify (needs a human look)", "structural"),
    Rule("unsafe-keras-lambda", CATEGORY_UNSAFE, Severity.HIGH,
         "Keras Lambda layer: serialised Python code that runs when the model is loaded without safe mode", "structural"),
    Rule("pickle-unresolved-global", CATEGORY_UNSAFE, Severity.MEDIUM,
         "Pickle imports a global whose name could not be determined statically", "structural"),
    Rule("pickle-incomplete-analysis", CATEGORY_UNSAFE, Severity.MEDIUM,
         "Pickle stream could not be analysed to its end; the remainder is unchecked", "structural"),
    Rule("pickle-trailing-data", CATEGORY_UNSAFE, Severity.LOW,
         "Unexpected data after the end of a pickle stream", "structural"),
    Rule("unscannable-model-archive", CATEGORY_UNSAFE, Severity.MEDIUM,
         "PyTorch-style archive that the scanner could not read; its pickle was not checked and PyTorch may still load it", "structural"),
    Rule("onnx-external-data-path", CATEGORY_UNSAFE, Severity.MEDIUM,
         "ONNX external data points outside the model's folder (absolute path, drive, home directory, URL or '..')", "structural"),
    Rule("unsafe-yaml-python-tag", CATEGORY_UNSAFE, Severity.MEDIUM,
         "YAML type tag that makes Python's unsafe YAML loaders build or import Python objects", "structural"),
    Rule("scan-error", CATEGORY_SCAN, Severity.INFO, "File could not be fully scanned", "structural"),
    # user review before launch: actions/checkout does not fetch Git LFS files by default, so a CI check of a repository
    # that keeps its weights in LFS read only these small text pointers and passed
    Rule("git-lfs-pointer", CATEGORY_SCAN, Severity.MEDIUM,
         "Git LFS pointer, not the file: the real file was not downloaded, so it was not checked (actions/checkout: "
         "lfs: true; locally: git lfs pull)", "structural"),
    Rule("archive-path-traversal", CATEGORY_UNSAFE, Severity.MEDIUM,
         "Archive member name leaves the extraction folder ('..', an absolute path or a drive): unpacking it can "
         "write outside the target folder", "structural"),
]

RULES_BY_ID: Dict[str, Rule] = {}
for _r in ALL_REGEX_RULES + EXTRA_RULES + [CRED_FIELD_RULE]:
    RULES_BY_ID.setdefault(_r.id, _r)

# Several rules share an id (one per platform/address family); the catalogue shows one general title.
_GENERAL_TITLES = {
    "username-in-path": "User name inside a home-directory path (C:\\Users\\<name>, /home/<name>, /Users/<name>, Kaggle input owner folders)",
    "ip-address": "IP address (IPv4, or IPv6 in full form)",
    "hostname": "Internal or machine host name",
    "generic-credential": "Value assigned to a password/secret/token/api-key style name",
}
for _rid, _title in _GENERAL_TITLES.items():
    RULES_BY_ID[_rid] = replace(RULES_BY_ID[_rid], title=_title)


# --------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------
_ENT_RE = re.compile(r"(?<![A-Za-z0-9+/_\-])[A-Za-z0-9+/_\-]{%d,%d}={0,2}(?![A-Za-z0-9+/_=\-])" % (ENTROPY_MIN_LEN, ENTROPY_MAX_LEN))
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HEX_RE = re.compile(r"[0-9a-fA-F]+")

# Strings carved out of binary files (float bytes!) can look like "x@y.Kq". There, an e-mail must end in a real TLD.
KNOWN_TLDS = frozenset(
    "com org net edu gov mil int io ai co app dev info biz me tv xyz online site tech cloud store blog page link live "
    "news email team space shop world today network systems solutions digital media group company services studio "
    "ac ad ae af ag al am ao ar as at au aw az ba bb bd be bf bg bh bi bj bm bn bo br bs bt bw by bz ca cd cf cg ch "
    "ci ck cl cm cn cr cu cv cy cz de dj dk dm do dz ec ee eg er es et eu fi fj fm fo fr ga gd ge gf gg gh gi gl gm "
    "gn gp gq gr gt gu gw gy hk hn hr ht hu id ie il im in iq ir is it je jm jo jp ke kg kh ki km kn kp kr kw ky kz "
    "la lb lc li lk lr ls lt lu lv ly ma mc md mg mh mk ml mm mn mo mp mq mr ms mt mu mv mw mx my mz na nc ne nf ng "
    "ni nl no np nr nu nz om pa pe pf pg ph pk pl pm pn pr ps pt pw py qa re ro rs ru rw sa sb sc sd se sg sh si sk "
    "sl sm sn so sr ss st su sv sx sy sz tc td tg th tj tk tl tm tn to tr tt tw tz ua ug uk us uy uz va vc ve vg vi "
    "vn vu wf ws ye yt za zm zw".split()
)

MODE_FULL = "full"  # everything
MODE_SECRETS = "secrets"  # provider-specific secrets and key blocks only (opaque binary data)
MODE_BINARY = "binary"  # strings carved from binary files (same detectors as MODE_FULL)


_SEP = re.compile(r"[-_/.+]")
_WORD_PLUS_DIGITS = re.compile(r"[A-Za-z]+\d+|\d+[A-Za-z]+")
_URLISH_WORD = re.compile(r"^(?:www\.|/|[\w.\-]+\.[a-z]{2,6}/)", re.I)


def _wordlike(core: str) -> bool:
    """Identifiers, paths and repo ids: at least 30% of the characters sit in alphabetic tokens of 3+ letters (random base64 almost never does)."""
    alpha = sum(len(t) for t in _SEP.split(core) if len(t) >= 3 and t.isalpha())
    return alpha * 10 >= len(core) * 3


def _in_url_or_path(text: str, start: int, end: int) -> bool:
    ws = start
    lo = max(0, start - 300)
    while ws > lo and not text[ws - 1].isspace() and text[ws - 1] not in "\"'`<>(":
        ws -= 1
    word = text[ws:end]
    return "://" in word or bool(_URLISH_WORD.match(word))


def _entropy_candidates(text: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for m in _ENT_RE.finditer(text):
        s = m.group(0)
        core = s.rstrip("=")
        if _WORD_PLUS_DIGITS.fullmatch(core) or _wordlike(core) or _in_url_or_path(text, m.start(), m.end()):
            continue
        if _UUID_RE.fullmatch(core):
            continue
        if "/" in core and ("//" in core or core.startswith("/") or core.count("/") >= 2 and not re.search(r"\d", core)):
            continue
        has_digit = any(c.isdigit() for c in core)
        has_alpha = any(c.isalpha() for c in core)
        if not (has_digit and has_alpha):
            continue
        if _HEX_RE.fullmatch(core):
            if len(core) in HASH_HEX_LENGTHS:
                continue
            if shannon_entropy(core) < ENTROPY_HEX_THRESHOLD:
                continue
        else:
            if shannon_entropy(core) < ENTROPY_BASE64_THRESHOLD:
                continue
            # camelCase / snake_case identifiers with a few digits are not secrets
            if re.fullmatch(r"[A-Za-z]+(?:[_\-][A-Za-z0-9]+)+", core) and shannon_entropy(core) < 4.8:
                continue
        out.append((core, m.start(), m.start() + len(core)))
    return out


#: At most this many matches of one rule are reported per text. Retest of release candidate 8: the limit was silent (a
#: CSV with 400 e-mail addresses showed 200 and said nothing); Engine.scan now tells the caller which rules reached it.
MAX_HITS_PER_RULE = 200


class Engine:
    """Runs the catalogue over one string. Stateless and reusable."""

    def scan(self, text: str, key: Optional[str] = None, mode: str = MODE_FULL,
             capped: Optional[set] = None) -> List[Hit]:
        """The hits in `text`. The ids of rules with more than MAX_HITS_PER_RULE matches (the rest not reported) are added
        to `capped` when it is given."""
        if len(text) < MIN_SCAN_LEN:
            return []
        hits: List[Hit] = []
        low = text.lower()
        for rule in PROVIDER_RULES:
            hits.extend(self._run(rule, text, low, capped))
        if mode != MODE_SECRETS:
            for rule in GENERIC_RULES + PII_RULES + HOST_RULES:
                hits.extend(self._run(rule, text, low, capped))
            if key:
                hits.extend(self._key_hits(key, text))
            if len(text) >= ENTROPY_MIN_LEN:  # also in carved binary strings: 24+ base64 chars in a row never occur by chance
                hits.extend(self._entropy(text))
        if mode == MODE_BINARY:
            hits = [h for h in hits if not (h.rule.id == "email-address" and h.value.rpartition(".")[2] not in KNOWN_TLDS)]
        return self._resolve(hits)

    # -- helpers -------------------------------------------------------------------
    _BIG = 100_000  # above this, regexes only run in windows around their hints
    _WIN_BEFORE = 120
    _WIN_AFTER = 5000
    _MAX_HINT_HITS = 5000

    def _spans(self, rule: Rule, text: str, low: str):
        """Windows (start, end) around every occurrence of one of the rule's hints; None = scan everything."""
        hay = low if rule.ci else text
        spans = []
        n = len(text)
        count = 0
        for h in rule.hints:
            h = h.lower() if rule.ci else h
            i = hay.find(h)
            while i >= 0:
                count += 1
                if count > self._MAX_HINT_HITS:
                    return None
                spans.append((max(0, i - self._WIN_BEFORE), min(n, i + len(h) + self._WIN_AFTER)))
                i = hay.find(h, i + len(h))
        if not spans:
            return []
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return merged

    def _run(self, rule: Rule, text: str, low: str, capped: Optional[set] = None) -> List[Hit]:
        pat = rule.pattern
        if pat is None:
            return []
        if rule.hints and len(text) > self._BIG:
            spans = self._spans(rule, text, low)
            if spans is not None:
                out: List[Hit] = []
                for i, (a, b) in enumerate(spans):
                    out.extend(self._matches(rule, pat.finditer(text, a, b), text, capped))
                    if len(out) >= MAX_HITS_PER_RULE:
                        if i + 1 < len(spans) and capped is not None:
                            capped.add(rule.id)
                        break
                return out
        elif rule.hints:
            hay = low if rule.ci else text
            if not any(h.lower() in hay if rule.ci else h in hay for h in rule.hints):
                return []
        return self._matches(rule, pat.finditer(text), text, capped)

    def _matches(self, rule: Rule, matches, text: str, capped: Optional[set] = None) -> List[Hit]:
        out: List[Hit] = []
        for m in matches:
            if rule.tier == "generic":
                value = m.group("val")
                start, end = m.start("val"), m.end("val")
            else:
                value = m.group(rule.group)
                start, end = m.start(rule.group), m.end(rule.group)
            sev = rule.severity
            if rule.check is not None:
                res = rule.check(value, m, text)
                if res is False:
                    continue
                if isinstance(res, Severity):
                    sev = res
            if len(out) >= MAX_HITS_PER_RULE:          # a real match past the limit: not reported, but said
                if capped is not None:
                    capped.add(rule.id)
                break
            if rule.id == "private-key-pem":
                # the fingerprint must tell two keys apart yet survive JSON/YAML escaping of the newlines
                tail = re.sub(r"\\n|\\r|\s", "", text[m.end(): m.end() + 400])[:128]
                value = value + tail
                masked = rule.masker(m.group(0), 0)
            else:
                masked = rule.masker(value, rule.prefix_len)
            out.append(Hit(rule, value, start, end, masked, sev))
        return out

    def _key_hits(self, key: str, text: str) -> List[Hit]:
        out: List[Hit] = []
        v = text.strip()
        if _USERNAME_KEYS.match(key):
            if v and not _bad_username(v) and not is_placeholder(v) and len(v) <= 64 and not re.search(r"\s{2,}", v):
                out.append(Hit(USERNAME_FIELD_RULE, v, 0, len(text), mask_word(v), USERNAME_FIELD_RULE.severity))
        elif _HOSTNAME_KEYS.match(key):
            if (v and len(v) <= 253 and not is_placeholder(v) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-_]*", v)
                    and v.lower() not in {"localhost", "0.0.0.0", "127.0.0.1", "::1", "any", "all", "auto", "default"}
                    and not v.replace(".", "").isdigit()):
                out.append(Hit(HOSTNAME_FIELD_RULE, v, 0, len(text), mask_host(v), HOSTNAME_FIELD_RULE.severity))
        if _CRED_KEY_RE.search(key) and not _TOKENIZER_KEY_RE.search(key):
            fake = re.match(r"(?P<key>.*)", key)
            if fake is not None and _check_generic_cred(v, fake, text):
                out.append(Hit(CRED_FIELD_RULE, v, 0, len(text), mask_secret(v), Severity.MEDIUM))
        return out

    def _entropy(self, text: str) -> List[Hit]:
        out = []
        for core, s, e in _entropy_candidates(text):
            out.append(Hit(ENTROPY_RULE, core, s, e, mask_secret(core), ENTROPY_RULE.severity))
            if len(out) >= 50:
                break
        return out

    @staticmethod
    def _resolve(hits: List[Hit]) -> List[Hit]:
        """Drop weaker hits that overlap stronger ones; dedupe identical (rule, value)."""
        if len(hits) <= 1:
            return hits
        tier_rank = {"provider": 0, "structural": 0, "key": 1, "generic": 2, "pii": 3, "entropy": 4}
        order = sorted(hits, key=lambda h: (tier_rank.get(h.rule.tier, 5), -int(h.severity), h.start))
        kept: List[Hit] = []
        for h in order:
            clash = False
            for k in kept:
                if k.rule.category == h.rule.category == CATEGORY_PII:
                    continue  # PII hits of different kinds may legitimately overlap (path + user)
                if h.start < k.end and k.start < h.end:
                    clash = True
                    break
            if not clash:
                kept.append(h)
        # a pii hit fully inside a kept secret hit is dropped as well
        final = []
        for h in kept:
            if h.rule.category == CATEGORY_PII and any(
                k.rule.category == CATEGORY_SECRET and k.start <= h.start and h.end <= k.end for k in kept
            ):
                continue
            final.append(h)
        final.sort(key=lambda h: h.start)
        seen = set()
        uniq = []
        for h in final:
            sig = (h.rule.id, h.value, h.start)
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(h)
        return uniq


ENGINE = Engine()
