"""Rule-level tests. All secret-shaped values are assembled from fragments (no secret literals in the repo)."""
import pytest

from limnex_preflight.detectors import ENGINE, shannon_entropy
from limnex_preflight.models import Severity


def ids(text, **kw):
    return sorted({h.rule.id for h in ENGINE.scan(text, **kw)})


F28 = "FAKEfakeFAKEfakeFAKEfakeFAKE"


PROVIDER_CASES = [
    ("key=" + "AKIA" + "IOSFODNN7EXAMPLE", "aws-access-key-id"),
    ("ASIA" + "ABCDEFGHIJKLMNOP", "aws-access-key-id"),
    ("aws_secret_access_key = " + "wJalrXUtnFEMI/K7MDENG/" + "bPxRfiCYEXAMPLEKEY", "aws-secret-access-key"),
    ("AIza" + F28 + "0123456", "gcp-api-key"),
    ("ghp" + "_" + F28 + "01234567", "github-token"),
    ("glpat" + "-" + "FAKEfakeFAKEfake0123", "gitlab-token"),
    ("hf" + "_" + F28 + "012345", "huggingface-token"),
    ("sk" + "-proj-" + F28 + "FAKEfakeFAKEfake0123", "openai-api-key"),
    ("sk-ant" + "-api03-" + "FAKEfakeFAKEfake" * 5, "anthropic-api-key"),
    ("KAGGLE_KEY=" + "0123456789abcdef0123456789abcdef", "kaggle-api-key"),
    ("xoxb" + "-123456789012-1234567890123-" + "FAKEfakeFAKEfake0123", "slack-token"),
    ("sk_live" + "_" + F28 + "0123", "stripe-secret-key"),
    ("-----BEGIN OPENSSH " + "PRIVATE KEY-----\n" + "FAKEfakeFAKEfake" * 3, "private-key-pem"),   # a body follows (test 5, T8)
    ("postgres://u:" + "Zr7qLm2x" + "@h.example.org/db", "password-in-url"),
    ("npm" + "_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8", "npm-token"),
]


@pytest.mark.parametrize("text,rule", PROVIDER_CASES, ids=[f"{r}-{i}" for i, (_, r) in enumerate(PROVIDER_CASES)])
def test_provider_patterns(text, rule):
    assert rule in ids(text)


def test_stripe_test_key_is_medium():
    h = [x for x in ENGINE.scan("sk_test" + "_" + F28 + "0123") if x.rule.id == "stripe-secret-key"][0]
    assert h.severity == Severity.MEDIUM


def test_jwt_requires_a_json_header_with_alg():
    import base64

    def b(x):
        return base64.urlsafe_b64encode(x).decode().rstrip("=")

    good = b(b'{"alg":"HS256"}') + "." + b(b'{"sub":"x"}') + "." + "sigsigsigsig"
    bad = b(b'{"foo":"bar"}') + "." + b(b'{"sub":"x"}') + "." + "sigsigsigsig"
    assert "jwt" in ids(good)
    assert "jwt" not in ids(bad)


@pytest.mark.parametrize("text,expected", [
    ("C:\\Users\\alice_ml\\Desktop\\x.py", True),
    ("C:\\\\Users\\\\alice_ml\\\\Desktop", True),
    ("/home/alice_ml/data", True),
    ("/Users/alice.ml/Library", True),
    ("/kaggle/input/datasets/someowner/xray/img.png", True),
    ("/home/ubuntu/work", False),
    ("C:\\Users\\Public\\Documents", False),
    ("/home/$USER/data", False),
    ("/home/runner/work/repo", False),
    ("/usr/local/home/alice_ml/x", False),
])
def test_username_in_path(text, expected):
    assert ("username-in-path" in ids(text)) is expected


def test_platform_paths():
    assert "platform-path" in ids("/content/drive/MyDrive/my_secret_thesis/ckpt.pt")
    assert "platform-path" in ids("/kaggle/input/some-dataset-slug/train.csv")
    assert "platform-path-generic" in ids("saved to /kaggle/working/model.bin")


@pytest.mark.parametrize("text,expected", [
    ("write to jane.person@gmail.com please", True),
    ("noreply@github.com", False),
    ("user@example.com", False),
    ("git@github.com:org/repo.git", False),
    ("logo@2x.png", False),
    ("12345+jane@users.noreply.github.com", True),
])
def test_email(text, expected):
    assert ("email-address" in ids(text)) is expected


def test_role_mailbox_is_low():
    h = [x for x in ENGINE.scan("contact: info@somecorp.io") if x.rule.id == "email-address"][0]
    assert h.severity == Severity.LOW


@pytest.mark.parametrize("text,expected", [
    ("server at 192.168.1.20 port", True),
    ("10.0.0.5", True),
    ("127.0.0.1", False),
    ("0.0.0.0", False),
    ("version 1.2.3.4", False),
    ("numpy==1.2.3.4", False),
    ("1.02.3.4 build", False),
    ("203.0.113.9", False),
    ("169.254.169.254", False),
    ("999.1.1.1", False),
    # pre-release test 3, R3: a version word inside another word, or the "@" of user@host, hid private addresses
    ("server 10.0.0.5", True),
    ("--server 10.0.0.5", True),
    ("SERVER: 10.0.0.5", True),
    ("dev 10.0.0.5", True),
    ("driver 172.16.0.9", True),
    ("ssh ubuntu@10.0.1.23", True),
    ("v10.0.0.5", False),
    ("rev 10.0.0.5", False),
    ("opset: 10.0.0.1", False),
])
def test_ipv4(text, expected):
    assert ("ip-address" in ids(text)) is expected


def test_hostnames():
    assert "hostname" in ids("trained on gpu-node-07.corp.internal")
    assert "hostname" in ids("Johns-MacBook-Pro.local")
    assert "hostname" in ids("ec2-1-2-3-4.compute-1.amazonaws.com")
    assert "hostname" in ids("http://buildserver:8080/job")
    assert "hostname" not in ids("http://localhost:8080/job")
    assert "hostname" not in ids("self.local_rank")


class TestEntropy:
    def test_random_token_flagged(self):
        assert "high-entropy-string" in ids("tok=q8Zr3KmX7vLp2NwYt5HcJd9BfGs4AeU6oIiR1PzV")

    def test_hash_uuid_and_words_not_flagged(self):
        assert ids("sha256 " + "ab12" * 16) == []
        assert ids("id 123e4567-e89b-12d3-a456-426614174000") == []
        assert ids("BertForSequenceClassificationHeadWithPooling") == []
        assert ids("encoder_layer_norm_weight_scale_factor_0") == []

    def test_threshold_is_documented_constant(self):
        from limnex_preflight import detectors

        assert detectors.ENTROPY_BASE64_THRESHOLD == 4.5
        assert detectors.ENTROPY_HEX_THRESHOLD == 3.0
        assert shannon_entropy("aaaa") == 0.0


class TestGenericCredential:
    def test_assignment_flagged(self):
        assert "generic-credential" in ids("DB_PASSWORD=Xk9-Passw0rd-4471")
        assert "cli-secret-argument" in ids("--api_key Zr7qLm2xPd9s")

    def test_placeholders_and_tokenizer_tokens_not_flagged(self):
        assert ids("api_key: YOUR_API_KEY_HERE") == []
        assert ids("password = <your-password>") == []
        assert ids('"pad_token": "<pad>"') == []
        assert ids("--max_token_length 128") == []
        assert ids("--tokenizer_name bert-base-uncased") == []

    def test_key_context_for_structured_values(self):
        hits = ENGINE.scan("Xk9Passw0rd4471", key="db_password")
        assert [h.rule.id for h in hits] == ["generic-credential"]
        assert ENGINE.scan("<s>", key="bos_token") == []
        assert ENGINE.scan("Xk9Passw0rd4471", key="unk_token") == []


def test_username_field_keys():
    assert [h.rule.id for h in ENGINE.scan("fx_alice_ml", key="username")] == ["username-field"]
    assert ENGINE.scan("user", key="username") == []


def test_binary_carved_strings_need_a_real_tld_for_emails():
    from limnex_preflight.detectors import MODE_BINARY

    assert ids("d1x@j7.Kq", mode=MODE_BINARY) == []
    assert ids("fx.person@gmail.com", mode=MODE_BINARY) == ["email-address"]


def test_vcs_metadata_block_email_is_not_the_publishers():
    block = "Branch: tags/v1.2.10\n    Author: some dev <some.dev@corp-tools.io>\n    Summary: Release 1.2.10. commit_hash:f8e0038b"
    assert ids(block) == []
    assert "email-address" in ids("Author: someone, mail me: some.dev@corp-tools.io please")


def test_public_ip_needs_host_context_but_private_does_not():
    assert ids("Connecting to raw.githubusercontent.com... 185.199.110.133, 185.199.111.133") == []
    assert ids("typing-extensions (3.10.0.2)") == []
    assert "ip-address" in ids("remote_host = 8.8.4.4")
    assert "ip-address" in ids("see http://8.8.4.4/path")
    assert "ip-address" in ids("lives at 192.168.1.20 somewhere")


def test_paths_urls_and_identifiers_are_not_high_entropy():
    assert ids("see https://huggingface.co/production/uploads/5f0e1d2c3b4a59687c8d9e0f/Qm7kLp2xRt-VwZ") == []
    assert ids("wav2vec2-xls-r-300m-xx-250/checkpoint-16800/pytorch_model") == []
    assert ids("1/attention/self/Sqrt_2_output_0_CUDAExecutionProvider") == []
    assert ids("api_key=os.environ['HF_TOKEN']") == []


def test_code_references_are_not_credential_values():
    assert ids("user_secrets = UserSecretsClient()") == []
    assert ids("client = OpenAI(api_key=self.api_key)") == []
    assert ids("api_key=os.environ['HF_TOKEN']") == []
    assert "generic-credential" in ids("DB_PASSWORD=Xk9-Passw0rd-4471")
    assert "generic-credential" in ids("DB_PASSWORD=Passw0rd123Xyz")


def test_trainer_run_directory_leaks_host_name():
    assert "hostname" in ids("logging_dir: HF_kmeans_model_saved/runs/Mar04_22-56-44_ZX9q0b0001")
    assert "hostname" not in ids("runs/Mar04_22-56-44_localhost")


@pytest.mark.parametrize("text,rule", PROVIDER_CASES, ids=[f"big-{r}-{i}" for i, (_, r) in enumerate(PROVIDER_CASES)])
def test_large_text_path_finds_the_same_secrets(text, rule):
    """Above 100 000 characters regexes only run in windows around their hints; results must not change."""
    padded = ("filler line without anything of interest\n" * 4000) + " " + text + " " + ("more filler text here\n" * 4000)
    assert len(padded) > 100_000
    assert rule in ids(padded)


# --- found by dogfooding on the author's own projects (2026-10-29) ---------------------------
@pytest.mark.parametrize("text", [
    "behaves identically to the real 'http://trusted.com@evil.com' attack",  # URL user-info after //
    "abc12@secure-bank-login.com/verify/account",  # scheme-less URL: the host is followed by a path
    "fx-deploy7@build-host.com:8080/job/1",  # ... or by a port
])
def test_url_user_info_is_not_an_email_address(text):
    assert "email-address" not in ids(text)


def test_an_email_at_the_end_of_a_sentence_is_still_one():
    assert "email-address" in ids("Questions? Write to jane.person@gmail.com.")
    assert "email-address" in ids("mailto:jane.person@gmail.com")
    assert "email-address" in ids("https://site.org/users/jane.person@gmail.com/settings")  # inside a path


def test_a_file_name_containing_api_key_is_not_a_credential_name():
    digest = "f8" + "0123456789abcdef" * 3 + "0123456789ab" + "a2"  # 64 hex, a sha256
    assert ENGINE.scan(digest, key="scripts/mint_api_key.py") == []
    assert "generic-credential" not in ids('"scripts/mint_api_key.py": "' + digest + '"')
    assert "generic-credential" not in ids("mint_api_key.py: " + digest)


def test_a_call_taking_a_string_literal_is_code_not_a_credential_value():
    assert "generic-credential" not in ids('    "api_key": re.compile(r"[A-Za-z0-9]{32}"),')
    assert "generic-credential" not in ids('private_key = load_pem(b"...")')
    # a real value that merely contains a parenthesis is still a credential
    assert "generic-credential" in ids("DB_PASSWORD=Xk9(Passw0rd)4471")
