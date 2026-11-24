from limnex_preflight.masking import mask_email, mask_host, mask_ip, mask_secret, mask_word


def test_hf_style_mask_keeps_prefix_and_two_chars_each_side():
    value = "hf_" + "A" * 2 + "x" * 29 + "9Z"
    assert mask_secret(value, prefix_len=3) == "hf_AA...9Z"


def test_short_values_fully_hidden():
    assert mask_secret("abcdefgh") == "********"


def test_mask_never_reveals_more_than_a_third():
    v = "K" * 12
    m = mask_secret(v)
    assert len(m.replace("...", "")) <= len(v) // 3


def test_masks_of_personal_values():
    assert mask_word("alice") == "a****"
    assert mask_email("jane.doe@gmail.com") == "j***@g***.com"
    assert mask_ip("10.20.30.40") == "10.20.*.*"
    assert mask_host("gpu-07.corp.internal") == "g*****.corp.internal"


def test_control_characters_are_neutralised():
    assert "\n" not in mask_secret("abc\ndefghijklmnop")
