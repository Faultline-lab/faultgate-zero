import pytest

from faultgate_zero import redact


@pytest.mark.parametrize("text,expected", [
    ("mail jane.doe+x@example.co.uk now", "mail [REDACTED:email] now"),
    ("card 4111 1111 1111 1111 ok", "card [REDACTED:card] ok"),
    ("card 4111-1111-1111-1112 fails luhn", "card 4111-1111-1111-1112 fails luhn"),
    ("order 1042 amount 900", "order 1042 amount 900"),
    ("key sk-proj-abcdefghijklmnop1234 end", "key [REDACTED:key] end"),
    ("key sk-ant-api03-abcdefghijk end", "key [REDACTED:key] end"),
    ("aws AKIAABCDEFGHIJKLMNOP", "aws [REDACTED:key]"),
    ("gh ghp_abcdefghijklmnopqrstuvwxyz123", "gh [REDACTED:key]"),
    ("Authorization: Bearer abc.def-ghi_123", "Authorization: [REDACTED:token]"),
    ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sflKxwRJSMeKKF2QT4", "jwt [REDACTED:jwt]"),
    ("timestamp 1729500000000000000", "timestamp 1729500000000000000"),
])
def test_builtin(text, expected):
    assert redact.builtin(text) == expected


def test_off_and_unknown():
    assert redact.get("off")("a@b.com") == "a@b.com"
    with pytest.raises(ValueError):
        redact.get("presidio")
