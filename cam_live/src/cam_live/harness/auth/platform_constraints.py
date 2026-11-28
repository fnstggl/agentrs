"""Hard platform constraints: no automation of ordinary or owner user accounts, no circumvention of CAPTCHA,
human verification or anti-automation measures (operator amendment 2026-11-27 "hard platform constraints";
spec 12.5 "Do not automate ordinary Discord user accounts", 12.6 "Do not permit ... circumvention of platform
restrictions", 12A.13 "Do not automate ordinary user accounts ... Human verification ... must respect independent
authorization requirements. The model must never learn to circumvent access controls").

Unlike the rules of the policy file (data the operator edits), these constraints are fixed in code. No policy
file, task, live authorization record, approval, pre-authorization or ``interactive`` / ``owned`` list can permit
them: :func:`apply` adds ``deny`` findings, which no later stage of :class:`~cam_live.harness.auth.PolicyService`
turns into anything else. It is called by :meth:`Evaluator.node <cam_live.harness.auth.evaluate.Evaluator.node>`,
so it applies to every decision path -- ``authorize`` (every step of a program or skill) and ``check_primitive``
(every operation an executor reports: browser navigations of the page and its frames, the page's write requests,
HTTP hops after redirects, sandbox connections, skill steps, visual gestures).

What is refused (policy ids):

``X-HUMAN-VERIFICATION``
    * an operation whose name is about CAPTCHAs / human verification / bot challenges (a "solve" tool);
    * any request to a CAPTCHA-solving service, by any modality and with any method, also when generated code
      or a raw request only names the service;
    * an interaction (click, typing, upload) or write (POST of an answer, a token request) on a human-verification
      endpoint -- reCAPTCHA, hCaptcha, Cloudflare Turnstile and challenge platform, Arkose / FunCaptcha, GeeTest,
      DataDome, PerimeterX, AWS WAF, Friendly Captcha, MTCaptcha, Akamai's challenge path. Loading a page that
      embeds a challenge is a read and stays possible; answering it is not;
    * a solved-challenge token in the arguments, in a JSON / form body sent as a string, or in a URL query
      (``g-recaptcha-response``, ``h-captcha-response``, ``cf-turnstile-response``, ``captcha_token`` ...);
    * program text or arguments that use a CAPTCHA-solver or bot-detection-evasion library, or override
      ``navigator.webdriver`` (anti-automation measures are respected, not defeated);
    * routing requests through a proxy / IP rotation named in the arguments (evading a site's rate limits or
      blocks): the harness's own egress is the only network path.
``X-USER-ACCOUNT``
    * the browser (structured or visual) interacting with or writing to Discord, or opening its client
      application (``/login``, ``/register``, ``/channels/...``, ``/app``, invites, OAuth2 authorization, user
      pages, settings) or its API with the browser's session: the persistent browser profile may hold the owner's
      personal session, and Discord forbids automating user accounts ("self-bots"). Reading Discord's public
      documentation, guidelines and policies stays possible;
    * raw HTTP (the ``api``, ``code`` and ``skill`` modalities) to Discord's user-account-only endpoints
      (``/auth/*``, relationships, private channels, user settings / billing / connections / sessions, accepting
      an invite, OAuth2 authorization) and raw HTTP to Discord's API with any credential other than the ``bot``
      scheme or with a literal ``Authorization`` header: programmatic Discord operations go through the official
      bot integration (``discord.*`` tools, ``Authorization: Bot <token>``);
    * generated code that connects to Discord at all, or that uses self-bot libraries / user-token login.

``X-PLATFORM-ADAPTER`` (G2-harness D1)
    * any request to an integrated platform's data -- Discord's REST API, Gateway or attachment CDN -- that is not
      an operation of the platform's own adapter (``discord.*``), by any modality (generic HTTP, browser, code,
      skills): the adapter applies the data-use rules (message content, member list, personal data) and the
      operator's scope, a generic request would bypass them (spec 12A.7: an access restriction is never a reason
      to switch modalities). The adapter's own transport requests realise its operation and stay covered;
    * a platform credential (by its vault scope or its known name, e.g. ``discord_bot_token``) named by any other
      operation; the authorization service also refuses to *release* it except to an action that is, or contains,
      an operation of that adapter.

Human-verification tests are also recognised structurally (G2-harness D4): an answer key matches by substring
(``captcha_input``, ``recaptcha_response_field``, ``captchaResponseToken``, ``human_check`` ...), the browser reports
the accessible name of the element an input or click goes to ("CAPTCHA answer", "I am not a robot", "Verify you
are human" are refused) and the field names of every form a page submits.

The lists below are this module's data; they are deliberately not part of the operator-editable policy file.
Anything they do not name is still subject to every rule of the policy (default deny of writes to destinations
the operator does not own, approvals, rate limits ...): these constraints only add refusals.
"""
from __future__ import annotations

import fnmatch
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlsplit

from ... import contracts as C
from . import destinations as D

PID_HUMAN = "X-HUMAN-VERIFICATION"
PID_ACCOUNT = "X-USER-ACCOUNT"
PID_ADAPTER = "X-PLATFORM-ADAPTER"
PIDS = (PID_HUMAN, PID_ACCOUNT, PID_ADAPTER)

#: integrated platforms: the operations of their official adapter. A platform's data (its API, gateway, attachment
#: CDN) is reached only through these operations, whose data-use rules apply, and the platform's credential is
#: released only to them (G2-harness D1; spec 12.5, 12.6, 12A.7 "never interpret an access restriction as a reason
#: to switch execution modalities")
PLATFORM_ADAPTER_OPS: Dict[str, Tuple[str, ...]] = {"discord": ("discord.*",)}
#: credential names that belong to a platform even when the operator stored them without its scope
PLATFORM_CREDENTIALS: Dict[str, str] = {"discord_bot_token": "discord"}

_BROWSER = (C.Modality.STRUCTURED_BROWSER, C.Modality.VISUAL)
_RAW = (C.Modality.API, C.Modality.CODE, C.Modality.SKILL)
_ACTIVE_EFFECTS = ("interact", "external_write")

# ------------------------------------------------------------------------------------------- human verification
#: CAPTCHA-solving services (human or automated solvers): never contacted, never named in a payload
SOLVER_HOSTS: Tuple[str, ...] = (
    "2captcha.com", "*.2captcha.com", "rucaptcha.com", "*.rucaptcha.com", "anti-captcha.com", "*.anti-captcha.com",
    "capsolver.com", "*.capsolver.com", "capmonster.cloud", "*.capmonster.cloud", "deathbycaptcha.com",
    "*.deathbycaptcha.com", "nopecha.com", "*.nopecha.com", "azcaptcha.com", "*.azcaptcha.com", "captchas.io",
    "*.captchas.io", "bestcaptchasolver.com", "*.bestcaptchasolver.com", "imagetyperz.com", "*.imagetyperz.com",
    "endcaptcha.com", "*.endcaptcha.com", "captchaai.com", "*.captchaai.com", "cap.guru", "*.cap.guru",
    "solvecaptcha.com", "*.solvecaptcha.com", "anycaptcha.com", "*.anycaptcha.com", "noopcaptcha.com",
    "captchasolver.com", "*.captchasolver.com", "truecaptcha.org", "*.truecaptcha.org", "capbypass.com",
    "*.capbypass.com", "ez-captcha.com", "*.ez-captcha.com", "captcha.guru", "*.captcha.guru")

#: human-verification / bot-challenge endpoints: (host glob, path glob or "" for every path)
CHALLENGE_ENDPOINTS: Tuple[Tuple[str, str], ...] = (
    ("google.com", "/recaptcha/*"), ("recaptcha.google.com", ""), ("recaptcha.net", ""), ("*.recaptcha.net", ""),
    ("gstatic.com", "/recaptcha/*"), ("hcaptcha.com", ""), ("*.hcaptcha.com", ""),
    ("challenges.cloudflare.com", ""), ("*", "/cdn-cgi/challenge-platform/*"),
    ("arkoselabs.com", ""), ("*.arkoselabs.com", ""), ("*.arkoselabs.cn", ""), ("funcaptcha.com", ""),
    ("*.funcaptcha.com", ""), ("geetest.com", ""), ("*.geetest.com", ""), ("*.geevisit.com", ""),
    ("captcha-delivery.com", ""), ("*.captcha-delivery.com", ""), ("*.perimeterx.net", ""), ("*.px-cloud.net", ""),
    ("*.px-cdn.net", ""), ("*.awswaf.com", ""), ("friendlycaptcha.com", ""), ("*.friendlycaptcha.com", ""),
    ("*.friendlycaptcha.eu", ""), ("mtcaptcha.com", ""), ("*.mtcaptcha.com", ""), ("*", "/_sec/cp_challenge/*"),
    ("*", "/.well-known/sgcaptcha/*"), ("captcha.com", ""), ("*.captcha.com", ""))

#: normalised (lower case, only [a-z0-9]) argument / body / query keys that carry a solved challenge
TOKEN_FIELDS = frozenset((
    "grecaptcharesponse", "recaptcharesponse", "recaptchatoken", "recaptcha", "hcaptcharesponse", "hcaptchatoken",
    "cfturnstileresponse", "turnstileresponse", "turnstiletoken", "captcha", "captcharesponse", "captchatoken",
    "captchasolution", "captchaanswer", "captchacode", "captchatext", "captchavalue", "captchakey", "captchaid",
    "arkosetoken", "fctoken", "funcaptchatoken", "geetestvalidate", "geetestseccode", "geetestchallenge",
    "datadomecaptcha", "pxcaptcha", "awswaftoken", "frccaptchasolution", "mtcaptchaverifiedtoken"))

#: parts of a normalised key (argument, body field, URL query, form field) that make it a human-verification
#: answer: matched as substrings, so ``captcha_input``, ``recaptcha_response_field``, ``captchaResponseToken``,
#: ``human_check`` are recognised without an exact list (G2-harness D4)
TOKEN_KEY_PARTS = ("captcha", "recaptcha", "hcaptcha", "turnstile", "humancheck", "humanverif", "notarobot",
                   "imnotarobot", "botcheck", "antibot", "arkose", "funcaptcha", "geetest", "frccaptcha",
                   "challengeresponse", "challengeanswer", "challengetoken", "proofofhuman")
#: accessible names / labels of page elements that are a human-verification test (structural, any site)
_HUMAN_TEST_TEXT = re.compile(
    r"(?i)captcha|\bturnstile\b|human[\s_\-]*(verification|check|test)|verify\s+(that\s+)?you(\s+are|'re)\s+(a\s+)?"
    r"(human|real|not\s+a\s+(ro)?bot)|prove\s+(that\s+)?you(\s+are|'re)\s+(a\s+)?human|(i\s*am|i'm)\s+not\s+a\s+"
    r"(ro)?bot|are\s+you\s+(a\s+)?(human|(ro)?bot)|not\s+a\s+robot|bot\s+(check|detection|protection)|"
    r"security\s+(check|challenge)|type\s+the\s+(characters|letters|text|code)\s+(you\s+see|shown|in\s+the\s+image)|"
    r"select\s+all\s+(images|squares)")


def is_token_key(key: str) -> bool:
    """A normalised key that names a human-verification answer (substring match on :data:`TOKEN_KEY_PARTS`)."""
    k = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return k in TOKEN_FIELDS or any(part in k for part in TOKEN_KEY_PARTS)


def is_human_test_label(text: str) -> bool:
    return bool(text) and bool(_HUMAN_TEST_TEXT.search(str(text)))


#: normalised operation-name parts that make an operation a human-verification "solver"
_OP_WORDS = ("captcha", "humanverification", "botchallenge", "solvechallenge", "bypasschallenge", "antibot",
             "botdetection", "turnstile")

#: markers in program text / arguments: solver services and libraries, detection evasion
_SOLVER_TEXT = re.compile(
    r"(?i)(?<![a-z0-9])(2captcha|twocaptcha|rucaptcha|anti-?captcha(official)?|python3?_anticaptcha|capsolver|"
    r"capmonster|deathbycaptcha|nopecha|azcaptcha|bestcaptchasolver|imagetyperz|endcaptcha|captchaai|"
    r"captcha[_\-\s]?solver|recaptcha[_\-]?solver|hcaptcha[_\-]?solver|pypasser|playwright[_\-]recaptcha|"
    r"puppeteer-extra-plugin-recaptcha|undetected[_\-]chromedriver|playwright[_\-]stealth|selenium[_\-]stealth|"
    r"puppeteer-extra-plugin-stealth|cloudscraper|cfscrape|flaresolverr|nodriver)(?![a-z0-9])")
_SOLVE_PHRASE = re.compile(
    r"(?i)\b(solv\w*|bypass\w*|crack\w*|break\w*|defeat\w*|circumvent\w*|evad\w*|beat\w*|get\s+around|skip\w*)"
    r"[\s_\-]{0,3}(the\s+|a\s+|an\s+|this\s+)?(re|h)?captchas?\b|\b(re|h)?captcha[\s_\-]{0,3}"
    r"(solv\w*|bypass\w*|crack\w*|breaker|defeat\w*|farm\w*)\b|\bsolve[\s_\-]{0,3}(the\s+)?(human[\s_\-]verification|"
    r"bot[\s_\-]challenge|turnstile|challenge[\s_\-]page)")
_WEBDRIVER_OVERRIDE = re.compile(
    r"(?i)defineproperty\s*\(\s*navigator\s*,\s*['\"]webdriver['\"]|navigator\.webdriver\s*=|"
    r"['\"]?--disable-blink-features['\"]?\s*[=,]?\s*['\"]?automationcontrolled|excludeswitches.{0,40}enable-automation")

#: argument keys that route traffic around the harness's egress (proxy chains, IP rotation)
PROXY_FIELDS = frozenset(("proxy", "proxies", "proxyurl", "proxyserver", "proxyhost", "httpproxy", "httpsproxy",
                          "socksproxy", "socks5", "residentialproxy", "rotateip", "iprotation", "rotatingproxy",
                          "rotateproxies", "torproxy", "usetor"))

# ------------------------------------------------------------------------------------------------- user accounts
DISCORD_HOSTS: Tuple[str, ...] = ("discord.com", "*.discord.com", "discordapp.com", "*.discordapp.com",
                                  "discord.gg", "*.discord.gg", "discord.new", "discord.gift", "*.discord.gift",
                                  "discord.co", "*.discord.co", "discordapp.net", "*.discordapp.net")
#: Discord hosts that are documentation / policy pages only (no user session is ever used there)
_DISCORD_DOC_HOSTS = ("support.discord.com", "support-dev.discord.com", "support-apps.discord.com",
                      "dis.gd", "status.discord.com", "blog.discord.com")
#: paths of Discord's web client (a user account's session): not opened by the browser
DISCORD_CLIENT_PATHS: Tuple[str, ...] = (
    "/login", "/login/*", "/register", "/register/*", "/channels", "/channels/*", "/app", "/app/*", "/activity",
    "/activity/*", "/invite/*", "/oauth2/*", "/users/*", "/settings*", "/library*", "/store*", "/shop*",
    "/discovery*", "/guild-discovery*", "/quest-home*", "/nitro*", "/billing*", "/ra/*", "/reset*", "/verify*",
    "/authorize*", "/api", "/api/*", "/gift/*", "/template/*", "/events/*")
#: API paths (after ``/api`` and an optional ``/v<n>``) that exist only for user accounts
DISCORD_USER_ENDPOINTS: Tuple[str, ...] = (
    "/auth", "/auth/*", "/users/@me/relationships*", "/users/@me/channels*", "/users/@me/settings*",
    "/users/@me/billing*", "/users/@me/connections*", "/users/@me/mfa*", "/users/@me/notes*",
    "/users/@me/library*", "/users/@me/affinities*", "/users/@me/sessions*", "/users/@me/devices*",
    "/users/@me/consent*", "/users/@me/harvest*", "/users/@me/email*", "/users/@me/password*",
    "/users/@me/remote-auth*", "/users/@me/disable*", "/users/@me/delete*", "/users/@me/activities*",
    "/users/@me/entitlements*", "/users/@me/guilds/premium*", "/users/@me/referrals*", "/users/@me/phone*",
    "/users/@me/profile*", "/users/@me/survey*", "/users/@me/collectibles*", "/users/@me/applications/*/role-connection*",
    "/invites/*", "/invite/*", "/oauth2/authorize*", "/oauth2/tokens*", "/users/*/profile*", "/users/*/relationships*",
    "/friend-suggestions*", "/science", "/track", "/experiments*", "/hypesquad*", "/gifts*", "/entitlements/gift-codes*",
    "/channels/*/recipients*", "/report*", "/safety-hub*", "/promotions*", "/activities*", "/quests*",
    "/applications/detectable*", "/discoverable-guilds*", "/guilds/*/requests*", "/guilds/*/member-verification*",
    "/guilds/*/members/@me", "/guilds/*/members/@me/*", "/guilds/*/premium*", "/guilds/*/profile*",
    "/guilds/*/join-requests*", "/guilds/*/ack*", "/channels/*/messages/*/ack", "/read-states*", "/sticker-packs*")
#: Discord's attachment CDN (files of the guild's messages): platform data like the API
_DISCORD_CDN = (("cdn.discordapp.com", ("/attachments/*", "/ephemeral-attachments/*")),
                ("media.discordapp.net", ("/attachments/*", "/ephemeral-attachments/*")),
                ("cdn.discord.com", ("/attachments/*", "/ephemeral-attachments/*")))
_API_PREFIX = re.compile(r"^/api(/v\d+)?(?=/|$)")
_SELFBOT_TEXT = re.compile(
    r"(?i)(?<![a-z0-9])(self[_\-\s]?bots?|discord\.py-self|discord-self|discord\.js-selfbot(-v\d+)?|"
    r"discum)(?![a-z0-9])|/api(/v\d+)?/auth/(login|register|mfa)")
#: only together with a reference to Discord: a user token is the credential of a user account
_USER_TOKEN_TEXT = re.compile(r"(?i)(?<![a-z0-9])user[_\-\s]?token(?![a-z0-9])")
#: resource types of page requests that scripts issue (their responses are readable by the page)
_SCRIPT_READS = ("xhr", "fetch", "eventsource", "websocket", "other", "ping", "beacon")
_GATEWAY_HOSTS = ("gateway.discord.gg", "*.discord.gg", "gateway.discord.com", "*.gateway.discord.gg")


def _host_in(host: str, globs: Iterable[str]) -> bool:
    h = D.bare_host((host or "").lower())
    return any(fnmatch.fnmatchcase(h, g) or fnmatch.fnmatchcase(host or "", g) for g in globs)


def _path_in(path: str, globs: Iterable[str]) -> bool:
    p = path or "/"
    return any(fnmatch.fnmatchcase(p, g) for g in globs)


def is_discord(d: D.Destination) -> bool:
    return d.kind == "url" and _host_in(d.host, DISCORD_HOSTS)


def is_solver(d: D.Destination) -> bool:
    return d.kind == "url" and _host_in(d.host, SOLVER_HOSTS)


def is_challenge(d: D.Destination) -> bool:
    if d.kind != "url":
        return False
    for host, path in CHALLENGE_ENDPOINTS:
        if host != "*" and not _host_in(d.host, (host,)):
            continue
        if not path or fnmatch.fnmatchcase(d.path or "/", path):
            return True
    return False


def discord_api_path(d: D.Destination) -> Optional[str]:
    """``/users/@me`` for ``https://discord.com/api/v10/users/@me``; None when ``d`` is not Discord's REST API."""
    if not is_discord(d):
        return None
    m = _API_PREFIX.match(d.path or "")
    if not m:
        return None
    return (d.path[m.end():] or "/")


def is_discord_data(d: D.Destination) -> bool:
    """Discord's REST API, its Gateway, or its attachment CDN: the platform's data."""
    if d.kind != "url" or not is_discord(d):
        return False
    if discord_api_path(d) is not None or _host_in(d.host, _GATEWAY_HOSTS):
        return True
    return any(_host_in(d.host, (h,)) and _path_in(d.path, paths) for h, paths in _DISCORD_CDN)


def _discord_client_path(path: str) -> bool:
    p = path or "/"
    if _path_in(p, DISCORD_CLIENT_PATHS):
        return True
    # the developer portal (applications, bots and their tokens, teams, servers) runs with a user's session; only
    # the documentation below /developers/docs is public reading (G2-harness D3)
    low = p.lower()
    return low.startswith("/developers/") and not (low == "/developers/" or low.startswith("/developers/docs"))


def adapter_op(operation: str, platform: str) -> bool:
    return any(fnmatch.fnmatchcase(operation or "", g) for g in PLATFORM_ADAPTER_OPS.get(platform, ()))


def credential_platform(name: str, scopes: Sequence[str] = ()) -> str:
    """The integrated platform a stored credential belongs to ('' for none)."""
    n = str(name or "")
    n = n[len("secret:"):] if n.startswith("secret:") else n
    for sc in scopes or ():
        if str(sc).lower() in PLATFORM_ADAPTER_OPS:
            return str(sc).lower()
    return PLATFORM_CREDENTIALS.get(n, "")


def _query_keys(d: D.Destination) -> List[str]:
    if d.kind != "url" or "?" not in (d.raw or ""):
        return []
    try:
        q = urlsplit(d.raw if "://" in d.raw else "https://" + d.raw).query
    except ValueError:
        return []
    return [re.sub(r"[^a-z0-9]", "", k.lower()) for k, _v in parse_qsl(q, keep_blank_values=True)]


def _walk_keys(obj: Any, depth: int = 0) -> Iterable[Tuple[str, Any]]:
    if depth > 16:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield re.sub(r"[^a-z0-9]", "", str(k).lower()), v
            yield from _walk_keys(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_keys(v, depth + 1)


def _truthy(v: Any) -> bool:
    return v not in (None, "", False, 0, 0.0) and v != [] and v != {}


def findings(node: C.ActionSpec, dests: Sequence[D.Destination], effect: str, text_all: str,
             expanded_args: Any, raw_tool: bool, opaque: bool,
             secret_platform: Optional[Callable[[str], str]] = None,
             realises: Sequence[str] = ()) -> List[Tuple[str, str, Optional[D.Destination]]]:
    """``(policy id, reason, destination)`` of every hard constraint ``node`` violates.

    ``effect`` is the node's resolved base effect (``external_read`` / ``external_write`` / ``interact`` /
    ``local`` / ``none``); ``raw_tool`` is true for raw HTTP tools (``http.*``, ``net.request``, ``api.*``);
    ``opaque`` for tools whose payload is program text; ``realises`` the operations of the granted action a
    primitive belongs to (the adapter's own transport requests realise an adapter operation)."""
    out: List[Tuple[str, str, Optional[D.Destination]]] = []
    op_norm = re.sub(r"[^a-z0-9]", "", (node.operation or "").lower())
    modality = node.modality
    active = effect in _ACTIVE_EFFECTS

    # ---- human verification ------------------------------------------------------------------------------------
    if any(w in op_norm for w in _OP_WORDS):
        out.append((PID_HUMAN, f"{node.operation!r} is an operation on CAPTCHAs / human verification; the harness "
                               f"never solves, bypasses or automates them", None))
    url_dests = [d for d in dests if d.kind == "url"]
    for d in url_dests:
        if is_solver(d):
            out.append((PID_HUMAN, f"{d.host} is a CAPTCHA-solving service; it is never contacted", d))
        elif is_challenge(d) and active:
            out.append((PID_HUMAN, f"{d.key} is a human-verification challenge: loading a page that embeds it is "
                                   f"permitted, interacting with or answering it is not", d))
        if any(is_token_key(k) for k in _query_keys(d)):
            out.append((PID_HUMAN, "the URL carries a human-verification answer", d))
    keys = [(k, v) for k, v in _walk_keys(expanded_args) if _truthy(v)]
    tok = sorted({k for k, _ in keys if is_token_key(k)})
    a0 = node.arguments if isinstance(node.arguments, dict) else {}
    ff = a0.get("form_fields")
    if isinstance(ff, (list, tuple)):          # a page's form submission / write request, as the browser reports it
        hit = sorted({str(f) for f in ff if isinstance(f, str) and is_token_key(f)})
        if hit:
            out.append((PID_HUMAN, f"the request submits a human-verification form field ({', '.join(hit[:3])})",
                        None))
    el = a0.get("element")
    if isinstance(el, dict) and (active or effect in ("interact",)):
        label = " ".join(str(el.get(k) or "") for k in ("name", "label", "placeholder", "id", "field"))
        if is_human_test_label(label):
            out.append((PID_HUMAN, f"the element {str(el.get('role') or '')} {str(el.get('name') or '')[:60]!r} is a "
                                   f"human-verification test; it is never answered or ticked", None))
    if tok:
        out.append((PID_HUMAN, f"the arguments carry a human-verification token field ({', '.join(tok[:3])})", None))
    prox = sorted({k for k, _ in keys if k in PROXY_FIELDS})
    if prox:
        out.append((PID_HUMAN, f"routing through a proxy / IP rotation ({', '.join(prox[:3])}) would evade a "
                               f"site's rate limits or blocks; the harness's own egress is the only network path",
                    None))
    m = _SOLVER_TEXT.search(text_all) or _WEBDRIVER_OVERRIDE.search(text_all)
    if m:
        out.append((PID_HUMAN, f"the payload uses a CAPTCHA-solver or bot-detection-evasion tool ({m.group(0)[:40]!r})",
                    None))
    elif _SOLVE_PHRASE.search(text_all) and (opaque or raw_tool or active):
        out.append((PID_HUMAN, "the payload is about solving or bypassing a CAPTCHA / human verification", None))
    for d in D.urls_in_text(text_all) if (opaque or raw_tool) else ():
        if is_solver(d):
            out.append((PID_HUMAN, f"the payload refers to {d.host}, a CAPTCHA-solving service", d))
            break

    # ---- integrated platforms: data and credentials only through the platform's adapter ------------------------
    in_adapter = adapter_op(node.operation, "discord") or any(adapter_op(o, "discord") for o in realises)
    for d in url_dests:
        if is_discord_data(d) and not in_adapter:
            out.append((PID_ADAPTER, f"{d.key} is Discord's API / Gateway / attachment store: Discord data is reached "
                                     f"only through the Discord adapter (discord.* operations), whose data-use and "
                                     f"scope rules apply; no other modality may read or write it", d))
    if (opaque or raw_tool) and not in_adapter:
        for d in D.urls_in_text(text_all):
            if is_discord_data(d):
                out.append((PID_ADAPTER, f"the payload addresses Discord's API / Gateway / attachments ({d.host}); "
                                         f"Discord data is reached only through the Discord adapter", d))
                break
    a1 = node.arguments if isinstance(node.arguments, dict) else {}
    auth = a1.get("auth") if isinstance(a1.get("auth"), dict) else None
    names = [auth.get("secret")] if auth and isinstance(auth.get("secret"), str) else []
    names += [v for k, v in _walk_keys(expanded_args) if isinstance(v, str) and v.startswith("secret:")]
    for n in names:
        plat = (secret_platform(n) if secret_platform is not None else "") or credential_platform(n)
        if plat and not (adapter_op(node.operation, plat) or any(adapter_op(o, plat) for o in realises)):
            out.append((PID_ADAPTER, f"{n!r} is the {plat} credential: it is released only to the {plat} adapter's "
                                     f"operations, never to {node.operation!r}", None))
            break

    # ---- user accounts (Discord) -------------------------------------------------------------------------------
    for d in url_dests:
        if not is_discord(d):
            continue
        docs = _host_in(d.host, _DISCORD_DOC_HOSTS)
        if modality in _BROWSER:
            rtype = str((node.arguments or {}).get("resource_type") or "").lower() if isinstance(node.arguments, dict) else ""
            if (not active and rtype in _SCRIPT_READS and not docs
                    and ((d.path or "/") == "/" or _discord_client_path(d.path) or discord_api_path(d) is not None)):
                out.append((PID_ACCOUNT, f"a page script's read of {d.key} would use the browser's Discord session; "
                                         f"Discord is never read with a user's session (G2-harness D6)", d))
                continue
            if active:
                out.append((PID_ACCOUNT, f"the browser does not interact with Discord ({d.host}): Discord user "
                                         f"accounts are never automated; programmatic operations use the official "
                                         f"bot integration", d))
            elif not docs and (_discord_client_path(d.path) or _host_in(d.host, ("discord.gg", "*.discord.gg",
                                                                                              "discord.new",
                                                                                              "discord.gift"))):
                out.append((PID_ACCOUNT, f"{d.key} is Discord's client application (a user account's session); the "
                                         f"browser only reads Discord's public documentation and policies", d))
            continue
        if modality == C.Modality.CODE and opaque:
            out.append((PID_ACCOUNT, f"generated code does not connect to Discord ({d.host}); programmatic Discord "
                                     f"operations use the official bot integration", d))
            continue
        api = discord_api_path(d)
        if api is not None and (raw_tool or modality in _RAW) and _path_in(api, DISCORD_USER_ENDPOINTS):
            out.append((PID_ACCOUNT, f"{api} is a Discord endpoint of user accounts; it is never called", d))
        if _host_in(d.host, _GATEWAY_HOSTS) and modality == C.Modality.CODE:
            out.append((PID_ACCOUNT, "generated code does not open Discord Gateway sessions", d))
    if raw_tool and any(is_discord(d) for d in url_dests):
        a = node.arguments if isinstance(node.arguments, dict) else {}
        auth = a.get("auth") if isinstance(a.get("auth"), dict) else None
        if auth and str(auth.get("scheme", "bearer")).lower() != "bot":
            out.append((PID_ACCOUNT, "a raw request to Discord with a non-bot credential scheme would act as a user "
                                     "account; only 'Authorization: Bot <token>' is ever sent", None))
        hdrs = a.get("headers") if isinstance(a.get("headers"), dict) else {}
        if any(re.sub(r"[^a-z0-9]", "", str(k).lower()) == "authorization" for k in hdrs):
            out.append((PID_ACCOUNT, "a raw request to Discord with a literal Authorization header; the credential "
                                     "is attached by the executor with the bot scheme only", None))
    if (opaque or raw_tool or modality in _BROWSER) and (
            _SELFBOT_TEXT.search(text_all) or ("discord" in text_all.lower() and _USER_TOKEN_TEXT.search(text_all))):
        out.append((PID_ACCOUNT, "the payload automates a user account (self-bot / user-token login)", None))
    if opaque and modality == C.Modality.CODE:
        for d in D.urls_in_text(text_all):
            if is_discord(d) and not _host_in(d.host, _DISCORD_DOC_HOSTS) and discord_api_path(d) is not None:
                out.append((PID_ACCOUNT, f"generated code calls Discord's API ({d.host}) directly; programmatic "
                                         f"Discord operations use the official bot integration", d))
                break
    seen, uniq = set(), []
    for f in out:
        k = (f[0], f[1])
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq
