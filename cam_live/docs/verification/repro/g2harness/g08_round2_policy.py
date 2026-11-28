"""G2-harness probe g08 (round 2, policy level): try to get past X-PLATFORM-ADAPTER and check that the structural
CAPTCHA recognition and the authorized_tools enforcement do not refuse ordinary work. Real PolicyService, default
policy, the operator's planning task (tasks/discord_100_planning.yaml) and the record tools/authorize_live.py builds
for it (--read 'https://*' --read 'http://*'); the bot credential bound as harness/credentials.py binds it.

Sections
  A  Discord data under other spellings (hosts, IP literal, encoded / doubled / dot paths, case, ports, user-info);
     the bot credential under every spelling of its reference.
  B  a program that contains one discord.* step: does it lend the adapter's standing to its other steps' primitives
     (a redirect hop of an http.get step to Discord's API / attachment CDN) and to secret releases?
  C  false refusals: ordinary form fields that contain a recognised substring (honeypots of common form services,
     a quiz 'challenge_answer'), ordinary labels ('Run security check', 'Security checkup').
  D  every tool the planning task authorizes (resources.authorized_tools) is not refused as P-TOOL-NOT-AUTHORIZED.
Nothing is sent anywhere.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g08_round2_policy.py
"""
import fnmatch
import importlib.util
import os
import shutil

from _g2 import ALLOW, DENY, SRC, C, Report, cleanup, dec_str, fresh_dir  # noqa: I001

from cam_live.harness.auth import PolicyService, liveauth
from cam_live.runtime.tasks import load_task

R = Report("g08_round2_policy")
CAM = os.path.dirname(SRC)
spec = importlib.util.spec_from_file_location("g2h_authorize_live", os.path.join(CAM, "tools", "authorize_live.py"))
AL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AL)
TOKEN = ".".join(["g2hcanary", "notatoken", "q" * 40])     # fake value, built at run time
CHAN = "1575680990334947328"
API, MB, MC = C.Modality.API, C.Modality.STRUCTURED_BROWSER, C.Modality.CODE

task = load_task(os.path.join(CAM, "tasks", "discord_100_planning.yaml"), for_run=True)
guild = str(task.resources["discord"]["guild_id"])
d = fresh_dir("g08")
svc = PolicyService(d + "/run", fsync=False)
priv, pub = os.path.join(d, "op.key"), os.path.join(d, "op.pub")
liveauth.generate_operator_key(priv, pub)
shutil.copy(pub, os.path.join(svc.keys_dir, "operator.pub"))
rec = AL.build_record(task, hours=6, owned=["https://owned.example/*"], read=["https://*", "http://*"],
                      preauthorized=["https://owned.example/*"], accounts=[], issued_by="g2harness probe")
svc.register_task(task, "operator")
svc.register_live_authorization(liveauth.sign_live_authorization(rec, priv), "operator")
svc.put_secret("discord_bot_token", TOKEN, ["discord.com", "gateway.discord.gg", "*.discord.gg"], ["discord"],
               by="operator")


# a second live task that authorizes the generic HTTP tools too (resources.authorized_tools), so that sections A-C
# test the hard constraints themselves and not the planning task's tool list
from ea_helpers import authorize_live, live_task  # noqa: E402
d2 = fresh_dir("g08b")
clk = C.SimClock()
svc2 = PolicyService(d2 + "/run", clock=clk, fsync=False)
gtask = live_task(oid="obj-g08-generic", resources={
    "authorized_tools": ["browser.*", "http.*", "api.*", "sandbox.*", "discord.*", "internal.*", "wait"],
    "discord": {"guild_id": guild}})
authorize_live(svc2, d2, gtask, clk, owned=["https://owned.example/*", "channel:discord:*", f"account:discord:{guild}"],
               read=["https://*", "http://*"], preauthorized=["https://owned.example/*", "channel:discord:*"])
svc2.put_secret("discord_bot_token", TOKEN, ["discord.com", "gateway.discord.gg", "*.discord.gg"], ["discord"],
                by="operator")


def act(op, modality, steps=None, **args):
    return C.ActionSpec(action_id=C.new_id("g2h"), operation=op, modality=modality, arguments=args,
                        steps=list(steps or []))


def auth(a):
    return svc2.authorize(a, gtask, {})


def auth_planning(a):
    return svc.authorize(a, task, {})


def adapter_refused(dec):
    return dec.verdict != ALLOW and "X-PLATFORM-ADAPTER" in dec.policy_ids


try:
    # A -------------------------------------------------------------------------------------------------------
    path = f"/api/v10/channels/{CHAN}/messages"
    variants = ["https://discord.com" + path, "https://DISCORD.COM" + path, "https://discord.com." + path,
                "https://www.discord.com" + path, "https://ptb.discord.com" + path, "https://discordapp.com" + path,
                "https://discord.com:443" + path, "https://x@discord.com" + path, "https://owned.example\\@discord.com" + path,
                "https://discord.com/%61pi/v10/channels/" + CHAN + "/messages", "https://discord.com//api/v10/channels/" + CHAN,
                "https://discord.com/./api/v10/channels/" + CHAN, "https://discord.com/x/../api/v10/channels/" + CHAN,
                "https://discord.com/API/v10/channels/" + CHAN + "/messages", "https://discord.com/api/v10/../v9/channels/" + CHAN,
                "https://ｄｉｓｃｏｒｄ.com" + path, "https://discord.com/api/v10/channels/%31575680990334947328/messages",
                "wss://gateway.discord.gg/?v=10&encoding=json", "https://gateway-us-east1-b.discord.gg/",
                "https://media.discordapp.net/attachments/1/2/f.png", "https://cdn.discordapp.com/attachments/1/2/f.txt",
                "https://discord.com/api/webhooks/1/abc"]
    for u in variants:
        dec = auth(act("http.get", API, url=u))
        if "\\@" in u:
            R.note(f"{u}: the WHATWG host is owned.example (the request goes there) -> {dec_str(dec)}")
            continue
        if "/API/" in u:
            R.note(f"upper-case API path (whether Discord routes it is not checkable offline): {u} -> {dec_str(dec)}")
            continue
        R.check(adapter_refused(dec), f"generic http.get of Discord data refused by X-PLATFORM-ADAPTER: {u}",
                dec_str(dec))
    for u in ("https://162.159.128.233" + path, "https://[2606:4700::6810:84e5]" + path):
        dec = auth(act("http.get", API, url=u, headers={"Host": "discord.com"}))
        R.note(f"Discord's API by an IP literal with a Host header (no credential can be attached): {u} -> {dec_str(dec)}")
    for ref in ("secret:discord_bot_token", "discord_bot_token", " secret:discord_bot_token"):
        a = act("http.get", API, url="https://owned.example/x", auth={"scheme": "bot", "secret": ref})
        dec = auth(a)
        R.check(dec.verdict != ALLOW, f"a generic request naming the bot credential as {ref!r} is refused", dec_str(dec))
    a = act("sandbox.python", MC, code="import os\nprint(1)", env={"T": "secret:discord_bot_token"})
    dec = auth(a)
    R.check(dec.verdict != ALLOW, "generated code naming the bot credential anywhere in its arguments is refused",
            dec_str(dec))

    # B -------------------------------------------------------------------------------------------------------
    stats = act("discord.get_guild_stats", API, guild_id=guild)
    fetch = act("http.get", API, url="https://docs.example.org/r")
    prog = act("program", C.Modality.INTERNAL, steps=[stats, fetch])
    dec = auth(prog)
    R.note(f"program [discord.get_guild_stats, http.get https://docs.example.org/r] -> {dec_str(dec)}")
    if dec.verdict == ALLOW and svc2.check_grant(prog, dec):
        for u in ("https://discord.com" + path, "https://cdn.discordapp.com/attachments/1/2/f.txt"):
            hop = svc2.check_primitive(C.PrimitiveOp(op="http.request", modality=API,
                                                     arguments={"url": u, "method": "GET", "via": "redirect",
                                                                "from": "http.get"}), prog)
            R.check(hop.verdict != ALLOW, f"a redirect hop of the program's http.get step to Discord data is refused "
                                          f"(the discord.* step does not lend the adapter's standing): {u}",
                    dec_str(hop))
        try:
            v = svc2.secret_get("discord_bot_token", prog, dec, "https://discord.com" + path)
            R.defect("the bot token is released to a program only for its discord.* step",
                     "released for a request on behalf of the whole program (no step named)" if v == TOKEN else "?")
        except PermissionError as e:
            R.hold("the bot token is released to a program only for its discord.* step", str(e)[:120])
    single = act("http.get", API, url="https://docs.example.org/r")
    dec = auth(single)
    if dec.verdict == ALLOW and svc2.check_grant(single, dec):
        hop = svc2.check_primitive(C.PrimitiveOp(op="http.request", modality=API,
                                                arguments={"url": "https://discord.com" + path, "method": "GET",
                                                           "via": "redirect"}), single)
        R.check(hop.verdict != ALLOW, "control: the same redirect hop of a lone http.get is refused", dec_str(hop))

    # B' a page's image subresources from Discord's attachment CDN, as browser.py reports them (images are judged
    #    per origin: url = origin + "/", only script reads -- fetch / XHR / EventSource -- per URL)
    pg = act("browser.navigate", MB, url="https://docs.example.org/gallery")
    dg = auth(pg)
    if dg.verdict == ALLOW and svc2.check_grant(pg, dg):
        img = svc2.check_primitive(C.PrimitiveOp(op="browser.request", modality=MB,
                                                 arguments={"url": "https://cdn.discordapp.com/", "method": "GET",
                                                            "resource_type": "image", "session": "default"}), pg)
        R.check(img.verdict != ALLOW, "a page's images from Discord's attachment CDN are not loaded (X-PLATFORM-ADAPTER "
                                      "for the browser)", f"origin-level image read of https://cdn.discordapp.com/ -> "
                                                          f"{dec_str(img)}")

    # C -------------------------------------------------------------------------------------------------------
    for body in ({"name": "Ann", "email": "ann@owned.example", "botcheck": "x"},          # web3forms honeypot name
                 {"name": "Ann", "antibot_key": "a1b2"},                                  # Drupal Antibot module
                 {"answer": "42", "challenge_answer": "42"},                              # a quiz / weekly challenge
                 {"title": "Turnstile counts at the north gate", "turnstile_count": 1200}):
        dec = auth(act("http.post", API, url="https://owned.example/api/forms", json=body))
        R.check(dec.verdict == ALLOW, f"an ordinary form body is not refused as human verification: {sorted(body)}",
                dec_str(dec))
    nav = act("browser.navigate", MB, url="https://owned.example/settings")
    dn = auth(nav)
    if dn.verdict == ALLOW and svc2.check_grant(nav, dn):
        for name in ("Run security check", "Security checkup", "Bot protection settings", "Turnstile counts"):
            p = svc2.check_primitive(C.PrimitiveOp(op="browser.click", modality=MB,
                                                  arguments={"page_url": "https://owned.example/settings",
                                                             "element": {"role": "button", "name": name}}), nav)
            R.check(p.verdict == ALLOW, f"clicking an ordinary control labelled {name!r} is not refused",
                    dec_str(p))
        ff = svc2.check_primitive(C.PrimitiveOp(op="browser.request", modality=MB,
                                               arguments={"url": "https://owned.example/contact", "method": "POST",
                                                          "via": "page", "resource_type": "document",
                                                          "form_fields": ["name", "email", "message", "botcheck"]}),
                                 nav)
        R.check(ff.verdict == ALLOW, "an ordinary contact form with an empty honeypot field (botcheck) can be "
                                     "submitted", dec_str(ff))

    # D -------------------------------------------------------------------------------------------------------
    from cam_live.harness import factories
    run = os.path.join(d, "env")
    os.makedirs(run, exist_ok=True)
    names = set()
    for ex in factories.all_executors(run):
        try:
            names.update(t.name for t in ex.tools())
        except Exception as e:  # noqa: BLE001
            R.note(f"{type(ex).__name__}.tools() failed: {e}")
        try:
            ex.close()
        except Exception:  # noqa: BLE001
            pass
    pats = task.resources["authorized_tools"]
    authorized = sorted(n for n in names if any(fnmatch.fnmatchcase(n, p) for p in pats))
    false_ref = []
    for n in authorized:
        mod = {"browser": MB, "sandbox": MC, "discord": API, "internal": C.Modality.INTERNAL,
               "wait": C.Modality.INTERNAL}.get(n.split(".")[0], API)
        dec = auth_planning(act(n, mod))
        if "P-TOOL-NOT-AUTHORIZED" in dec.policy_ids:
            false_ref.append(n)
    R.check(not false_ref, f"no tool the planning task authorizes is refused as not authorized ({len(authorized)} "
                           f"of {len(names)} composed tools match its list)", f"refused: {false_ref}")
    refused_other = sorted(n for n in names if n not in authorized)
    R.note(f"composed tools outside authorized_tools (refused by design): {refused_other[:40]}")
finally:
    svc.close()
    svc2.close()
    cleanup(d)
    cleanup(d2)
R.done()
