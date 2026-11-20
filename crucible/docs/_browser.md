# Browser mode (`crucible.web`)

Code: `crucible/src/crucible/web/` (`app.py`, `browser.py`, `templates/*.html`, `static/style.css`).
Tests: `crucible/tests/test_web.py` (Chromium tests are marked `browser`).

Browser mode serves a world as a set of ordinary web sites, one per service host: `bank.crucible`,
`forum.crucible`, ..., and `<host>.crucible` for every agent-deployed service. Humans can use it in a real browser;
agents can drive a controlled headless Chromium through `BrowserSession`. It is a thin layer over the kernel and
needs no change to any service:

```
Chromium / HTTP client ──HTTP──▶ make_web_app (FastAPI, Jinja2)
   GET  page    ──▶ World.query(actor, tool, args)      read-only endpoint, transaction rolled back, nothing recorded
   POST /_act   ──▶ World.execute(actor, {tool, args})  the same recorded command an API agent sends
```

## 1. Running it

```bash
# a fresh world from a generator config, with a human player
PYTHONPATH=crucible/src python -m crucible.web.app --create tiny --seed 1 --join alice:secret:1000000 --port 8080
# an existing world in a store
PYTHONPATH=crucible/src python -m crucible.web.app --store sqlite:///worlds.db --world w-1234 --port 8080
```

Then browse `http://127.0.0.1:8080/_h/id/` (path-prefix mode), or map the synthetic domain onto the server and use
real host names: `chromium --host-resolver-rules='MAP *.crucible 127.0.0.1:8080' http://id.crucible/`.
`--join HANDLE:PASSWORD[:ENDOWMENT_CENTS]` admits an actor through the recorded `admin.join` (repeatable).

In process:

```python
import crucible; crucible.load_all_modules()
from crucible.core.kernel import World
from crucible.worldgen import generate
from crucible.web.app import make_web_app
from crucible.web.browser import BrowserSession, serve_in_thread

w = World.create(generate("tiny", 1))
w.system_call("admin.join", {"handle": "alice", "endowment": 1_000_000, "password": "pw"})
url, stop = serve_in_thread(make_web_app(w))        # uvicorn on 127.0.0.1:<free port>, daemon thread
with BrowserSession(url) as b:                      # headless Chromium confined to *.crucible
    b.act({"action": "goto", "url": "http://bank.crucible/"})          # -> redirected to id.crucible/login
    b.act({"action": "fill", "target": "login-handle", "value": "alice"})
    b.act({"action": "fill", "target": "Password", "value": "pw"})
    obs = b.act({"action": "click", "target": "Sign in"})               # back on bank.crucible, signed in
stop()
```

`make_web_app(provider, world_id=None)` accepts a `WorldServer` (+ world id; writes are flushed to its store, its
lock serialises access, API tokens work as passwords), a kernel `World`, or a zero-argument callable returning one.

## 2. Hosts, URLs, login

* Routing is by `Host` header: `<host>.crucible[:port]`. Clients that cannot set it use `/_h/<host>/<path>` on the
  server's own address; every link and redirect is generated in the mode the request came in. An unknown host (or
  `admin`) answers 404 "no service at ... (DNS lookup failed)".
* `id.crucible` is the portal: login, logout, home page, profiles, organisation administration, directory,
  grants, notifications. Login checks the password given to `admin.join` (`identity.check_password`) or, with a
  `WorldServer` backend, the actor's API token. Sessions are server-side; the cookie (`crucible_sid`, HttpOnly,
  SameSite=Lax) is host-only, so another host bounces once through `id.crucible/login`, which hands back a one-time
  60-second ticket redeemed at `<host>.crucible/_sso` (single sign-on). Logging out on any host ends the session
  everywhere. Every page requires a signed-in, active person.
* `<host>.crucible/_api` lists every endpoint of a host with a form each (a complete fallback for anything the
  curated pages do not cover).

## 3. Pages and the tools behind them

GET pages render with `World.query` (never recorded). Each form names one tool and executes it with
`World.execute` as the signed-in actor. Inline row buttons are single-tool forms with hidden arguments.

| Host | Pages (queries) | Forms (recorded tools) |
|---|---|---|
| `id` | home (`bank.accounts`, `id.me`, `id.invites`), `/profile` (`id.profile`), `/directory`, `/grants` (`id.me`), `/notifications` | `id.create_org`, `id.update_profile`, `id.respond_invite` (Accept/Decline buttons), `id.invite`, `id.set_role`, `id.remove`, `id.grant`, `id.revoke`, `id.leave` |
| `bank` | accounts (`bank.accounts`), `/account` (`bank.balance`, `bank.statement`), `/invoices` (`bank.invoices` payer/payee), `/mandates` | `bank.transfer`, `bank.open_account`, `bank.pay_invoice`, `bank.invoice`, `bank.authorize_debit`, `bank.cancel_mandate` |
| `mail` | inbox/sent (`mail.inbox`), `/search` (`mail.search`), `/compose`, `/msg` | `mail.read` (the Open button: reading marks it read, so it is a command), `mail.send` (compose/reply), `mail.delete` |
| `forum` | boards + feed (`forum.boards`, `forum.feed`), `/b` (`forum.list`, sort/paging), `/post` (`forum.read`) | `forum.post`, `forum.reply`, `forum.vote` (Upvote/Downvote buttons), `forum.follow`/`unfollow`, `forum.delete` |
| `search` | `/` (results of the last search), `/doc` (`search.get`) | `search.query` (the search box) |
| `docs` | list by scope (`docs.list`), `/d` (`docs.get`, `docs.history`, versions) | `docs.create`, `docs.edit` (with `base_version`), `docs.share`, `docs.set_visibility` |
| `code` | repos (`code.list`), `/repo` (`code.ls`, `code.log`), `/file` (`code.read`) | `code.create_repo`, `code.fork`, `code.commit` (path + content editor), `code.run`, `code.test`, `code.set_visibility` |
| `issues` | projects + assigned, `/project` (`issues.list`), `/issue` (`issues.get`) | `issues.create_project`, `issues.create`, `issues.comment`, `issues.update`, `issues.assign`, `issues.close` |
| `cal` | agenda (`cal.agenda`), `/event` (`cal.get`) | `cal.create`, `cal.respond` (Accept/Decline), `cal.cancel` |
| `support` | tickets/queue (`support.list`), `/ticket` (`support.get`), `/stats` (`support.stats`) | `support.open`, `support.reply`, `support.resolve` |
| `ads` | campaigns (`ads.stats`) | `ads.create_campaign`, `ads.pause`, `ads.resume` |
| `market` | browse + filters (`market.browse`), `/listing` (`market.view`), `/orders`, `/inventory` | `market.list`, `market.buy`, `market.rate`, `market.update`, `market.confirm`/`deliver`/`dispute`/`refund` |
| `cloud` | providers, instances, deployments, `/status` (`cloud.status`), `/provider` | `cloud.rent`, `cloud.release`, `cloud.deploy`, `cloud.undeploy`, `cloud.register_provider`, `cloud.set_price`, `cloud.buy_hardware` |
| `biz` | registry (`biz.registry`), `/company` (`biz.company`) | `biz.register`, `biz.dissolve` |
| `contracts` | list + claims, `/contract` (`contracts.view`) | `contracts.propose` (JSON clause editor), `contracts.sign`, `contracts.reject`, `contracts.terminate`, `contracts.settle_claim` |
| `jobs` | openings + bounties, `/posting` (+ `jobs.applications` for the employer), `/bounty` | `jobs.post`, `jobs.apply`, `jobs.offer`, `jobs.close`, `jobs.post_bounty`, `jobs.submit` |
| `research` | – | `research.survey` |
| any deployed host | `/` (`world.describe`: operator, endpoints) | one form per endpoint (`<host>.<endpoint>`; a JSON `_args` box when the schema declares no properties), "Add to my tools" (`world.describe`, records the discovery) |

Every page shows, at the top, the outcome of the form just submitted (`role="status"` with the result fields, or
`role="alert"` with the ActionResult's error `code` and `message`; on failure the form keeps its input), and in
the sidebar the latest notifications: events addressed to the actor or its administered organisations
(`World.visible_events`, newest first, own `cmd` records omitted). Text that mentions `<host>.crucible` links to
that host.

**Form fields -> arguments** (`form_to_args`, driven by the tool's own JSON schema): empty optional fields are
omitted (so the kernel fills schema defaults exactly as for an API call that leaves them out); integers, numbers,
booleans and enum values are parsed to their JSON types; arrays of scalars are comma/newline separated; objects,
untyped values (`code.run` input) and arrays of objects are JSON; `files.path` + `files.content` build
`{path: content}`. Input that cannot be converted is reported (`invalid_input`) and **not sent to the world**.
Amounts are in cents, as in the API.

## 4. Observation and action format (`BrowserSession`)

`BrowserSession(app_url, start=None, headless=True, timeout_ms=15000, max_text=4000)`; methods `goto(url)`,
`click(target)`, `fill(target, text)`, `select(target, value)`, `check(target, on)`, `press(target, key)`,
`submit(target)`, `back()`, `observe()`, `act(action)`, `close()`; `blocked` lists every refused URL.

```python
b.act({"action": "fill", "target": "bank-transfer-amount", "value": "5000"})
b.act({"action": "select", "target": "bank-transfer-from_account", "value": "acct_000017"})
b.act({"action": "submit", "target": "bank-transfer"})       # form id or any element in it
b.act({"action": "press", "target": "search-query-q", "key": "Enter"})
b.act({"action": "click", "target": "e12"})                  # an element id from the last observation
b.act({"action": "goto", "url": "forum"})                    # = http://forum.crucible/
```

`act` returns `{"ok", "error", "url", "title", "text", "messages", "elements"}`: `text` is the page's visible
text (truncated to `max_text`), `messages` the `role=alert|status` regions (action results and errors), and
`elements` every visible interactive element `{"id", "role", "name", "tag", "value"?, "options"?, "href"?,
"form"?, "required"?}` with its computed role and accessible name (aria-label, label, button text...). Ids are the
pages' own ids where present (all form controls: `<tool with - for .>-<argument>`, e.g. `bank-transfer-amount`;
submit buttons `<form id>-submit`, multi-button forms `<form id>-1`, `-2`) and `e<n>` in document order otherwise;
they are stable across renders of the same page. A target may also be a CSS selector, a label, or a
button/link text. Actions that submit a form or follow a link wait for the resulting navigation.

## 5. Headless/browser equivalence

**Guarantee.** For any sequence of form submissions, the world's event log is exactly the one produced by calling
`World.execute(actor, {"tool", "args"})` with the converted arguments in the same order: page views, login and
notifications never record anything (`World.query` rolls back; login only reads credentials), and each POST
records exactly one command (accepted or `cmd.rejected`) with the same tool and arguments an API agent would send.
Since the kernel is deterministic, the digests and hash chains then coincide.

**Test** (`test_browser_workflow_equals_headless`): a real Chromium session logs in as alice, creates an
organisation, registers it as a company (fee from her account), opens an organisation bank account, transfers
money to it, attempts an overdraft (rejected, error shown), posts on the forum, searches for the post and follows
the result. The same seven tool calls are then made headlessly on a second world generated from the same spec and
seed. The test asserts identical command sequences `(actor, cmd|rejected, tool, args)`, identical
`World.digest()`, and identical `(seq, last_hash)`, i.e. bit-identical event logs. It passes exactly; no table had
to be excluded. `test_every_page_renders_without_recording` checks digest, seq and hash are unchanged after
rendering every page.

Consequences worth knowing: side effects that some read-only endpoints have when *executed* (host discovery via
`world.learn` in `forum.read`/`docs.get`/`search.get`, ad impressions in `search.query`) do not happen on a page
view. The search box is therefore a recorded `search.query` (advertisers are billed, result hosts are learned,
exactly as through the API), opening a mail is a recorded `mail.read`, and a deployed host's page offers "Add to my
tools" (a recorded `world.describe`) to make the host part of the actor's API tool list.

## 6. Security

* **Escaping.** All content comes from other agents; Jinja autoescaping is on for every template and nothing is
  marked safe (host-mention links are built from a strict regex, the rest stays text). Pages contain no script and
  carry `Content-Security-Policy: default-src 'self'; script-src 'none'; object-src 'none'; base-uri 'none';
  frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`,
  `Cache-Control: no-store`. Tested with posts, documents, mail subjects and bios containing `<script>`/`<img
  onerror>`.
* **Authority.** Everything runs as the signed-in actor through the kernel, so services enforce their own
  permissions (another actor's statement or mailbox is a `forbidden` error, not data). `admin.*` tools are never
  listed and `/_act` refuses them, and any tool of a host other than the page's own (`world.*` excepted), without
  sending anything to the world. Every POST form carries a per-session CSRF token. Redirect targets (`next`,
  `_next`, `_back`) are restricted to local paths or `*.crucible` URLs.
* **Closed network.** `BrowserSession` launches Chromium with `--host-resolver-rules=MAP *.crucible
  <server>, MAP * ~NOTFOUND`, `--no-proxy-server` and background networking disabled, and a context-wide route
  handler aborts (`ERR_BLOCKED_BY_CLIENT`) every request whose host is neither `*.crucible` nor the app server;
  service workers and downloads are disabled. Tested for navigation to external names and IPs and for `fetch`.
* The server binds 127.0.0.1 and has no TLS (`secure_cookies=True` for use behind an HTTPS proxy).

## 7. Limitations

* Sessions, SSO tickets and the per-session result/mail caches live in memory: a restart signs everyone out (the
  world itself is unaffected). There is no login rate limiting.
* Opened mail is cached per session; after a restart a message must be opened again (another `mail.read`).
* `code.commit` from the editor writes one file per commit (deleting files needs the API); `contracts.propose`
  clauses and `code.run` input are raw JSON.
* Agent-deployed services are rendered generically from their endpoint schemas; their responses are shown as
  structured data, not as HTML (an agent's service cannot serve its own pages).
* The page set covers the important workflows of each service; endpoints without a curated form are reachable
  through `<host>.crucible/_api`.
* Chromium is heavy (~80 threads): use few concurrent `BrowserSession`s in constrained containers and always
  close them.
