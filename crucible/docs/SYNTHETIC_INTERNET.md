# The synthetic internet

Code: `crucible/src/crucible/services/`, `economy/`, `actors/`, `worlds/digital_civilization.py`, `web/`.
Tests: `tests/test_services_comm.py`, `test_economy.py`, `test_goals.py`, `test_web.py`.

Seventeen built-in services share one authoritative state (one identity system, one ledger, one permission
model, one event log). Agents reach them as tools (`<host>.<endpoint>`), through HTTP/MCP/OpenEnv, or through
real web pages in Chromium (browser mode); all three paths execute the same endpoint handlers.

| host | service | what is real about it |
|---|---|---|
| `id` | identity, organisations, roles, delegation | principals, membership, invites that must be accepted, capability grants with spending limits and expiry |
| `bank` | payments and banking | double-entry ledger, invoices, direct-debit mandates, statements; conservation invariant |
| `mail` | email/messaging | delayed delivery, bounces, org mailboxes, privacy |
| `forum` | social discussion | boards, posts, replies, votes, follows, attention-limited feed; indexed by search |
| `search` | search engine | BM25 over public documents × ranking signals (verified ratings), sponsored results; results are how hosts are discovered |
| `docs` | documents and public pages | versions, sharing, visibility; the standards institute publishes the need formats here |
| `code` | repositories and execution | content-addressed commits, licences, deps; sandboxed `code.run`/`code.test` with compute charges |
| `issues` | project management | projects bound to repositories, assignment that simulated developers act on |
| `cal` | calendars | invitations, responses, conflicts |
| `support` | customer support | tickets opened by real service failures, SLA statistics |
| `ads` | advertising | keyword campaigns, per-impression billing, budgets; impressions served to agents and to simulated consumers |
| `market` | marketplace | API listings (consumer demand), code licences, goods, services; escrow, arbitration, verified ratings |
| `cloud` | compute and hosting | providers, rented instances, deployments, DNS, metering, overload, billing, suspension, outages, hardware purchases |
| `biz` | company registry | registration, cap tables, fees, insolvency and bankruptcy |
| `contracts` | contracts | composable clauses enforced by the scheduler and by predicates over state |
| `jobs` | labour market | postings, applications, employment contracts, verified bounties |
| `research` | market research | paid, noisy consumer surveys |
| `goals` | private self-goals | agents' own experiments/subgoals checked by the kernel |
| `world` | kernel | `whoami`, `clock`, `describe(host)` |

## Required integrations (implemented and tested)

* A user registers a company (`id.create_org` → `biz.register` → `bank.open_account`), which owns a repository
  (`code.create_repo`); code from it is deployed (`cloud.deploy`) onto a rented instance (`cloud.rent`) whose
  hourly bill is debited from the company account; requests consume compute units; customers discover the service
  through search, ads, forum posts and word of mouth; their verified experience changes satisfaction, retention,
  ratings and tickets; revenue enters the company account; the company can hire (`jobs.*` → employment contract →
  membership), and employees can access permitted repositories and commit code; outages and overload trigger
  support tickets and lost revenue; bankruptcy terminates services, contracts and employment.
  (`tests/test_economy.py`, `tests/test_goals.py`, `examples/demo_lifetime.py`.)
* Agent-deployed services are first-class: discoverable, callable by any principal (as tools once discovered),
  priced per call, usable by other agents' programs (`crucible_sdk.call`), and listed for consumer demand.

## Discovery and dynamic tools

Built-in services are the "well-known" internet and always listed. A host deployed by any agent becomes a tool
for a principal once that principal learns of it: from search results, listings, forum posts, mail, docs or
`world.describe`. `World.available_tools(principal)` is therefore dynamic, and MCP clients receive
`notifications/tools/list_changed`.

## Extending the internet (admission of new modules)

New services are new kernel modules (code reviewed by humans), never agent-written code with kernel authority.
A module is admitted when: its endpoints have strict schemas; it moves money only via the bank; its appliers are
deterministic (the replay **and** regeneration tests pass on a scripted session using it); `check_invariants` is
implemented and empty after its tests; private data never appears in other principals' visible events; its web
pages escape content; and it does not touch evaluator or reward code. The communication services below were
added through exactly this procedure (`tests/test_services_comm.py`).

### Communication and information services

Code: `crucible/src/crucible/services/{search,ads,forum,mail,docs,calendar,issues,support}.py`.
Tests: `crucible/tests/test_services_comm.py`.

These eight modules are the information substrate of the simulated economy. They decide **who knows what**:
which services an agent has discovered (and therefore which tools it can call), which products simulated
consumers hear about, how firms coordinate, and how service failures become visible to operators. Each module
is a `@register_module` `Module` whose name equals its host (`search`, `ads`, `forum`, `mail`, `docs`, `cal`,
`issues`, `support`; `calendar` is registered as an alias of `cal`). All follow the module rules of
WORLD_KERNEL.md §3: handlers emit events, appliers are deterministic, iteration is sorted, randomness comes from
`ctx.rng` and is recorded in payloads, authorisation goes through `ident(world).authorize/allowed`, money moves
only through `bank.move`.

### Causal map

```
 forum.post / docs (public) / market, biz, jobs, cloud, code ──index()──▶ search_docs ◀── rank_signal hook
                                                                            │
 ads campaigns ──sponsored hook──▶ search.query ──▶ results ──learn(host)──▶ knowledge ──▶ available_tools
        ▲                               │
        └──── record_impressions ◀──────┘──── bank.move(advertiser → operator)     (pause on no funds)

 forum.read / mail.read / docs.get / search.get ──learn(host)──▶ knowledge        (word of mouth)
 mail.send ──latency──▶ mail.deliver ──▶ inbox ──on_mail_delivered──▶ NPC actors wake up
 cal.create ──on_calendar_invite──▶ NPCs;  issues.assign ──on_issue_assigned──▶ NPCs / workers
 service failures (cloud/market/consumers) ──open_ticket──▶ support ──resolve──▶ on_ticket_resolved ──▶ economy
 support.daily ──▶ sup_stats (public SLA)  ──▶ reputation / demand (read by economy modules)
```

Discovery rule: a host referenced in text as `<name>.crucible` is kept only if `world.hooks["is_host"](state,
name)` is true (or `name` is a built-in host); the result is stored in the artefact's `hosts` list at write time
(so replay never re-evaluates the hook). Seeing an artefact through `search.query`, `search.get`, `forum.read`,
`mail.read` or `docs.get` calls `world.learn(ctx, actor, "host", h)` for each of its hosts. Shared helpers:
`search.tokenize(text)`, `search.is_host(world, name)`, `search.extract_hosts(world, text, extra=())`.

### search (host `search`)

* **Data.** `search_docs[doc_id] = {id, kind, title, text (<= 8000 chars), owner, ref, hosts (sorted, <= 16), t
  (last index time), seq (event seq of last index), created}`; indexes `kind`, `owner`. Kinds in use: `post`,
  `doc`, `listing`, `company`, `job`, `service`, `repo`.
* **Internal API.** `index(ctx, doc_id, kind, title, text, owner, ref=None, hosts=())` (insert or replace; no-op
  if unchanged), `unindex(ctx, doc_id)` (both also available as module-level functions that are no-ops if search
  is not installed); pure `query(world, q, kind=None, limit=10) -> list[dict]` (sponsored first, then organic);
  `Search.search(world, q, kind, limit) -> (sponsored, organic, matched)`.
* **Ranking.** BM25 (k1 = 1.2, b = 0.75) over lowercase `[a-z0-9]+` tokens; title tokens weigh 2, host names 1.
  Score × `world.hooks["rank_signal"](world, record)` (default 1; <= 0 delists). Ties: newer `t`, then later
  `seq`, then id. Empty query = browse most recent (optionally of a kind). Up to 3 sponsored results from
  `world.hooks["sponsored"](world, q, kind, min(limit, 3))`, labelled `sponsored=True`, placed first.
* **Endpoints.** `search.query(q<=200, kind?, limit 1..20)` → `{results, matched}`; learns hosts of every
  returned result; charges sponsored results through `ads.record_impressions` and drops any that could not be
  charged. `search.get(id)` → full indexed text (learns hosts).
* **Events.** `search.index`, `search.unindex` (internal).
* **Runtime cache.** The inverted index lives in `world.runtime["search.index"]`. Appliers only mark doc ids
  dirty; before each query the cache reconciles dirty ids against the authoritative records by object identity,
  and keeps re-checking ids touched inside the current transaction until that transaction ends, so command
  rollbacks and savepoint rollbacks of nested calls cannot leave it stale. It is dropped in `on_load` (replay,
  load, fork). Maintained statistics are integers and scores accumulate in sorted term order, so results equal a
  from-scratch computation.
* **Invariants.** ids consistent; hosts sorted/unique/bounded; size limits; cache == from-scratch rebuild.

### ads (host `ads`)

* **Data.** `ads_campaigns[adc_*] = {advertiser, account, keywords (tokens), kind (target or null), bid_cents,
  daily_budget_cents, status active|paused, reason, spent_today, total_spent, impressions, landing {title, text,
  hosts, ref}, created, by}` (indexes `advertiser`, `status`); `ads_kw["<kw>|<campaign>"]` keyword index.
* **Endpoints.** `create_campaign(account, keywords<=20, bid_cents 1..10000, daily_budget_cents, title, text?,
  hosts<=5 (must be hosts), ref?, kind?)` — requires `bank.spend` on the account's owner (the advertiser) and
  funds for one impression; `pause`, `resume` (bank.spend; resume needs funds); `stats(campaign?)`
  (`analytics.read`, or all campaigns of your principals).
* **Hook provided.** `world.hooks["sponsored"](world, q, kind, limit)`: active campaigns with a keyword among the
  query tokens, matching target kind (or none), `spent_today + bid <= daily_budget`; ordered by bid, keyword hits,
  age; one per advertiser. Returns `{id: "ad:<cid>", campaign, kind, title, text, hosts, ref, owner, t, bid_cents}`.
* **Internal API.** `record_impressions(ctx, campaign_ids, n=1) -> charged ids`: charges `bid * k` now with
  `k = min(n, remaining budget, affordable)` via `bank.move(account → operator)` (`config["operator_account"]`,
  default `acct_fees`, kind `ad_impressions`, ref = campaign id). Campaigns that cannot pay are paused
  (`insufficient_funds` / `account_closed`). Consumers modules read campaign state directly for exposure.
* **Process.** `ads.daily` (every 86400 s from t = 86400) emits `ads.reset` for campaigns with spend today.
* **Events.** `ads.campaign` (advertiser, creator), `ads.status` (advertiser), `ads.impressions`, `ads.reset`
  (internal).
* **Invariants.** `total_spent == impressions * bid == Σ ad_impressions bank txns for the campaign`;
  `0 <= spent_today <= daily_budget`; keyword rows reference campaigns; no charges for unknown campaigns.

### forum (host `forum`)

* **Data.** `forum_boards[name] = {title, description, posts, created}` (genesis from `config["boards"]`,
  default general/dev/market/jobs/help); `forum_posts[fpost_*] = {board, author, title, body, tags, hosts, t,
  score, up, down, replies, last_activity, status visible|deleted}`; `forum_replies[freply_*] = {post, author,
  body, hosts, t}`; `forum_votes["<post>|<voter>"] = {value ±1}`; `forum_follows["<follower>|<target>"]` (target is
  a principal or `board:<name>`).
* **Internal API.** `post(ctx, author, board, title, body, tags=(), hosts=()) -> post_id` (simulated consumers
  post reviews with it; `hosts` are trusted extras), `reply(ctx, author, post_id, body, hosts=())`,
  `create_board(ctx, name, title, description)`, `feed(world, actor, limit)`.
* **Endpoints.** `boards`; `post(board, title, body, tags<=5, as_org?)` (as_org needs `forum.post`);
  `reply(post, body, as_org?)`; `read(post, replies<=100)` (learns hosts); `list(board, sort new|top|active,
  limit<=30, offset)`; `vote(post, ±1)` (one per principal, changeable, never on own posts or posts of an org
  you belong to); `follow/unfollow(target)`; `feed(limit<=30)`: hot score
  `(max(score,0)+1+0.5·replies)/(age_h+2)^1.5`, ×2 for followed principals, over the latest 50 posts of each
  followed principal, followed board and the 3 most active boards; `delete(post)` (author or `forum.post` on the
  author org; unindexes post and replies).
* **Search.** Posts and replies are indexed as kind `post` (replies: title `Re: …`, ref = post id).
* **Events.** `forum.board/post/reply/delete` public; `forum.vote` (voter); `forum.follow` (follower, target);
  `forum.unfollow` (follower).
* **Invariants.** score/up/down equal the votes; vote ids canonical (uniqueness); no self-votes; reply counters;
  board post counters.

### mail (host `mail`)

* **Data.** `mail_msgs[mail_*] = {from, to, subject, body, hosts, in_reply_to, t, delays, status {rcpt:
  pending|delivered|bounced}, dtime {rcpt: t}, kind mail|bounce}`; `mail_box["<msg>|<owner>|<folder>"] = {msg,
  owner, folder inbox|sent, read, t}`; `mail_unread[owner] = {n}`.
* **Internal API.** `send(ctx, sender, to, subject, body, in_reply_to=None) -> msg_id` (no authorisation;
  `sender` may be `sys`).
* **Endpoints.** `send(to 1..20, subject, body, in_reply_to?, from_org?)` (from_org needs `mail.send`) →
  `{msg, to, eta}`; `inbox(folder, unread_only, limit<=50, mailbox?)`; `read(msg, mailbox?)` (marks read, learns
  hosts, sender copies show per-recipient delivery status); `search(q, limit<=20)` over own mail;
  `delete(msg, folder, mailbox?)`. Organisation mailboxes are visible to owners/admins (`principals_of`) or with
  a `mail.read` grant.
* **Process.** `mail.deliver(msg, to)` at `t_send + delay`, delay = `round(mail_latency_s · U[0.5, 1.5))` (param
  `mail_latency_s`, default 60; 0 = next clock advance), drawn at send time and recorded in `mail.sent`. Bounce
  if the recipient is inactive (`recipient_inactive`), a dissolved organisation (`organisation_dissolved`), or
  with probability `mail_failure_p` (`transient_failure`, default 0). A bounce adds an "Undeliverable" notice
  (`kind=bounce`, from `sys`) to the sender's inbox. After delivery: `world.hooks["on_mail_delivered"](ctx, msg_id,
  recipient)`.
* **Events / visibility.** `mail.sent` (sender + acting principal, includes body); `mail.delivered` (recipient
  only; sender and subject, no body); `mail.bounced` (sender); `mail.read`, `mail.delete` (mailbox owner).
  Non-parties never see a message; reading one you cannot access is `not_found`.
* **Invariants.** pending recipients have their keyed delivery job (`mail|<msg>|<rcpt>`); delivery/bounce at or
  after sending; inbox entries only for delivered recipients; unread counters equal a recount.

### docs (host `docs`)

* **Data.** `documents[doc_*] = {owner, title, body, visibility private|org|public, version, created, updated,
  hosts, by}`; `doc_versions["<doc>|<v:06d>"]` (every version, full text); `doc_shares["<doc>|<principal>"] =
  {perm read|write}`.
* **Access.** admin: owner, owners/admins of an owning org, `docs.share` grantees; write: `write` shares, members of
  the owning org for `org`/`public` docs, `docs.write` grantees; read: public docs, `read` shares (a share to an
  org covers its members), `docs.read` grantees. Members of an owning org do not see its private docs unless
  shared. No access ⇒ `not_found` (existence hidden).
* **Internal API.** `create_doc(ctx, owner, title, body, visibility="public") -> doc_id`; `access(ctx, actor,
  doc_id, record) -> 0..3`.
* **Endpoints.** `create(title, body, visibility, org?)` (org needs `docs.write`; `org` visibility only for
  org-owned docs); `get(doc, version?)` (learns hosts); `history(doc)`; `list(scope mine|org|shared|public)`;
  `edit(doc, title?, body?, base_version?)` (optimistic concurrency → `conflict`); `share(doc, principal,
  read|write|none)`; `set_visibility(doc, visibility)`.
* **Search.** Public docs are indexed as kind `doc` (re-indexed on edit; unindexed when made non-public).
* **Events.** `docs.create/edit` public for public docs, otherwise owner + actor + share holders; `docs.share`
  (owner, grantee, actor); `docs.visibility` (owner, actor).
* **Invariants.** versions 1..n present and the latest equals the document; `org` visibility only on org-owned
  docs; **a document is in the search index iff it is public** (hidden-information guarantee).

### cal (host `cal`)

* **Data.** `cal_events[cev_*] = {organiser, creator, title, description, start, duration, attendees, status
  active|cancelled, created}`; `cal_invites["<event>|<attendee>"] = {response pending|accepted|declined, t}`.
* **Endpoints.** `create(title, start>=now, duration 60 s..14 d, attendees<=20 persons, description?, org?,
  allow_conflict)` (org needs `cal.write`; the creator attends, accepted; fails with `conflict` if it overlaps the
  creator's accepted events); `respond(event, accept, allow_conflict)` (accepting an overlapping event fails with
  `conflict` + `details.conflicts`); `cancel(event, reason?)` (creator or `cal.write` on the organiser);
  `agenda(from_t?, to_t?, include_declined)` (<= 90 days, marks overlaps); `get(event)` (parties only).
* **Hook.** `world.hooks["on_calendar_invite"](ctx, event_id, attendee)` for every invitee.
* **Events.** `cal.create`, `cal.respond`, `cal.cancel` — organiser, creator and attendees only.
* **Invariants.** invites reference events and attendees; valid responses; creator is an attendee; every attendee
  has an invite; positive durations.

### issues (host `issues`)

* **Data.** `iss_projects[proj_*] = {owner, name (unique per owner), repo, public, description, open, issues}`;
  `iss_issues[iss_*] = {project, title, body, status open|in_progress|closed, labels, assignee, creator, t,
  updated, closed_t, comments}`; `iss_comments[icmt_*]`.
* **Authorisation.** write = `issues.write` on the project owner (org members have it); read = public project, or
  `issues.write`/`issues.read`. Public projects: anyone may file issues and comment; an issue's creator may edit
  its title/body and close it. Private projects are `not_found` to outsiders.
* **Endpoints.** `create_project(name, owner?, repo?, public, description)`; `projects(owner?)`;
  `create(project, title, body, labels<=10, assignee?)`; `list(project, status|any, assignee?, label?,
  limit<=50)`; `assigned()`; `get(issue)`; `comment(issue, body)`; `assign(issue, assignee?)`;
  `update(issue, status?, labels?, title?, body?)`; `close(issue, comment?)`.
* **Hook.** `world.hooks["on_issue_assigned"](ctx, issue_id, assignee)` on assignment (create or assign).
* **Events.** `issues.project/create/update/comment` — public for public projects, otherwise owner, creator,
  assignee and actor.
* **Invariants.** per-project open/total counters equal recounts; `closed` ⇔ `closed_t`; comment counters;
  issues reference projects.

### support (host `support`)

* **Data.** `sup_tickets[tkt_*] = {host, owner, customer (principal or opaque ref like seg:12), subject, body,
  severity 1..4, status open|answered|resolved, opened, first_response, resolved, msgs, last_activity}`;
  `sup_msgs["<ticket>|<n>"] = {author, role staff|customer, body, t}`; `sup_owners[owner]` live counters (open,
  answered, resolved, total) and today counters; `sup_stats["<owner>|<day>"]` daily SLA rows.
* **Internal API.** `open_ticket(ctx, host, owner, customer, subject, body, severity=1) -> ticket_id` (used by
  modules that observe real service failures), `resolve(ctx, ticket_id, note="")`, `stats(state, owner)`.
* **Endpoints.** `open(host, subject, body, severity)` (owner from `world.hooks["host_owner"](state, host)`);
  `list(owner?, status, limit)` (customer view, or the operator queue with `support.reply`); `get(ticket)`;
  `reply(ticket, body)` (staff → `answered`, records first response; customer → `open`); `resolve(ticket, note)`
  (staff); `stats(owner, days<=30)` (public: live counters + daily history).
* **Hook.** `world.hooks["on_ticket_resolved"](ctx, ticket_id)`.
* **Process.** `support.daily` (every 86400 s from t = 86400) emits internal `support.stats` per owner with
  unresolved tickets or activity that day: open, answered, unresolved, median and max age of unresolved tickets,
  opened/resolved that day, mean first-response time.
* **Events.** `support.open/msg/status` — owner and customer (if a principal); `support.stats` internal.
* **Invariants.** live counters equal recounts; `resolved` ⇔ resolution time; first response not before opening;
  message counters.

### Hooks summary

| hook | provided by | called by |
|---|---|---|
| `sponsored(world, q, kind, limit)` | ads | search |
| `rank_signal(world, doc_record) -> float` | economy/reputation modules | search |
| `is_host(state, name) -> bool` | cloud/hosting modules | forum, mail, docs, ads (host extraction/validation) |
| `host_owner(state, host) -> principal` | cloud/hosting modules | support.open |
| `on_mail_delivered(ctx, msg_id, recipient)` | NPC actors | mail |
| `on_calendar_invite(ctx, event_id, attendee)` | NPC actors | cal |
| `on_issue_assigned(ctx, issue_id, assignee)` | NPC actors / workers | issues |
| `on_ticket_resolved(ctx, ticket_id)` | economy / consumers | support |

Hooks run inside the calling transaction; anything they change must be emitted as events, and hooks that
influence handler outcomes must be installed by a module's `install` so replayed and regenerated worlds have
them too.

## Browser mode



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

### 1. Running it

```bash

PYTHONPATH=crucible/src python -m crucible.web.app --create tiny --seed 1 --join alice:secret:1000000 --port 8080

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

### 2. Hosts, URLs, login

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

### 3. Pages and the tools behind them

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

### 4. Observation and action format (`BrowserSession`)

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

### 5. Headless/browser equivalence

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

### 6. Security

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

### 7. Limitations

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
