## Communication and information services

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
