"""Mandatory duplicate-fact check over every table of every module (Crucible 5.0, D-21; MASTER_SPEC Part 2.3 "world
modules should not maintain incompatible duplicate versions of shared facts").

The 5.0 check of `primitives.check_no_duplicates` saw a module only if it *declared* `shared_facts`, so it was
vacuous (audit A6). This module scans what modules actually create - every state table (`declare_table`), every
column table and every column a module adds to one (`crucible.substrate.columns`), and the dense arrays a module
keeps (4.0 background populations) - and classifies each of them:

* **owner** - the one table of a shared kind (`primitives.OWNERS`);
* **reviewed** - listed in `REVIEWED` with its role: `part` (detail rows of another classified table, e.g. a
  community's votes), `extension` (module-private attributes keyed by an owner's ids, e.g. a company's equity keyed
  by the organisation), `private` (not a shared fact; the entry says why the name looked like one) or `duplicate`
  (a second copy of a shared kind's facts);
* **unreviewed** - a table whose name carries the signature of a shared kind (`SIGNATURES`) but that nobody has
  classified: a problem (the check fails until workstream A reviews it);
* anything else is module-private by default (no signature of a shared kind).

Every `duplicate` must appear in `KNOWN_DUPLICATES` - the explicit, reviewed allow-list with the decision, the
target owner after migration, the owning workstream, the migration and an expiry date - or the check fails. An
expired entry fails too. Nothing is ignored silently: known duplicates present in a world are reported with how
many facts each copy actually holds (the data-level view: e.g. 292 forum follows vs 0 social-graph edges).

* **legacy** - (the migration target for 4.0 tables, which cannot be retired: 4.0 worlds must stay byte-identical)
  a 4.0 table that holds a kind's facts only in worlds WITHOUT the kind's owner; where the owner is installed, the
  4.0 module writes through the owner instead. The rule is enforced on data in every world (`world_problems`): a
  legacy table that holds facts while the kind's owner is in the world is a problem (split facts). A legacy entry
  names the test that shows its module delegating (`verified_by`); `audit` fails for an entry without one. A legacy
  table does not make its kind a known duplicate (no world can hold two versions of the facts).

* `world_problems(world)` / `world_report(world)` - over the modules of one world (used by
  `primitives.problems` / `check_no_duplicates`).
* `scan_registry()` / `audit()` - over EVERY registered kernel module, each installed on its own (with its
  dependencies) so that what it declares is attributed to it alone.
"""
from __future__ import annotations

import datetime
import re

import numpy as np

KINDS_WITH_SIGNATURES = ("identity", "population", "organization", "product", "listing", "subscription",
                         "relationship", "community", "institution", "contract", "employment", "need")

#: name signatures of the shared kinds that modules are most likely to copy (audit A6)
SIGNATURES: dict[str, str] = {
    "identity": r"person|people|user|individual|profile|customer|voice",
    "population": r"crowd|consumer|patron|worker|npc|mind|population|^pop$|agents?$",
    "organization": r"(^|_)orgs?($|_)|compan|firm|venture|business|employer",
    "product": r"product|catalog|(^|_)items?($|_)",
    "listing": r"listing|offer",
    "subscription": r"subscri|(^|_)subs?($|_)|member|nlsub|tier",
    "relationship": r"follow|friend|(^|_)ties?($|_)|block|graph|(^|_)edges?($|_)|contact",
    "community": r"communit|board|club|group|cmember",
    "institution": r"constitution|charter|institution|(^|_)inst_|(^|_)gov|office|ballot|vote",
    "contract": r"contract|commitment|(^|_)deals?($|_)|agreement|mandate|claim",
    "employment": r"employ|(^|_)jobs?($|_)|labor|labour|posting|hire|applica",
    "need": r"(^|_)needs?($|_)",
}

_P, _X, _V, _D, _L = "part", "extension", "private", "duplicate", "legacy"


def _r(kind, role, why, of=None, verified_by=None):
    d = {"kind": kind, "role": role, "why": why}
    if of:
        d["of"] = of
    if verified_by:
        d["verified_by"] = verified_by
    return d


#: classification of every table that carries a signature (or is a known copy) and is not its kind's owner.
#: Keys: "<module>.<table>" (state table), "<module>.cols:<table>" (the columns a module declares on a column
#: table), "<module>.cols:<table>.<column>" (one column), "<module>.arrays:<prefix>" (dense state arrays).
REVIEWED: dict[str, dict] = {
    # ---------------------------------------------------------------- identity / population
    "identity.members": _r("organization", _P, "membership and roles of organisations", "identity.orgs"),
    "social.social_profiles": _r("identity", _X, "profile text and links of a principal, keyed by principal"),
    "social_ids.social_node_ids": _r("identity", _X, "social node index of principals and population rows (an "
                                                     "index into the owners' ids; holds no facts of its own)"),
    "social_ids.social_nodes": _r("identity", _X, "social node records (principal or population row) for the "
                                                  "column edge tables"),
    "aud.cols:pop": _r("population", _X, "audience attributes of each individual (keyed by population row)"),
    "aud.cols:aud_creators": _r("identity", _X, "audience aggregates per creator principal (pop_index link)"),
    "feed.cols:pop": _r("population", _X, "feed inbox and topic views of each individual (population row)"),
    "community.cols:cmember": _r("community", _P, "community memberships (columnar)", "community.communities"),
    "crowd.cols:pop": _r("population", _D, "the social stand-in population (refuses to coexist with pop)"),
    "crowd.crowd_creators": _r("population", _P, "creators of the stand-in population", "crowd.cols:pop"),
    "consumers.arrays:c_": _r("population", _D, "4.0 anonymous consumers (dense arrays, pooled household)"),
    "consumers.cons_voices": _r("population", _P, "persona accounts of the 4.0 consumers", "consumers.arrays:c_"),
    "consumers.cons_offers": _r("listing", _P, "offerings the 4.0 consumers know", "consumers.arrays:c_"),
    "consumers.cons_summary": _r(None, _V, "4.0 consumer demand statistics"),
    "patrons.patrons": _r("population", _D, "commerce customers (scripted NPC class with fixed needs)"),
    "npcs.npc_workers": _r("population", _D, "4.0 stochastic NPC workers (persons with their own policy state)"),
    "npcs.npc_firms": _r("organization", _X, "policy state of NPC incumbent firms, keyed by their organisation"),
    "npcs.npc_lenders": _r("organization", _X, "policy state of NPC lenders, keyed by their organisation"),
    "npcs.npc_decisions": _r(None, _V, "log of NPC policy decisions"),
    "society.minds": _r("population", _D, "4.0 society minds (simulated people outside the population)"),
    "labor.labor_workers": _r("population", _D, "simulated workers with skills and reservation wages"),
    "sci.sci_npc": _r("population", _D, "simulated researchers of the science domain"),
    "adopt.adopt_people": _r("population", _D, "need-driven capability adopters (innovation)"),
    "adopt.adopt_decisions": _r(None, _V, "adoption decision log of the adopters", "adopt.adopt_people"),
    "market.api_customers": _r("listing", _P, "customers of API plans", "market.listings"),
    "bank.invoices": _r("money", _P, "invoices (money claims) held by the bank", "bank.txns"),
    "pop.cols:pop.employer": _r("employment", _D, "200 abstract integer employers with random job loss/finding"),
    "pop_adaptive.pop_groups": _r(None, _V, "neural policy-sharing groups (parameter sharing, not communities)"),
    "pop_strategic.pop_strat_groups": _r(None, _V, "neural policy-sharing groups of strategic participants"),
    "pop_strategic.pop_strat_members": _r(None, _V, "membership of strategic participants in policy groups"),
    "popweb.cols:res": _r(None, _V, "web resources the population fetched (page cache, visit statistics)"),
    "pop_social.cols:pop": _r("population", _X, "content-creation counters of each individual (items written, "
                                                "views last seen, last item), keyed by population row"),
    # columns of the population table scanned one by one (wave 3, A): facts of other kinds kept per individual
    "pop.cols:pop.contacts": _r("relationship", _D, "who each individual knows (contact rows, read by word of mouth "
                                                    "in civgoods/work): a fourth tie graph next to forum follows, the "
                                                    "social graph and the audience's follower sets"),
    "pop.cols:pop.contact_trust": _r("relationship", _P, "trust per contact", "pop.cols:pop.contacts"),
    "pop.cols:pop.communities": _r("community", _D, "boards each individual takes part in (forum board indices), "
                                                    "a membership record of its own next to community.cols:cmember"),
    "pop.cols:pop.policy_group": _r(None, _V, "index of the neural policy-sharing group (parameter sharing)"),
    "pop.cols:pop.wallet_cents": _r("money", _X, "income accrued from the exogenous economy and not yet deposited "
                                                 "(a claim on acct_external, not ledger money): an individual pays "
                                                 "only after money_deposit moves it into its own bank account "
                                                 "(bank.move), so no payment bypasses the ledger"),
    "pop_social.cols:pop.soc_items": _r("content_artifact", _X, "how many pieces the individual published "
                                                                "(counter over its catalogue items)"),
    "pop_social.cols:pop.soc_last_item": _r("content_artifact", _X, "catalogue index of the individual's latest "
                                                                    "piece"),
    "aud.cols:pop.aud_c_item": _r("content_artifact", _X, "per-individual pointer into the catalogue (audience "
                                                          "accounting state)"),
    "aud.cols:pop.aud_i_item": _r("content_artifact", _X, "per-individual pointer into the catalogue (audience "
                                                          "accounting state)"),
    "aud.cols:aud_creators.followers": _r("relationship", _D, "follower sets of creators kept by the audience "
                                                              "engine (bitsets fo:<creator>) and their counts: a "
                                                              "follow graph of its own"),
    **{f"aud.cols:{t}.{c}": _r("relationship", _P, "counts of follows decided in audience accounting",
                               "aud.cols:aud_creators.followers")
       for t, c in (("aud_creators", "follows"), ("aud_items", "follows"), ("aud_items", "pub_follows"),
                    ("aud_items", "stg_follows"))},
    # ---------------------------------------------------------------- content
    "content.content": _r("content_artifact", _X, "content items of the catalogue (creator, platform, artifact "
                                                  "REFs)"),
    "aud.cols:aud_items": _r("content_artifact", _X, "audience accounting per catalogue item (exposures, "
                                                     "viewers, completions), keyed by catalogue index"),
    "aud.aud_item_info": _r("content_artifact", _D, "creator, platform, kind, duration and title of content items "
                                                    "copied from the catalogue at indexing"),
    "space.space_items": _r("content_artifact", _X, "recorded shared-space representations of artifacts (vector, "
                                                    "chunks, intrinsic features) keyed by artifact REF + encoder: "
                                                    "derived from the bytes, never a second copy of the artifact"),
    "space_read.space_read_items": _r(None, _V, "the fixed reading list of the space_read measurement module "
                                                "(catalogue index -> artifact REF; no shared facts)"),
    # ---------------------------------------------------------------- organization
    "business.companies": _r("organization", _X, "company registration and equity of an organisation"),
    "business.shares": _r("asset", _X, "equity shares of companies (4.0 asset class owner)"),
    "ventures.ventures": _r("organization", _X, "business lifecycle (books, wind-down) of an organisation"),
    "firms.firms": _r("organization", _X, "policy state (goal, band, books) of Tier 2 autonomous firms, keyed by "
                                          "their identity organisation"),
    "firms.firms_decisions": _r(None, _V, "decision log of the autonomous firms", "firms.firms"),
    "firms.firms_sales": _r(None, _V, "sales of the firms' internal work to the rest of the world", "firms.firms"),
    "fleet.fleet_jobs": _r(None, _V, "batch compute jobs of the infrastructure world's tenants (workloads, not "
                                     "employment)"),
    # ---------------------------------------------------------------- product / listing / subscription
    "products.prod_offers": _r("product", _P, "price plans of a product", "products.products"),
    "products.prod_entitlements": _r("product", _P, "access entitlements to products", "products.products"),
    "products.prod_purchases": _r("product", _P, "purchases of products", "products.products"),
    "products.prod_ratings": _r("product", _P, "ratings by paying customers", "products.products"),
    "products.prod_releases": _r("product", _P, "releases of a product", "products.products"),
    "products.prod_use_days": _r("product", _P, "measured use per day", "products.products"),
    "products.prod_use_holders": _r("product", _P, "measured use per holder", "products.products"),
    "assets.asset_offers": _r("asset", _P, "offers to sell an asset", "assets.assets"),
    "industry.offers": _r("listing", _D, "4.0 industrial supply offers of goods at facilities"),
    "xchg.xchg_listings": _r("listing", _D, "listings of launched products on participant-run exchanges"),
    "xchg.xchg_exchanges": _r(None, _V, "participant-run exchanges (venues, take rates)"),
    "xchg.xchg_orders": _r("listing", _P, "orders on exchange listings", "xchg.xchg_listings"),
    "xchg.xchg_ratings": _r("listing", _P, "ratings of exchange orders", "xchg.xchg_listings"),
    "subs.sub_plans": _r("subscription", _P, "subscription plans of a product", "subs.sub_subs"),
    "subs.sub_coupons": _r("subscription", _P, "coupons of subscription plans", "subs.sub_subs"),
    "subs.sub_invoices": _r("subscription", _P, "subscription invoices", "subs.sub_subs"),
    "creators.creator_members": _r("subscription", _D, "creator memberships: monthly charges, lapse, cancel"),
    "creators.creator_tiers": _r("subscription", _P, "creator membership tiers", "creators.creator_members"),
    "letters.cols:nlsub": _r("subscription", _D, "newsletter subscriptions (columnar)"),
    "letters.newsletters": _r("content_artifact", _X, "newsletters (publications of a creator)"),
    "letters.newsletter_issues": _r("content_artifact", _X, "issues of a newsletter (content items)"),
    "letters.newsletter_index": _r("content_artifact", _X, "index of newsletters"),
    # ---------------------------------------------------------------- relationship / community
    "social.cols:follow": _r("relationship", _D, "the v5 social graph's follow edges (columnar)"),
    "social.cols:share": _r(None, _V, "shares of content items (distribution events, not ties)"),
    "msg.msg_blocks": _r(None, _V, "per-pair messaging blocks (a messaging setting, not a social tie)"),
    "msg.msg_members": _r(None, _V, "participants of a conversation"),
    "community.communities": _r("community", _D, "v5 communities (membership, moderation, governance)"),
    **{f"community.community_{x}": _r("community", _P, f"community {x}", "community.communities")
       for x in ("events", "history", "index", "modlog", "proposals", "reports", "rsvps", "votes")},
    "forum.forum_posts": _r("content_artifact", _X, "forum posts (discussion content of boards)"),
    "forum.forum_replies": _r("content_artifact", _X, "forum replies"),
    "forum.forum_votes": _r(None, _V, "up/down votes on forum posts (content signals, not institutions)"),
    "popweb.pop_boards": _r("community", _X, "the population's view of forum boards (keyed by board)"),
    "dev.dev_votes": _r(None, _V, "votes on developer posts (content signals)"),
    "kb.kb_votes": _r(None, _V, "votes on knowledge-base posts (content signals)"),
    "launch.launch_votes": _r(None, _V, "votes on product launches (content signals)"),
    "jobs.bounties": _r("employment", _P, "bounties of the 4.0 job board", "jobs.postings"),
    "caps.cap_deps": _r("tool_definition", _P, "dependencies between capabilities", "caps.capabilities"),
    "industry.facilities": _r("resource_inventory", _X, "facilities that hold stock (resource sites)"),
    "needs.needs_edges": _r("need", _P, "relations of the needs graph", "needs.needs_nodes"),
    "needs.needs_open": _r("need", _P, "open needs", "needs.needs_nodes"),
    "needs.needs_requests": _r("need", _P, "requests attached to needs", "needs.needs_nodes"),
    # ---------------------------------------------------------------- institution
    "gov.proposals": _r("institution", _P, "governance proposals", "gov.constitutions"),
    "gov.votes": _r("institution", _P, "governance votes", "gov.constitutions"),
    "gov.gov_spend": _r("institution", _P, "governed spending", "gov.constitutions"),
    "inst.inst_charters": _r("institution", _D, "v5 adaptable institutions (charters)"),
    **{f"inst.inst_{x}": _r("institution", _P, f"institution {x.replace('_', ' ')}", "inst.inst_charters")
       for x in ("ballots", "contributions", "decisions", "delegations", "merge_offers", "office_spend", "offices",
                 "partners", "projects", "tasks", "unit_members", "units")},
    # ---------------------------------------------------------------- contract / employment
    "contracts.claims": _r("contract", _P, "claims of creditors after a breach", "contracts.contracts"),
    "contracts.contract_parties": _r("contract", _P, "parties of a contract", "contracts.contracts"),
    "bank.mandates": _r("account", _P, "standing payment authorisations held by the bank", "bank.accounts"),
    "code.commits": _r("software_repository", _P, "commits of repositories", "code.repos"),
    "commit.commitments": _r("contract", _D, "escrowed commitments (agreements paid on delivery)"),
    "commit.commit_offers": _r("contract", _P, "negotiation of commitments", "commit.commitments"),
    "deal.deals": _r("contract", _P, "negotiation threads whose acceptance creates contracts", "contracts.contracts"),
    "rig.rig_claims": _r(None, _V, "4.0 mechanism-rig result claims (evaluator-side records)"),
    "jobs.postings": _r("employment", _D, "4.0 job board postings (gig work paid by bounty submissions)"),
    "jobs.applications": _r("employment", _P, "applications to 4.0 postings", "jobs.postings"),
    "labor.labor_posts": _r("employment", _P, "labour-market postings (recruitment)", "contracts.contracts"),
    "labor.labor_apps": _r("employment", _P, "applications in the labour market", "contracts.contracts"),
    "labor.labor_offers": _r("employment", _P, "salary offers; acceptance creates the employment contract",
                             "contracts.contracts"),
    "labor.labor_perf": _r("employment", _P, "verified work record of workers", "contracts.contracts"),
    "industry.labor_days": _r(None, _V, "industrial labour hours per facility and day (production accounting)"),
    "cal.cal_invites": _r(None, _V, "calendar invitations"),
    "identity.invites": _r("organization", _P, "invitations to organisations", "identity.orgs"),
}

#: the explicit, reviewed allow-list of duplicates that exist today. Each names the decision, the owner the facts
#: move to, the workstream that migrates them and the expiry after which the check fails again.
_WAVE2 = "2027-03-31"
KNOWN_DUPLICATES: dict[str, dict] = {
    # D-15: one population
    **{k: {"target_owner": "pop.cols:pop", "decision": "D-15", "workstream": ws, "expires": _WAVE2,
           "migration": mig} for k, ws, mig in (
        ("crowd.cols:pop", "B+C", "retire `crowd`; the social engine registers its activities on `pop` "
                                  "(population.register_activity) and gates C/D/I run on `pop`"),
        ("consumers.arrays:c_", "B", "4.0 consumers stay only in 4.0 regression families; v5 demand is decided "
                                     "by `pop` individuals (D-15)"),
        ("patrons.patrons", "B+F", "commerce customers become `pop` individuals choosing products through the one "
                                   "decision engine; `patrons` kept only for its historical tests"),
        ("npcs.npc_workers", "B+F", "workers are `pop` individuals in the labour market; 4.0 NPC workers stay in "
                                    "4.0 families"),
        ("society.minds", "B", "4.0 society minds stay in 4.0 families; v5 worlds use `pop` classes 2-4"),
        ("labor.labor_workers", "B+F", "a worker profile becomes a view of a `pop` individual (skills, reservation "
                                       "wage as population columns)"),
        ("sci.sci_npc", "B+I", "researchers become `pop` individuals with research activities"),
        ("adopt.adopt_people", "B+G", "adopters become `pop` individuals whose activities hit bottlenecks (D-18)"),
    )},
    "pop.cols:pop.employer": {"target_owner": "contracts.contracts", "decision": "D-15/D-21", "workstream": "B+F",
                              "expires": _WAVE2,
                              "migration": "employment of individuals comes from accepted labour-market offers "
                                           "(employment contracts); the employer column becomes a cache derived "
                                           "from them, and random job loss/finding is removed"},
    # found by scanning every column on its own (wave 3, A): per-individual ties and memberships, audience follows
    "pop.cols:pop.contacts": {"target_owner": "social.cols:follow", "decision": "D-15/D-21", "workstream": "B+C",
                              "expires": _WAVE2,
                              "migration": "an individual's contacts are edges of the one social graph (a contact "
                                           "tie with trust as an edge attribute) that word of mouth, recommenders "
                                           "and follows all read; the column becomes a cache derived from the "
                                           "edges or is removed"},
    "pop.cols:pop.communities": {"target_owner": "community.communities", "decision": "D-15/D-21",
                                 "workstream": "B+C", "expires": _WAVE2,
                                 "migration": "memberships of individuals are the community module's membership "
                                              "columns (community.cols:cmember); a forum board is a community's "
                                              "discussion space; popweb reads memberships from there"},
    "aud.cols:aud_creators.followers": {"target_owner": "social.cols:follow", "decision": "D-21", "workstream": "C",
                                        "expires": _WAVE2,
                                        "migration": "a follow decided in the consumption engine is a follow edge "
                                                     "of the social graph (written through it); audience "
                                                     "accounting keeps only counts derived from those edges"},
    # relationships, communities, institutions, contracts, subscriptions, listings: D-21 ownership decisions
    "social.cols:follow": {"target_owner": "social.cols:follow", "decision": "D-21", "workstream": "C+A",
                           "expires": _WAVE2,
                           "migration": "ONE follow graph: forum.follow/unfollow become endpoints that write social "
                                        "follow edges (population and agents alike); forum_follows is retired; then "
                                        "OWNERS.relationship = social.cols:follow"},
    "community.communities": {"target_owner": "community.communities", "decision": "D-21", "workstream": "C+A",
                              "expires": _WAVE2,
                              "migration": "a forum board becomes the discussion space of a community (board -> "
                                           "community id); forum_boards keeps no membership of its own; then "
                                           "OWNERS.community = community.communities"},
    "inst.inst_charters": {"target_owner": "inst.inst_charters", "decision": "D-21", "workstream": "F+A",
                           "expires": _WAVE2,
                           "migration": "a 4.0 gov constitution becomes a charter (its rules as charter rules, "
                                        "proposals/votes as inst decisions); gov keeps 4.0 families only; then "
                                        "OWNERS.institution = inst.inst_charters"},
    "commit.commitments": {"target_owner": "contracts.contracts", "decision": "D-21", "workstream": "F",
                           "expires": _WAVE2,
                           "migration": "a commitment is a contract with an escrow clause (or references one): "
                                        "parties, status and breach live in contracts"},
    "creators.creator_members": {"target_owner": "subs.sub_subs", "decision": "D-21", "workstream": "C+F",
                                 "expires": _WAVE2,
                                 "migration": "creator tiers become subscription plans of the creator's product; "
                                              "memberships become subscriptions"},
    "letters.cols:nlsub": {"target_owner": "subs.sub_subs", "decision": "D-21", "workstream": "C+F",
                           "expires": _WAVE2,
                           "migration": "newsletter subscriptions (free or paid) are subscriptions to the "
                                        "newsletter's plan; the column table remains an index"},
    "xchg.xchg_listings": {"target_owner": "market.listings", "decision": "D-11/D-21", "workstream": "F",
                           "expires": _WAVE2,
                           "migration": "a listing on an exchange is a market listing with the exchange as venue "
                                        "(product reference kept); xchg keeps venues, orders and take rates"},
    "industry.offers": {"target_owner": "market.listings", "decision": "D-11/D-21", "workstream": "J (4.0 "
                        "industry)", "expires": _WAVE2,
                        "migration": "industrial supply offers become listings of goods (sku, facility) or stay in "
                                     "4.0 industrial families only"},
    "aud.aud_item_info": {"target_owner": "content.content", "decision": "D-16/D-21", "workstream": "C+D",
                          "expires": _WAVE2,
                          "migration": "audience accounting reads item metadata from the catalogue (content.content "
                                       "and the artifact) by catalogue index instead of keeping a copy"},
    "jobs.postings": {"target_owner": "contracts.contracts", "decision": "D-21", "workstream": "F",
                      "expires": _WAVE2,
                      "migration": "v5 worlds post work through the labour market (labor posts -> offers -> "
                                   "employment contracts); the 4.0 job board stays in 4.0 families"},
}


# ---------------------------------------------------------------------------------------------------- classify
def _owners_by_key() -> dict[str, list[str]]:
    from .primitives import KERNEL, OWNERS
    out: dict[str, list[str]] = {}
    for kind, o in OWNERS.items():
        if o["module"] == KERNEL or o["table"].startswith("@"):
            continue
        out.setdefault(f"{o['module']}.{o['table']}", []).append(kind)
    return out


def _signature(name: str) -> list[str]:
    return [k for k, p in SIGNATURES.items() if re.search(p, name)]


def classify(key: str) -> dict:
    """{key, status: owner|reviewed|unreviewed|private, kind(s), role, why} of one item."""
    owners = _owners_by_key()
    if key in owners:
        return {"key": key, "status": "owner", "kinds": owners[key], "role": "owner"}
    if key in REVIEWED:
        r = REVIEWED[key]
        return {"key": key, "status": "reviewed", "kinds": [r["kind"]] if r["kind"] else [], **r}
    name = key.split(".", 1)[1]
    name = name.split(":", 1)[1] if ":" in name else name
    sig = _signature(name)
    if sig:
        return {"key": key, "status": "unreviewed", "kinds": sig, "role": None,
                "why": f"name carries the signature of {', '.join(sig)} and nobody classified it"}
    return {"key": key, "status": "private", "kinds": [], "role": "private", "why": "no shared-kind signature"}


def _expired(entry: dict, today: datetime.date | None) -> bool:
    today = today or datetime.date.today()
    return datetime.date.fromisoformat(entry["expires"]) < today


def _judge(items: list[dict], today=None) -> list[str]:
    probs = []
    for c in items:
        if c["status"] == "unreviewed":
            probs.append(f"unreviewed table {c['key']}: {c['why']} (classify it in substrate/facts.py REVIEWED)")
        elif c.get("role") == _D:
            k = KNOWN_DUPLICATES.get(c["key"])
            if k is None:
                probs.append(f"duplicate {c['kind']}: {c['key']} keeps {c['kind']} facts outside their owner and is "
                             "not on the reviewed allow-list")
            elif _expired(k, today):
                probs.append(f"duplicate {c['kind']}: {c['key']} - its allow-list entry expired on {k['expires']} "
                             f"({k['workstream']}: {k['migration'][:80]})")
        elif c.get("role") == _L:
            from .primitives import OWNERS
            o = OWNERS.get(c.get("kind") or "")
            if o is None or f"{o['module']}.{o['table']}" == c["key"] or c["key"].split(".", 1)[0] == o["module"]:
                probs.append(f"legacy {c.get('kind')}: {c['key']} must be a 4.0 table of another module than the "
                             f"kind's owner")
            elif not c.get("verified_by"):
                probs.append(f"legacy {c['kind']}: {c['key']} names no test showing that its module writes through "
                             f"{o['module']}.{o['table']} where that owner is installed (verified_by)")
    return probs


def legacy_split(world) -> list[str]:
    """The data-level exclusivity rule of legacy tables in one world: a legacy table of kind k must hold no facts
    while k's owner module is installed (its module writes through the owner there)."""
    from .primitives import OWNERS
    probs = []
    for key in world_items(world):
        r = REVIEWED.get(key)
        if not r or r["role"] != _L:
            continue
        o = OWNERS.get(r["kind"])
        if o is None or o["module"] not in world.modules:
            continue
        n = _rows(world, key) or 0
        if n:
            probs.append(f"split {r['kind']} facts: legacy table {key} holds {n} while the owner "
                         f"{o['module']}.{o['table']} is in this world")
    return probs


# ---------------------------------------------------------------------------------------------------- one world
def world_items(world) -> list[str]:
    """Every item a world's modules created: state tables, column tables and columns per declaring module, and
    dense arrays of the modules with reviewed array entries."""
    from . import columns
    keys = []
    for table, mods in sorted(world.state.declared_by.items()):
        for m in mods:
            if m != "kernel":
                keys.append(f"{m}.{table}")
    for table, rec in sorted(columns._registry(world).items()):
        by = rec.get("by", {})
        for m in sorted(set(by.values())):
            if m == "kernel":
                continue
            keys.append(f"{m}.cols:{table}")
            for c, who in sorted(by.items()):
                # every column is scanned on its own name too (the owner's and extensions' columns included: a
                # population column of contacts is a relationship fact whoever declares it)
                if who == m and (f"{m}.cols:{table}.{c}" in REVIEWED or _signature(c)):
                    keys.append(f"{m}.cols:{table}.{c}")
    for k in REVIEWED:
        if ".arrays:" in k:
            m, prefix = k.split(".arrays:")
            if m in world.modules:
                keys.append(k)
    return keys


def _rows(world, key: str) -> int | None:
    """How many facts an item holds in this world's state (the data-level view)."""
    from . import columns
    m, rest = key.split(".", 1)
    st = world.state
    if rest.startswith("cols:"):
        name = rest[5:]
        table, _, col = name.partition(".")
        rec = columns._registry(world).get(table)
        if rec is None:
            return None
        cols = [col] if col else [c for c, who in rec.get("by", {}).items() if who == m]
        hit = None
        for c in cols:
            a = st.arrays.get(f"{table}.{c}")
            if a is None:
                continue
            fill = rec["columns"][c][2]
            nz = np.any((a != fill).reshape(a.shape[0], -1), axis=1)
            hit = nz if hit is None else (hit | nz)
        return int(hit.sum()) if hit is not None else 0
    if rest.startswith("arrays:"):
        prefix = rest[7:]
        sizes = [a.shape[0] for k, a in st.arrays.items() if k.startswith(prefix) and a.ndim >= 1]
        return max(sizes) if sizes else 0
    t = st.tables.get(rest)
    return None if t is None else len(t)


def world_problems(world, today=None) -> list[str]:
    return _judge([classify(k) for k in world_items(world)], today) + legacy_split(world)


def world_report(world, today=None) -> dict:
    """{items, owners: {kind: [key, rows]}, known_duplicates: [...], unreviewed: [...], problems: [...]}."""
    from .primitives import OWNERS
    items = [classify(k) for k in world_items(world)]
    owners = {}
    for c in items:
        if c["status"] == "owner":
            for k in c["kinds"]:
                owners[k] = {"table": c["key"], "rows": _rows(world, c["key"])}
    known = []
    for c in items:
        if c.get("role") == _D and c["key"] in KNOWN_DUPLICATES:
            e = KNOWN_DUPLICATES[c["key"]]
            o = OWNERS.get(c["kind"], {})
            known.append({"kind": c["kind"], "table": c["key"], "rows": _rows(world, c["key"]),
                          "owner": f"{o.get('module')}.{o.get('table')}",
                          "owner_rows": owners.get(c["kind"], {}).get("rows"), **e})
    return {"items": len(items), "owners": owners, "known_duplicates": known,
            "unreviewed": [c["key"] for c in items if c["status"] == "unreviewed"],
            "legacy": [{"table": c["key"], "kind": c["kind"], "rows": _rows(world, c["key"])} for c in items
                       if c.get("role") == _L],
            "problems": _judge(items, today) + legacy_split(world)}


# ---------------------------------------------------------------------------------------------------- registry
def _with_deps(names):
    from ..core.kernel import MODULES
    order = []

    def add(m):
        if m in order:
            return
        for d in MODULES[m].depends:
            add(d)
        order.append(m)
    for m in names:
        add(m)
    return order


_BASES = (("identity", "bank", "admin"), ("identity", "bank", "admin", "artifacts", "cols", "pop"),
          ("identity", "bank", "admin", "artifacts", "cols", "crowd"))


def _install(name: str):
    """A world with `name` installed (no genesis) and the items its dependencies alone create."""
    from ..core.kernel import World, WorldSpec
    errs = []
    for base in _BASES:
        try:
            deps = _with_deps([b for b in base if b != name])
            w = World(WorldSpec("facts-scan", 0, [{"name": m, "config": {}} for m in _with_deps([*deps, name])]),
                      "facts-scan")
            return w, None
        except Exception as e:   # noqa: BLE001 - try the next population base; reported when none works
            errs.append(f"{type(e).__name__}: {e}"[:200])
    return None, "; ".join(errs)


_SKIP_PACKAGES = ("crucible.redteam",)     # adversarial probe modules: not world mechanics


def shipped_modules() -> list[str]:
    """Every kernel module the crucible package registers: `load_all_modules`, the Tier 2 domain modules, and any
    other module class found by importing every submodule of the package (modules registered by test files and by
    the red-team probes are excluded)."""
    import importlib
    import pkgutil

    import crucible
    from ..core.kernel import MODULES
    crucible.load_all_modules()
    from .. import domains
    domains.load_all()
    for info in pkgutil.walk_packages(crucible.__path__, "crucible."):
        if info.name.startswith(_SKIP_PACKAGES) or info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception:   # noqa: BLE001 - optional dependencies (psycopg, playwright, ...): nothing to register
            pass
    return sorted(n for n, cls in MODULES.items()
                  if cls.__module__.startswith("crucible.") and not cls.__module__.startswith(_SKIP_PACKAGES))


def scan_registry(names=None) -> dict:
    """{module: [items it creates]} for every shipped module (or `names`), each installed on its own with its
    dependencies; modules that cannot be installed at all are listed under '!uninstallable' with the reason."""
    out: dict = {"!uninstallable": {}}
    for name in sorted(names or shipped_modules()):
        w, err = _install(name)
        if w is None:
            out["!uninstallable"][name] = err
            continue
        out[name] = sorted({k for k in world_items(w) if k.split(".", 1)[0] == name})
    return out


def audit(names=None, today=None) -> dict:
    """The registry-wide check: every item of every module classified. {modules, items, by_status, owners,
    duplicates, known_duplicates, unreviewed, legacy, uninstallable, problems}."""
    scan = scan_registry(names)
    bad = scan.pop("!uninstallable")
    items = [classify(k) for m in sorted(scan) for k in scan[m]]
    probs = _judge(items, today) + [f"module {m} could not be installed for the scan: {e}" for m, e in sorted(bad.items())]
    for k in KNOWN_DUPLICATES:
        r = REVIEWED.get(k)
        if r is None or r["role"] != _D:
            probs.append(f"allow-list entry {k} is not a reviewed duplicate")
    by = {}
    for c in items:
        by[c["status"]] = by.get(c["status"], 0) + 1
    dups = [c for c in items if c.get("role") == _D]
    return {"modules": len(scan), "items": len(items), "by_status": by,
            "owners": sorted(c["key"] for c in items if c["status"] == "owner"),
            "duplicates": sorted(c["key"] for c in dups),
            "known_duplicates": sorted(c["key"] for c in dups if c["key"] in KNOWN_DUPLICATES),
            "unreviewed": sorted(c["key"] for c in items if c["status"] == "unreviewed"),
            "legacy": sorted(c["key"] for c in items if c.get("role") == _L),
            "uninstallable": bad, "problems": probs}
