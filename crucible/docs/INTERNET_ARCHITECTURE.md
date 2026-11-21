# Synthetic internet architecture (Crucible 3.0)

This document describes the open-internet layer added on `feat/crucible-internet`. It sits on top of the v1
services documented in [SYNTHETIC_INTERNET.md](SYNTHETIC_INTERNET.md): identity, bank, code, cloud, market,
search, forum, mail, docs, issues, support, ads, jobs and contracts. Everything here is ordinary world state in
the one event-sourced kernel. There is no second server, no second database and no path to the real internet.

## Layers

| Layer | What it is | Code |
|---|---|---|
| Names | `<host>.crucible` for every live deployment; organisations register names (`net.register`) and point them at deployments they operate (`net.point`); `net.whois` | `services/net.py` |
| HTTP | `net.fetch(url, method, form/json)` is **one recorded command** whatever the client: an API agent, the web front end, or real Chromium. Status codes 200/302/400/401/403/404/429/500/503; DNS failure for unknown names | `services/net.py` |
| Applications | any deployment whose manifest has a `web` section: routes such as `GET /item/{id}`, static files, a per-minute rate limit and a login mode. Handlers run **inside the sandbox** as `handler(request, caller)` | `services/cloud.py` (`validate_web`, `execute_handler`) |
| Sessions and auth | server-side sessions keyed `host|visitor`; "Log in with Crucible ID" (`net.login`) reveals the visitor's identity to a site only after a consent page | `services/net.py`, `web/app.py` |
| Storage | `appdb` (`db.put/get/query/delete`): each application has its own namespace, callable only by its service principal | `services/net.py` (`AppDB`) |
| Money | sites bill through `bank` invoices; payment state is readable by payer and payee (`bank.invoice_status`) | `services/bank.py` |
| Failure | outages and overload from `cloud`; handler errors and timeouts from the sandbox; rate limiting; pages that disappear (`net.page_gone`) | `services/cloud.py`, `services/net.py` |
| Discovery | a daily crawler (`p:webcrawler`) follows links from seeds, indexes public pages (HTTP 200 to an anonymous visitor, no `noindex`) into the shared search index, and builds a link graph that adds a ranking signal; the index lags the web and is incomplete | `services/net.py` (`pr_crawl`), `services/search.py` |
| Browser | real Chromium (Playwright). The host resolver maps only `*.crucible` to the local front end, and every other name fails (`MAP * ~NOTFOUND`). Request routing blocks everything else. DOM, accessibility tree, screenshots and console errors are all available | `web/browser.py`, `web/app.py` |

## Who builds sites

* **Agents**: any actor commits code with a `web` manifest, rents compute and deploys. `SiteAuthor`,
  `EvidenceAggregator` and the LLM agents use the same path.
* **The world generator** (`worldgen/sitegen.py`, module `webgen`). It has schema-first application specs (12
  themes and extra fields), one application template, seeding with real records, and an operator organisation
  with cash.
* **Adversaries** (`worldgen/adversarial.py`, module `adversary`): injection pages, a fraudulent capability,
  spam, and false documentation.

Generated sites are accepted only if the **site critic** passes. The critic uses the site like a QA engineer
would and checks results against authoritative state:
* `web/critic.py api_critic` runs journeys through `net.fetch`;
* `browser_critic` repeats the main journey in Chromium, with labelled controls and no console errors.

## Equivalence of interfaces

A browser click on a form submits through the web front end, which calls `net.fetch` as the logged-in
principal. An API agent calls `net.fetch` directly. Both produce the same `cmd` event, the same handler
execution and the same state changes.
`tests/test_internet.py::test_real_chromium_browses_an_agent_web_app_with_the_same_effects_as_the_api` checks
this. A Chromium session through login, consent and form submission records only `net.fetch` and `net.login`
commands. Re-executing those commands through the API in a twin world yields an identical world digest.

## Trust boundaries

In-world text is data. Pages addressed to "AI agents" are ordinary content created by simulated actors. Nothing
inside a world can:
* reach the operator's session;
* read environment variables;
* change the evaluator, whose verifiers and metrics are evaluator-side objects.

`evaluation/trust_boundary.assess(world, actor)` measures whether an actor acted on adversarial content, using
the actor's recorded commands and authoritative state. See [SECURITY.md](SECURITY.md) for isolation, the
sandbox and modes A/B/C (`crucible/modes.py`).

## Limits (see LIMITATIONS.md)

* No JavaScript execution model for agent-written client code beyond what Chromium runs from static files; most
  applications are server-rendered.
* No real DNS hierarchy, TLS, cookies across sites, or CDN.
* One template underlies generated sites; agent-written sites are arbitrary.
* The crawler is daily and bounded (6 pages per site by default).
* Simulated people do not browse web sites. They read the forum, mail and the capability registry. Web traffic
  comes from agents, the crawler and site seeding, so audience-type goals depend on agents.
