"""Measured costs of the synthetic-internet layer (Part 27). Writes docs/benchmarks/internet.json.

* API browsing: `net.fetch` GETs against a generated site (requests per wall second, mean ms).
* Chromium: page loads and a login + form submission on the same site (wall ms per action), including the cost
  the browser adds over the API for the same recorded commands.
* Concurrency: N light scripted agents on independent clocks (AsyncRunner) issuing `net.fetch` reads; simulated
  actions per wall second as N grows. These are reasoning-free agents; they measure the kernel, not LLM scale.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import AsyncRunner, Policy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402


def world(seed=1):
    return World.create(generate({"family": "open_digital", "overrides": {
        "n_consumers": 500, "module_config": {"webgen": {"sites": 2}}}}, seed))


class Reader(Policy):
    name, kind = "reader", "scripted"

    def __init__(self, url):
        self.url = url

    def act(self, obs, tools):
        return {"tool": "net.fetch", "args": {"url": self.url}}


def main():
    out = {}
    w = world()
    site = sorted(h for h, _ in w.state.scan("hosts") if w.modules["net"].web_dep(w.state, h)[1] is not None)[0]
    url = f"http://{site}.crucible/"
    w.system_call("admin.join", {"handle": "bob", "endowment": 100_000})
    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        assert w.execute("p:bob", {"tool": "net.fetch", "args": {"url": url}}).ok
        w.advance_time(w.now + 20)   # below the per-actor limit of 240 actions per simulated hour
    dt = time.perf_counter() - t0
    out["api_fetch"] = {"site": site, "n": n, "per_s": round(n / dt, 1), "mean_ms": round(1000 * dt / n, 2)}
    # Chromium
    from crucible.web.app import make_web_app
    from crucible.web.browser import BrowserSession, playwright_available, serve_in_thread
    if playwright_available():
        wapp, stop = serve_in_thread(make_web_app(w))
        b = BrowserSession(wapp, timeout_ms=20_000)
        try:
            t = time.perf_counter()
            b.act({"action": "goto", "url": url})
            first = time.perf_counter() - t
            ts = []
            for _ in range(20):
                t = time.perf_counter()
                o = b.act({"action": "goto", "url": url})
                ts.append(time.perf_counter() - t)
                assert o["ok"]
            out["chromium_page_load"] = {"first_ms": round(1000 * first, 1), "n": len(ts),
                                         "mean_ms": round(1000 * sum(ts) / len(ts), 1),
                                         "overhead_vs_api": round((sum(ts) / len(ts)) / (dt / n), 1)}
        finally:
            b.close()
            stop()
    # concurrency
    conc = []
    for k in (10, 50, 200):
        w2 = world(2)
        ag = []
        for i in range(k):
            h = f"r{i:03d}"
            w2.system_call("admin.join", {"handle": h, "endowment": 10_000})
            ag.append((f"p:{h}", Reader(url)))
        r = AsyncRunner(w2, ag, seed=1, think_mean_s=600)
        t = time.perf_counter()
        acts = r.run_until(w2.now + 6 * 3600)
        el = time.perf_counter() - t
        conc.append({"agents": k, "sim_hours": 6, "actions": acts, "wall_s": round(el, 2),
                     "actions_per_wall_s": round(acts / el, 1)})
    out["async_agents"] = conc
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmarks", "internet.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
