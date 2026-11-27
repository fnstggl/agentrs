"""Realistic (context, action, goal, query) inputs for the decision path, shared by the CPU and GPU tests."""
import torch

from cam_live.contracts import ActionSpec, ExperienceItem, ItemKind, OutcomeKind, OutcomeQuery, Quantity


def realistic_decisions(model, n_ctx=48, seed=0):
    """phi over realistic (context, action, goal, query) combinations through the whole pipeline: contexts of 1-6
    items of mixed kinds with texts, quantities spanning many magnitudes and irregular timestamps; eight different
    actions with arguments; four goals; six queries of every fast-layer kind. -> phi [n_ctx, 8, 6, F]."""
    g = torch.Generator().manual_seed(seed)
    kinds = [ItemKind.OBSERVATION, ItemKind.ACTION, ItemKind.OUTCOME, ItemKind.EVENT, ItemKind.DOCUMENT,
             ItemKind.FAILURE, ItemKind.SOFTWARE]
    words = ("checkout form export job search results login page error log pricing table signup funnel api "
             "response unit tests").split()
    rint = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=g))
    ctxs = []
    for i in range(n_ctx):
        its = []
        for t in range(rint(1, 7)):
            w = " ".join(words[rint(0, len(words))] for _ in range(rint(2, 9)))
            v = float(torch.sinh(torch.randn(1, generator=g) * 2.0))
            its.append(ExperienceItem(kind=kinds[rint(0, len(kinds))], text=w,
                                      ts=1000.0 * i + 10.0 ** float(torch.rand(1, generator=g) * 6) + 10.0 * t,
                                      quantities=[Quantity(f"{w.split()[0]}.count", round(v, 2))] if t % 2 == 0 else []))
        ctxs.append(sorted(its, key=lambda it: it.ts))
    acts = [ActionSpec("a1", "browser.click", arguments={"ref": "e3"}),
            ActionSpec("a2", "sandbox.python", modality="code", arguments={"code": "print(1)"}),
            ActionSpec("a3", "internal.wait", arguments={"seconds": 600}),
            ActionSpec("a4", "browser.type", arguments={"ref": "e7", "text": "hello"}),
            ActionSpec("a5", "api.call", modality="api", arguments={"endpoint": "/export", "rows": 100}),
            ActionSpec("a6", "skill.run", modality="skill", level="procedure", arguments={"skill": "publish_page"}),
            ActionSpec("a7", "browser.navigate", arguments={"url": "https://example.org/docs"}),
            ActionSpec("a8", "visual.click", modality="visual", arguments={"x": 310, "y": 88})]
    qs = [OutcomeQuery("q1", "signups", kind=OutcomeKind.COUNT, horizon_s=3600.0),
          OutcomeQuery("q2", "signups", kind=OutcomeKind.COUNT, horizon_s=30 * 86400.0),
          OutcomeQuery("q3", "latency", horizon_s=30.0, description="seconds until the job is done"),
          OutcomeQuery("q4", "done", kind=OutcomeKind.BINARY, horizon_s=600.0),
          OutcomeQuery("q5", "time_to_first_reply", kind=OutcomeKind.TIME_TO_EVENT, horizon_s=86400.0),
          OutcomeQuery("q6", "revenue", horizon_s=7 * 86400.0, description="money received")]
    goals = ["Reach the stated target within the time limit.", "Find out why the export job fails.",
             "Increase voluntary sign-ups.", "Reduce the page load time."]
    with torch.no_grad():
        o = model.core_forward(model.tokenize(ctxs))
        e_a, _ = model.encode_actions([acts] * n_ctx)
        e_g = model.encode_goals([goals[i % 4] for i in range(n_ctx)])
        e_q = model.encode_queries([qs] * n_ctx)[0]
        return model.features(model.decision_latent(o.b, e_a, e_g), e_q)


def gram_stats(phi):
    """(max / min eigenvalue, participation ratio) of the uncentred second-moment matrix of phi."""
    P = phi.reshape(-1, phi.shape[-1]).double().cpu()
    ev = torch.linalg.eigvalsh(P.T @ P / P.shape[0]).flip(0)
    return float(ev[0] / ev[-1]), float(ev.sum() ** 2 / ev.pow(2).sum())
