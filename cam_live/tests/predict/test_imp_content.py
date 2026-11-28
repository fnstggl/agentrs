"""IMP / ST request: a generic, content-dependent, self-supervised predictive target — the content of what an action
returns (frozen-backbone text embedding + hashed symlog quantities), predicted from the decision latent with an
ensemble Gaussian (NLL, epistemic variance, information gain), trained by the continual learner, recorded per
candidate, and readable as a record's content NLL (realised information value)."""
import math

import pytest
import torch

from cam_live.contracts import ExperienceItem, ItemKind, Quantity
from cam_live.predict.content import (HEAD_CONTENT, ContentHead, ContentLoss, ContentTarget, content_target_for,
                                      record_content_nll, returned_items)
from cam_live.predict.providers import default_providers

F64 = torch.float64


class _Embed:
    """Deterministic stand-in for model.embed_texts (a hash of the text) — the target map is under test here."""
    d_sem = 32

    def embed_texts(self, texts):
        out = []
        for t in texts:
            g = torch.Generator().manual_seed(abs(hash(("e", t))) % (2 ** 31))
            out.append(torch.randn(self.d_sem, generator=g))
        return torch.stack(out) if out else torch.zeros(0, self.d_sem)


def test_content_target_depends_on_text_and_exact_numbers_and_masks_nothing_returned():
    m = _Embed()
    t = ContentTarget(m.d_sem, seed=0)
    a = [ExperienceItem(kind=ItemKind.OBSERVATION, text="reading", quantities=[Quantity("probe.x", 1.0)])]
    b = [ExperienceItem(kind=ItemKind.OBSERVATION, text="reading", quantities=[Quantity("probe.x", 50.0)])]
    c = [ExperienceItem(kind=ItemKind.OBSERVATION, text="other text", quantities=[Quantity("probe.x", 1.0)])]
    y, mask = t(m, [a, b, c, []])
    assert mask.tolist() == [True, True, True, False] and torch.equal(y[3], torch.zeros(t.k, dtype=F64))
    assert float((y[0] - y[1]).norm()) > 0.5                        # the number is part of the content
    assert float((y[0] - y[2]).norm()) > 0.1                        # so is the text
    y2, _ = ContentTarget(m.d_sem, seed=0)(m, [a])
    assert torch.allclose(y2[0], y[0])                              # a fixed map: the learner cannot move it


def test_content_head_learns_reducible_content_and_its_information_gain_tracks_what_is_still_unknown(measure):
    torch.manual_seed(0)
    d, k = 12, 4
    head = ContentHead(d, k=k, hidden=32, n_members=5, seed=1)
    w = torch.randn(d, k)
    seen = torch.randn(256, d)
    unseen = 4.0 * torch.randn(64, d) + 3.0
    noise_z = torch.randn(64, d)
    ys = (seen @ w).double() * 0.5
    with torch.no_grad():
        nll0 = float(head(seen).predictive_nll(ys).mean())
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(700):
        idx = torch.randint(0, 256, (64,))
        out = head(seen[idx])
        loss = out.member_nll(ys[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        o_seen, o_unseen = head(seen), head(unseen)
        nll1 = float(o_seen.predictive_nll(ys).mean())
        ig_seen, ig_unseen = float(o_seen.information_gain().mean()), float(o_unseen.information_gain().mean())
    measure("imp_content_nll_untrained_trained", [nll0, nll1])
    measure("imp_content_ig_seen_unseen", [ig_seen, ig_unseen])
    assert nll1 < nll0 - 2.0
    assert ig_unseen > 2.0 * ig_seen and ig_seen >= 0.0
    # pure noise: the head learns a large aleatoric variance, so observing it carries little information
    head2 = ContentHead(d, k=k, hidden=32, n_members=5, seed=2)
    opt = torch.optim.Adam(head2.parameters(), lr=3e-3)
    for _ in range(700):
        out = head2(noise_z)
        loss = out.member_nll(3.0 * torch.randn(64, k, dtype=F64)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ig_noise = float(head2(noise_z).information_gain().mean())
    measure("imp_content_ig_noise", ig_noise)
    assert ig_noise < ig_unseen


def test_content_head_is_registered_trained_recorded_and_readable_on_the_real_model(lab_rig):
    rig = lab_rig(n_lifetimes=3, n_decisions=8)
    model, heads = rig.model, rig.heads
    assert model.head(HEAD_CONTENT) is heads.content
    provs = default_providers(heads, rig.check_signature)
    loss_p = next(p for p in provs if isinstance(p, ContentLoss))
    recs = [r for r in rig.records if r.post_item_refs]
    assert all(loss_p.eligible(r) for r in recs) and all(returned_items(rig.store, r) for r in recs)
    assert all(it.kind != ItemKind.OUTCOME for r in recs for it in returned_items(rig.store, r))
    from cam_live.learning_api import encode_decisions
    batch = encode_decisions(model, rig.store, recs[:8], grad=True)
    loss, m = loss_p.loss(model, batch, rig.store)
    loss.backward()
    assert math.isfinite(float(loss)) and m["content_n"] == 8.0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in heads.content.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for n, p in model.named_parameters() if n.startswith("exp."))     # it shapes the shared trunk
    nll = record_content_nll(model, rig.store, recs[:4])
    assert all(v is not None and math.isfinite(v) for v in nll)
    evs = rig.evaluate()
    assert all("content_ig" in e.extras and e.extras["content_ig"] >= 0.0 for e in evs)
    assert content_target_for(model) is content_target_for(model)


def test_a_checkpoint_without_the_content_head_loads_and_reports_it_fresh(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    state = {k: v for k, v in rig.model.trainable_state().items() if not k.startswith("heads." + HEAD_CONTENT)
             and HEAD_CONTENT not in k}
    rep = rig.model.load_trainable_state(state, strict=False)
    assert rep["missing"] and all(HEAD_CONTENT in k for k in rep["missing"]) and not rep["unexpected"]
