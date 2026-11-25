"""QwenSemantic: token states, chunking, embeddings, cache/version semantics, LoRA bookkeeping."""
from __future__ import annotations

import torch


def test_hidden_shapes_mask_and_padding(enc):
    texts = ["hello world", "The service returned HTTP 503 after 2000 ms.", ""]
    h, m = enc.hidden(texts)
    n = [len(enc.tok(t, add_special_tokens=False)["input_ids"]) or 1 for t in texts]
    assert h.shape == (3, max(n), enc.sem_dim) and m.shape == (3, max(n))
    assert m.sum(1).tolist() == n
    assert torch.all(h[~m] == 0)
    assert torch.isfinite(h[m].float()).all()
    # max_tokens cuts rows and the cut is counted
    before = enc.stats["truncated_tokens"]
    h2, m2 = enc.hidden(texts, max_tokens=3)
    assert h2.shape[1] == 3 and enc.stats["truncated_tokens"] > before
    assert torch.equal(h2[1, :3], h[1, :3])


def test_layer_is_configurable_and_forward_stops_early(enc):
    # the encoder layer is an interior layer; its states differ from the final-norm states
    assert 0 <= enc.layer < enc.n_layers - 1
    ids = enc.tok("A cat sat on the mat.", return_tensors="pt")["input_ids"].to(enc.device)
    with torch.no_grad(), enc._encoder_ctx():
        out = enc.lm.model(input_ids=ids, output_hidden_states=True)
    ref = out.hidden_states[enc.layer + 1][0]                  # hidden_states[i+1] = output of layer i
    got = enc._forward_layer([ids[0].tolist()])[0]
    assert torch.nn.functional.cosine_similarity(got.float(), ref.float(), dim=-1).min() > 0.999


def test_long_text_is_chunked_with_mapping(enc):
    long = " ".join(f"item{i} costs {i * 7} cents;" for i in range(400))
    ids = enc._token_ids(long)
    assert len(ids) > 2 * enc.chunk_tokens
    states = enc.token_states([long])[0]
    assert states.shape[0] == len(ids)
    ch = enc.hidden_chunks(["short text", long])
    n_long = -(-len(ids) // enc.chunk_tokens)
    assert ch.h.shape[0] == 1 + n_long
    assert ch.text_index.tolist() == [0] + [1] * n_long
    assert ch.chunk_start.tolist()[1:] == [k * enc.chunk_tokens for k in range(n_long)]
    # every chunk is encoded independently: chunk 2 equals encoding its token window alone
    c2 = ids[enc.chunk_tokens: 2 * enc.chunk_tokens]
    alone = enc._forward_layer([c2])[0]
    cos = torch.nn.functional.cosine_similarity(states[enc.chunk_tokens: 2 * enc.chunk_tokens].float(), alone.float(),
                                                dim=-1)
    assert cos.mean() > 0.999 and cos.min() > 0.95              # identical up to bf16 batch-composition noise


def test_cache_hits_misses_and_version_semantics(enc):
    text = "Cache probe: the reactor gauge reads 451 units."
    m0, h0 = enc.stats["misses"], enc.stats["hits"]
    a = enc.token_states([text])[0]
    assert enc.stats["misses"] == m0 + 1
    b = enc.token_states([text])[0]
    assert enc.stats["hits"] == h0 + 1 and b is a
    v0, dv0 = enc.version, enc.decoder_version
    saved = enc.trainable_state_dict()
    probe_ids = enc.tok("Answer:", return_tensors="pt")["input_ids"].to(enc.device)

    def dec_logits():
        with torch.no_grad():
            e = enc.embed_ids(probe_ids)
            return enc.lm_logits(enc.decoder_hidden(e, torch.ones_like(probe_ids)))[0, -1]

    l0 = dec_logits()
    try:
        with torch.no_grad():                                    # a "training step" on the LoRA weights
            for p in enc.lora_parameters():
                p.add_(torch.randn_like(p) * 0.05)
        enc.mark_updated()
        # encoder path uses base weights -> version unchanged, cache still served, recomputation identical
        assert enc.version == v0 and enc.decoder_version != dv0
        h1 = enc.stats["hits"]
        assert enc.token_states([text])[0] is a and enc.stats["hits"] == h1 + 1
        fresh = enc._encode_texts([text])[0]
        assert torch.equal(fresh, a)
        # ... while the decoder path (LoRA on) changed
        assert (dec_logits() - l0).abs().max() > 1e-3
        # with the encoder on LoRA (a later, versioned condition) the version changes and the cache is not served
        enc.encoder_lora = True
        enc.mark_updated()
        assert enc.version != v0
        m1 = enc.stats["misses"]
        c = enc.token_states([text])[0]
        assert enc.stats["misses"] == m1 + 1
        assert (c.float() - a.float()).abs().max() > 1e-3
    finally:
        enc.encoder_lora = False
        enc.load_trainable_state_dict(saved)
    assert enc.version == v0
    assert torch.allclose(dec_logits(), l0, atol=1e-3)


def test_lru_is_bounded_by_bytes(enc):
    old = enc.cache_max_bytes
    try:
        enc.cache_max_bytes = 200_000
        ev = enc.stats["evicted"]
        for i in range(20):
            enc.token_states([f"bounded cache entry number {i} " * 3])
        assert enc._cache_bytes <= enc.cache_max_bytes
        assert enc.stats["evicted"] > ev
    finally:
        enc.cache_max_bytes = old


def test_embed_pooled_and_cached(enc):
    e = enc.embed(["profit margin", "def f(x):\n    return x * 2\n", "profit margin"])
    assert e.shape == (3, enc.sem_dim) and e.dtype == torch.float32
    assert torch.equal(e[0], e[2])
    assert not torch.allclose(e[0], e[1])


def test_parameter_counts_lora_targets_and_frozen_base(enc, cfg):
    counts = enc.parameter_counts()
    assert counts["pretrained"] == sum(p.numel() for n, p in enc.peft.named_parameters() if "lora_" not in n)
    expect = 0
    for name, mod in enc.lm.named_modules():
        if any(name.endswith(t) for t in cfg.sem_lora_targets) and hasattr(mod, "base_layer"):
            base = mod.base_layer
            expect += cfg.sem_lora_rank * (base.in_features + base.out_features)
    assert counts["lora"] == expect > 0
    for n, p in enc.peft.named_parameters():
        assert p.requires_grad == ("lora_" in n), n
    emb = enc.input_embeddings()
    assert emb.weight.shape[1] == enc.sem_dim and emb.weight.shape[0] >= len(enc.tok)
    assert 0 < enc.embedding_rms < 1
