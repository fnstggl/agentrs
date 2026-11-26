"""Crucible 5.0 - Semantic and multimodal intelligence: content encoders with caching, the shared small-model inference service, semantic participants.

Owned by workstream D (docs/v5/WORKSTREAMS.md). Kernel modules of this package live in
`semantic/modules.py` or are imported from it; `crucible.load_all_modules()` imports that file when it exists.

Content encoders (INTERFACE_CONTRACTS section 4): `encode`, `encode_many`, `encoder_info`, `cache_key` from
`crucible.semantic.encoders`; media tooling in `crucible.semantic.media`. Importing this package does not load torch or
any model; models load on first use.
"""
from .encoders import cache_key, encode, encode_many, encoder_info  # noqa: F401
