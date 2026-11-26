"""The audience of the one population responds to meaning, not to tokens (MASTER_SPEC 4.3/4.4; audit C_social p3):
informative posts reach more people than keyword-stuffed or token-optimised gibberish posts on the same topics, and
a content.update re-indexes the distributed representation so the item's later audience follows its new subject."""
import numpy as np
import pytest

from crucible.social import meaning

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


def test_informative_content_beats_stuffing_and_gibberish():
    r = meaning.token_gaming(0, n=1500, run_h=48)
    a = r["arms"]
    assert a["informative"]["unique_viewers"] > a["stuffed"]["unique_viewers"]
    assert a["informative"]["unique_viewers"] > a["gibberish"]["unique_viewers"]
    assert a["informative"]["view_rate"] > a["stuffed"]["view_rate"]
    # the posts differ in their bytes only: legibility is computed from them
    assert min(a["informative"]["legibility"]) > max(a["gibberish"]["legibility"])


def test_update_moves_the_audience_with_the_representation():
    r = meaning.artifact_change(0, n=1500, run_h=36)
    assert r["viewers_before"] > 0 and r["viewers_after"] > 0
    # viewers after the update lean towards the new subject more than those before it
    assert r["lean_to_b_after"] > r["lean_to_b_before"]
