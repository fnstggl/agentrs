# Wave 3 merge progress (integrator)

branch | merge commit | tests run | result
---|---|---|---
W3 worktree-wf_477a68f9-804-3@ca7958b | 013e854 | core/, dom/test_dom_semantic_xattn.py, integration/ (not slow) | 107+15+119 passed, 1 skipped, 0 failed
W4 worktree-wf_477a68f9-804-4@dd5cdc6 | f04aa2f | semantic/ (16 files), dom/test_dom_decoder_xattn.py, dom/test_dom_semantic_xattn.py, core/test_core_entities_satellite.py, integration/ | 1 failure fixed (test_value_channel used W3's interim set_semantic; now sem=); after fix all pass: semantic 114, dom 19, core 3, integration 119 (+1 skip)
W6 cam-unified/W6-pred@152b7d1 | 7557e48 | world/ (13 files), core/, capabilities/, integration/, learn/, dom/ | 2 failures fixed (test_w6_rows_and_ranges doubles assumed the seam core); after fix: world 87 pass (slow deselected), core 119, capabilities 9, integration 119 (+1 skip), learn 93, dom 19
