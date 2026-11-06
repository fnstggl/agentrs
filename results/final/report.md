| configuration | mode | seeds | val success | test success | test_family success | invalid-action rate (test) | real env steps | train PFLOP | inference PFLOP | wall (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| A policy-only | policy | 2 | 0.025 [0.03,0.03] | 0.025 [0.03,0.03] | 0.031 [0.03,0.03] | 0.000 | 0 | 0.000 | 0.007 | 0.7 |
| B imitation | policy | 2 | 0.325 [0.30,0.35] | 0.250 [0.23,0.28] | 0.000 [0.00,0.00] | 0.000 | 0 | 0.144 | 0.006 | 13.0 |
| C model-free RL | policy | 1 | 0.250 [0.12,0.40] | 0.300 [0.17,0.45] | 0.000 [0.00,0.00] | 0.000 | 4802 | 0.539 | 0.043 | 35.5 |
| D model-based | planner | 2 | 0.312 [0.28,0.35] | 0.275 [0.28,0.28] | 0.000 [0.00,0.00] | 0.000 | 4586 | 0.763 | 0.074 | 41.3 |
| D model-based | policy | 2 | 0.287 [0.25,0.33] | 0.287 [0.28,0.30] | 0.000 [0.00,0.00] | 0.000 | 4586 | 0.763 | 0.074 | 41.3 |
| D model-based +imagination | planner | 2 | 0.300 [0.28,0.33] | 0.250 [0.23,0.28] | 0.016 [0.00,0.03] | 0.000 | 4592 | 0.813 | 0.077 | 46.9 |
| D model-based +imagination | policy | 2 | 0.287 [0.28,0.30] | 0.263 [0.25,0.28] | 0.016 [0.00,0.03] | 0.000 | 4592 | 0.813 | 0.077 | 46.9 |
| D model-based -demos | planner | 1 | 0.025 [0.00,0.10] | 0.050 [0.00,0.12] | 0.000 [0.00,0.00] | 0.000 | 6029 | 0.407 | 0.102 | 39.7 |
| D model-based -demos | policy | 1 | 0.025 [0.00,0.10] | 0.050 [0.00,0.12] | 0.000 [0.00,0.00] | 0.000 | 6029 | 0.407 | 0.102 | 39.7 |
| D model-based -replay | planner | 1 | 0.225 [0.10,0.38] | 0.300 [0.17,0.45] | 0.031 [0.00,0.09] | 0.000 | 4802 | 0.787 | 0.083 | 48.9 |
| D model-based -replay | policy | 1 | 0.250 [0.12,0.40] | 0.300 [0.17,0.45] | 0.000 [0.00,0.00] | 0.000 | 4802 | 0.787 | 0.083 | 48.9 |

Held-out family success (test_family):

| configuration | mode | dialog_confirm | lookup_then_fill | scroll_find | toggle_settings |
|---|---|---|---|---|---|
| A policy-only | policy | 0.00 | 0.00 | 0.00 | 0.12 |
| B imitation | policy | 0.00 | 0.00 | 0.00 | 0.00 |
| C model-free RL | policy | 0.00 | 0.00 | 0.00 | 0.00 |
| D model-based | planner | 0.00 | 0.00 | 0.00 | 0.00 |
| D model-based | policy | 0.00 | 0.00 | 0.00 | 0.00 |
| D model-based +imagination | planner | 0.06 | 0.00 | 0.00 | 0.00 |
| D model-based +imagination | policy | 0.06 | 0.00 | 0.00 | 0.00 |
| D model-based -demos | planner | 0.00 | 0.00 | 0.00 | 0.00 |
| D model-based -demos | policy | 0.00 | 0.00 | 0.00 | 0.00 |
| D model-based -replay | planner | 0.12 | 0.00 | 0.00 | 0.00 |
| D model-based -replay | policy | 0.00 | 0.00 | 0.00 | 0.00 |
