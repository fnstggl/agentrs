# Licences of the fitted checkpoints

Every checkpoint under `models/` holds only fitted model parameters (NumPy arrays) and aggregate metrics, never rows
of a dataset. Each is nevertheless derived from a dataset whose licence governs its use. The licence of every
registered model follows from the dataset its card names (`registry.card_view(card)["datasets"]`, licence in
`datasets()[name]["license"]`):

| checkpoint(s) | dataset | licence of the dataset | use of the derived checkpoint |
|---|---|---|---|
| content_consumption, content_consumption_feed, watch_duration, return_daily, session_rate, and the E4 KuaiRand models | kuairand_pure (Gao et al., CIKM 2022, doi:10.5281/zenodo.10439422) | CC BY-SA 4.0 (LICENSE file in the archive) | attribution required; adaptations are shared under CC BY-SA 4.0 |
| news_click (@1, @2) | mind_small (Wu et al., ACL 2020) | Microsoft Research License Terms, **non-commercial research only** | **non-commercial research only**; acquisition route unresolved (BLOCKERS B-1) |
| commercial_response | criteo_uplift_v2.1 (Diemert et al., AdKDD 2018) | CC BY-NC-SA 4.0 | attribution required, **non-commercial only**, share alike |
| software_deep_engagement, software_repo_growth, and the E4 GitHub models | gharchive_2024-10-07_96h (GH Archive, public GitHub events) | GitHub Terms of Service and Acceptable Use Policies (research use allowed; no use of personal information to identify people); GH Archive code MIT | research use; no identifier is kept (pseudonymised extracts) |
| the E4 retail models | online_retail_ii (Chen, UCI 2012, doi:10.24432/C5CG6D) | CC BY 4.0 | attribution required |

The Crucible repository's own licence does not extend to these checkpoints: anyone redistributing Crucible with them
must keep this notice and respect the non-commercial terms of the MIND- and Criteo-derived files.
