"""Observability for CAM Unified (workstream F; spec Part 18 and Part 22).

accounting  ComputeMeter (GPU/CPU seconds, tokens, env steps, decisions, simulated days, peak memory, cost)
trace       TraceRecorder for DecisionTrace (append-only JSONL + patches, Parquet export, retrospective summaries)
dashboard   static single-file HTML learning dashboard over a run directory, with per-lifetime drill-down pages
"""
