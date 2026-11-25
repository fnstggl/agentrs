"""CAM Unified Stage 1: the genuine end-to-end learning path on the real Crucible OpenWorld (spec Part 21 Stage 1-2).

    system.py         assembly of the full real stack (A semantic + B core + C world + D learner + E memory + F traces)
    harness_agent.py  the integrated agent as workstream G's lifetime-harness AgentP, durable save / load (new process)
    branch.py         abandoned branches of a resumed lifetime (archive view for replay / consolidation)
    tasks.py          Stage-1 tasks (ow_tiny open economy, ow_works rig) at training-range dev seeds, split guard
    runner.py         run / resume one lifetime in chunks through `envs.harness.run_lifetime`
    probes.py         main-scale instantiation, act + update timing and memory
    evidence.py       compact evidence summaries of a run directory
    __main__.py       command line: python -m superlearner.cam_unified.stage1 {run,resume,summarize,dashboard,...}
"""
