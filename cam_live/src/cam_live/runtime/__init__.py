"""cam_live.runtime — the persistent live runtime (workstream F).

    loop            LiveRuntime: the 17-step event loop of spec 15.1 over AgentAPI / EnvironmentAdapter / PolicyEngine /
                    OutcomeVerifier, with limits, approvals, waiting and crash-safe resume
    logs            RunLog (append-only JSONL with redaction and provenance), RunReader, atomic file helpers
    journal         the human-readable research journal regenerated from the structured records
    tasks           YAML -> TaskSpec with validation (no strategy content, explicit limits for live runs)
    verifier        independent signed verifiers (membership, metric threshold) with a hash-chained log
    verifier_service  process entry + RemoteVerifier client (VerifyRPC)
    scheduler       pending-outcome queue, wake scheduling, interruptible sleeping
    control         operator commands through the run-directory control file and the PolicyEngine
    obstacles       ObstacleRecord creation and linking to later decisions and learning updates
    multi           several actors sharing one central learner
    dashboard       token-protected FastAPI monitoring interface and operator controls
    doctor / preflight   environment inspection and the live-readiness report
    cli             the ``cam-live`` command (spec Part 17); loop_service: the learner process entry
"""
