"""Distributed shared-weight learning for CAM Unified (workstream F; spec Parts 14 and 22).

queue      durable, versioned experience spool (ExperienceQueue, SpoolWriter, SpoolConsumer)
publish    atomic weights-only checkpoint publication (CheckpointPublisher) and actor refresh (PolicyClient)
offpolicy  per-decision policy-lag routing (LagRouter, validated max lag) and semi-Markov V-trace
lagsweep   the lag validation sweep behind offpolicy.validated_max_lag (lag_validation.json)
actor      ActorProcess / ActorSpec / ActorPolicyP
learner    CentralLearner / LearnerSpec / LearnerModelP / TrajectoryAssembler (bootstrapped segments)
semserve   shared semantic-model server (SemanticServer, SemanticClient)
launcher   Launcher (spawn + supervise learner, actors, semserve; compute accounting) and actor benchmarks
sinks      JsonlArchive (standalone ArchiveP sink)
toyworld   mechanism-proof environment, tiny recurrent policy and its learner (not CAM components)

Submodules are imported lazily so actor processes only load what they use.
"""
