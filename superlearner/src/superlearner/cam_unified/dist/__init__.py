"""Distributed shared-weight learning for CAM Unified (workstream F; spec Parts 14 and 22).

queue      durable, versioned experience spool (ExperienceQueue, SpoolWriter, SpoolConsumer)
publish    atomic weights-only checkpoint publication (CheckpointPublisher) and actor refresh (PolicyClient)
offpolicy  policy-lag routing (LagRouter) and semi-Markov V-trace
actor      ActorProcess / ActorSpec / ActorPolicyP
learner    CentralLearner / LearnerSpec / LearnerModelP / TrajectoryAssembler
semserve   shared semantic-model server (SemanticServer, SemanticClient)
launcher   Launcher (spawn + supervise learner, actors, semserve; compute accounting) and actor benchmarks
sinks      JsonlArchive (standalone ArchiveP sink)
toyworld   mechanism-proof environment, tiny recurrent policy and its learner (not CAM components)

Submodules are imported lazily so actor processes only load what they use.
"""
