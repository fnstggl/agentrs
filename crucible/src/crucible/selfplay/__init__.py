"""Multi-agent self-play on hidden mechanisms (Crucible 4.0).

    arena      `SelfPlayEnv` (every seat operated, per-seat observations and rewards), `KernelSelfPlayEnv` (the same
               match in a kernel world through `rig.*` tools), `Individual`, `Objective`, `play`
    learners   independent learners behind `research.learner.Learner`: `spsa`, `pg`, the scripted `climber`,
               fixtures; `make_learner` also builds any learner of the `research.learners` registry
    profiles   operator-side scripted settings computed from the hidden mechanism (experimental fixtures)
    modes      the six interaction modes as runnable configurations with their analyses
    league     a persistent population on disk: individuals, archived stages, matches, ratings, matchmaking, and
               a held-out opponent population that never trains anybody
    audit      transfers cannot move the objective, conservation, and the classification of a pair's joint
               behaviour as value-creating cooperation, tacit collusion or neither

See `docs/SELF_PLAY.md` for the 3.0 policy leagues in full economies (`training.league`), whose rating and
sampling code this package reuses.
"""
from .arena import (Individual, KernelSelfPlayEnv, Objective, SelfPlayEnv, arena_task, play, summarise,  # noqa: F401
                    with_noise)
from .learners import make_learner  # noqa: F401
