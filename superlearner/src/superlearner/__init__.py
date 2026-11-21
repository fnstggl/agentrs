"""CAM Superlearner 4.0.

Two connected learning systems built on the Crucible world simulator:

* the inner learner (CAM): a frozen language model plus a trainable recurrent controller that acts in Crucible
  through grounded, open-ended actions and is trained by reinforcement learning from verified outcomes;
* the outer learner (Meta-Researcher): proposes, implements, tests, trains and compares changes to the inner
  learner's learning algorithm in isolated candidate workspaces, and learns from its own search history.

Package layout (see docs/ARCHITECTURE.md):
    llm/        frozen language model service (generation, embeddings, cache)
    grounding/  capability discovery, validation, grounded execution, agent-created tools
    actions/    the action-program representation and its interpreter
    world/      Crucible adapter: tasks, sessions, splits
    evalkit/    protected evaluation code (never editable by candidates)
    cam/        the trainable controller, its features, memory and decision procedure   [candidate-editable]
    algo/       learning algorithms (recurrent PPO, behaviour cloning, learned rules)     [candidate-editable]
    train/      rollout workers, trajectory store, trainer, curriculum
    meta/       the Meta-Researcher
    selfplay/   several learners in one persistent world
"""
import os
import sys

__version__ = "4.0.0"

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
CRUCIBLE_SRC = os.path.join(REPO_ROOT, "crucible", "src")


def ensure_crucible() -> None:
    """Make the sibling Crucible package importable (it is used as a library; nothing in it is replaced)."""
    try:
        import crucible  # noqa: F401
    except ImportError:
        if CRUCIBLE_SRC not in sys.path:
            sys.path.insert(0, CRUCIBLE_SRC)
        import crucible  # noqa: F401
    import crucible as c
    c.load_all_modules()
