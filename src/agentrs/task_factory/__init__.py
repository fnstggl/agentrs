"""Continuous economic task factory: environment-native task generation, admission control,
curriculum and coverage accounting on top of the AGENTRS economic environments (``agentrs.econ``).

A task is not a prompt: every generated task is an executable starting state (SQLite business
data, a code repository, a MiniWeb application, a document collection) plus tools, business rules,
scripted counterparties, a step budget and a hidden verifier that inspects the final state. The
factory turns ``(family, split, index)`` into a :class:`TaskSpec`, admits it only if it is valid
(environment builds, the initial state does not already satisfy the verifier, the scripted expert can
solve it in the real environment, it is not a duplicate), and samples admitted tasks according to
a learning-progress curriculum balanced across economic domains.
"""
from agentrs.task_factory.factory import Admission, TaskFactory, TaskSpec, build_spec  # noqa: F401
