"""cam_live.harness.factories — the computer environment's executors, composed as one running system. Workstream EB.

:func:`all_executors` builds every modality executor the harness has, configured from the task:

========================  =============================================================================================
modality                  executor
========================  =============================================================================================
``internal``              :class:`~cam_live.harness.internal.InternalExecutor` (wait, inspect, notes, operator requests)
``structured_browser``    :class:`~cam_live.harness.browser.BrowserExecutor` (Chromium behind the network guard)
``code``                  :class:`~cam_live.harness.sandbox.SandboxExecutor` (+ background jobs)
``visual``                :class:`~cam_live.harness.visual.VisualExecutor` on the browser's screens
``api``                   :class:`~cam_live.harness.http_api.ApiModality` of ``HttpApiExecutor``,
                          ``PublishingExecutor.from_resources(task.resources)`` and
                          ``DiscordExecutor.from_task(task)``
``skill``                 :class:`~cam_live.harness.skills.SkillExecutor` over a library in ``<run_dir>/skills``
========================  =============================================================================================

An executor that cannot work in this deployment is still registered: its tools say ``available: false`` with the
exact reason (no Discord guild configured, no owned publishing destination, the sandbox cannot drop to its uid ...),
so ``discover_tools`` tells the learner what exists and why it cannot be used. Nothing here decides what CAM does.

Factory names for ``python -m cam_live.harness.service --executor MODULE:FACTORY`` (see
docs/HARNESS_ADAPTER_GUIDE.md): ``cam_live.harness.factories:all_executors`` (everything; also the service's
default when no ``--executor`` is given), and one per modality — ``internal``, ``browser``, ``code``, ``visual``,
``api``, ``skills``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..contracts import Clock, ModalityExecutor, TaskSpec, WallClock

#: modalities :func:`all_executors` composes, in registration order
ALL_MODALITIES = ("internal", "structured_browser", "code", "visual", "api", "skill")


def _resources(task: Optional[TaskSpec]) -> Dict[str, Any]:
    return dict(getattr(task, "resources", None) or {})


def default_sandbox_workspace(run_dir: str) -> str:
    """``<run>/sandbox``: the launcher prepares it for the sandbox uid next to the executor's own directory
    (``<run>/exec``) in the separated deployment; in-process it is ``<run_dir>/sandbox``."""
    run_dir = os.path.abspath(run_dir)
    parent = os.path.dirname(run_dir)
    if os.path.basename(run_dir) == "exec" and os.path.isdir(os.path.join(parent, "sandbox")):
        return os.path.join(parent, "sandbox")
    return os.path.join(run_dir, "sandbox")


def internal(run_dir: str, clock: Optional[Clock] = None, **_: Any) -> ModalityExecutor:
    from .internal import InternalExecutor
    return InternalExecutor(run_dir, clock=clock)


def browser(run_dir: str, clock: Optional[Clock] = None, browser_config: Any = None, **_: Any) -> ModalityExecutor:
    from .browser import BrowserExecutor
    return BrowserExecutor(run_dir, clock=clock, config=browser_config)


def code(run_dir: str, clock: Optional[Clock] = None, sandbox_config: Any = None, sandbox_workspace: str = "",
         **_: Any) -> ModalityExecutor:
    """The sandbox in this process when it can drop programs to the sandbox uid (root); otherwise, in the separated
    deployment, the sandbox host the launcher started as root (``$CAM_LIVE_SANDBOX_SOCKET``): the environment
    service itself never needs root (G2-harness D2)."""
    from .sandbox_host import ENV_SOCKET, RemoteSandboxExecutor
    remote = os.environ.get(ENV_SOCKET, "")
    if remote and os.geteuid() != 0 and sandbox_config is None:
        return RemoteSandboxExecutor(remote)
    from .sandbox import SandboxExecutor
    return SandboxExecutor(workspace=sandbox_workspace or default_sandbox_workspace(run_dir), cfg=sandbox_config)


def visual(run_dir: str = "", browser_executor: Any = None, visual_config: Any = None, **_: Any) -> ModalityExecutor:
    """The visual executor. Its screens are the browser executor's sessions (``screen`` names a session; a
    gesture's navigation is checked by the browser's network guard under the same action)."""
    from .visual import VisualExecutor
    if browser_executor is None:
        return VisualExecutor(cfg=visual_config)

    def screen_for(name: str, action: Any, ctx: Any) -> Any:
        tab = action.arguments.get("tab") if action is not None else None
        return browser_executor.screen(name or "default", tab, ctx, action)

    return VisualExecutor(screen_factory=screen_for, cfg=visual_config)


def credential_probe(policy: Any, task: Optional[TaskSpec]) -> Any:
    """``probe(name) -> bool``: does a stored secret of that name exist? Asked without releasing it. With the
    authorization service in this process its names are listed; over AuthRPC the executor uid may not list names
    (operator only), so a secret counts as present when the operator named it in the task — its release is still
    decided at the moment of use, for the granted action and destination only."""
    names_fn = getattr(policy, "secret_names", None)
    if callable(names_fn):
        def probe(name: str) -> bool:
            try:
                return any((n.get("name") if isinstance(n, dict) else n) == name for n in names_fn())
            except Exception:  # noqa: BLE001 — not allowed to list: fall back to the task's own statement
                return name in _named_secrets(task)
        return probe
    return lambda name: name in _named_secrets(task)


def _named_secrets(task: Optional[TaskSpec]) -> set:
    res = _resources(task)
    d = dict(res.get("discord") or {})
    return {str(x) for x in (d.get("token_secret"), res.get("discord_bot_credential_name")) if x}


def api(run_dir: str, clock: Optional[Clock] = None, secrets: Any = None, task: Optional[TaskSpec] = None,
        discord_base_url: str = "", policy: Any = None, **_: Any) -> ModalityExecutor:
    """Generic HTTP, publishing and Discord behind the one ``api`` modality. Discord is configured from
    ``task.resources["discord"]``; publishing destinations from ``task.resources`` (``owned_sites``,
    ``discord.channels``)."""
    from .discord import DiscordExecutor
    from .http_api import ApiModality, HttpApiExecutor
    from .publishing import PublishingExecutor
    task = task or TaskSpec(objective_id="", objective="")
    state = os.path.join(os.path.abspath(run_dir), "state")
    discord = DiscordExecutor.from_task(task, base_url=discord_base_url, secrets=secrets,
                                        credential_probe=credential_probe(policy, task))
    publishing = PublishingExecutor.from_resources(_resources(task), state, discord=discord, clock=clock)
    return ApiModality([HttpApiExecutor(), publishing, discord])


def skills(run_dir: str, clock: Optional[Clock] = None, **_: Any) -> ModalityExecutor:
    """Skills over a persistent library; their steps run through the environment's own executors."""
    from .skills import SkillExecutor, SkillLibrary
    return SkillExecutor(SkillLibrary(os.path.join(os.path.abspath(run_dir), "skills"), clock or WallClock()), [])


def all_executors(run_dir: str, clock: Optional[Clock] = None, env: Any = None, policy: Any = None,
                  secrets: Any = None, task: Optional[TaskSpec] = None, *, browser_enabled: bool = True,
                  browser_config: Any = None, sandbox_config: Any = None, visual_config: Any = None,
                  sandbox_workspace: str = "", discord_base_url: str = "",
                  exclude: tuple = ()) -> List[ModalityExecutor]:
    """Every executor of the harness for ``task`` (internal, structured browser, code, visual, api with HTTP /
    publishing / Discord, skills), in :data:`ALL_MODALITIES` order. ``secrets`` is the executor-side broker handle
    (Discord's token is fetched through it at the moment of use); ``env`` and ``policy`` are accepted so that the
    function can be used as a ``--executor`` factory."""
    clock = clock or getattr(env, "clock", None) or WallClock()
    if env is not None and getattr(env, "secrets", None) is not None:
        secrets = env.secrets                       # the environment's handle (it scrubs what it releases)
    out: List[ModalityExecutor] = []
    want = [m for m in ALL_MODALITIES if m not in exclude and (browser_enabled or m != "structured_browser")]
    b = None
    if "internal" in want:
        out.append(internal(run_dir, clock))
    if "structured_browser" in want:
        b = browser(run_dir, clock, browser_config=browser_config)
        out.append(b)
    if "code" in want:
        out.append(code(run_dir, clock, sandbox_config=sandbox_config, sandbox_workspace=sandbox_workspace))
    if "visual" in want:
        out.append(visual(run_dir, browser_executor=b, visual_config=visual_config))
    if "api" in want:
        out.append(api(run_dir, clock, secrets=secrets, task=task, discord_base_url=discord_base_url, policy=policy))
    if "skill" in want:
        out.append(skills(run_dir, clock))
    return out
