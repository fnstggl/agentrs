"""One adapter interface over realistic browser environments.

``RealisticWebEnvironmentAdapter``::

    reset(task_id, seed=None) -> Observation      observe() -> Observation
    available_actions() -> list[Action]           step(action) -> StepResult
    get_observable_events() -> list[dict]          verify_task() -> Verification
    snapshot() -> handle                           restore(handle)
    close()

Observations are what the page shows (accessibility-derived elements, visible text, URL) plus a
screenshot reference (PNG stored by SHA-256 in a blob directory). Verifier / backend truth never
enters observations.

Implementations and their honest limitations:

* ``SocialWebLabAdapter`` — our self-hosted multi-user app in real Chromium; real snapshot/restore
  (SQLite backup of the application database, including its virtual clock; the browser reloads).
* ``MiniWebChromiumAdapter`` — the 14 MiniWeb apps served as HTML in Chromium; no snapshot (the
  closest valid mechanism is a deterministic ``reset`` to the same task and seed).
* ``BrowserGymAdapter`` — BrowserGym tasks (MiniWoB++ runs here); reset by seed only.
* ``WebArenaAdapter`` / ``WorkArenaAdapter`` — thin wrappers over ``browsergym-webarena`` /
  ``browsergym-workarena``. They need the benchmark's self-hosted sites (WebArena: Docker images of
  shopping, admin, forum, GitLab, wiki, map) or a ServiceNow instance (WorkArena) — neither is
  available in this sandbox (no Docker daemon; external accounts are not authorised), so these raise
  ``EnvironmentUnavailable`` with the exact missing requirement instead of pretending to run.
"""

from __future__ import annotations

import abc
import hashlib
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

from agentrs.actions.schema import Action
from agentrs.environments.base import Observation, StepResult, TaskSpec, Verification


class EnvironmentUnavailable(RuntimeError):
    pass


class RealisticWebEnvironmentAdapter(abc.ABC):
    name = "abstract"
    supports_snapshot = False
    limitations: tuple[str, ...] = ()

    def __init__(self, blob_dir: str | Path | None = None, screenshots: bool = True):
        self.blob_dir = Path(blob_dir or tempfile.mkdtemp(prefix="agentrs-shots-"))
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots = screenshots
        self.env: Any = None
        self.task: TaskSpec | None = None

    # --------------------------------------------------------------- required
    @abc.abstractmethod
    def make_task(self, task_id: str, seed: int | None) -> TaskSpec: ...

    def reset(self, task_id: str, seed: int | None = None) -> Observation:
        self.task = self.make_task(task_id, seed)
        return self._with_shot(self.env.reset(self.task))

    def observe(self) -> Observation:
        return self._with_shot(self.env.observe())

    def available_actions(self) -> list[Action]:
        return self.env.available_actions()

    def step(self, action: Action) -> StepResult:
        r = self.env.step(action)
        r.observation = self._with_shot(r.observation)
        return r

    def get_observable_events(self) -> list[dict]:
        return []

    def verify_task(self) -> Verification:
        return self.env.verify_outcome()

    def snapshot(self) -> Any:
        raise NotImplementedError(f"{self.name}: snapshot not supported; use reset(task_id, seed) "
                                  "(deterministic) as the closest valid mechanism")

    def restore(self, handle: Any) -> None:
        raise NotImplementedError(f"{self.name}: restore not supported")

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    # --------------------------------------------------------------- screenshots
    def _with_shot(self, obs: Observation) -> Observation:
        if not self.screenshots or not hasattr(self.env, "run_in_browser"):
            return obs
        try:
            png = self.env.run_in_browser(lambda page: page.screenshot(type="png"))
        except Exception:  # noqa: BLE001 - screenshots are best-effort, never fatal
            return obs
        h = hashlib.sha256(png).hexdigest()
        p = self.blob_dir / f"{h}.png"
        if not p.exists():
            p.write_bytes(png)
        obs.screenshot_ref = h
        return obs

    def screenshot_path(self, ref: str) -> Path:
        return self.blob_dir / f"{ref}.png"


class SocialWebLabAdapter(RealisticWebEnvironmentAdapter):
    """task_id: '<family>/<split>/<index>' for the families in ``socialweblab.tasks``."""

    name = "socialweblab"
    supports_snapshot = True

    def __init__(self, lab, blob_dir=None, screenshots: bool = True, **env_kw):
        super().__init__(blob_dir, screenshots)
        from agentrs.socialweblab.browser_env import SocialWebLabBrowserEnv

        self.lab = lab
        self.env = SocialWebLabBrowserEnv(lab, **env_kw)
        self._snap_dir = Path(tempfile.mkdtemp(prefix="swl-snap-"))
        self._n = 0
        self._since = 0

    def make_task(self, task_id: str, seed: int | None) -> TaskSpec:
        from agentrs.socialweblab.tasks import make_task

        family, split, index = task_id.split("/")
        self._since = max([e["id"] for e in self.lab.public_events(0, 10 ** 9)] + [0])
        return make_task(self.lab, family, split, int(index) + (seed or 0) * 1000)

    def get_observable_events(self) -> list[dict]:
        evs = self.lab.public_events(self._since, 10 ** 6)
        return evs

    def snapshot(self) -> str:
        self._n += 1
        return self.env.snapshot(str(self._snap_dir / f"snap{self._n}.db"))

    def restore(self, handle: str) -> None:
        self.env.restore(handle)


class MiniWebChromiumAdapter(RealisticWebEnvironmentAdapter):
    """task_id: '<family>/<split>/<index>' from ``environments.miniweb.tasks``."""

    name = "miniweb-chromium"
    limitations = ("no snapshot/restore: reset(task_id, seed) is deterministic",)

    def __init__(self, blob_dir=None, screenshots: bool = True):
        super().__init__(blob_dir, screenshots)
        from agentrs.environments.web.playwright_env import LocalWebEnv

        self.env = LocalWebEnv()

    def make_task(self, task_id: str, seed: int | None) -> TaskSpec:
        from agentrs.environments.miniweb.tasks import task_set

        family, split, index = task_id.split("/")
        ts = task_set(split, [family], n_per_family=int(index) + 1)
        return ts[int(index)]


class BrowserGymAdapter(RealisticWebEnvironmentAdapter):
    """task_id: a BrowserGym task name, e.g. 'miniwob.click-test'."""

    name = "browsergym"
    limitations = ("reset by seed only; snapshot unsupported by BrowserGym",)

    def __init__(self, blob_dir=None, screenshots: bool = False, max_steps: int = 10):
        super().__init__(blob_dir, screenshots)
        self.max_steps = max_steps

    def make_task(self, task_id: str, seed: int | None) -> TaskSpec:
        from agentrs.environments.browsergym_adapter import BrowserGymEnv

        if self.env is not None:
            self.env.close()
        self.env = BrowserGymEnv(task_name=task_id, seed=seed or 0, max_steps=self.max_steps)
        return self.env.make_task() if hasattr(self.env, "make_task") else None  # type: ignore[return-value]

    def reset(self, task_id: str, seed: int | None = None) -> Observation:
        self.task = self.make_task(task_id, seed)
        return self.env.reset(self.task) if self.task is not None else self.env.reset(None)  # type: ignore[arg-type]


WEBARENA_SITES = ("WA_SHOPPING", "WA_SHOPPING_ADMIN", "WA_REDDIT", "WA_GITLAB", "WA_WIKIPEDIA", "WA_MAP",
                  "WA_HOMEPAGE")


def webarena_requirements() -> list[str]:
    missing = []
    if importlib.util.find_spec("browsergym.webarena") is None:
        missing.append("python package browsergym-webarena (pip install browsergym-webarena)")
    missing += [f"environment variable {v} (URL of the self-hosted site)" for v in WEBARENA_SITES if not os.environ.get(v)]
    return missing


class WebArenaAdapter(BrowserGymAdapter):
    """task_id: 'webarena.<n>' (or 'visualwebarena.<n>'). Needs self-hosted WebArena sites."""

    name = "webarena"

    def __init__(self, *a, **kw):
        missing = webarena_requirements()
        if missing:
            raise EnvironmentUnavailable("WebArena cannot run here; missing: " + "; ".join(missing))
        import browsergym.webarena  # noqa: F401 - registers the tasks

        super().__init__(*a, **kw)


class WorkArenaAdapter(BrowserGymAdapter):
    """task_id: a WorkArena task name. Needs a ServiceNow developer instance (external account)."""

    name = "workarena"

    def __init__(self, *a, **kw):
        missing = []
        if importlib.util.find_spec("browsergym.workarena") is None:
            missing.append("python package browsergym-workarena")
        for v in ("SNOW_INSTANCE_URL", "SNOW_INSTANCE_UNAME", "SNOW_INSTANCE_PWD"):
            if not os.environ.get(v):
                missing.append(f"environment variable {v} (an authorised ServiceNow instance)")
        if missing:
            raise EnvironmentUnavailable("WorkArena cannot run here; missing: " + "; ".join(missing))
        import browsergym.workarena  # noqa: F401

        super().__init__(*a, **kw)
