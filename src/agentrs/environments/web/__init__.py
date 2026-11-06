"""Real-browser web environments.

* ``LocalWebServer`` (``server.py``) hosts MiniWeb apps as real HTML pages.
* ``PlaywrightEnv`` / ``LocalWebEnv`` (``playwright_env.py``) drive Chromium via
  Playwright with a generic DOM extractor (``dom_extract.js``).

The Playwright-dependent classes are imported lazily so that importing this
package (e.g. to start the server) does not require Playwright.
"""

from agentrs.environments.web.server import LocalWebServer

_LAZY = {"PlaywrightEnv", "LocalWebEnv", "PlaywrightThread", "BrowserUnavailable", "candidate_actions",
         "launch_chromium", "chromium_candidates", "resolved_chromium_executable"}


def __getattr__(name):
    if name in _LAZY:
        from agentrs.environments.web import playwright_env

        return getattr(playwright_env, name)
    raise AttributeError(name)


__all__ = ["LocalWebServer", *sorted(_LAZY)]
