"""An unavailable tool the task does not authorize does not block live readiness; an authorized one does."""
from types import SimpleNamespace

from cam_live.contracts import TaskSpec
from cam_live.runtime.preflight import integration_items


class _Env:
    def __init__(self, tools):
        self._tools = tools

    def get_capabilities(self):
        return {"live": True, "provenance": "live", "simulated_time": False, "name": "computer"}

    def discover_tools(self):
        return self._tools


def _tool(name, modality, available, why=""):
    return SimpleNamespace(name=name, modality=modality, available=available, unavailable_reason=why)


def _task(authorized):
    return TaskSpec(objective_id="t", objective="x", mode="live", interfaces=["api", "internal"],
                    resources={"authorized_tools": authorized})


TOOLS = [_tool("publish.draft", "api", False, "no owned publishing destination is configured"),
         _tool("discord.get_identity", "api", True), _tool("internal.note", "internal", True)]


def test_an_unavailable_tool_outside_the_authorized_tools_does_not_block():
    items = integration_items(_task(["discord.*", "internal.*"]), _Env(TOOLS))
    assert not [i for i in items if i["item"] == "publish.draft"], items


def test_an_unavailable_authorized_tool_still_blocks():
    items = integration_items(_task(["discord.*", "internal.*", "publish.*"]), _Env(TOOLS))
    assert [i for i in items if i["item"] == "publish.draft"]


def test_without_an_authorized_tools_list_every_unavailable_tool_blocks():
    items = integration_items(_task([]), _Env(TOOLS))
    assert [i for i in items if i["item"] == "publish.draft"]
