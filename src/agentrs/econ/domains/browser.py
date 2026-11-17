"""Browser / administrative work: the existing MiniWeb simulated web applications (login, checkout,
form validation, search-and-lookup, data entry, dropdown forms) with their server-side verifiers.
The policy GENERATES the action string (e.g. type(e12, "Ines Costa")); no candidate list is used.
Toy single-click families (click_button, navigate_link, checkboxes, menu) are excluded."""
from __future__ import annotations

from agentrs.actions.schema import try_parse_action
from agentrs.agent.belief import BeliefState
from agentrs.econ.core import Family, StepOut, Task, register

TRAIN = ["login", "checkout", "form_validation", "search_lookup", "fill_field", "dropdown"]
HELDOUT = ["lookup_then_fill"]
GRAMMAR = """ACTIONS (write exactly one, on one line):
click(eN)                  click element eN
type(eN, "text")           type text into a field (replaces its content)
select(eN, "option")       choose an option of a dropdown
scroll(down) / scroll(up)  scroll the page
wait()                     wait for the page to update
finish() / finish("answer")  declare the task complete (with an answer when one is asked for)"""


class BrowserEnv:
    domain = "browser"
    action_format = "miniweb"

    def __init__(self, task: Task):
        from agentrs.demonstrations.teacher import ScriptedTeacher
        from agentrs.evaluation.benchmark import make_env

        self.task = task
        self.mt = task.private["miniweb_task"]
        self.env = make_env("miniweb")
        obs = self.env.reset(self.mt)
        self.belief = BeliefState(task=self.mt.public())
        self.belief.reset(obs)
        self.teacher = ScriptedTeacher(epsilon=0.0, seed=task.seed)
        self.teacher.reset()
        self.done = False
        self.log: list[dict] = []

    def system_prompt(self) -> str:
        return ("You are an operations assistant using web applications through a text browser. Elements are "
                "referenced by ids like e12.\n" + GRAMMAR)

    def initial_observation(self) -> str:
        return ""

    def state_text(self) -> str:
        return self.belief.prompt()

    def expert_action(self) -> str:
        cands = self.belief.augment_candidates(self.env.available_actions())
        return self.teacher.act(self.belief.task, self.belief.observation, cands).to_str()

    def step(self, action_text: str) -> StepOut:
        line = action_text.strip().split("\n")[0].strip() if action_text.strip() else ""
        a = try_parse_action(line)
        if a is None:
            self.log.append({"tool": "parse_error", "ok": False})
            return StepOut("Invalid action: could not parse. " + GRAMMAR.split("\n")[0], valid=False, error="parse")
        r = self.env.step(a)
        self.belief.update(a, r)
        self.log.append({"tool": a.type.value, "ok": bool(r.action_valid)})
        self.done = bool(r.terminated or r.truncated)
        return StepOut(r.error or "ok", valid=bool(r.action_valid), error=r.error, done=self.done)

    def verify(self):
        v = self.env.verify_outcome()
        return bool(v.success), {"progress": getattr(v, "progress", None)}


def _maker(fam, heldout):
    def make(split, seed):
        from agentrs.econ.core import SPLIT_SEED_BASE
        from agentrs.environments.miniweb.tasks import task_set

        idx = seed - SPLIT_SEED_BASE[split]
        mw_split = "test_family" if heldout else split
        start = idx + {"train": 0, "val": 0, "test": 0}[split] + (5000 if heldout and split == "val" else 0) + \
            (9000 if heldout and split == "test" else 0)
        mt = task_set(mw_split, [fam], n_per_family=1, start=start)[0]
        return Task("browser", fam, split, seed, mt.goal, {"miniweb_task": mt}, max_steps=min(12, mt.max_steps))
    return make


for _f in TRAIN:
    register(Family("browser", _f, _maker(_f, False), BrowserEnv, None))
for _f in HELDOUT:
    register(Family("browser", _f, _maker(_f, True), BrowserEnv, None, heldout=True))
