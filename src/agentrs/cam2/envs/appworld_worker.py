"""AppWorld worker: runs inside AppWorld's own virtualenv (AppWorld pins its own dependency versions and freezes
the process clock, so it must not share a process with the learner). Protocol: one JSON object per line on stdin,
one JSON reply per line on the original stdout (AppWorld's own prints go to stderr).

Only agent-visible information leaves this process: the task instruction, the supervisor's public profile, API
documentation, and the outputs of the calls the agent made. Ground truth stays inside AppWorld and is used only by
AppWorld's own evaluator, whose result (success, passed / failed test counts) is returned for reward and
evaluation.

Standalone on purpose: stdlib + appworld only.
"""
import json
import os
import sys


def main() -> None:
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # anything AppWorld prints goes to stderr
    sys.stdout = sys.stderr
    from appworld import AppWorld, load_task_ids

    world = None
    exp = os.environ.get("CAM2_APPWORLD_EXPERIMENT", f"cam2_{os.getpid()}")

    def reply(ok, value):
        out.write(json.dumps({"ok": ok, "value": value}, default=str) + "\n")
        out.flush()

    for line in sys.stdin:
        try:
            msg = json.loads(line)
            cmd = msg["cmd"]
            if cmd == "tasks":
                reply(True, load_task_ids(msg["split"]))
            elif cmd == "reset":
                if world is not None:
                    world.close()
                world = AppWorld(task_id=msg["task_id"], experiment_name=exp, max_interactions=10_000,
                                 random_seed=int(msg.get("seed", 100)))
                t = world.task
                sup = t.supervisor
                reply(True, {"instruction": t.instruction,
                             "supervisor": {k: sup[k] for k in ("first_name", "last_name", "email", "phone_number")},
                             "allowed_apps": list(t.allowed_apps), "datetime": str(t.datetime)})
            elif cmd == "api_docs":
                docs = []
                for app, apis in world.task.api_docs.items():
                    for name, d in apis.items():
                        docs.append({"app": app, "api": name, "description": d.get("description", ""),
                                     "parameters": d.get("parameters", [])})
                reply(True, docs)
            elif cmd == "call":
                code = ("import json as __j\n"
                        f"__r = apis.{msg['app']}.{msg['api']}(**__j.loads({json.dumps(json.dumps(msg['args']))}))\n"
                        "print(__j.dumps(__r, default=str))")
                o = world.execute(code)
                reply(True, {"output": o[: int(msg.get("max_chars", 6000))], "completed": world.task_completed()})
            elif cmd == "evaluate":
                r = world.evaluate().to_dict()
                reply(True, {"success": bool(r["success"]), "num_tests": int(r["num_tests"]),
                             "passes": int(len(r["passes"]) if isinstance(r["passes"], list) else r["passes"]),
                             "failures": int(len(r["failures"]) if isinstance(r["failures"], list) else r["failures"])})
            elif cmd == "save":
                world.save_state(msg["state_id"])
                reply(True, msg["state_id"])
            elif cmd == "load":
                world.load_state(msg["state_id"])
                reply(True, msg["state_id"])
            elif cmd == "close":
                if world is not None:
                    world.close()
                reply(True, None)
                break
            else:
                reply(False, f"unknown command {cmd}")
        except Exception as e:  # noqa: BLE001
            reply(False, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
