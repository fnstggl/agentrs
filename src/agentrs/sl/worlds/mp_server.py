"""Melting Pot world server: runs inside Melting Pot's own interpreter (/workspace/ext/venv-mp, CPU TensorFlow) and
speaks length-prefixed pickle over stdin/stdout with the learner process. Only numpy + meltingpot are imported.

Only Melting Pot's PERMITTED_OBSERVATIONS reach the learner (WORLD.RGB, debug/puppeteer signals, events and Lua
properties are never sent). RGB is summarised to a grid of 8x8-pixel cells (2x2 average colour per cell).

    request  ("init", {"kind": "substrate"|"scenario", "name": str, "seed": int})  -> spec + first observations
             ("reset", None) -> observations          ("step", [int, ...]) -> observations, rewards, last
             ("close", None)
"""
import os
import pickle
import struct
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402

OUT = sys.stdout.buffer
IN = sys.stdin.buffer
sys.stdout = sys.stderr  # library prints must not corrupt the protocol stream


def send(obj):
    b = pickle.dumps(obj, protocol=4)
    OUT.write(struct.pack("<Q", len(b)) + b)
    OUT.flush()


def recv():
    h = IN.read(8)
    if len(h) < 8:
        return None
    (n,) = struct.unpack("<Q", h)
    return pickle.loads(IN.read(n))


def cells(rgb: np.ndarray, cell: int = 8) -> np.ndarray:
    H, W, _ = rgb.shape
    g = rgb[:H - H % cell, :W - W % cell].reshape(H // cell, 2, cell // 2, W // cell, 2, cell // 2, 3)
    return g.mean(axis=(2, 5)).transpose(0, 2, 1, 3, 4).reshape(H // cell, W // cell, 12).astype(np.uint8)


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    assert not tf.config.get_visible_devices("GPU")
    from meltingpot import scenario, substrate

    permitted = set(scenario.PERMITTED_OBSERVATIONS) - {"WORLD.RGB"}
    env = None

    def obs_list(ts):
        out = []
        for o in ts.observation:
            d = {"cells": cells(np.asarray(o["RGB"])) if "RGB" in o else None, "nums": {}}
            for k, v in o.items():
                if k == "RGB" or k not in permitted:
                    continue
                a = np.asarray(v, dtype=np.float32).ravel()
                if a.size <= 8:
                    for i, x in enumerate(a):
                        d["nums"][f"{k}{i}" if a.size > 1 else k] = float(x)
                else:
                    d["nums"][f"{k}_sum"] = float(a.sum())
                    d["nums"][f"{k}_max"] = float(a.max())
            out.append(d)
        return out

    while True:
        msg = recv()
        if msg is None or msg[0] == "close":
            break
        kind, arg = msg
        try:
            if kind == "init":
                if env is not None:
                    env.close()
                if arg["kind"] == "substrate":
                    cfg = substrate.get_config(arg["name"])
                    env = substrate.build(arg["name"], roles=tuple(cfg.default_player_roles))
                    action_set = cfg.action_set
                else:
                    cfg = scenario.get_config(arg["name"])
                    env = scenario.build(arg["name"])
                    action_set = substrate.get_config(cfg.substrate).action_set
                labels = []
                for a in action_set:
                    parts = [f"{k}={v}" for k, v in sorted(dict(a).items()) if v]
                    labels.append("_".join(parts) if parts else "noop")
                ts = env.reset()
                send(("ok", {"n": len(ts.observation), "labels": labels, "obs": obs_list(ts)}))
            elif kind == "reset":
                ts = env.reset()
                send(("ok", {"obs": obs_list(ts)}))
            elif kind == "step":
                ts = env.step([int(x) for x in arg])
                send(("ok", {"obs": obs_list(ts), "rewards": [float(r) for r in ts.reward], "last": bool(ts.last())}))
        except Exception as e:  # noqa: BLE001 - report to the learner, keep serving
            import traceback
            send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"))
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
