"""cam_live.harness.service_boot — how the launcher starts a separated service's Python (G2-harness R2-C1).

Run as ``<python> -I <this file> <import roots> -m <module> <args>`` (or ``<script> <args>``) with cwd ``/``.
Isolated mode (``-I`` = ``-E -s -P``): no PYTHON* environment variable is honoured, no user site-packages, and
neither the cwd nor this file's directory is put on ``sys.path``. Exactly the launcher's import roots (the
``cam_live`` source tree, root-owned) are put in front, then the module (or script) runs as ``__main__``. So no file
a service uid can write -- the run directory, a workspace, a home directory -- is ever importable by a service,
in particular not by a root one (the sandbox host). This file imports only the standard library.
"""
import runpy
import sys


def main() -> None:
    roots = [x for x in sys.argv[1].split(":") if x]
    rest = sys.argv[2:]
    sys.path[:0] = roots
    if rest[:1] == ["-m"]:
        sys.argv = [rest[1]] + rest[2:]
        runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = rest
        runpy.run_path(rest[0], run_name="__main__")


if __name__ == "__main__":
    main()
