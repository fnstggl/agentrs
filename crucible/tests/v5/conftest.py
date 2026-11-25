"""Crucible 5.0 gate tests. A gate test measures a requirement and records evidence with
`crucible.acceptance.v5.evidence.record`; `python -m crucible.acceptance.v5 run` attaches the JUnit result.

Set CRUCIBLE_V5_AUTHOR to the producing workstream (for example `ws-B`) so reviews can be checked for independence."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
