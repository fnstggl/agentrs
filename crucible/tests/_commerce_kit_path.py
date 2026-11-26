"""Import the commerce gate kit (tests/v5/_commerce_kit.py) from unit tests."""
import importlib
import os
import sys


def kit():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v5")
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("_commerce_kit")
