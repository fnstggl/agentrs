"""Local fixtures for the ES tests (tests/harness/skills)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from es_support import AppServer, NotesApp  # noqa: E402


@pytest.fixture
def notes():
    app = NotesApp()
    server = AppServer(app)
    app.url = server.url
    yield app
    server.close()


@pytest.fixture
def other_origin():
    app = NotesApp()
    server = AppServer(app)
    app.url = server.url
    yield app
    server.close()
