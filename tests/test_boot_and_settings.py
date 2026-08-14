"""Startup banner and settings tests.

`src/core/boot.py` had no tests, and its two interesting branches are exactly
the ones that misbehave in a container: whether output is paced (TTY only) and
whether the check mark can be encoded at all. Under `docker logs` without `-t`,
or on a cp1252 console, getting either wrong produces a hang-looking delay or a
`UnicodeEncodeError` during startup.
"""

import os

import pytest

from src.config import Settings
from src.core import boot


class _Stream:
    """A stderr stand-in with controllable tty-ness and encoding.

    Not a `StringIO` subclass: `encoding` is a read-only attribute on the real
    text-IO types, and overriding it is the whole point here.
    """

    def __init__(self, tty=False, encoding="utf-8"):
        self._chunks = []
        self._tty = tty
        self.encoding = encoding

    def write(self, text):
        self._chunks.append(text)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return self._tty

    def getvalue(self):
        return "".join(self._chunks)


@pytest.fixture
def stderr(monkeypatch):
    def _install(**kwargs):
        stream = _Stream(**kwargs)
        monkeypatch.setattr(boot.sys, "stderr", stream)
        return stream

    return _install


def test_output_is_paced_only_on_a_tty(stderr, monkeypatch):
    """Under `docker logs` there is nobody watching the frames, so the pacing
    sleep must not run -- 30 art lines at 0.03s each is a second of dead time
    per start, and every k8s Job pays it."""

    slept = []
    monkeypatch.setattr(boot.time, "sleep", lambda d: slept.append(d))

    stderr(tty=False)
    boot.stream_banner(art="a\nb\nc", delay=0.03)
    assert slept == [], "paced output on a non-tty"

    stderr(tty=True)
    boot.stream_banner(art="a\nb\nc", delay=0.03)
    assert slept, "did not pace output on a tty"


def test_check_mark_degrades_when_the_stream_cannot_encode_it(stderr):
    """A cp1252 console cannot encode U+2713. Writing it raises during startup,
    so the checklist falls back to ASCII."""

    out = stderr(encoding="cp1252")
    boot.boot_step("Device", "CPU", ok=True)
    assert "OK" in out.getvalue()
    assert "\u2713" not in out.getvalue()

    out = stderr(encoding="cp1252")
    boot.boot_step("Weights", "missing", ok=False)
    assert "--" in out.getvalue()


def test_check_mark_is_used_when_the_stream_can_encode_it(stderr):
    out = stderr(encoding="utf-8")
    boot.boot_step("Device", "CUDA", ok=True)
    assert "\u2713" in out.getvalue()


def test_boot_step_writes_the_label_and_detail(stderr):
    out = stderr(encoding="utf-8")
    boot.boot_step("Curriculum", "depth 1 -> 15")
    assert "Curriculum" in out.getvalue()
    assert "depth 1 -> 15" in out.getvalue()


def test_banner_emits_every_line(stderr):
    out = stderr(encoding="utf-8")
    boot.stream_banner(art="alpha\nbeta\ngamma", delay=0)
    written = out.getvalue()
    assert "alpha" in written and "beta" in written and "gamma" in written


def test_settings_read_overrides_from_the_environment(monkeypatch):
    """`Settings` is what every deployment path configures runs through, and no
    test covered that an override actually lands."""

    monkeypatch.setenv("RUN_ID", "sweep-a")
    monkeypatch.setenv("SEARCH_MODE", "muzero")
    monkeypatch.setenv("TRAIN_STEPS_PER_EPISODE", "7")
    monkeypatch.setenv("SAVE_BEST_ONLY", "false")

    settings = Settings()

    assert settings.RUN_ID == "sweep-a"
    assert settings.SEARCH_MODE == "muzero"
    assert settings.TRAIN_STEPS_PER_EPISODE == 7
    assert settings.SAVE_BEST_ONLY is False


def test_settings_default_artefact_paths_are_repo_relative(monkeypatch):
    """No absolute developer paths: both directories derive from the repo root."""

    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("LOG_DIR", raising=False)

    settings = Settings()

    assert os.path.isabs(settings.MODEL_DIR)
    assert settings.MODEL_DIR.replace("\\", "/").endswith("data/models")
    assert settings.LOG_DIR.replace("\\", "/").endswith("data/logs")


def test_unknown_environment_variables_are_ignored(monkeypatch):
    """`extra = "ignore"`, so a stray variable must not blow up startup."""

    monkeypatch.setenv("MY_CUSTOM_PATH", "D:/somewhere")

    assert Settings().RUN_ID is not None
