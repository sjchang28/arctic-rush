"""Best-weights checkpointing.

One weights file, written on merit. These pin the two ways "save the best" gets
it wrong: overwriting a good checkpoint with a degraded one, and freezing the
file forever because an early easy level set an unbeatable bar.
"""

import json
import os

import pytest

from src.config import settings
from src.model.storage import SharedStorage


class _FakeNetwork:
    def __init__(self):
        self.saves = 0

    def save_model(self):
        self.saves += 1


def _storage(tmp_path, monkeypatch, level=0, score=float("-inf")):
    """A SharedStorage with the persistence wired up but no real network."""

    monkeypatch.setattr(settings, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "RUN_ID", "test")
    monkeypatch.setattr(settings, "SAVE_BEST_ONLY", True)
    monkeypatch.setattr(settings, "CHECKPOINT_WARMUP_EPISODES", 2)

    storage = SharedStorage.__new__(SharedStorage)
    storage.latest_network = _FakeNetwork()
    storage.best_level = level
    storage.best_score = score
    return storage


def test_warmup_episodes_always_save(tmp_path, monkeypatch):
    """An early crash must still leave a usable file."""

    storage = _storage(tmp_path, monkeypatch)

    assert storage.save_if_best(level=1, score=0.0, episode=1)
    assert storage.save_if_best(level=1, score=0.0, episode=2)
    assert storage.latest_network.saves == 2


def test_degradation_does_not_overwrite(tmp_path, monkeypatch):
    """The whole point: a run that peaks and declines keeps its peak."""

    storage = _storage(tmp_path, monkeypatch)

    assert storage.save_if_best(level=1, score=0.90, episode=10)
    before = storage.latest_network.saves

    assert not storage.save_if_best(level=1, score=0.60, episode=11)
    assert not storage.save_if_best(level=1, score=0.10, episode=12)
    assert storage.latest_network.saves == before
    assert storage.best_score == pytest.approx(0.90)


def test_improvement_within_a_level_saves(tmp_path, monkeypatch):
    storage = _storage(tmp_path, monkeypatch)

    assert storage.save_if_best(level=1, score=0.50, episode=10)
    assert storage.save_if_best(level=1, score=0.75, episode=11)
    assert storage.latest_network.saves == 2
    assert storage.best_score == pytest.approx(0.75)


def test_a_deeper_level_resets_the_bar(tmp_path, monkeypatch):
    """Solving 95% at depth 2 is not better than 60% at depth 6.

    Ranking on the solved rate alone would freeze the file at the first easy
    level and never write again, which looks identical to a working checkpoint
    until the run ends and the weights turn out to be from episode 30.
    """

    storage = _storage(tmp_path, monkeypatch)

    assert storage.save_if_best(level=2, score=0.95, episode=10)

    # Promotion: lower rate, harder puzzles, and these are the weights that
    # earned the promotion.
    assert storage.save_if_best(level=3, score=0.20, episode=11)
    assert storage.best_level == 3
    assert storage.best_score == pytest.approx(0.20)

    # ...and within the new level the ratchet applies again.
    assert not storage.save_if_best(level=3, score=0.15, episode=12)
    assert storage.save_if_best(level=3, score=0.40, episode=13)


def test_marker_round_trips_so_a_resume_cannot_clobber(tmp_path, monkeypatch):
    """A resumed run inherits the bar, rather than replacing hours of training
    with whatever its first window happened to score."""

    storage = _storage(tmp_path, monkeypatch)
    storage.save_if_best(level=4, score=0.80, episode=50)

    marker = os.path.join(str(tmp_path), "test", "best.json")
    assert os.path.exists(marker)
    with open(marker) as handle:
        assert json.load(handle)["score"] == pytest.approx(0.80)

    resumed = _storage(tmp_path, monkeypatch)
    resumed.best_level, resumed.best_score = resumed._read_best_marker()

    assert resumed.best_level == 4
    assert not resumed.save_if_best(level=4, score=0.30, episode=51)
    assert resumed.latest_network.saves == 0


def test_missing_marker_is_not_an_unbeatable_bar(tmp_path, monkeypatch):
    """A deleted or never-written marker must not lock saving out."""

    storage = _storage(tmp_path, monkeypatch)
    level, score = storage._read_best_marker()

    assert (level, score) == (0, float("-inf"))


def test_save_best_only_off_restores_every_episode_saving(tmp_path, monkeypatch):
    storage = _storage(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "SAVE_BEST_ONLY", False)

    assert storage.save_if_best(level=1, score=0.9, episode=10)
    assert storage.save_if_best(level=1, score=0.1, episode=11)
    assert storage.latest_network.saves == 2
