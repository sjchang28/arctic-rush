"""Run knobs: the settings that actually differ between one run and the next.

This file is deliberately small. Everything that is a fixed property of the
task or the architecture -- board size, the action space, the trunk width, the
unroll length -- is a constant in the package that owns it, because dressing a
constant up as an env-overridable field implies it is something you tune, and
that is how a sweep ends up varying eleven things when it meant to vary one:

    src/game/config.py    board, colours, directions, pygame presentation
    src/model/config.py   action space, observation planes, rewards, network
                          architecture and replay/loss constants
    src/core/config.py    boot banner, console logging cadence

What stays here is what a deployment sets: which run this is, where it writes,
which search mode it plans with, and how much compute it spends.
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    """Every per-run tunable, in one place.

    This is the single source of truth for run configuration. Values were
    previously split between here and `docker-compose.yaml`, which meant the
    defaults in this file described a run nobody was doing: the container
    overrode a dozen of them, so reading either file alone gave a wrong picture
    of what was actually training.

    Everything here is env-overridable -- that is what lets parallel Docker/k8s
    runs differ without editing code. The rule is that an override should exist
    only where a value genuinely differs *between deployments*:

      * `docker-compose.yaml` sets `RUN_ID`, because that is what separates one
        run's checkpoints and logs from another's.
      * `Dockerfile` sets `MODEL_DIR` / `LOG_DIR`, because those are container
        paths onto the mounted volume.
      * Tuning knobs belong here, not in either of those.
    """

    # Run identity / output dirs
    RUN_ID: str = Field("local", description="Identifier for this training run")
    # Defaults point at the repository-level `data/` tree (one level above
    # `src/`), which is what the Docker/k8s volume mounts onto. Code lives in
    # `src/` and run artefacts live in `data/`; nothing writes into the package.
    MODEL_DIR: str = Field(
        default_factory=lambda: os.path.join(_REPO_ROOT, "data", "models"),
        description="Directory model checkpoints are written to/read from",
    )
    LOG_DIR: str = Field(
        default_factory=lambda: os.path.join(_REPO_ROOT, "data", "logs"),
        description="Directory training logs are written to",
    )

    # Budget. Sized off the 2026-08-13 `alphazero` run: the rolling 100-episode
    # solve rate climbed from 18% to ~47% by episode 900 and then sat flat in a
    # 26-49% band for the remaining 14k episodes, while training loss drifted from
    # 0.40 up to 0.87. Learning was done inside the first hour; the other 27 hours
    # bought nothing. A budget that outruns the plateau by an order of magnitude
    # only spends GPU time re-fitting a buffer the policy has stopped improving on.
    TRAINING_EPISODES: int = Field(2_000, description="Number of self-play/training episodes")
    NUM_ACTORS: int = Field(1, description="Number of parallel self-play actors")
    
    # 50 simulations is thin for a puzzle whose solutions run ~4 moves but whose
    # failures ran the full 25-move cap: search that shallow rarely reaches the
    # solutions the policy then has to learn from. Doubling it roughly doubles
    # wall clock per episode, which the smaller episode budget pays for.
    TOTAL_MCTS_EPISODES: int = Field(100, description="Simulations per move")
    
    # Gradient steps taken per self-play episode. This used to be hard-coded to 1,
    # which meant a default 120-episode run performed 120 SGD steps in total while
    # the LR schedule was written against a 500k-step run. 40 overcorrected: with
    # the solve rate flat and the loss climbing, those extra steps were re-fitting
    # stale buffer targets rather than tracking a policy that was still moving.
    TRAIN_STEPS_PER_EPISODE: int = Field(10, description="Gradient steps per self-play episode")

    # Checkpointing. One weights file, overwritten only when the model improves:
    # saving every episode meant a run that peaked and then degraded ended with
    # the degraded weights, since the last episode always won.
    SAVE_BEST_ONLY: bool = Field(True, description="Overwrite weights only on improvement")
    CHECKPOINT_WARMUP_EPISODES: int = Field(
        20, description="Episodes saved unconditionally so an early crash still leaves a file"
    )

    # Search. Ricochet Robots is deterministic and fully observable with a cheap
    # exact simulator, so MuZero's learned dynamics is pure cost here -- it has to
    # rediscover wall and robot blocking before its search means anything.
    # Measured head to head on this task: 86% solved vs 33% at equal wall clock.
    # SEARCH_MODE=muzero still works if you want the comparison back.
    SEARCH_MODE: str = Field("alphazero", description="'muzero' (learned dynamics) or 'alphazero' (real environment)")

    # Reanalyse re-searches stored positions with the *learned* model, which is
    # not the model SEARCH_MODE=alphazero plans with, so it stays off by default.
    # Raise it alongside SEARCH_MODE=muzero.
    REANALYSE_FRACTION: float = Field(
        0.0, description="Fraction of training batches drawn from reanalysed (refreshed) targets"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


settings: Settings = Settings()
