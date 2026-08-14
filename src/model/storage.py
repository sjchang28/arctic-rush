"""Checkpointing, best-model selection and TensorBoard scalars.

Split out of `network.py`. This is run bookkeeping -- what is on disk, what the
best result so far was, what to plot -- and has nothing in common with the torch
module it used to sit beside.
"""

import json
import math
import os

from torch.utils.tensorboard import SummaryWriter

from src.config import settings
from src.core.logger import logger
from src.model.config import WEIGHTS_FILE_PATH
from src.model.network import Network


class SharedStorage:

    def __init__(self, config, path=WEIGHTS_FILE_PATH):
        self.networks = {}  # Dictionary to store networks by step
        self.latest_step = 0

        model_path = os.path.join(settings.MODEL_DIR, settings.RUN_ID, path)

        self.latest_network = Network(config)

        # What the weights on disk actually scored, so a resumed run has to beat
        # the checkpoint it inherited rather than replacing it with its first
        # noisy window. Only meaningful alongside the weights it describes -- a
        # marker left behind by a deleted checkpoint would otherwise set a bar
        # that a freshly initialised network can never clear.
        if os.path.exists(model_path):
            self.latest_network.load_model(path)
            self.best_level, self.best_score = self._read_best_marker()
        else:
            self.best_level, self.best_score = 0, float("-inf")

        self._writer = None


    @property
    def writer(self) -> SummaryWriter:
        """TensorBoard writer, opened on first use.

        Constructing a `SummaryWriter` creates its log directory, so building it
        in `__init__` meant merely constructing a `SharedStorage` -- in a test,
        say -- left an empty `tensorboard/` directory behind. Deferring it means
        the directory appears when something is actually logged.
        """

        if self._writer is None:
            self._writer = SummaryWriter(
                log_dir=os.path.join(settings.LOG_DIR, settings.RUN_ID, "tensorboard"))

        return self._writer


    def get_latest_network(self) -> Network:

        return self.latest_network


    def save_network(self, step: int, network: Network):

        self.networks[step] = network
        if step > self.latest_step:
            self.latest_step = step
            self.latest_network = network


    def get_network(self, step: int) -> Network:

        return self.networks.get(step, self.latest_network)


    def log_scalars(self, step, reward=None, loss=None, elapsed_min=None,
                    solved_rate=None, solution_length=None, loss_parts=None,
                    curriculum_moves=None, optimal_depth=None, excess_moves=None):

        if reward is not None:
            self.writer.add_scalar("episode/reward", reward, step)
        # Episodes before the buffer fills record no loss; NaN poisons the chart.
        if loss is not None and not math.isnan(loss):
            self.writer.add_scalar("episode/loss", loss, step)
        if elapsed_min is not None:
            self.writer.add_scalar("episode/elapsed_minutes", float(elapsed_min), step)
        if solved_rate is not None:
            self.writer.add_scalar("episode/solved_rate", solved_rate, step)
        if solution_length is not None:
            self.writer.add_scalar("episode/solution_length", solution_length, step)
        if curriculum_moves is not None:
            self.writer.add_scalar("episode/curriculum_depth", curriculum_moves, step)
        # Plot against curriculum_depth: the two diverging means the scramble is
        # generating easier positions than the level it is labelled with.
        if optimal_depth is not None:
            self.writer.add_scalar("episode/optimal_depth", optimal_depth, step)
        # Moves spent above the optimum. Zero is optimal play; solved_rate alone
        # cannot distinguish that from a win that wandered.
        if excess_moves is not None:
            self.writer.add_scalar("episode/excess_moves", excess_moves, step)
        if loss_parts:
            for name, value in loss_parts.items():
                if not math.isnan(value):
                    self.writer.add_scalar(f"loss/{name}", value, step)


    def save_if_best(self, level, score, episode):

        """Overwrite the single weights file only when the model has improved.

        One file, as before -- but written on merit rather than on every episode.
        A run that peaks and then degrades used to end with the degraded weights,
        because the last episode always won.

        "Best" cannot be the solved rate alone. Solving 95% at curriculum depth 2
        is not better than solving 60% at depth 6, so a plain rate comparison
        would freeze the file at the first easy level and never write again.
        Progress is therefore ranked by (level, rate) lexicographically:

          * A deeper level always beats a shallower one, and resets the bar --
              the first score at a new depth is by definition its best so far.
              This also means the weights that earned a promotion are saved
              immediately, which is exactly when they are worth keeping.
          * Within a level, only an improved rate writes. Degradation cannot
              overwrite a better checkpoint.

        Early episodes save unconditionally so that a crash in the first few
        minutes still leaves a usable file.
        """

        if not settings.SAVE_BEST_ONLY:
            self.latest_network.save_model()
            return True

        warming_up = episode <= settings.CHECKPOINT_WARMUP_EPISODES

        # A new depth is a new scale; last level's rate is not a bar to clear.
        if level > self.best_level:
            self.best_level = level
            self.best_score = float("-inf")

        improved = score > self.best_score

        if not (warming_up or improved):
            return False

        if improved:
            self.best_score = score

        self.latest_network.save_model()
        self._write_best_marker(episode)

        if improved and not warming_up:
            logger.debug(
                f"[Checkpoint] Episode {episode}: saved at level {level}, "
                f"solved {score:.0%}."
            )

        return True


    def _write_best_marker(self, episode):

        """Record what the saved weights scored, next to the weights themselves.

        Without this a resumed run starts with `best_score` unset, and its first
        noisy window overwrites a checkpoint that took hours to earn.
        """

        run_dir = os.path.join(settings.MODEL_DIR, settings.RUN_ID)
        os.makedirs(run_dir, exist_ok=True)

        marker = {"level": self.best_level, "score": self.best_score, "episode": episode}

        with open(os.path.join(run_dir, "best.json"), "w") as handle:
            json.dump(marker, handle)


    def _read_best_marker(self):

        path = os.path.join(settings.MODEL_DIR, settings.RUN_ID, "best.json")

        try:
            with open(path) as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            return 0, float("-inf")

        return marker.get("level", 0), marker.get("score", float("-inf"))


    def update_elapsed_time(self, new_time):

        run_log_dir = os.path.join(settings.LOG_DIR, settings.RUN_ID)
        os.makedirs(run_log_dir, exist_ok=True)
        elapsed_time_file = os.path.join(run_log_dir, "elapsed_time.txt")

        try:
            # Read the current elapsed time from the file (if it exists)
            with open(elapsed_time_file, 'r') as file:
                current_time = file.read().strip()

            # Update the time by adding the new value
            updated_time = float(current_time) + float(new_time) if current_time else new_time
            logger.debug(f"Total Elapsed Time: {updated_time} minutes [+{new_time} min/ep]")

        except FileNotFoundError:
            # If the file doesn't exist, start with the new time
            updated_time = new_time
            logger.warning(f"File not found. Starting with new time: {updated_time} minutes")

        # Write the updated time back to the file
        with open(elapsed_time_file, 'w') as file:
            file.write(str(updated_time))

        # Only flush a writer that exists: touching the property here would
        # create the TensorBoard directory purely to flush nothing.
        if self._writer is not None:
            self._writer.flush()


##### End Helpers ########
##########################
