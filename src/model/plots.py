"""End-of-run plots.

Split out of `muzero.py`, which held configuration construction and matplotlib
rendering in one module. Two unrelated reasons to change, and one concrete cost:
`matplotlib.use("Agg")` sat at the top of `muzero.py`, so importing a *config
object* switched the matplotlib backend for the whole process. Anything else in
the interpreter that wanted a different backend silently lost.

That call now lives here, next to the only code that needs it, and runs when
plotting is imported rather than when a config is built.
"""

import os

import matplotlib

# Headless by construction: training runs in containers with no display, and the
# figures are written to disk rather than shown.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import settings  # noqa: E402
from src.core.logger import logger  # noqa: E402


def display_final_stats(rewards, losses):
    """Write the loss-vs-reward scatter and the two boxplots for this run."""

    # Episodes before the replay buffer fills record no loss; drop those
    # pairs rather than plotting NaNs.
    paired = [(r, l) for r, l in zip(rewards, losses) if not np.isnan(l)]
    if not paired:
        logger.warning("No completed training steps to plot.")
        return

    # Sort by rewards, preserving index alignment
    sorted_rewards, sorted_losses = zip(*sorted(paired))
    losses = list(sorted_losses)

    plt.figure()
    plt.plot(sorted_rewards, sorted_losses, marker='o', label='Loss vs Reward')

    # Adding labels and title
    plt.xlabel('Rewards')
    plt.ylabel('Losses')
    plt.title('Loss Decreases as Rewards Increase')
    plt.legend()
    plt.grid(True)

    # Histogram - Rewards
    plt.figure()
    plt.boxplot([rewards], tick_labels=['Rewards'])

    plt.title("Boxplot of Rewards")
    plt.ylabel("Value")

    # Histogram - Losses
    plt.figure()
    plt.boxplot([losses], tick_labels=['Losses'])

    plt.title("Boxplot of Losses")
    plt.ylabel("Value")

    out_dir = os.path.join(settings.LOG_DIR, settings.RUN_ID)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(1).savefig(os.path.join(out_dir, "loss_vs_reward.png"))
    plt.figure(2).savefig(os.path.join(out_dir, "rewards_boxplot.png"))
    plt.figure(3).savefig(os.path.join(out_dir, "losses_boxplot.png"))
    plt.close("all")
