import collections
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.config import settings
from src.core.boot import stream_banner
from src.model.config import (
    EMPTY_CACHE_EVERY_N_EPISODES,
    MIN_REPLAY_GAMES,
)
from src.model.device import gpu_device
from src.model.logging_lines import excess_moves, log_episode, log_moving_averages
from src.model.mcts import (
    Node,
    RealEnvironmentModel,
    expand_node,
    expand_root,
    run_gumbel_mcts,
    run_mcts,
)
from src.model.muzero import RicochetRobotsConfig, make_ricochet_config
from src.model.network import Network
from src.model.plots import display_final_stats
from src.model.promotion import maybe_promote_curriculum
from src.model.replay import ReplayBuffer
from src.model.reporting import report_ready
from src.model.state import RicochetRobotsGame
from src.model.storage import SharedStorage
from src.model.support import scalar_to_support, support_to_scalar
from src.model.types import ActionHistory


def softmax_sample(distribution, temperature: float):
    """Sample an action from (visit_count, action) pairs.

    Returns the *action id*, not its position in `distribution` -- the two differ
    whenever the root is expanded over a subset of the action space, which is
    always, since the root is masked to legal actions.

    The distribution is `count ** (1 / T)` normalised, per MuZero. Exponentiating
    the raw counts instead collapses it to effectively one-hot (counts run up to
    `num_simulations`), which made the temperature schedule inert.
    """

    visit_counts = np.array([count for count, _ in distribution], dtype=np.float64)
    actions = [action for _, action in distribution]

    if temperature <= 0:
        return actions[int(np.argmax(visit_counts))]

    policy = visit_counts ** (1 / temperature)
    total = policy.sum()

    if total <= 0:
        # No simulation reached any child; fall back to uniform over legal actions.
        policy = np.ones_like(policy) / len(policy)
    else:
        policy = policy / total

    return actions[int(np.random.choice(len(policy), p=policy))]


def select_action(config: RicochetRobotsConfig,
                  num_moves: int,
                  node: Node,
                  network: Network):

    visit_counts = [(child.visit_count, action)
                    for action, child in node.children.items()]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
    action = softmax_sample(visit_counts, t)
    return action


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.

def add_exploration_noise(config: RicochetRobotsConfig, node: Node):

    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

##### End Helpers ########
##########################


##################################
####### Part 1: Self-Play ########

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.

def run_selfplay_worker(config: RicochetRobotsConfig,
                        storage: SharedStorage,
                        replay_buffer: ReplayBuffer,
                        render_ai=False):

    # for episode in range(config.training_episodes):
    network = storage.get_latest_network()
    game = play_game(config=config, network=network, render_game=render_ai)

    # The finished game lives in the replay buffer for hundreds of episodes, so
    # drop its pygame environment before archiving it.
    game.release_environment()

    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.

@torch.no_grad()
def play_game(config: RicochetRobotsConfig, network: Network, render_game: bool = False) -> RicochetRobotsGame:

    # Self-play is inference only. Without no_grad every hidden state produced
    # here keeps its autograd graph — and the GPU activations behind it — alive
    # for as long as the search tree (and then the replay buffer) holds it.
    game = config.new_game()

    # SEARCH_MODE=alphazero searches the real simulator instead of the learned
    # dynamics network. Both back-ends share this loop, the tree, the trunk and
    # the training code (see mcts.RealEnvironmentModel).
    model = RealEnvironmentModel(game.environment) if config.search_mode == "alphazero" else None

    while not game.is_terminal() and len(game.history) < config.max_moves:

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        legal_actions = expand_root(config, root, network, game, model)

        if not legal_actions:
            # No robot can move: the position is a dead end, not a solve.
            break

        # We then run a Monte Carlo Tree Search using only action sequences and
        # whichever model of the environment this run is configured with.
        if config.use_gumbel:
            # Gumbel noise at the root *is* the exploration, so Dirichlet noise is
            # not applied; the policy target is the completed-Q improved policy
            # rather than visit fractions.
            action, improved_policy = run_gumbel_mcts(
                config, root, game.action_history(), network, model)
            game.apply(action)
            game.store_search_statistics(root, policy=improved_policy)
        else:
            add_exploration_noise(config, root)
            run_mcts(config, root, game.action_history(), network, model)
            action = select_action(config, len(game.history), root, network)
            game.apply(action)
            game.store_search_statistics(root)

        if render_game:
            game.environment.render()
    
    return game


##################################
####### Part 2: Training #########


def scale_gradient(tensor, scale: float):

    # Scales the gradient for the backward pass.
    return tensor * scale + tensor.detach() * (1 - scale)


def categorical_loss(logits, target_scalars, support_size):
    """Cross entropy of a categorical head against two-hot encoded targets.

    Replaces the previous `F.mse_loss` on raw scalars. With rewards on the old
    1000 scale that MSE dwarfed the policy loss by orders of magnitude and drove
    the run to NaN within a handful of episodes; a categorical head over a
    squashed support is scale-free and can also represent a bimodal return.
    """

    target = scalar_to_support(target_scalars, support_size)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def consistency_loss(predicted, target):
    """EfficientZero's self-supervised consistency loss (negative cosine).

    The dynamics network's predicted next hidden state is pulled towards the
    representation of the observation that actually followed. Without this, the
    dynamics network is only ever supervised through value and reward error and
    is free to drift into any latent that fits those, rather than modelling wall
    and robot geometry. `target` is detached, SimSiam-style, so gradients flow
    only into the predicted branch.
    """

    predicted = F.normalize(predicted, dim=-1, eps=1e-8)
    target = F.normalize(target.detach(), dim=-1, eps=1e-8)

    return -(predicted * target).sum(dim=-1)


def update_weights(optimizer, network, batch, config):
    """One gradient step over a stacked batch.

    The whole batch is unrolled together. The previous implementation looped over
    batch elements and ran a separate forward pass per sample.
    """

    network.train()
    optimizer.zero_grad(set_to_none=True)

    support_size = network.support_size
    unroll_steps = batch["actions"].shape[1]
    weights = batch["weights"]

    hidden_state = network.representation(batch["observations"])
    policy_logits = network.policy_net(hidden_state)
    value_logits = network.value_net(hidden_state)

    # Step 0 has no incoming reward, so only value and policy are supervised.
    value_losses = categorical_loss(value_logits, batch["target_values"][:, 0], support_size)
    reward_losses = torch.zeros_like(value_losses)
    policy_losses = (
        -(batch["target_policies"][:, 0] * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1)
        * batch["policy_mask"][:, 0]
    )
    consistency_losses = torch.zeros_like(value_losses)

    # Value error at the root is what the replay priority is keyed on.
    with torch.no_grad():
        root_value = support_to_scalar(value_logits, support_size)
        priorities = (root_value - batch["target_values"][:, 0]).abs().cpu().numpy()

    gradient_scale = 1.0 / max(unroll_steps, 1)

    # Value-prefix LSTM state, carried across the unroll and reset every
    # `lstm_horizon_len` steps. The reward target is the cumulative reward over
    # the same window (see RicochetRobotsGame.make_target), so the model only has
    # to know a reward arrived somewhere in the window, not exactly when.
    reward_hidden = network.initial_reward_hidden(
        batch["observations"].shape[0], hidden_state.device)

    for k in range(unroll_steps):

        if k > 0 and k % config.lstm_horizon_len == 0:
            reward_hidden = network.initial_reward_hidden(
                batch["observations"].shape[0], hidden_state.device)

        action = batch["actions"][:, k]

        hidden_state = network.dynamics(hidden_state, action)
        reward_logits, reward_hidden = network.value_prefix(hidden_state, reward_hidden)
        value_logits = network.value_net(hidden_state)
        policy_logits = network.policy_net(hidden_state)

        step_value = categorical_loss(value_logits, batch["target_values"][:, k + 1], support_size)
        step_reward = categorical_loss(reward_logits, batch["target_rewards"][:, k + 1], support_size)
        step_policy = (
            -(batch["target_policies"][:, k + 1] * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1)
            * batch["policy_mask"][:, k + 1]
        )

        # Consistency: compare against the representation of the observation that
        # actually followed this action. Masked past the end of the trajectory.
        if config.consistency_loss_weight > 0:
            with torch.no_grad():
                target_state = network.representation(batch["next_observations"][:, k])
                target_projection = network.project(target_state, with_predictor=False)

            predicted_projection = network.project(hidden_state, with_predictor=True)
            step_consistency = (
                consistency_loss(predicted_projection, target_projection)
                * batch["next_mask"][:, k]
            )
            consistency_losses = consistency_losses + scale_gradient(step_consistency, gradient_scale)

        value_losses = value_losses + scale_gradient(step_value, gradient_scale)
        reward_losses = reward_losses + scale_gradient(step_reward, gradient_scale)
        policy_losses = policy_losses + scale_gradient(step_policy, gradient_scale)

        # Halve the gradient flowing back through the recurrent path, per MuZero.
        hidden_state = scale_gradient(hidden_state, 0.5)

    per_sample = (
        value_losses
        + reward_losses
        + policy_losses
        + config.consistency_loss_weight * consistency_losses
    )

    # Importance-sampling weights correct the bias from prioritised sampling.
    loss = (per_sample * weights).mean()

    loss.backward()

    # Clip before stepping: the unrolled graph occasionally produces a large
    # gradient early on, and one such step is enough to wreck the run.
    grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)

    optimizer.step()

    parts = {
        "value": value_losses.mean().item(),
        "reward": reward_losses.mean().item(),
        "policy": policy_losses.mean().item(),
        "consistency": consistency_losses.mean().item(),
        "grad_norm": float(grad_norm),
    }

    return loss.item(), priorities, parts


@torch.no_grad()
def reanalyse_game(config, network, game):
    """Refresh a stored trajectory's value and policy targets with the current network.

    MuZero Reanalyse. With a 100-game window and one game added per episode, a
    trajectory's targets are otherwise generated by a network hundreds of steps
    stale; re-running the search over stored positions fixes that at the cost of
    inference only, with no new environment interaction. This is the cheapest
    large sample-efficiency gain available to this codebase.

    Search here runs on the *learned* model, so no environment is needed -- which
    is what makes it applicable to games already released from their environment.
    """

    network.eval()

    if not game.child_visits:
        return

    for index in range(len(game.child_visits)):

        root = Node(0)
        observation = game.make_image(index)

        # The stored trajectory does not record which actions were legal at each
        # step, so reanalyse expands over the full action space. Illegal actions
        # are still absent from the *behaviour* targets that self-play recorded.
        expand_node(root,
                    list(range(config.action_space_size)),
                    network.initial_inference(observation))

        history = ActionHistory(game.history[:index], config.action_space_size)

        # Reanalyse must produce the same *kind* of target self-play produces,
        # otherwise the network is trained against two different objectives.
        if config.use_gumbel:
            _, improved_policy = run_gumbel_mcts(config, root, history, network)
            game.child_visits[index] = improved_policy
        else:
            add_exploration_noise(config, root)
            run_mcts(config, root, history, network)

            total_visits = sum(child.visit_count for child in root.children.values())
            if total_visits == 0:
                continue

            game.child_visits[index] = [
                root.children[a].visit_count / total_visits if a in root.children else 0
                for a in range(config.action_space_size)
            ]

        game.root_values[index] = root.value()



def maybe_reanalyse(config, network, replay_buffer):

    fraction = settings.REANALYSE_FRACTION
    if fraction <= 0 or len(replay_buffer) == 0:
        return

    if config.search_mode != "muzero":
        # Reanalyse re-searches stored positions with the learned model. Under
        # SEARCH_MODE=alphazero the search needs a live environment, and stored
        # games have already released theirs -- refreshing targets with a
        # different model than self-play used would train against two objectives.
        return

    count = max(1, int(round(fraction * len(replay_buffer))))
    indices = np.random.choice(len(replay_buffer), size=min(count, len(replay_buffer)), replace=False)

    for index in indices:
        reanalyse_game(config, network, replay_buffer.buffer[index])


def train_network(config: RicochetRobotsConfig,
                  storage: SharedStorage,
                  replay_buffer: ReplayBuffer,
                  optimizer,
                  num_steps: int):
    """Take `num_steps` gradient steps and return the mean loss over them.

    This used to take exactly one step per call and be called once per episode,
    so a default 120-episode run performed 120 SGD steps in total. The LR
    schedule is now driven by the network's global step count rather than the
    episode index, which is what `lr_decay_steps` was always written against.
    """

    network = storage.get_latest_network()
    step_losses = []
    last_parts = {}

    for _ in range(num_steps):

        # Apply the decay schedule in place. Rebuilding the optimizer every step
        # would throw away its moment estimates and re-allocate their state.
        learning_rate = config.lr_init * \
            (config.lr_decay_rate ** (network.tot_training_steps / config.lr_decay_steps))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate

        batch = replay_buffer.sample_batch(
            config.num_unroll_steps, config.td_steps, config.action_space_size)

        loss, priorities, last_parts = update_weights(optimizer, network, batch, config)
        replay_buffer.update_priorities(batch["indices"], priorities)

        step_losses.append(loss)

        del batch

        network.tot_training_steps += 1

    mean_loss = float(np.mean(step_losses)) if step_losses else float('nan')
    return mean_loss, last_parts


##############################
####### Part 3: MuZero #######

def launch_selfplay_jobs(config, storage, replay_buffer, render_ai):

    threads = []

    for _ in range(config.num_actors):
        t = threading.Thread(target=run_selfplay_worker, args=(
            config, storage, replay_buffer, render_ai))
        t.start()
        threads.append(t)

    # Wait for all to finish
    for t in threads:
        t.join()



# MuZero training is split into two independent parts:
# Network training and self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.

def muzero(config: RicochetRobotsConfig, render_ai: bool = False):

    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    # One optimizer for the whole run so its state survives between episodes.
    # AdamW rather than SGD+momentum: the exponential decay schedule was tuned
    # for a 500k-step run that never happened, and the categorical losses here
    # have very different curvature across heads.
    optimizer = torch.optim.AdamW(storage.get_latest_network().parameters(),
                                  lr=config.lr_init,
                                  weight_decay=config.weight_decay)

    # Reported here rather than in the banner: the network is built by now, so
    # every line is a fact about this run instead of a promise.
    report_ready(config, storage.get_latest_network(), gpu_device())

    rewards = collections.deque(maxlen=100)
    losses = collections.deque(maxlen=100)
    # Reward alone is a poor progress signal; track whether the puzzle was
    # actually solved and in how many moves.
    solved = collections.deque(maxlen=100)
    # ...and how many moves it *needed*, so a solved rate can be read against the
    # difficulty it was earned on rather than against a scramble-depth label.
    depths = collections.deque(maxlen=100)

    for i in range(config.training_episodes):

        t = time.time()

        launch_selfplay_jobs(config, storage, replay_buffer, render_ai)

        # print and plot rewards
        game = replay_buffer.last_game()
        reward_e = game.total_rewards()

        rewards.append(reward_e)

        # Every episode counts towards its own depth's record, which is what
        # decides the depth is worth rehearsing.
        config.record_result(game.optimal_depth, game.is_terminal())

        # Rehearsals are trained on like any other game but kept out of the
        # curriculum statistics: they are practice at a level the agent has
        # already left, so counting them would measure the wrong thing twice over
        # -- inflating the solved rate with easy wins, and dragging the mean
        # measured depth below the gate's own min-depth ratio.
        if not game.was_rehearsal:
            solved.append(1.0 if game.is_terminal() else 0.0)
            depths.append(game.optimal_depth)

        # training. Hold off until the buffer has a few games, otherwise the
        # first batches are every position of a single trajectory.
        loss_parts = {}
        if len(replay_buffer) >= MIN_REPLAY_GAMES:

            # Refresh stale targets on stored games before training on them.
            maybe_reanalyse(config, storage.get_latest_network(), replay_buffer)

            loss, loss_parts = train_network(config, storage, replay_buffer, optimizer,
                                             settings.TRAIN_STEPS_PER_EPISODE)
        else:
            loss = float('nan')

        losses.append(loss)

        # empty_cache() forces a device sync, so it is not worth doing every episode.
        if (i + 1) % EMPTY_CACHE_EVERY_N_EPISODES == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        recent_rewards, recent_losses = list(rewards), list(losses)
        solved_rate = float(np.mean(list(solved)))

        log_episode(i + 1, config.training_episodes, game, reward_e, loss)
        log_moving_averages(i + 1, recent_rewards, recent_losses, list(solved), list(depths))

        maybe_promote_curriculum(config, solved, depths, i + 1)

        total_episode_time = str((time.time() - t) / 60)
        storage.log_scalars(step=i + 1, reward=reward_e, loss=loss, elapsed_min=total_episode_time,
                            solved_rate=solved_rate, solution_length=game.total_moves(),
                            loss_parts=loss_parts, curriculum_moves=config.curriculum_moves,
                            optimal_depth=game.optimal_depth,
                            excess_moves=excess_moves(game))
        # Ranked by (curriculum level, rolling solved rate). Called after the
        # promotion check on purpose: the weights that just earned a promotion
        # are the ones worth keeping, and they are saved on that episode.
        storage.save_if_best(level=config.curriculum_moves, score=solved_rate, episode=i + 1)

        storage.update_elapsed_time(total_episode_time)


    config.finish_game()

    display_final_stats(rewards=list(rewards), losses=list(losses))


######### End Training ###########
##################################


# Entry-point function
if __name__ == "__main__":
    
    stream_banner()
    muzero(config=make_ricochet_config(render_ai=False), render_ai=False)
