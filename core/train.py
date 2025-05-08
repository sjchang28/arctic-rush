import time
import numpy as np
import threading

import torch
import torch.nn.functional as F

from core.mcts import Node, run_mcts, expand_node
from core.muzero import RicochetRobotsConfig, make_ricochet_config
from core.network import Network, SharedStorage, ReplayBuffer
from core.state import RicochetRobotsGame

GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax_sample(distribution, temperature: float):

    visit_counts = np.array([visit_counts for visit_counts, _ in distribution])
    visit_counts_exp = np.exp(visit_counts)
    policy = visit_counts_exp / np.sum(visit_counts_exp)
    policy = (policy ** (1 / temperature)) / \
        (policy ** (1 / temperature)).sum()
    action_index = np.random.choice(range(len(policy)), p=policy)

    return action_index


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
                        pid,
                        episode_number,
                        render_ai=False):

    # for episode in range(config.training_episodes):
    network = storage.get_latest_network()
    game = play_game(config=config, network=network, render_game=render_ai)
    replay_buffer.save_game(game)
    print(f"[Worker {pid} @ Episode {episode_number}] Thread Finished.")


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.

def play_game(config: RicochetRobotsConfig, network: Network, render_game: bool = False) -> RicochetRobotsGame:

    game = config.new_game()

    while not game.is_terminal() and len(game.history) < config.max_moves:

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root,
                    game.legal_actions(),
                    network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)

        if render_game:
            game.environment.render()
    
    return game


##################################
####### Part 2: Training #########


def scalar_loss(prediction, target) -> float:

    # MSE in board games, cross entropy between categorical values in Atari.
    prediction = prediction.view(-1)
    return F.mse_loss(prediction, torch.tensor(target, dtype=torch.float32).to(GPU_DEVICE))


def scale_gradient(tensor, scale: float):

    # Scales the gradient for the backward pass.
    return tensor * scale + tensor.detach() * (1 - scale)


def update_weights(optimizer, network, batch, weight_decay):

    network.train()
    optimizer.zero_grad()

    loss = 0

    for image, actions, targets in batch:
        
        value, reward, _, policy_t, hidden_state = network.initial_inference(
            image)
        predictions = [(1.0, value, reward, policy_t)]

        for action in actions:
            value, reward, _, policy_t, hidden_state = network.recurrent_inference(
                hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_t))
            hidden_state = scale_gradient(hidden_state, 0.5)

        for k, (prediction, target) in enumerate(zip(predictions, targets)):
            gradient_scale, value, reward, policy_t = prediction
            target_value, target_reward, target_policy = target

            l_a = scalar_loss(value, [target_value])
            l_b = scalar_loss(reward, [target_reward]) if k > 0 else 0

            if target_policy == []:
                l_c = 0
            else:
                target_tensor = torch.tensor(
                    [target_policy], dtype=torch.float32).to(GPU_DEVICE)
                l_c = F.cross_entropy(policy_t, target_tensor)

            l = l_a + l_b + l_c
            loss += scale_gradient(l, gradient_scale)

    loss /= len(batch)

    # Weight decay
    for param in network.parameters():
        loss += weight_decay * torch.sum(param ** 2)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_network(config: RicochetRobotsConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, iterations: int) -> float:
    
    network = storage.get_latest_network()
    learning_rate = config.lr_init * \
        (config.lr_decay_rate ** (iterations / config.lr_decay_steps))
    optimizer = torch.optim.SGD(
        network.parameters(), lr=learning_rate, momentum=config.momentum)

    batch = replay_buffer.sample_batch(
        config.num_unroll_steps, config.td_steps, config.action_space_size)
    loss = update_weights(optimizer, network, batch, config.weight_decay)

    network.tot_training_steps += 1
    return loss


##############################
####### Part 3: MuZero #######

def launch_selfplay_jobs(config, storage, replay_buffer, episode_number, render_ai):

    threads = []

    for i in range(config.num_actors):
        print(f"[Worker {i+1} @ Episode {episode_number}] Thread Launched ...")
        t = threading.Thread(target=run_selfplay_worker, args=(
            config, storage, replay_buffer, i+1, episode_number, render_ai))
        t.start()
        threads.append(t)

    # Wait for all to finish
    for t in threads:
        t.join()

    # run_selfplay_worker(config,storage,replay_buffer,1000)


# MuZero training is split into two independent parts:
# Network training and self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.

def muzero(config: RicochetRobotsConfig, render_ai: bool = False):

    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    rewards = []
    losses = []

    for i in range(config.training_episodes):

        t = time.time()
        
        launch_selfplay_jobs(config, storage, replay_buffer, i+1, render_ai)

        # print and plot rewards
        game = replay_buffer.last_game()
        reward_e = game.total_rewards()
        
        if len(rewards) > 100:
            rewards.pop()
        rewards.append(reward_e)
    
        # training
        loss = train_network(config, storage, replay_buffer, i)

        if len(losses) > 100:
            losses.pop()
        losses.append(loss)
        
        
        print(torch.cuda.memory_summary())
        torch.cuda.empty_cache()
        
        print(f'Episode: ({i+1}/{config.training_episodes}) [+Reward: {reward_e} | -Loss: {loss}]')
        print(f'* Moving Average: (20) [+Rewards: {np.mean(rewards[-20:])} | -Losses: {np.mean(losses[-20:])}]')
        print(f'* Moving Average: (100) [+Rewards: {np.mean(rewards[-100:])} | -Losses: {np.mean(losses[-100:])}]')
        
        total_episode_time = str((time.time() - t) / 60)
        storage.update_elapsed_time(total_episode_time)


    config.finish_game()

    config.display_final_stats(rewards=rewards, losses=losses)


######### End Training ###########
##################################


# Entry-point function
if __name__ == "__main__":
    
    muzero(config=make_ricochet_config(render_ai=False), render_ai=False)
