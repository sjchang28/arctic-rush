# Arctic Rush
## GYM Output
### rollout/
Metric | Meaning
ep_len_mean | Average episode length over the rollout. In your case, it's ~1500 steps per episode. This depends on your environment's max_steps.
ep_rew_mean | Average total reward per episode. Yours is around -2020, which means on average, the agent is still being penalized more than rewarded.
### time/
Metric | Meaning
fps | Frames per second (training speed) — 170 means your model is processing 170 steps/second.
iterations | Number of training iterations (model.learn() updates).
time_elapsed | Time passed (in seconds) since training began.
total_timesteps | Total number of steps taken in the environment.
### train/
Metric | Meaning
approx_kl | Approximate KL divergence (how much the policy changed). Good values are around 0.01–0.02.
clip_fraction | Fraction of actions where the policy was clipped (helps stabilize training). Around 0.1–0.2 is typical.
clip_range | PPO parameter: usually 0.2 (default).
entropy_loss | Measures policy randomness. Lower values → less exploration.
explained_variance | Measures how well the value function explains reward. 1.0 is perfect. You're at 0, meaning the value function isn’t learning well.
learning_rate | Learning rate used by the optimizer.
loss | Combined loss (includes value + policy + entropy losses).
n_updates | Number of training updates performed.
policy_gradient_loss | Loss from policy optimization (negative = minimizing).
value_loss | Loss of the value function — your model is struggling here, 5.42 is moderately high.