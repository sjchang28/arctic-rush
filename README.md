?# Arctic Rush

MuZero-family agent for Ricochet Robots.

## Running

```bash
python -m src.core.train          # train
python -m tests.test          # play one episode with a checkpoint, rendered
python -m pytest tests -q     # unit + learning tests
```

Everything is configured through environment variables (or a `.env` file) read by
`Settings` in `src/config.py`, so parallel Docker / k8s runs do not collide. Set
`RUN_ID` per run: checkpoints, logs and TensorBoard scalars are all namespaced by it.

```bash
RUN_ID=my_run CURRICULUM_START_MOVES=2 TRAIN_STEPS_PER_EPISODE=100 python -m src.core.train
tensorboard --logdir data/logs
```

### Docker

`docker compose up -d` runs both search modes side by side on the GPU, writing to
`./data/models/<RUN_ID>` and `./data/logs/<RUN_ID>`:

```bash
docker compose up -d --build
docker compose logs -f train-muzero
tensorboard --logdir data/logs        # both runs, side by side
```

The image ships **no seed checkpoint** — every `RUN_ID` starts from a freshly
initialised network. It previously baked in a checkpoint that predates the
network rewrite and can no longer be loaded into it; that file has been removed.

## Search modes

| `SEARCH_MODE` | What the tree expands with | When to use |
|---|---|---|
| `muzero` | the learned dynamics network | the actual MuZero objective |
| `alphazero` | the real simulator, via `snapshot()` / `restore()` | the honest control |

Ricochet Robots is deterministic and fully observable and the simulator is a cheap
loop, so a learned dynamics model buys nothing here and costs a great deal — it
has to rediscover wall and robot blocking before its search means anything. Both
modes share the trunk, the replay buffer and the training loop, so running the
same episode budget under each answers "how much of the difficulty is MuZero?"

## Key settings

[`src/config.py`](src/config.py) is the **single source of truth** for run configuration.
Defaults there are the run you get; nothing restates them elsewhere.
`docker-compose.yaml` sets only `RUN_ID` and the GPU wiring, and the `Dockerfile`
sets only the container paths (`MODEL_DIR`, `LOG_DIR`). Every setting is still
env-overridable for sweeps — but an override that duplicates a default is how
the file you read stops describing the run you got.

| Setting | Default | Notes |
|---|---|---|
| `TRAINING_EPISODES` | 4000 | self-play episodes; the training loop bound |
| `TRAIN_STEPS_PER_EPISODE` | 40 | gradient steps per self-play episode |
| `TOTAL_MCTS_EPISODES` | 50 | simulations per move |
| `MAX_TOTAL_MOVES_PER_GAME` | 25 | move cap per episode |
| `USE_GUMBEL` | true | Gumbel root search; guarantees policy improvement at small budgets |
| `GUMBEL_NUM_CONSIDERED` | 16 | root actions entering sequential halving |
| `REANALYSE_FRACTION` | 0.0 | off: it re-searches with the learned model, which `alphazero` does not plan with |
| `CONSISTENCY_LOSS_WEIGHT` | 2.0 | EfficientZero self-supervised consistency loss |
| `LSTM_HORIZON_LEN` | 5 | steps the value-prefix LSTM accumulates before its state resets |
| `VALUE_PREFIX_DIM` | 128 | width of the value-prefix LSTM |
| `USE_HER` / `HER_FRACTION` | true / 0.5 | hindsight relabelling of failed episodes |
| `CURRICULUM_START_MOVES` | 1 | curriculum depth as an *optimal* solution length; 0 = fully random starts |
| `CURRICULUM_MAX_MOVES` | 15 | deepest level the ramp will try for |
| `CURRICULUM_VERIFY_DEPTH` | true | solve every generated position exactly before using it |
| `CURRICULUM_POOL_SIZE` / `_MIN` / `_REFRESH` | 256 / 24 / 0.15 | reuse of verified positions |
| `SOLVER_MAX_DEPTH` / `SOLVER_NODE_BUDGET` | 16 / 15000 | BFS reach, and expansions before it reports "unknown" |
| `SAVE_BEST_ONLY` | true | overwrite weights only on improvement |
| `CHECKPOINT_WARMUP_EPISODES` | 20 | episodes saved unconditionally at the start |
| `NUM_CHANNELS` / `NUM_BLOCKS` | 64 / 4 | residual trunk size |
| `VALUE_SUPPORT_SIZE` | 10 | categorical value/reward support spans `[-10, 10]` transformed |
| `NUM_UNROLL_STEPS` | 5 | dominates training VRAM |
| `SEARCH_MODE` | `alphazero` | see above |

### Checkpointing

One weights file per run — `MODEL_DIR/RUN_ID/leela.pth` — overwritten only when
the model improves. Saving every episode meant a run that peaked and then
degraded ended with the degraded weights, because the last episode always won.

"Best" is ranked by **(curriculum level, rolling solved rate)**, lexicographically.
The solved rate alone is not a usable score: solving 95% at depth 2 is not better
than 60% at depth 6, so a plain rate comparison would freeze the file at the
first easy level and never write again — which looks exactly like working
checkpointing until the run ends and the weights turn out to be from episode 30.

* A deeper level always wins and resets the bar, so the weights that earned a
  promotion are saved on that episode.
* Within a level, only an improved rate writes.
* The first `CHECKPOINT_WARMUP_EPISODES` save unconditionally, so an early crash
  still leaves a usable file.

A sidecar `best.json` records what the saved weights scored, so a resumed run has
to beat the checkpoint it inherited instead of overwriting it with its first
noisy window. Set `SAVE_BEST_ONLY=false` to go back to saving every episode.

Note the optimizer state is still not saved, so a resumed run restarts AdamW's
moments from zero and is not equivalent to an uninterrupted one.

### Curriculum

`CURRICULUM_START_MOVES=N` generates start positions whose **optimal** solution
is `N` moves, and the level deepens once the agent solves 75% of a 30-episode
window. On a sparse-reward puzzle this is what gives the value network a gradient
before forward exploration would ever reach a goal.

`N` is a measured quantity, not a requested one. Positions are generated, solved
exactly by BFS (`src/game/solver.py`), and kept only if the optimum matches the
level. This is not incidental rigour — it is the whole mechanism. A run using
unverified scramble depth ramped 2 → 12 in ten consecutive windows, never
stalling once, and reported a 96% solved rate on positions that were mostly one
move from the goal at every level.

Three generators are sampled and the solver arbitrates between them. Measured on
`level_01` against BFS:

| generator | mean optimal depth | needs 8+ moves | needs 10+ |
|---|---|---|---|
| backward scramble, any length 2–32 | 1.1 | 0% | 0% |
| forward walk, n=16 | 4.5 | 22% | 5% |
| forward walk, n=64 | 6.4 | 31% | 6% |
| uniform random placement | 5.8 | 21% | 3% |

* **Backward scramble** (sampling predecessor states) lands one move from solved
  ~85% of the time and *does not respond to its length at all* — 2 moves and 32
  moves give the same distribution. Sliding is long-range, so a robot anywhere
  along the target's row or column is still one move out, and a backward random
  walk barely leaves that set. Kept only as a depth-1–2 source.
* **Forward walk** — start on the goal, play `n` random legal moves — does
  respond to `n`, and is what reaches the deep tail. Moves are not invertible, so
  `n` is a mixing parameter, not a distance: the walk can even wander back onto
  the goal. The solver settles it.
* **Uniform placement** is mixed in throughout so positions do not all share the
  solved state as a common ancestor.

Verified positions are pooled and resampled, because proving a depth-6 position
takes longer than the episode that uses it.

Two things worth expecting:

* **The ramp stalls, around depth 8–10** on `level_01` — the level that requests
  8 achieves a mean of 7.5, and deeper levels fall further behind until the
  min-depth gate holds them. That is a generator limit, not a solver one:
  raising `SOLVER_NODE_BUDGET` will not move it, a denser board or a
  depth-directed generator would. Unverified episodes cannot promote a level
  either — an unmeasured position is not a hard one.
* **Verification is not free**: ~0.2 s per episode at depth 4, ~0.8 s at depth 6,
  ~1.3 s at depth 8, against a ~1.5 s episode. Lower `CURRICULUM_POOL_REFRESH` to
  spend less of it, at the cost of variety in the positions the agent sees.
* **`episode/optimal_depth` is the metric to watch**, plotted against
  `episode/curriculum_depth`. The two diverging means the generator is not
  producing the difficulty its level claims. `episode/excess_moves` — moves spent
  above the optimum — is what distinguishes solving from solving *well*.

Levels with bounce pads are scrambled conservatively: a pad rewrites direction
mid-slide, so squares behind one are not valid predecessors.

## Observation and action spaces

The observation is a `13 x 16 x 16` plane stack: 4 wall planes (one per
direction), 4 robot-occupancy planes (one per colour), 4 target planes (one per
colour; an "any" target marks all four), and a broadcast move counter.

The action space is 16: action `i * 4 + d` moves robot `i` in direction `d` until
it is blocked. There is no SWITCH action — tab-to-select is a human interface
affordance, and modelling it made 16 of 20 action ids alias onto 4 distinct
effects while letting the agent burn its move budget on no-ops.

### Value prefix

The reward head does not predict the reward of one specific step. It predicts the
**cumulative** reward accumulated since the LSTM state was last reset, so the
model only has to know a reward arrived somewhere in the window rather than
pinning down exactly when. The LSTM state is carried along a search path and
across the training unroll, and reset every `LSTM_HORIZON_LEN` steps in both — the
reward targets in `make_target` use the same windowing. A node's own reward is
recovered by differencing its prefix against its parent's.

## Rewards

Rewards sit on a `[-1, 1]` scale: `+1` for solving, `-0.01` per move, `-0.04` for
re-entering a previously seen configuration. Solve length is priced through the
per-move cost rather than a large terminal bonus, which keeps the value target
inside the categorical head's support.

## TensorBoard scalars

`episode/reward`, `episode/loss`, `episode/solved_rate`, `episode/solution_length`,
`episode/elapsed_minutes`, and per-head `loss/{value,reward,policy,consistency,grad_norm}`.
Reward alone is a poor progress signal — watch `solved_rate` and `solution_length`.
