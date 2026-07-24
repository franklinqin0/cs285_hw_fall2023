# DQN & Double Q-Learning — Commit Notes

Explanation of commit `a2d4812` ("hw3: dqn, double Q learning"), which fills in the core DQN
implementation for HW3, including Double Q-learning, with the RL math behind each change.

## 1. Epsilon-greedy action selection (`dqn_agent.py`)

```python
if np.random.rand() < epsilon:
    action = np.random.randint(0, self.num_actions)
else:
    qa_values = self.critic(observation)
    action = torch.argmax(qa_values, dim=1).item()
```

This implements the exploration policy:

$$
a_t = \begin{cases} \text{random action} & \text{with prob. } \epsilon \\ \arg\max_a Q_\phi(s_t, a) & \text{with prob. } 1-\epsilon \end{cases}
$$

With probability $\epsilon$ you explore uniformly; otherwise you act greedily w.r.t. the current
Q-network $Q_\phi$. $\epsilon$ is annealed over training (`exploration_schedule`) so the agent
explores early and exploits later.

## 2. The Bellman target (`dqn_agent.py`)

```python
next_qa_values = self.target_critic(next_obs)
if self.use_double_q:
    next_action = torch.argmax(self.critic(next_obs), dim=1)
else:
    next_action = torch.argmax(next_qa_values, dim=1)
next_q_values = torch.gather(next_qa_values, dim=1, index=next_action.unsqueeze(1)).squeeze(1)
target_values = reward + self.discount * next_q_values * (1 - done.float())
```

This builds the TD target $y$ for the regression. **Standard DQN** uses the target network
$Q_{\phi'}$ for both the argmax and the value:

$$
y = r + \gamma \, (1-d) \max_{a'} Q_{\phi'}(s', a')
$$

The $(1 - d)$ term (`1 - done.float()`) zeros out the bootstrap on terminal transitions — there is
no future value after the episode ends, so the target is just $r$.

**Double Q-learning** (the `use_double_q` branch) decouples *action selection* from *action
evaluation*:

$$
y = r + \gamma \, (1-d) \, Q_{\phi'}\!\big(s', \; \arg\max_{a'} Q_\phi(s', a')\big)
$$

The **online** network $Q_\phi$ picks the argmax action; the **target** network $Q_{\phi'}$
evaluates it. Why: the plain-DQN target $\max_{a'} Q_{\phi'}$ takes a max over noisy estimates, and
$\mathbb{E}[\max] \ge \max \mathbb{E}$, so it systematically **overestimates** action values. Using
two independent networks for selection vs. evaluation breaks this correlation and reduces the upward
bias.

`torch.gather` here picks out the Q-value of the chosen `next_action` from the full
`(batch, num_actions)` tensor.

## 3. The critic loss (`dqn_agent.py`)

```python
qa_values = self.critic(obs)                      # (batch, num_actions)
q_values = torch.gather(qa_values, dim=1, index=action.long().unsqueeze(1)).squeeze(1)
loss = self.critic_loss(q_values, target_values)
```

This minimizes the mean squared Bellman error over the sampled batch:

$$
\mathcal{L}(\phi) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}} \Big[ \big( Q_\phi(s,a) - y \big)^2 \Big]
$$

Note `target_values` was computed under `torch.no_grad()` — gradients flow only through
$Q_\phi(s,a)$, not the target. This is the "semi-gradient" nature of Q-learning; treating $y$ as a
fixed regression target (rather than differentiating through it) is what makes the update
stable-ish.

## 4. Target network update (`dqn_agent.py`)

```python
critic_stats = self.update_critic(obs, action, reward, next_obs, done)
if (step + 1) % self.target_update_period == 0:
    self.update_target_critic()
```

The target network parameters $\phi'$ are periodically copied from the online network:
$\phi' \leftarrow \phi$ every `target_update_period` steps. Holding $\phi'$ fixed between updates
gives the regression a stationary target and prevents the "chasing a moving target" divergence that
plagues naive online Q-learning.

## 5. Gymnasium API migration (env step)

In `run_hw3_dqn.py` and `run_hw3_sac.py`:

```python
next_observation, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

The newer Gym/Gymnasium API splits the old `done` into two flags. This distinction is
**RL-critical**, not cosmetic:

- **`terminated`** — the MDP reached a true terminal state (pole fell, agent died). There is
  genuinely no future value → bootstrap must be cut.
- **`truncated`** — the episode was cut off artificially by a time limit, but the underlying state
  still *has* future value.

Notice the replay buffer stores `done=terminated` (not `done`), so the $(1-d)$ mask in the Bellman
target only zeros the bootstrap on **true terminations**. On a time-limit truncation you still
bootstrap $\gamma \max_{a'} Q(s',a')$, which is correct — otherwise you'd wrongly teach the agent
that reaching the time limit has zero future value.

## 6. Supporting fixes

- `utils.py` — guards `max_length` against `None` so evaluation rollouts with no length cap don't
  crash on `steps > None`.
- `cartpole.yaml` — sets `learning_rate: 0.05`, a hyperparameter tweak for the CartPole DQN
  experiment.

## Summary

The commit implements epsilon-greedy exploration, the (Double) DQN Bellman target with proper
terminal masking, the MSE critic loss with a frozen target, and periodic target-network sync — plus
the Gymnasium `terminated`/`truncated` split that keeps bootstrapping semantically correct.
