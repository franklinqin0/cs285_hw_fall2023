# SAC actor-loss nuances

Notes on the subtle choices in `cs285/agents/soft_actor_critic.py`. All line refs are to
`actor_loss_reinforce`, `actor_loss_reparametrize`, and `update_critic`.

## 1. `sample()` vs `rsample()`

Both draw from the *same* distribution (identical sample values, statistically). The only
difference is the gradient path:

| method       | grad w.r.t. distribution params? | availability                          |
| ------------ | -------------------------------- | ------------------------------------- |
| `.sample()`  | no (detached)                    | every distribution, incl. `Categorical` |
| `.rsample()` | yes (pathwise / reparam trick)   | only if `has_rsample` (e.g. `Normal`) |

So `rsample` is **not** a superset of `sample`: it adds a gradient path but drops support for
non-reparameterizable (discrete) distributions.

### REINFORCE branch -> `sample()`
REINFORCE is a **score-function** estimator. The gradient rides entirely on
`log_prob(action)`; the action itself must be treated as a constant. Using `rsample` would be
semantically wrong. In the current code the sample is inside `torch.no_grad()`, so `rsample`
would *happen* to give the same numbers (the grad path is severed) — but it's fragile: if that
`no_grad` were removed, `rsample` would let `log_prob(action)` backprop through the action too,
injecting a spurious term and silently corrupting the estimator.

### Reparametrize branch -> `rsample()`
Here we *want* the pathwise gradient. `a = mu_theta(s) + sigma_theta(s) * eps` with `eps` a
fixed noise draw, and `a` stays differentiable w.r.t. theta. This branch is deliberately **not**
wrapped in `no_grad`.

The two branches exist precisely because you sometimes can't reparametrize (discrete actions);
using `rsample` in the REINFORCE branch defeats the reason it's there.

## 2. Why the reparametrized gradient is correct

`loss = -mean(Q(s, a_theta))` with `a_theta` from `rsample`. On `loss.backward()`, autograd
applies the chain rule:

```
grad_theta Q(s, a_theta) = grad_a Q(s,a) * grad_theta a_theta
                         = grad_a Q * (grad_theta mu_theta + eps * grad_theta sigma_theta)
```

This is the one-sample Monte Carlo estimate of `grad_theta E_eps[Q(s, mu_theta + sigma_theta*eps)]`.
Two conditions make it work, both satisfied: (a) `rsample` keeps `a` differentiable; (b) no
`no_grad` on this path.

Note: `backward()` also populates grads on the **critic** weights (Q depends on them), but
`update_actor` only steps `self.actor_optimizer`, whose param group holds actor params only. The
critic's accumulated grads are never applied and get zeroed at the next critic update. So Q's
*weights* are effectively fixed and we only differentiate through its *action input* — the
`grad_a Q` we wanted.

## 3. `mean` (actor) vs `q_backup_strategy` / `min` (critic target)

Two places consume Q values, for different purposes:

- **Critic target** (`update_critic`, via `q_backup_strategy`): uses min/clip to fight Q-value
  **overestimation bias**. Targets are bootstrapped into the Bellman target, so a chance
  overestimate compounds through self-bootstrapping. `min` is a pessimistic estimate that
  suppresses that positive bias.
- **Actor loss** (here): just needs a reasonable point estimate of Q to push the policy toward
  high-Q actions. No bootstrap, so no compounding to fight — the ensemble **mean** is fine.
  `-torch.mean(q_values)` over the whole `(num_critic_networks, batch_size)` tensor collapses
  both the ensemble and batch dims at once (equivalent to mean-over-ensemble then mean-over-batch).

(Some SAC implementations do use `min` for the actor too — a different design choice. This
assignment deliberately splits: mean for the actor, backup strategy for the target.)

## 4. Entropy bonus consistency

`actor_loss_reparametrize` returns `(loss, mean_entropy)` and `update_actor` does
`loss -= temperature * entropy`, giving `loss = -mean(Q) - temp*entropy`, i.e. maximizing
`Q + temp*entropy` — the correct SAC reparametrized objective.
