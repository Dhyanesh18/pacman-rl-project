import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys, os, random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from S20230010071_RL_Assignment1 import (
    PacmanMDP, ValueIteration, evaluate_policy, LiveVisualizer
)
from S20230010071_RL_Assignment3 import (
    state_features, phi_dim, safe_starts
)
from S20230010071_RL_Assignment2 import compare_policy_heatmaps


class FeatureEncoder:
    def __init__(self, mdp):
        self.mdp = mdp
        self.dim = phi_dim(mdp)

    def encode(self, state):
        return state_features(state, self.mdp).astype(np.float32)


class RawStateEncoder:
    def __init__(self, mdp):
        self.mdp = mdp
        rows, cols = mdp.grid_size
        self.norm_r = rows - 1
        self.norm_c = cols - 1
        self.dim = 4 + mdp.num_food   # pac(r,c) + ghost(r,c) + food bits

    def encode(self, state):
        pac, ghost, fm = self.mdp.state_to_components(state)
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = pac[0]   / self.norm_r
        vec[1] = pac[1]   / self.norm_c
        vec[2] = ghost[0] / self.norm_r
        vec[3] = ghost[1] / self.norm_c
        for i in range(self.mdp.num_food):
            vec[4 + i] = float((fm >> i) & 1)
        return vec


def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class MLP:
    def __init__(self, layer_sizes, seed=None):
        rng = np.random.default_rng(seed)
        self.params = []
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialisation: std = sqrt(2 / fan_in)
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out, dtype=np.float32)
            self.params.extend([W, b])


    def forward(self, x, return_cache=False):
        """
        x: 1-D float32 array of length layer_sizes[0]
        returns output vector of length layer_sizes[-1]
        If return_cache=True also returns intermediates needed for backprop
        """
        cache = []
        h = x.astype(np.float32)
        for i in range(0, len(self.params) - 2, 2):
            W, b = self.params[i], self.params[i + 1]
            z = h @ W + b
            a = relu(z)
            if return_cache:
                cache.append((h, z, W))
            h = a
        # final layer: linear (no activation)
        W_last, b_last = self.params[-2], self.params[-1]
        z_out = h @ W_last + b_last
        if return_cache:
            cache.append((h, z_out, W_last))
        if return_cache:
            return z_out, cache
        return z_out

    def backward(self, x, loss_grad_out, lr):
        """
        Backprop from a gradient on the output layer.
        loss_grad_out: dL/d(z_out), shape (output_dim,)
        Updates params in-place
        """
        _, cache = self.forward(x, return_cache=True)
        num_layers = len(self.params) // 2
        delta = loss_grad_out.astype(np.float32)

        for i in reversed(range(num_layers)):
            h_in, z, W = cache[i]
            dW = np.outer(h_in, delta)
            db = delta
            if i > 0:
                # propagate delta back through ReLU of previous layer
                delta = (delta @ W.T) * relu_grad(cache[i - 1][1])
            # gradient norm clipping — prevents exploding gradients that caused
            # NaN in the smoke test when the actor loss was large early on
            grad_norm = np.sqrt(np.sum(dW ** 2) + np.sum(db ** 2))
            if grad_norm > 1.0:
                scale = 1.0 / grad_norm
                dW *= scale
                db  = db * scale
            self.params[2 * i]     -= lr * dW
            self.params[2 * i + 1] -= lr * db

    
    def compute_gradient(self, x, loss_grad_out):
        """Return list of (dW, db) tuples for each layer, without updating"""
        _, cache = self.forward(x, return_cache=True)
        num_layers = len(self.params) // 2
        delta = loss_grad_out.astype(np.float32)
        grads = [None] * num_layers

        for i in reversed(range(num_layers)):
            h_in, z, W = cache[i]
            dW = np.outer(h_in, delta)
            db = delta.copy()
            grads[i] = (dW, db)
            if i > 0:
                delta = (delta @ W.T) * relu_grad(cache[i - 1][1])
        return grads

    def apply_gradients(self, grads, lr):
        """Apply pre-computed gradients with per-layer gradient norm clipping"""
        for i, (dW, db) in enumerate(grads):
            grad_norm = np.sqrt(np.sum(dW ** 2) + np.sum(db ** 2))
            if grad_norm > 1.0:
                scale = 1.0 / grad_norm
                dW = dW * scale
                db = db * scale
            self.params[2 * i]     -= lr * dW
            self.params[2 * i + 1] -= lr * db

    def copy_weights_from(self, other):
        """Hard-copy weights from another MLP (used for target network)"""
        for i in range(len(self.params)):
            self.params[i] = other.params[i].copy()

    def clone(self):
        cloned = MLP(self.layer_sizes)
        cloned.copy_weights_from(self)
        return cloned



class AdamOptimizer:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def apply(self, params, grads, max_grad_norm=1.0):
        self.t += 1
        alpha = (self.lr
                 * np.sqrt(1.0 - self.beta2 ** self.t)
                 / (1.0 - self.beta1 ** self.t))
        for i, (dW, db) in enumerate(grads):
            # per-layer gradient norm clipping before momentum
            gnorm = np.sqrt(np.sum(dW ** 2) + np.sum(db ** 2))
            if gnorm > max_grad_norm:
                scale = max_grad_norm / gnorm
                dW = dW * scale
                db = db * scale
            wi, bi = 2 * i, 2 * i + 1
            self.m[wi] = self.beta1 * self.m[wi] + (1.0 - self.beta1) * dW
            self.m[bi] = self.beta1 * self.m[bi] + (1.0 - self.beta1) * db
            self.v[wi] = self.beta2 * self.v[wi] + (1.0 - self.beta2) * dW * dW
            self.v[bi] = self.beta2 * self.v[bi] + (1.0 - self.beta2) * db * db
            params[wi] -= alpha * self.m[wi] / (np.sqrt(self.v[wi]) + self.eps)
            params[bi] -= alpha * self.m[bi] / (np.sqrt(self.v[bi]) + self.eps)

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, state_enc, action, reward, next_enc, done):
        self.buf.append((state_enc, action, reward, next_enc, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, nexts, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.array(nexts,  dtype=np.float32),
                np.array(dones,  dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQN:
    def __init__(self, mdp, encoder,
                 hidden=(128, 64),
                 lr=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=15000,
                 buffer_capacity=20000,
                 batch_size=64,
                 target_update_freq=200,
                 name='DQN'):
        self.mdp = mdp
        self.encoder = encoder
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.name = name

        in_dim = encoder.dim
        layer_sizes = [in_dim] + list(hidden) + [mdp.num_actions]
        self.online_net = MLP(layer_sizes, seed=42)
        self.target_net = self.online_net.clone()

        self.optim = AdamOptimizer(self.online_net.params, lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        self._steps = 0

    def pick_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.num_actions)
        enc = self.encoder.encode(state)
        q_vals = self.online_net.forward(enc)
        return int(np.argmax(q_vals))

    def _update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, nexts, dones = self.buffer.sample(self.batch_size)

        # accumulate gradients over the full batch before applying
        num_layers = len(self.online_net.params) // 2
        acc_grads = [(np.zeros_like(self.online_net.params[2 * j]),
                      np.zeros_like(self.online_net.params[2 * j + 1]))
                     for j in range(num_layers)]

        batch_loss = 0.0
        for i in range(self.batch_size):
            q_pred = self.online_net.forward(states[i])
            q_target_all = self.target_net.forward(nexts[i])
            td_target = (rewards[i]
                         + (1.0 - dones[i]) * self.gamma * q_target_all.max())

            # Huber loss gradient for the chosen action
            td_err = td_target - q_pred[actions[i]]
            delta = np.clip(td_err, -10.0, 10.0)   # Huber clip (wide for large rewards)

            grad_out = np.zeros(self.mdp.num_actions, dtype=np.float32)
            grad_out[actions[i]] = -delta / self.batch_size

            grads = self.online_net.compute_gradient(states[i], grad_out)
            for j in range(num_layers):
                acc_grads[j] = (acc_grads[j][0] + grads[j][0],
                                acc_grads[j][1] + grads[j][1])
            batch_loss += 0.5 * td_err ** 2

        # single weight update with accumulated gradient (Adam)
        self.optim.apply(self.online_net.params, acc_grads)
        return batch_loss / self.batch_size

    # ---- main training loop --------------------------------------------------
    def run(self, num_episodes=3000, max_steps=150):
        rewards_per_ep = []
        losses = []
        starts = safe_starts(self.mdp)

        for ep in range(num_episodes):
            start_pos = starts[np.random.randint(len(starts))]
            state = self.mdp.reset(start_pos)
            total_r = 0.0
            ep_loss = 0.0
            n_updates = 0

            for _ in range(max_steps):
                action = self.pick_action(state)
                next_state, reward, done = self.mdp.step(state, action)
                total_r += reward

                enc  = self.encoder.encode(state)
                nenc = self.encoder.encode(next_state)
                self.buffer.push(enc, action, reward, nenc, float(done))

                loss = self._update()
                if loss > 0:
                    ep_loss  += loss
                    n_updates += 1

                self._steps += 1

                # decay epsilon linearly
                self.epsilon = max(self.epsilon_end,
                                   self.epsilon - self.epsilon_decay)

                # sync target network every C environment steps
                if self._steps % self.target_update_freq == 0:
                    self.target_net.copy_weights_from(self.online_net)

                if done:
                    break
                state = next_state

            rewards_per_ep.append(total_r)
            losses.append(ep_loss / max(n_updates, 1))

            if (ep + 1) % 500 == 0:
                avg = np.mean(rewards_per_ep[-500:])
                print(f"  [{self.name}] ep {ep+1}/{num_episodes} "
                      f"| avg reward: {avg:.1f} | ε: {self.epsilon:.3f}")

        # build a greedy policy array for compatibility with A1-A3 helpers
        policy = np.array([
            int(np.argmax(self.online_net.forward(self.encoder.encode(s))))
            for s in range(self.mdp.num_states)
        ])
        return policy, rewards_per_ep, losses


def smooth(data, window=100):
    if len(data) < window:
        return np.array(data, dtype=float)
    out = np.empty(len(data))
    for i in range(len(data)):
        lo = max(0, i - window // 2)
        hi = min(len(data), i + window // 2)
        out[i] = np.mean(data[lo:hi])
    return out


def plot_learning_curves(curves_dict, title="Learning Curves",
                         ylabel="Total Episode Reward", window=100):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for (label, data), color in zip(curves_dict.items(), colors):
        sm = smooth(data, window)
        ax.plot(sm,   label=label, color=color, linewidth=2)
        ax.plot(data, color=color, alpha=0.12, linewidth=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dqn_loss(losses_dict, title="DQN Training Loss"):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['tab:blue', 'tab:orange']
    for (label, losses), color in zip(losses_dict.items(), colors):
        ax.plot(smooth(losses, 50), label=label, color=color, linewidth=2)
        ax.plot(losses, color=color, alpha=0.1, linewidth=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Huber Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_final_bar(eval_results, title="Policy Evaluation Comparison"):
    """
    Bar chart comparing win rate and average reward across all trained agents.
    eval_results: list of (name, win_rate, avg_reward) tuples
    """
    names      = [r[0] for r in eval_results]
    win_rates  = [r[1] for r in eval_results]
    avg_rewards= [r[2] for r in eval_results]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    bars1 = ax1.bar(x, win_rates, width, color='steelblue', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate (ate all food)")
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars1, win_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    bars2 = ax2.bar(x, avg_rewards, width, color='darkorange', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax2.set_ylabel("Avg Episode Reward")
    ax2.set_title("Average Reward (200 test episodes)")
    for bar, val in zip(bars2, avg_rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.5 if val >= 0 else -2),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_encoder_comparison(curves_fa, curves_raw, algo_label, window=100):
    """Side-by-side: FA-encoded input vs raw-state input for the same algorithm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{algo_label}: Feature Encoding Comparison', fontsize=13, fontweight='bold')

    for ax, (label, data) in zip(axes, [('20-dim FA features', curves_fa),
                                         ('Raw state (8-dim)',  curves_raw)]):
        ax.plot(smooth(data, window), color='tab:blue', linewidth=2, label='smoothed')
        ax.plot(data, color='tab:blue', alpha=0.15, linewidth=0.7, label='raw')
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  EVALUATION HELPER
# =============================================================================

def eval_policy_stats(mdp, policy, num_episodes=200, max_steps=100,
                       start_pos=None):
    """
    Run num_episodes rollouts and return (win_pct, avg_reward, avg_len).
    """
    if start_pos is None:
        start_pos = mdp.initial_pacman_pos
    rewards, wins, lengths = [], 0, []

    for _ in range(num_episodes):
        state = mdp.reset(start_pos)
        total_r = 0.0
        for t in range(max_steps):
            ns, r, done = mdp.step(state, policy[state])
            total_r += r
            if done:
                _, _, fm = mdp.state_to_components(ns)
                if fm == 0:
                    wins += 1
                lengths.append(t + 1)
                break
            state = ns
        else:
            lengths.append(max_steps)
        rewards.append(total_r)

    win_pct = 100.0 * wins / num_episodes
    return win_pct, np.mean(rewards), np.mean(lengths)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Assignment 4 — Deep RL: DQN on PacManMDP")
    print("  (builds on A1-A3; justifies transition to neural approximators)")
    print("=" * 70)

    mdp = PacmanMDP(grid_size=(5, 5), discount=0.9,
                    action_success_prob=0.8, ghost_chase_prob=0.4)

    print(f"\nenvironment: {mdp.num_valid} valid cells, "
          f"{mdp.num_states} states, {mdp.num_actions} actions")

    # encoders
    fa_enc  = FeatureEncoder(mdp)
    raw_enc = RawStateEncoder(mdp)
    print(f"FA encoder dim : {fa_enc.dim}  (20-dim hand-crafted from A3)")
    print(f"Raw encoder dim: {raw_enc.dim}  (normalised coords + food bits)")

    # =========================================================================
    #  DP Baseline (from A1 — used for comparison)
    # =========================================================================
    print("\n--- Value Iteration (DP baseline from A1) ---")
    vi = ValueIteration(mdp, theta=1e-4)
    V_vi, policy_vi = vi.solve()

    # =========================================================================
    #  DQN — FA features
    # =========================================================================
    print("\n" + "=" * 50)
    print("  DQN  |  input: 20-dim FA features")
    print("=" * 50)
    dqn_fa = DQN(mdp, fa_enc,
                 hidden=(128, 64),
                 lr=5e-4,
                 gamma=0.9,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=1_200_000,
                 buffer_capacity=100_000,
                 batch_size=64,
                 target_update_freq=200,
                 name='DQN-FA')
    print("running 100000 episodes...")
    policy_dqn_fa, rew_dqn_fa, loss_dqn_fa = dqn_fa.run(num_episodes=100_000)
    print("done.")

    # =========================================================================
    #  DQN — Raw state
    # =========================================================================
    print("\n" + "=" * 50)
    print("  DQN  |  input: 8-dim raw state")
    print("=" * 50)
    dqn_raw = DQN(mdp, raw_enc,
                  hidden=(128, 64),
                  lr=5e-4,
                  gamma=0.9,
                  epsilon_start=1.0,
                  epsilon_end=0.05,
                  epsilon_decay_steps=1_200_000,
                  buffer_capacity=100_000,
                  batch_size=64,
                  target_update_freq=200,
                  name='DQN-Raw')
    print("running 100000 episodes...")
    policy_dqn_raw, rew_dqn_raw, loss_dqn_raw = dqn_raw.run(num_episodes=100_000)
    print("done.")


    # 1. All learning curves together
    plot_learning_curves(
        {
            'DQN  (FA features)' : rew_dqn_fa,
            'DQN  (raw state)'   : rew_dqn_raw,
        },
        title="Assignment 4 — Learning Curves: DQN (FA vs Raw)")

    # 2. DQN-specific: Huber loss convergence
    plot_dqn_loss(
        {'DQN-FA': loss_dqn_fa, 'DQN-Raw': loss_dqn_raw},
        title="DQN Training Loss (Huber) — shows target network stabilisation")
    
    plot_encoder_comparison(rew_dqn_fa, rew_dqn_raw, 'DQN')

    print("\n" + "=" * 50)
    print("  Policy Evaluation (200 test episodes each)")
    print("=" * 50)

    agents = [
        ("Value Iteration (DP)", policy_vi),
        ("DQN - FA features",    policy_dqn_fa),
        ("DQN - Raw state",      policy_dqn_raw)
    ]

    eval_results = []
    for name, pol in agents:
        win_pct, avg_r, avg_len = eval_policy_stats(mdp, pol, num_episodes=200)
        print(f"\n  {name}")
        print(f"    win rate  : {win_pct:.1f}%")
        print(f"    avg reward: {avg_r:.2f}")
        print(f"    avg steps : {avg_len:.1f}")
        eval_results.append((name, win_pct, avg_r))

    plot_final_bar(eval_results,
                   title="Assignment 4 — Final Evaluation: All Agents")
    compare_policy_heatmaps(
        mdp,
        {name: pol for name, pol in agents},
    )

    print("\n" + "=" * 50)
    print("  Gameplay Visualisation")
    print("=" * 50)

    for name, pol in agents:
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")
        vis = LiveVisualizer(mdp, pol, algo_name=name)
        vis.run_episode(start_pos=(0, 0), max_steps=200, delay=0.3)


if __name__ == "__main__":
    main()
