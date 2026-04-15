import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from S20230010071_RL_Assignment1 import PacmanMDP, evaluate_policy, plot_policy_grid, ValueIteration, LiveVisualizer



def state_features(state, mdp):
    pac, ghost, fm = mdp.state_to_components(state)
    rows, cols = mdp.grid_size
    norm = rows + cols


    pr = pac[0] / (rows - 1)
    pc = pac[1] / (cols - 1)


    d_ghost  = mdp.manhattan(pac, ghost) / norm
    danger   = 1.0 if mdp.manhattan(pac, ghost) <= mdp.proximity_threshold else 0.0
    dr_ghost = (pac[0] - ghost[0]) / norm
    dc_ghost = (pac[1] - ghost[1]) / norm


    food_left = bin(fm).count('1') / mdp.num_food


    food_feats = []
    nearest_d = float('inf')
    nearest_dr, nearest_dc = 0.0, 0.0
    for i, fp in enumerate(mdp.food_positions):
        if (fm >> i) & 1:
            d  = mdp.manhattan(pac, fp) / norm
            dr = (fp[0] - pac[0]) / norm
            dc = (fp[1] - pac[1]) / norm
            if mdp.manhattan(pac, fp) < nearest_d:
                nearest_d = mdp.manhattan(pac, fp)
                nearest_dr = dr
                nearest_dc = dc
        else:
            d, dr, dc = 0.0, 0.0, 0.0
        food_feats.extend([d, dr, dc])



    n_d  = nearest_d / norm if nearest_d < float('inf') else 0.0
    n_dr = nearest_dr
    n_dc = nearest_dc

    phi = np.array([1.0, pr, pc,
                    d_ghost, danger, dr_ghost, dc_ghost,
                    food_left,
                    n_d, n_dr, n_dc] + food_feats)
    return phi


def action_features(state, action_idx, mdp):
    phi_s = state_features(state, mdp)
    d     = phi_dim(mdp)
    out   = np.zeros(mdp.num_actions * d)
    out[action_idx * d : (action_idx + 1) * d] = phi_s
    return out



def phi_dim(mdp):
    return 11 + 3 * mdp.num_food    # 20 for the standard 3-food env


def qa_dim(mdp):
    return mdp.num_actions * phi_dim(mdp)


def safe_starts(mdp):
    ghost = mdp.initial_ghost_pos
    excluded = set()
    excluded.add(ghost)
    for act in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        excluded.add(mdp.get_next_position(ghost, act))
    return [p for p in mdp.valid_positions if p not in excluded]


# MC FA

class MCFuncApprox:
    def __init__(self, mdp, alpha=0.002, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.9997, min_epsilon=0.05):
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.w = np.zeros((mdp.num_actions, phi_dim(mdp)))
        self._starts = safe_starts(mdp)

    def q_value(self, state, action_idx):
        phi = state_features(state, self.mdp)
        return self.w[action_idx].dot(phi)

    def pick_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.num_actions)
        qs = [self.q_value(state, a) for a in range(self.mdp.num_actions)]
        return int(np.argmax(qs))

    def run(self, num_episodes=15000, max_steps=150):
        rewards_per_ep = []

        for ep in range(num_episodes):

            start_pos = self._starts[np.random.randint(len(self._starts))]
            state = self.mdp.reset(start_pos)
            episode = []

            s = state
            for _ in range(max_steps):
                a = self.pick_action(s)
                ns, r, done = self.mdp.step(s, a)
                episode.append((s, a, r))
                if done:
                    break
                s = ns

            total_r = sum(r for _, _, r in episode)
            rewards_per_ep.append(total_r)


            G = 0.0
            visited = set()
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = self.gamma * G + r

                if (s, a) not in visited:
                    visited.add((s, a))
                    phi = state_features(s, self.mdp)
                    q_hat = self.w[a].dot(phi)


                    self.w[a] += self.alpha * (G - q_hat) * phi

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        policy = np.array([int(np.argmax([self.q_value(s, a)
                           for a in range(self.mdp.num_actions)]))
                           for s in range(self.mdp.num_states)])
        return policy, rewards_per_ep


# SARSA FA

class SarsaFA:
    def __init__(self, mdp, alpha=0.01, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.9997, min_epsilon=0.05):
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.w = np.zeros((mdp.num_actions, phi_dim(mdp)))
        self._starts = safe_starts(mdp)

    def q_value(self, state, action_idx):
        phi = state_features(state, self.mdp)
        return self.w[action_idx].dot(phi)

    def pick_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.num_actions)
        qs = [self.q_value(state, a) for a in range(self.mdp.num_actions)]
        return int(np.argmax(qs))

    def run(self, num_episodes=30000, max_steps=150):
        rewards_per_ep = []
        alpha_start = self.alpha
        alpha_end   = self.alpha / 10.0

        for ep in range(num_episodes):

            current_alpha = alpha_start + (alpha_end - alpha_start) * (ep / num_episodes)

            start_pos = self._starts[np.random.randint(len(self._starts))]
            state = self.mdp.reset(start_pos)
            action = self.pick_action(state)
            total_r = 0.0

            for _ in range(max_steps):
                next_state, reward, done = self.mdp.step(state, action)
                total_r += reward
                next_action = self.pick_action(next_state)

                phi = state_features(state, self.mdp)
                q_cur = self.w[action].dot(phi)

                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * self.q_value(next_state, next_action)

                td_error = td_target - q_cur
                self.w[action] += current_alpha * td_error * phi

                if done:
                    break
                state = next_state
                action = next_action

            rewards_per_ep.append(total_r)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        policy = np.array([int(np.argmax([self.q_value(s, a)
                           for a in range(self.mdp.num_actions)]))
                           for s in range(self.mdp.num_states)])
        return policy, rewards_per_ep


# LSPI

class LSPI:
    def __init__(self, mdp, gamma=0.9, num_samples=30000, max_iter=20):
        self.mdp = mdp
        self.gamma = gamma
        self.num_samples = num_samples
        self.max_iter = max_iter
        self.d = qa_dim(mdp)   # feature dimension
        self.w = np.zeros(self.d)

    def _collect_samples(self):
        print(f"  collecting {self.num_samples} samples...")
        starts = safe_starts(self.mdp)
        samples = []
        state = self.mdp.reset(starts[np.random.randint(len(starts))])
        epsilon = 0.5

        for _ in range(self.num_samples):
            if np.random.rand() < epsilon:
                action = np.random.randint(self.mdp.num_actions)
            else:
                qs = [action_features(state, a, self.mdp).dot(self.w)
                      for a in range(self.mdp.num_actions)]
                action = int(np.argmax(qs))
            next_state, reward, done = self.mdp.step(state, action)
            samples.append((state, action, reward, next_state, done))
            if done:
                state = self.mdp.reset(starts[np.random.randint(len(starts))])
            else:
                state = next_state
        return samples

    def _lstdq(self, samples, policy):
        A = np.zeros((self.d, self.d))
        b = np.zeros(self.d)

        for s, a, r, ns, done in samples:
            phi = action_features(s, a, self.mdp)
            if done:
                phi_next = np.zeros(self.d)
            else:
                best_a = policy[ns]
                phi_next = action_features(ns, best_a, self.mdp)

            A += np.outer(phi, phi - self.gamma * phi_next)
            b += phi * r


        A += 1e-5 * np.eye(self.d)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w

    def _greedy_policy(self):
        policy = np.zeros(self.mdp.num_states, dtype=int)
        for s in range(self.mdp.num_states):
            qs = [action_features(s, a, self.mdp).dot(self.w)
                  for a in range(self.mdp.num_actions)]
            policy[s] = int(np.argmax(qs))
        return policy

    def run(self):
        samples = self._collect_samples()


        policy = np.random.randint(0, self.mdp.num_actions, self.mdp.num_states)

        for it in range(self.max_iter):
            print(f"  LSPI iteration {it + 1}/{self.max_iter}")
            new_w = self._lstdq(samples, policy)
            self.w = new_w
            new_policy = self._greedy_policy()


            changes = np.sum(policy != new_policy)
            print(f"    policy changes: {changes}")
            policy = new_policy
            if changes == 0:
                print("    converged!")
                break

        return policy


# REINFORCE

class REINFORCE:
    def __init__(self, mdp, alpha=0.005, alpha_v=0.01, gamma=0.9):
        self.mdp = mdp
        self.alpha = alpha
        self.alpha_v = alpha_v
        self.gamma = gamma
        self.theta = np.zeros((mdp.num_actions, phi_dim(mdp)))
        self.w_v   = np.zeros(phi_dim(mdp))
        self._starts = safe_starts(mdp)

    def _softmax_probs(self, state):
        phi_s = state_features(state, self.mdp)
        scores = self.theta.dot(phi_s)
        scores -= scores.max()
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum()

    def pick_action(self, state):
        probs = self._softmax_probs(state)
        return np.random.choice(self.mdp.num_actions, p=probs)

    def run(self, num_episodes=20000, max_steps=150):
        rewards_per_ep = []

        for ep in range(num_episodes):
            start_pos = self._starts[np.random.randint(len(self._starts))]
            state = self.mdp.reset(start_pos)
            episode = []

            s = state
            for _ in range(max_steps):
                a = self.pick_action(s)
                ns, r, done = self.mdp.step(s, a)
                episode.append((s, a, r))
                if done:
                    break
                s = ns

            total_r = sum(r for _, _, r in episode)
            rewards_per_ep.append(total_r)


            returns = []
            G = 0.0
            for _, _, r in reversed(episode):
                G = self.gamma * G + r
                returns.insert(0, G)


            for t, (s, a, _) in enumerate(episode):
                phi_s = state_features(s, self.mdp)
                v_s   = self.w_v.dot(phi_s)
                advantage = returns[t] - v_s


                self.w_v += self.alpha_v * (returns[t] - v_s) * phi_s


                probs = self._softmax_probs(s)
                grad = -np.outer(probs, phi_s)
                grad[a] += phi_s
                self.theta += self.alpha * advantage * grad

        policy = np.array([int(np.argmax(self._softmax_probs(s)))
                           for s in range(self.mdp.num_states)])
        return policy, rewards_per_ep


# Actor-Critic

class ActorCritic:
    def __init__(self, mdp, alpha_actor=0.002, alpha_critic=0.01, gamma=0.9):
        self.mdp = mdp
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        self.theta = np.zeros((mdp.num_actions, phi_dim(mdp)))
        self.w = np.zeros(phi_dim(mdp))
        self._starts = safe_starts(mdp)

    def _softmax_probs(self, state):
        phi_s = state_features(state, self.mdp)
        scores = self.theta.dot(phi_s)
        scores -= scores.max()
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(), phi_s

    def pick_action(self, state):
        probs, _ = self._softmax_probs(state)
        return np.random.choice(self.mdp.num_actions, p=probs)

    def run(self, num_episodes=20000, max_steps=150):
        rewards_per_ep = []

        for ep in range(num_episodes):
            start_pos = self._starts[np.random.randint(len(self._starts))]
            state = self.mdp.reset(start_pos)
            total_r = 0.0

            for _ in range(max_steps):
                probs, phi_s = self._softmax_probs(state)
                action = np.random.choice(self.mdp.num_actions, p=probs)

                next_state, reward, done = self.mdp.step(state, action)
                total_r += reward


                v_cur = self.w.dot(phi_s)
                if done:
                    v_next = 0.0
                else:
                    phi_ns = state_features(next_state, self.mdp)
                    v_next = self.w.dot(phi_ns)

                delta = reward + self.gamma * v_next - v_cur


                self.w += self.alpha_critic * delta * phi_s



                delta_actor = np.clip(delta, -5.0, 5.0)
                grad = -np.outer(probs, phi_s)
                grad[action] += phi_s
                self.theta += self.alpha_actor * delta_actor * grad

                if done:
                    break
                state = next_state

            rewards_per_ep.append(total_r)

        policy = np.array([int(np.argmax(self._softmax_probs(s)[0]))
                           for s in range(self.mdp.num_states)])
        return policy, rewards_per_ep


# Graphs

def smooth(data, window=50):
    if len(data) < window:
        return data
    out = []
    for i in range(len(data)):
        lo = max(0, i - window // 2)
        hi = min(len(data), i + window // 2)
        out.append(np.mean(data[lo:hi]))
    return out


def plot_learning_curves(curves_dict, title="Learning Curves"):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for (label, data), color in zip(curves_dict.items(), colors):
        ax.plot(smooth(data), label=label, color=color, linewidth=2)
        ax.plot(data, color=color, alpha=0.12, linewidth=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_value_heatmap(mdp, w_vec, title="Value Heatmap",
                       ghost_pos=None, food_mask=None):
    if ghost_pos is None:
        ghost_pos = mdp.initial_ghost_pos
    if food_mask is None:
        food_mask = (1 << mdp.num_food) - 1

    grid = np.full(mdp.grid_size, np.nan)
    for r in range(mdp.grid_size[0]):
        for c in range(mdp.grid_size[1]):
            pos = (r, c)
            if pos not in mdp.wall_positions:
                s = mdp.components_to_state(pos, ghost_pos, food_mask)
                grid[r][c] = state_features(s, mdp).dot(w_vec)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap='RdYlGn', aspect='equal')
    plt.colorbar(im, ax=ax, fraction=0.046)
    for r in range(mdp.grid_size[0]):
        for c in range(mdp.grid_size[1]):
            if not np.isnan(grid[r][c]):
                ax.text(c, r, f'{grid[r][c]:.1f}', ha='center', va='center',
                        fontsize=9, color='black')
            else:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color='dimgray'))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(range(mdp.grid_size[1]))
    ax.set_yticks(range(mdp.grid_size[0]))
    plt.tight_layout()
    plt.show()




def main():
    print("=" * 65)
    print("  Assignment 3 - Function Approximation & Policy Gradient")
    print("  Pacman Environment")
    print("=" * 65)

    mdp = PacmanMDP(grid_size=(5, 5), discount=0.9,
                    action_success_prob=0.8, ghost_chase_prob=0.4)
    print(f"\nenvironment: {mdp.num_valid} valid cells, "
          f"{mdp.num_states} states, {mdp.num_actions} actions")
    print(f"feature vector size (state): {phi_dim(mdp)}")
    print(f"feature vector size (Q(s,a)): {qa_dim(mdp)}")

    
    print("\n--- running Value Iteration for baseline comparison ---")
    vi = ValueIteration(mdp, theta=1e-4)
    V_vi, policy_vi = vi.solve()

    print("\n" + "=" * 50)
    print("  1. MC Function Approximation")
    print("=" * 50)
    mc_fa = MCFuncApprox(mdp, alpha=0.002, gamma=0.9)
    print("running 15000 episodes...")
    policy_mc_fa, rewards_mc_fa = mc_fa.run(num_episodes=15000)
    print("done.")

    print("\n" + "=" * 50)
    print("  2. SARSA (TD Function Approximation)")
    print("=" * 50)
    sg_sarsa = SarsaFA(mdp, alpha=0.01, gamma=0.9)
    print("running 30000 episodes...")
    policy_sg_sarsa, rewards_sg_sarsa = sg_sarsa.run(num_episodes=30000)
    print("done.")

    print("\n" + "=" * 50)
    print("  3. Least Squares Policy Iteration (LSPI)")
    print("=" * 50)
    lspi = LSPI(mdp, gamma=0.9, num_samples=80000, max_iter=20)
    policy_lspi = lspi.run()
    print("done.")

    print("\n" + "=" * 50)
    print("  4. REINFORCE (Policy Gradient)")
    print("=" * 50)
    reinforce = REINFORCE(mdp, alpha=0.005, alpha_v=0.01, gamma=0.9)
    print("running 20000 episodes...")
    policy_rf, rewards_rf = reinforce.run(num_episodes=20000)
    print("done.")

    print("\n" + "=" * 50)
    print("  5. Actor-Critic (Policy Gradient)")
    print("=" * 50)
    ac = ActorCritic(mdp, alpha_actor=0.002, alpha_critic=0.01, gamma=0.9)
    print("running 20000 episodes...")
    policy_ac, rewards_ac = ac.run(num_episodes=20000)
    print("done.")

    plot_learning_curves(
        {
            'MC FA': rewards_mc_fa,
            'SARSA FA': rewards_sg_sarsa,
            'REINFORCE': rewards_rf,
            'Actor-Critic': rewards_ac,
        },
        title="Assignment 3 - Learning Curves (FA & Policy Gradient)")

    plot_value_heatmap(mdp, ac.w,
                       title="Actor-Critic: Learned Value Function (V̂)")

    from S20230010071_RL_Assignment2 import compare_policy_heatmaps
    compare_policy_heatmaps(
        mdp,
        {
            'Value Iteration (DP)': policy_vi,
            'MC FA': policy_mc_fa,
            'SARSA FA': policy_sg_sarsa,
            'LSPI': policy_lspi,
            'REINFORCE': policy_rf,
            'Actor-Critic': policy_ac,
        })

    print("\n" + "=" * 50)
    print("  Policy Evaluation (200 test episodes each)")
    print("=" * 50)

    policies_to_eval = [
        ("Value Iteration (DP)", policy_vi),
        ("MC FA              ", policy_mc_fa),
        ("SARSA FA    ", policy_sg_sarsa),
        ("LSPI               ", policy_lspi),
        ("REINFORCE          ", policy_rf),
        ("Actor-Critic       ", policy_ac),
    ]

    for name, pol in policies_to_eval:
        print(f"\n--- {name.strip()} ---")
        evaluate_policy(mdp, pol, num_episodes=200, max_steps=50)

    print("\n" + "=" * 50)
    print("  Gameplay Visualization (1 episode each)")
    print("=" * 50)

    for name, pol in policies_to_eval:
        print(f"\n{'=' * 50}")
        print(f"  {name.strip()}")
        print(f"{'=' * 50}")
        vis = LiveVisualizer(mdp, pol, algo_name=name.strip())
        vis.run_episode(start_pos=(0, 0), max_steps=200, delay=0.3)


if __name__ == "__main__":
    main()
