import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

# MDP and helper stuff from assignment 1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from S20230010071_RL_Assignment1 import PacmanMDP, evaluate_policy, plot_policy_grid, ValueIteration, LiveVisualizer


#  MONTE CARLO METHODS
class MonteCarloPredict:
    def __init__(self, mdp, gamma=0.9):
        self.mdp = mdp
        self.gamma = gamma
        self.V = np.zeros(mdp.num_states)
        # keep a list of all returns we've seen for each state
        self.returns = defaultdict(list)

    def generate_episode(self, policy, start_state=None, max_steps=100):
        # play one episode from start to finish following the given policy
        if start_state is None:
            # exploring starts: random position ensures all states get value estimates,
            # not just those reachable from the fixed (0,0) start
            start_pos = self.mdp.valid_positions[np.random.randint(self.mdp.num_valid)]
            start_state = self.mdp.reset(start_pos)
        state = start_state
        episode = []   # will store (state, action, reward) tuples

        for _ in range(max_steps):
            action = policy[state]
            next_state, reward, done = self.mdp.step(state, action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def run(self, policy, num_episodes=2000, max_steps=100):
        delta_history = []  # track max change in V per episode, just to see convergence

        for ep in range(num_episodes):
            episode = self.generate_episode(policy, max_steps=max_steps)

            # walk backwards through the episode and compute returns
            G = 0.0
            visited = set()
            max_change = 0.0

            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = self.gamma * G + r

                if s not in visited:   # first visit only
                    visited.add(s)
                    self.returns[s].append(G)
                    old_v = self.V[s]
                    self.V[s] = np.mean(self.returns[s])
                    max_change = max(max_change, abs(old_v - self.V[s]))

            delta_history.append(max_change)

        return self.V, delta_history


class GLIE:
    def __init__(self, mdp, gamma=0.9, epsilon_schedule='1/k'):
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule  # '1/k', '1/sqrt(k)', 'harmonic'
        self.Q = np.zeros((mdp.num_states, mdp.num_actions))
        self.counts = defaultdict(int)  # visit counts for incremental mean
        self.policy = np.zeros(mdp.num_states, dtype=int)
        self.epsilon = 1.0

    def pick_action(self, state):
        # epsilon-greedy with current epsilon
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.num_actions)
        return int(np.argmax(self.Q[state]))

    def generate_episode(self, max_steps=100):
        start_pos = self.mdp.valid_positions[np.random.randint(self.mdp.num_valid)]
        state = self.mdp.reset(start_pos)
        episode = []
        for _ in range(max_steps):
            action = self.pick_action(state)
            next_state, reward, done = self.mdp.step(state, action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def run(self, num_episodes=5000, max_steps=100):
        rewards_per_ep = []

        for k in range(1, num_episodes + 1):
            if self.epsilon_schedule == '1/k':
                self.epsilon = 1.0 / k
            elif self.epsilon_schedule == '1/sqrt(k)':
                self.epsilon = 1.0 / np.sqrt(k)
            else:  # harmonic: 100/(100+k), stays high longer
                self.epsilon = 100.0 / (100.0 + k)

            episode = self.generate_episode(max_steps=max_steps)
            total_reward = sum(r for _, _, r in episode)
            rewards_per_ep.append(total_reward)

            # backward pass to compute returns, first-visit MC
            G = 0.0
            visited_pairs = set()

            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = self.gamma * G + r

                if (s, a) not in visited_pairs:
                    visited_pairs.add((s, a))
                    self.counts[(s, a)] += 1
                    n = self.counts[(s, a)]
                    self.Q[s][a] += (G - self.Q[s][a]) / n  # incremental mean
                    self.policy[s] = int(np.argmax(self.Q[s]))

        return self.Q, self.policy, rewards_per_ep



#  TEMPORAL DIFFERENCE 
class TDPrediction:
    def __init__(self, mdp, alpha=0.05, gamma=0.9):
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros(mdp.num_states)

    def run(self, policy, num_episodes=2000, max_steps=100):
        delta_history = []

        for ep in range(num_episodes):
            start_pos = self.mdp.valid_positions[np.random.randint(self.mdp.num_valid)]
            state = self.mdp.reset(start_pos)
            max_change = 0.0

            for _ in range(max_steps):
                action = policy[state]
                next_state, reward, done = self.mdp.step(state, action)

                if done:
                    target = reward          
                else:
                    target = reward + self.gamma * self.V[next_state]

                td_err = target - self.V[state]
                old_v = self.V[state]
                self.V[state] += self.alpha * td_err
                max_change = max(max_change, abs(self.V[state] - old_v))

                if done:
                    break
                state = next_state

            delta_history.append(max_change)

        return self.V, delta_history


class SARSA:
    def __init__(self, mdp, alpha=0.1, gamma=0.9, epsilon=0.1,
                 epsilon_decay=0.9998, min_epsilon=0.01):
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = np.zeros((mdp.num_states, mdp.num_actions))
        self.policy = np.zeros(mdp.num_states, dtype=int)

    def pick_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.num_actions)
        return int(np.argmax(self.Q[state]))

    def run(self, num_episodes=5000, max_steps=100):
        rewards_per_ep = []

        for ep in range(num_episodes):
            start_pos = self.mdp.valid_positions[np.random.randint(self.mdp.num_valid)]
            state = self.mdp.reset(start_pos)
            action = self.pick_action(state)   # pick first action before loop
            total_r = 0.0

            for _ in range(max_steps):
                next_state, reward, done = self.mdp.step(state, action)
                total_r += reward

                next_action = self.pick_action(next_state)

                # SARSA update - using next_action which is from same policy
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.Q[next_state][next_action]

                self.Q[state][action] += self.alpha * (target - self.Q[state][action])

                if done:
                    break
                state = next_state
                action = next_action

            rewards_per_ep.append(total_r)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # extract greedy policy from learned Q
        for s in range(self.mdp.num_states):
            self.policy[s] = int(np.argmax(self.Q[s]))

        return self.Q, self.policy, rewards_per_ep



#  TD LAMBDA
class TDLambda:
    def __init__(self, mdp, alpha=0.05, gamma=0.9, lam=0.8):
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.V = np.zeros(mdp.num_states)

    def run(self, policy, num_episodes=2000, max_steps=100):
        delta_history = []

        for ep in range(num_episodes):
            start_pos = self.mdp.valid_positions[np.random.randint(self.mdp.num_valid)]
            state = self.mdp.reset(start_pos)
            # eligibility traces reset at the start of each episode
            e = np.zeros(self.mdp.num_states)
            max_change = 0.0

            for _ in range(max_steps):
                action = policy[state]
                next_state, reward, done = self.mdp.step(state, action)

                if done:
                    td_error = reward - self.V[state]
                else:
                    td_error = reward + self.gamma * self.V[next_state] - self.V[state]
                e[state] = 1.0

                # update all V values weighted by trace
                old_V = self.V.copy()
                self.V += self.alpha * td_error * e

                max_change = max(max_change, np.max(np.abs(self.V - old_V)))

                # decay all traces
                e *= self.gamma * self.lam

                if done:
                    break
                state = next_state

            delta_history.append(max_change)

        return self.V, delta_history

# SARSA LAMBDA
class SARSALambda:
    def __init__(self, mdp, alpha=0.05, gamma=0.9, lam=0.8, epsilon=0.1,
                 epsilon_decay=0.9998, min_epsilon=0.01):
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = np.zeros((mdp.num_states, mdp.num_actions))
        self.policy = np.zeros(mdp.num_states, dtype=int)

    def pick_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.num_actions)
        return int(np.argmax(self.Q[state]))

    def run(self, num_episodes=5000, max_steps=100):
        rewards_per_ep = []

        for ep in range(num_episodes):
            start_pos = self.mdp.valid_positions[np.random.randint(self.mdp.num_valid)]
            state = self.mdp.reset(start_pos)
            action = self.pick_action(state)
            # traces for each (s, a) pair
            e = np.zeros((self.mdp.num_states, self.mdp.num_actions))
            total_r = 0.0

            for _ in range(max_steps):
                next_state, reward, done = self.mdp.step(state, action)
                total_r += reward
                next_action = self.pick_action(next_state)

                if done:
                    td_error = reward - self.Q[state][action]
                else:
                    td_error = (reward
                                + self.gamma * self.Q[next_state][next_action]
                                - self.Q[state][action])

                # replacing traces: cap at 1 to prevent exploding updates
                # accumulating traces (+=1) multiply the td_error by visit count
                # in an episode, causing high-variance Q updates on revisited states
                e[state][action] = 1.0

                # spread update across all (s,a) via traces
                self.Q += self.alpha * td_error * e
                e *= self.gamma * self.lam

                if done:
                    break
                state = next_state
                action = next_action

            rewards_per_ep.append(total_r)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        for s in range(self.mdp.num_states):
            self.policy[s] = int(np.argmax(self.Q[s]))

        return self.Q, self.policy, rewards_per_ep


#  PLOTTING / VISUALISATION HELPERS
def plot_glie_schedule_comparison(mdp, num_episodes=5000):
    """Run GLIE with three different epsilon schedules and compare win rates."""
    print("\ncomparing GLIE epsilon schedules...")
    schedules = ['1/k', '1/sqrt(k)', 'harmonic']
    labels = ['1/k  (standard GLIE)', '1/√k  (slower decay)', '100/(100+k)  (harmonic)']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # show how fast each schedule decays
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ks = np.arange(1, num_episodes + 1)
    eps_curves = {
        '1/k': 1.0 / ks,
        '1/sqrt(k)': 1.0 / np.sqrt(ks),
        'harmonic': 100.0 / (100.0 + ks),
    }
    for sched, label, color in zip(schedules, labels, colors):
        ax1.plot(ks, eps_curves[sched], label=label, color=color, linewidth=2)
    ax1.set_xlabel('Episode k')
    ax1.set_ylabel('ε')
    ax1.set_title('Epsilon Decay Schedules')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # train GLIE with each schedule and plot smoothed rewards
    policies = {}
    for sched, label, color in zip(schedules, labels, colors):
        print(f"[GLIE {sched}] running {num_episodes} episodes...")
        g = GLIE(mdp, gamma=0.9, epsilon_schedule=sched)
        _, pol, rewards = g.run(num_episodes=num_episodes)
        policies[sched] = pol
        print("  done.")
        ax2.plot(smooth(rewards), label=label, color=color, linewidth=2)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward (smoothed)')
    ax2.set_title('GLIE: Learning Curves by Epsilon Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return policies  # dict: schedule_name -> policy


def smooth(data, window=50):
    # simple moving average to make reward curves less noisy
    if len(data) < window:
        return data
    out = []
    for i in range(len(data)):
        lo = max(0, i - window // 2)
        hi = min(len(data), i + window // 2)
        out.append(np.mean(data[lo:hi]))
    return out


def plot_learning_curves(curves_dict, title="Learning Curves", ylabel="Episode Reward"):
    """Compare multiple algorithms on the same plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for (label, data), color in zip(curves_dict.items(), colors):
        sm = smooth(data)
        ax.plot(sm, label=label, color=color, linewidth=2)
        # faint raw curve in background
        ax.plot(data, color=color, alpha=0.15, linewidth=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_delta_curves(curves_dict, title="Value Convergence (Max Change per Episode)"):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for (label, data), color in zip(curves_dict.items(), colors):
        ax.plot(smooth(data, window=20), label=label, color=color, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Max |ΔV|")
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_value_functions(mdp, V_vi, V_mc, V_td, V_tdl,
                             ghost_pos=None, food_mask=None):
    """
    Show heatmaps of the value function learned by each method side by side.
    Fixing ghost position and food mask so we can actually visualise it.
    """
    if ghost_pos is None:
        ghost_pos = mdp.initial_ghost_pos
    if food_mask is None:
        food_mask = (1 << mdp.num_food) - 1

    # build grid values for each method
    def make_grid(V):
        grid = np.full(mdp.grid_size, np.nan)
        for r in range(mdp.grid_size[0]):
            for c in range(mdp.grid_size[1]):
                pos = (r, c)
                if pos not in mdp.wall_positions:
                    s = mdp.components_to_state(pos, ghost_pos, food_mask)
                    grid[r][c] = V[s]
        return grid

    labels = ['Value Iteration (DP)', 'MC Prediction', 'TD(0)', 'TD(λ=0.8)']
    Vs = [V_vi, V_mc, V_td, V_tdl]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Value Functions (ghost @ {ghost_pos}, food={bin(food_mask)})',
                 fontsize=13, fontweight='bold')

    for ax, label, V in zip(axes, labels, Vs):
        g = make_grid(V)
        im = ax.imshow(g, cmap='RdYlGn', aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046)
        # annotate cells
        for r in range(mdp.grid_size[0]):
            for c in range(mdp.grid_size[1]):
                if not np.isnan(g[r][c]):
                    ax.text(c, r, f'{g[r][c]:.0f}', ha='center', va='center',
                            fontsize=8, color='black')
                else:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               color='dimgray'))
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(range(mdp.grid_size[1]))
        ax.set_yticks(range(mdp.grid_size[0]))

    plt.tight_layout()
    plt.show()


def plot_lambda_comparison(mdp, policy, lambdas=(0.0, 0.3, 0.6, 0.9),
                           num_episodes=2000):
    """Run TD(lambda) for different lambda values and compare convergence."""
    print("\nrunning TD(lambda) for different lambda values...")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))

    for lam, color in zip(lambdas, colors):
        tdl = TDLambda(mdp, alpha=0.05, gamma=0.9, lam=lam)
        _, delta_hist = tdl.run(policy, num_episodes=num_episodes)
        ax.plot(smooth(delta_hist, window=30), label=f'λ={lam}',
                color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Max |ΔV|")
    ax.set_title("TD(λ) Convergence for Different λ Values")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_policy_heatmaps(mdp, policies_dict, ghost_pos=None, food_mask=None):
    """Show learned policies from different algorithms side by side as arrow grids."""
    if ghost_pos is None:
        ghost_pos = mdp.initial_ghost_pos
    if food_mask is None:
        food_mask = (1 << mdp.num_food) - 1

    arrow_map = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→', 'STAY': '●'}
    n = len(policies_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Learned Policies (ghost @ {ghost_pos})',
                 fontsize=13, fontweight='bold')

    for ax, (title, policy) in zip(axes, policies_dict.items()):
        ax.set_xlim(-0.5, mdp.grid_size[1] - 0.5)
        ax.set_ylim(mdp.grid_size[0] - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=11, fontweight='bold')

        for r in range(mdp.grid_size[0]):
            for c in range(mdp.grid_size[1]):
                pos = (r, c)
                if pos in mdp.wall_positions:
                    from matplotlib.patches import Rectangle
                    ax.add_patch(Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                           color='dimgray', ec='black'))
                else:
                    s = mdp.components_to_state(pos, ghost_pos, food_mask)
                    act = mdp.actions[policy[s]]
                    ax.text(c, r, arrow_map[act], ha='center', va='center',
                            fontsize=22, color='navy', fontweight='bold')

        # mark ghost
        ax.plot(ghost_pos[1], ghost_pos[0], 'rs', markersize=18, alpha=0.5)
        ax.text(ghost_pos[1], ghost_pos[0] - 0.38, 'ghost',
                ha='center', fontsize=9, color='red')

        # mark food
        for fp in mdp.food_positions:
            ax.plot(fp[1], fp[0], 'g*', markersize=14, alpha=0.6)

    plt.tight_layout()
    plt.show()




def main():
    print("=" * 65)
    print("  Assignment 2 - Model Free Prediction & Control")
    print("  Pacman Environment")
    print("=" * 65)

    # --- setup environment ---
    mdp = PacmanMDP(grid_size=(5, 5), discount=0.9,
                    action_success_prob=0.8, ghost_chase_prob=0.4)
    print(f"\nenvironment: {mdp.num_valid} valid cells, "
          f"{mdp.num_states} states, {mdp.num_actions} actions")

    # --- DP baseline from assignment 1 ---
    print("\n--- running Value Iteration (DP baseline) ---")
    vi = ValueIteration(mdp, theta=1e-4)
    V_vi, policy_vi = vi.solve()

    # we'll use the VI policy for prediction tasks
    # (prediction = estimating V under a fixed policy, not improving it)
    fixed_policy = policy_vi


    # PART 1: Prediction
    print("\n" + "=" * 50)
    print("  PART 1 - Prediction (estimating V under VI policy)")
    print("=" * 50)

    NUM_PRED_EPS = 10000

    # MC prediction
    print(f"\n[MC Prediction] running {NUM_PRED_EPS} episodes...")
    mc_pred = MonteCarloPredict(mdp, gamma=0.9)
    V_mc, mc_deltas = mc_pred.run(fixed_policy, num_episodes=NUM_PRED_EPS)
    print("  done.")

    # TD(0) prediction
    print(f"[TD(0) Prediction] running {NUM_PRED_EPS} episodes...")
    td0 = TDPrediction(mdp, alpha=0.05, gamma=0.9)
    V_td, td_deltas = td0.run(fixed_policy, num_episodes=NUM_PRED_EPS)
    print("  done.")

    # TD(lambda) prediction
    print(f"[TD(λ=0.8) Prediction] running {NUM_PRED_EPS} episodes...")
    tdl = TDLambda(mdp, alpha=0.05, gamma=0.9, lam=0.8)
    V_tdl, tdl_deltas = tdl.run(fixed_policy, num_episodes=NUM_PRED_EPS)
    print("  done.")

    # --- compare convergence ---
    plot_delta_curves(
        {'MC': mc_deltas, 'TD(0)': td_deltas, 'TD(λ=0.8)': tdl_deltas},
        title="Prediction Convergence - MC vs TD(0) vs TD(λ)")

    # --- compare value functions with DP ---
    compare_value_functions(mdp, V_vi, V_mc, V_td, V_tdl)

    # --- lambda comparison ---
    plot_lambda_comparison(mdp, fixed_policy, lambdas=(0.0, 0.3, 0.6, 0.9),
                           num_episodes=5000)

    # PART 2: Control
    print("\n" + "=" * 50)
    print("  PART 2 - Control (learning policy from scratch)")
    print("=" * 50)

    NUM_CTRL_EPS = 30000

    glie_policies = plot_glie_schedule_comparison(mdp, num_episodes=NUM_CTRL_EPS)

    glie_rewards_1k = []
    policy_glie = glie_policies['1/k']

    print(f"[SARSA] running {NUM_CTRL_EPS} episodes...")
    sarsa = SARSA(mdp, alpha=0.1, gamma=0.9, epsilon=0.1)
    _, policy_sarsa, sarsa_rewards = sarsa.run(num_episodes=NUM_CTRL_EPS)
    print("  done.")

    print(f"[SARSA(λ=0.8)] running {NUM_CTRL_EPS} episodes...")
    sarsa_lam = SARSALambda(mdp, alpha=0.05, gamma=0.9, lam=0.8, epsilon=0.1)
    _, policy_sarsa_lam, sarsal_rewards = sarsa_lam.run(num_episodes=NUM_CTRL_EPS)
    print("  done.")

    # --- learning curves ---
    plot_learning_curves(
        {
            'GLIE 1/k': glie_rewards_1k if glie_rewards_1k else [0],
            'SARSA': sarsa_rewards,
            'SARSA(λ)': sarsal_rewards,
        },
        title="Control: Episode Rewards During Training",
        ylabel="Total Episode Reward")

    # --- policy comparison heatmaps ---
    compare_policy_heatmaps(
        mdp,
        {
            'Value Iteration (DP)': policy_vi,
            'GLIE MC Control': policy_glie,
            'SARSA': policy_sarsa,
            'SARSA(λ)': policy_sarsa_lam,
        })

    # PART 3: Evaluation
    print("\n" + "=" * 50)
    print("  PART 3 - Policy Evaluation (200 test episodes each)")
    print("=" * 50)

    policies_to_eval = [
        ("Value Iteration (DP)  ", policy_vi),
        ("GLIE 1/k              ", glie_policies['1/k']),
        ("GLIE 1/sqrt(k)        ", glie_policies['1/sqrt(k)']),
        ("GLIE 100/(100+k)      ", glie_policies['harmonic']),
        ("SARSA                 ", policy_sarsa),
        ("SARSA(lambda)         ", policy_sarsa_lam),
    ]

    for name, pol in policies_to_eval:
        print(f"\n--- {name.strip()} ---")
        evaluate_policy(mdp, pol, num_episodes=200, max_steps=50)

    # PART 4: Gameplay Visualization
    print("\n" + "=" * 50)
    print("  PART 4 - Gameplay Visualization (1 episode each)")
    print("=" * 50)

    for name, pol in policies_to_eval:
        print(f"\n{'=' * 50}")
        print(f"  {name.strip()}")
        print(f"{'=' * 50}")
        vis = LiveVisualizer(mdp, pol, algo_name=name.strip())
        vis.run_episode(start_pos=(0, 0), max_steps=200, delay=0.3)
if __name__ == "__main__":
    main()