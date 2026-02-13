import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class PacmanMDP:
    def __init__(self, grid_size=(5, 5), discount=0.9,
                 action_success_prob=0.8, ghost_chase_prob=0.4):
        self.grid_size = grid_size
        self.discount = discount
        self.action_success_prob = action_success_prob
        self.ghost_chase_prob = ghost_chase_prob

        # 5 possible actions pacman can take
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        self.num_actions = len(self.actions)
        self._delta = {
            'UP': (-1, 0), 'DOWN': (1, 0),
            'LEFT': (0, -1), 'RIGHT': (0, 1),
            'STAY': (0, 0),
        }
        self._perpendicular = {
            'UP': ['LEFT', 'RIGHT'], 'DOWN': ['LEFT', 'RIGHT'],
            'LEFT': ['UP', 'DOWN'], 'RIGHT': ['UP', 'DOWN'],
            'STAY': [],
        }
        self.wall_positions = [(1, 1), (1, 3), (3, 1), (3, 3)]

        # food pellets - need to eat all of them to win
        self.food_positions = [(0, 4), (2, 2), (4, 0)]
        self.num_food = len(self.food_positions)
        self.num_food_states = 1 << self.num_food          # basically 2^num_food

        self.initial_pacman_pos = (0, 0)
        self.initial_ghost_pos = (2, 4)

        self.valid_positions = [
            (r, c)
            for r in range(grid_size[0])
            for c in range(grid_size[1])
            if (r, c) not in self.wall_positions
        ]
        self.num_valid = len(self.valid_positions)
        self._pos_idx = {p: i for i, p in enumerate(self.valid_positions)}

        self.num_states = self.num_valid * self.num_valid * self.num_food_states

        # reward structure
        self.food_reward = 10
        self.all_food_bonus = 100           # extra points for getting the last food
        self.collision_penalty = -100
        self.step_penalty = -1
        self.proximity_threshold = 2      # how close is too close to ghost
        self.proximity_penalty = -5

        # pre-calculate everything so we don't have to keep computing it
        self._transitions = {}
        self._precompute_transitions()

    def state_to_components(self, state):
        # break down the state into its parts
        food_mask = state % self.num_food_states
        rem = state // self.num_food_states
        gi = rem % self.num_valid
        pi = rem // self.num_valid
        return self.valid_positions[pi], self.valid_positions[gi], food_mask

    def components_to_state(self, pac_pos, ghost_pos, food_mask):
        # turn the components back into a single state number
        pi = self._pos_idx[pac_pos]
        gi = self._pos_idx[ghost_pos]
        return (pi * self.num_valid + gi) * self.num_food_states + food_mask

    def is_valid_position(self, pos):
        r, c = pos
        return (0 <= r < self.grid_size[0]
                and 0 <= c < self.grid_size[1]
                and pos not in self.wall_positions)

    def get_next_position(self, pos, action):
        dr, dc = self._delta[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        return nxt if self.is_valid_position(nxt) else pos

    @staticmethod
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _pacman_distribution(self, pac_pos, action):
        # pacman moves in intended direction 80% of the time
        # and slips perpendicular 10% each way
        # STAY is the only deterministic action
        if action == 'STAY':
            return {pac_pos: 1.0}
        dist = {}
        intended = self.get_next_position(pac_pos, action)
        dist[intended] = dist.get(intended, 0.0) + self.action_success_prob
        slip = (1.0 - self.action_success_prob) / 2.0
        for perp in self._perpendicular[action]:
            p = self.get_next_position(pac_pos, perp)
            dist[p] = dist.get(p, 0.0) + slip
        return dist

    def _ghost_distribution(self, ghost_pos, pac_pos):
        # ghost tries to chase pacman with some probability
        # otherwise moves randomly
        neighbours = set()
        for act in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
            neighbours.add(self.get_next_position(ghost_pos, act))
        neighbours.add(ghost_pos)                          # ghost can stay put too
        neighbours = list(neighbours)

        dist = {n: 0.0 for n in neighbours}

        # chasing behavior
        if self.ghost_chase_prob > 0 and ghost_pos != pac_pos:
            cur_d = self.manhattan(ghost_pos, pac_pos)
            chase = [n for n in neighbours
                     if self.manhattan(n, pac_pos) < cur_d]
            if chase:
                each = self.ghost_chase_prob / len(chase)
                for n in chase:
                    dist[n] += each
            else:                                           # stuck, can't get closer
                each = self.ghost_chase_prob / len(neighbours)
                for n in neighbours:
                    dist[n] += each

        # random movement component
        rand_each = (1.0 - self.ghost_chase_prob) / len(neighbours)
        for n in neighbours:
            dist[n] += rand_each
        return dist

    def get_reward(self, pac_pos, ghost_pos, food_mask):
        # calculate reward based on current state
        # food_mask is what food is available before eating
        if pac_pos == ghost_pos:
            return self.collision_penalty

        reward = self.step_penalty

        # check if we ate food
        for i, fp in enumerate(self.food_positions):
            if pac_pos == fp and (food_mask >> i) & 1:
                reward += self.food_reward
                # was this the last food pellet?
                new_mask = food_mask & ~(1 << i)
                if new_mask == 0:
                    reward += self.all_food_bonus
                break

        # penalty for being too close to ghost
        d = self.manhattan(pac_pos, ghost_pos)
        if 0 < d <= self.proximity_threshold:
            reward += self.proximity_penalty * (1.0 - d / (self.proximity_threshold + 1))
        return reward

    def is_terminal(self, state):
        pac, ghost, fm = self.state_to_components(state)
        return pac == ghost or fm == 0          # game over if caught or all food eaten

    def _update_food(self, food_mask, pac_pos):
        for i, fp in enumerate(self.food_positions):
            if pac_pos == fp and (food_mask >> i) & 1:
                food_mask &= ~(1 << i)
        return food_mask

    def _precompute_transitions(self):
        # build the transition table ahead of time
        print(f"precomputing transitions for {self.num_states} states and {self.num_actions} actions...")

        for state in range(self.num_states):
            pac, ghost, fm = self.state_to_components(state)

            if self.is_terminal(state):
                for ai in range(self.num_actions):
                    self._transitions[(state, ai)] = [(state, 1.0, 0.0)]
                continue

            for ai, action in enumerate(self.actions):
                pac_d = self._pacman_distribution(pac, action)
                ghost_d = self._ghost_distribution(ghost, pac)

                agg = {}                               # aggregate outcomes
                for np_, pp in pac_d.items():
                    for ng, gp in ghost_d.items():
                        prob = pp * gp
                        if prob < 1e-12:
                            continue
                        new_fm = self._update_food(fm, np_)
                        r = self.get_reward(np_, ng, fm)
                        ns = self.components_to_state(np_, ng, new_fm)
                        if ns in agg:
                            agg[ns] = (agg[ns][0] + prob, r)
                        else:
                            agg[ns] = (prob, r)

                self._transitions[(state, ai)] = [
                    (ns, p, r) for ns, (p, r) in agg.items()
                ]
        print("done precomputing transitions")

    def get_transitions(self, state, action_idx):
        # lookup transitions from our pre-computed table
        return self._transitions[(state, action_idx)]

    def reset(self, start_pos=None):
        if start_pos is None:
            start_pos = self.initial_pacman_pos
        food_mask = (1 << self.num_food) - 1               # all food starts available
        return self.components_to_state(start_pos, self.initial_ghost_pos,
                                        food_mask)

    def step(self, state, action):
        # take one step in the environment
        if isinstance(action, str):
            action = self.actions.index(action)
        transitions = self.get_transitions(state, action)
        states, probs, rewards = zip(*transitions)
        idx = np.random.choice(len(states), p=probs)
        ns = states[idx]
        return ns, rewards[idx], self.is_terminal(ns)

class ValueIteration:
    def __init__(self, mdp, theta=1e-4, max_iterations=1000):
        self.mdp = mdp
        self.theta = theta
        self.max_iterations = max_iterations
        self.values = np.zeros(mdp.num_states)
        self.policy = np.zeros(mdp.num_states, dtype=int)
        self.convergence_history = []

    def solve(self):
        for it in range(self.max_iterations):
            delta = 0.0
            new_values = self.values.copy()

            for s in range(self.mdp.num_states):
                if self.mdp.is_terminal(s):
                    continue
                best = -np.inf
                for ai in range(self.mdp.num_actions):
                    q = sum(p * (r + self.mdp.discount * self.values[ns])
                            for ns, p, r in self.mdp.get_transitions(s, ai))
                    if q > best:
                        best = q
                new_values[s] = best
                delta = max(delta, abs(self.values[s] - best))

            self.values = new_values
            self.convergence_history.append(delta)

            if delta < self.theta:
                print(f"converged after {it + 1} iterations (delta={delta:.2e})")
                break

        self._extract_policy()
        return self.values, self.policy

    def _extract_policy(self):
        for s in range(self.mdp.num_states):
            if self.mdp.is_terminal(s):
                continue
            best_q, best_a = -np.inf, 0
            for ai in range(self.mdp.num_actions):
                q = sum(p * (r + self.mdp.discount * self.values[ns])
                        for ns, p, r in self.mdp.get_transitions(s, ai))
                if q > best_q:
                    best_q, best_a = q, ai
            self.policy[s] = best_a


class PolicyIteration:
    def __init__(self, mdp, theta=1e-4, max_eval_iters=500,
                 max_iterations=100):
        self.mdp = mdp
        self.theta = theta
        self.max_eval_iters = max_eval_iters
        self.max_iterations = max_iterations
        self.values = np.zeros(mdp.num_states)
        self.policy = np.random.randint(0, mdp.num_actions,
                                        size=mdp.num_states)
        self.convergence_history = []

    def solve(self):
        for it in range(self.max_iterations):
            self._policy_evaluation()
            changed = self._policy_improvement()
            self.convergence_history.append(changed)
            if changed == 0:
                print(f"policy converged in {it + 1} iterations")
                break
        return self.values, self.policy

    def _policy_evaluation(self):
        for _ in range(self.max_eval_iters):
            delta = 0.0
            new_values = self.values.copy()
            for s in range(self.mdp.num_states):
                if self.mdp.is_terminal(s):
                    continue
                ai = self.policy[s]
                v = sum(p * (r + self.mdp.discount * self.values[ns])
                        for ns, p, r in self.mdp.get_transitions(s, ai))
                new_values[s] = v
                delta = max(delta, abs(self.values[s] - v))
            self.values = new_values
            if delta < self.theta:
                break

    def _policy_improvement(self):
        changed = 0
        for s in range(self.mdp.num_states):
            if self.mdp.is_terminal(s):
                continue
            old = self.policy[s]
            best_q, best_a = -np.inf, 0
            for ai in range(self.mdp.num_actions):
                q = sum(p * (r + self.mdp.discount * self.values[ns])
                        for ns, p, r in self.mdp.get_transitions(s, ai))
                if q > best_q:
                    best_q, best_a = q, ai
            self.policy[s] = best_a
            if old != best_a:
                changed += 1
        return changed

class LiveVisualizer:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy
        self.fig = None
        self.ax = None
        self.trajectory = []
        self.total_reward = 0.0

    def run_episode(self, start_pos=(0, 0), max_steps=50, delay=0.5):
        # make new window for each run
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        state = self.mdp.reset(start_pos)
        self.trajectory = []
        self.total_reward = 0.0

        print(f"\nstarting from position {start_pos}")
        print("-" * 50)

        for step in range(max_steps):
            pac, ghost, fm = self.mdp.state_to_components(state)
            self.trajectory.append((pac, ghost))

            action_idx = self.policy[state]
            action = self.mdp.actions[action_idx]
            next_state, reward, done = self.mdp.step(state, action_idx)
            self.total_reward += reward

            self._draw(pac, ghost, fm, step, action, reward)
            plt.pause(delay)

            food_bits = ''.join(
                '1' if (fm >> i) & 1 else '0'
                for i in range(self.mdp.num_food)
            )
            print(f"step {step}: pacman at {pac}, ghost at {ghost}, "
                  f"action: {action}, reward: {reward:.1f}, food: {food_bits}")

            if done:
                np_, ng_, fm_ = self.mdp.state_to_components(next_state)
                self._draw(np_, ng_, fm_, step + 1, "DONE", 0)
                plt.pause(delay)
                if fm_ == 0:
                    print("*** WON! ate all the food! ***")
                else:
                    print("*** GAME OVER - caught by ghost ***")
                break

            state = next_state

        print(f"total reward: {self.total_reward:.1f}")
        print("-" * 50)
        plt.ioff()
        plt.show()              # wait for user to close the window

    def _draw(self, pac, ghost, fm, step, action, reward):
        ax = self.ax
        ax.clear()
        gs = self.mdp.grid_size
        ax.set_xlim(-0.5, gs[1] - 0.5)
        ax.set_ylim(gs[0] - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'step {step} | action: {action} | '
            f'reward: {reward:.1f} | total: {self.total_reward:.1f}',
            fontsize=13, fontweight='bold')

        food_str = "  ".join(
            f"food{i}: {'available' if (fm >> i) & 1 else 'eaten'}"
            for i in range(self.mdp.num_food))
        ax.set_xlabel(food_str, fontsize=11)

        # shade the danger zone around ghost
        for r in range(gs[0]):
            for c in range(gs[1]):
                if self.mdp.manhattan((r, c), ghost) <= self.mdp.proximity_threshold:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color='red', alpha=0.07))

        # draw walls, food, empty cells
        for r in range(gs[0]):
            for c in range(gs[1]):
                pos = (r, c)
                if pos in self.mdp.wall_positions:
                    ax.add_patch(Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                           color='dimgray', ec='black', lw=2))
                elif pos in self.mdp.food_positions:
                    idx = self.mdp.food_positions.index(pos)
                    if (fm >> idx) & 1:
                        ax.add_patch(Circle((c, r), 0.2,
                                            color='yellow', alpha=0.8))
                    else:
                        ax.add_patch(Circle((c, r), 0.15,
                                            color='gray', alpha=0.3))
                else:
                    ax.text(c, r, '.', ha='center', va='center',
                            fontsize=18, color='lightgray', alpha=0.5)

        # draw path taken so far
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                p1, p2 = self.trajectory[i][0], self.trajectory[i + 1][0]
                a = 0.3 + 0.5 * (i / len(self.trajectory))
                ax.plot([p1[1], p2[1]], [p1[0], p2[0]],
                        'b-', alpha=a, linewidth=2)

        # draw ghost
        ax.add_patch(Circle((ghost[1], ghost[0]), 0.35, color='red',
                             alpha=0.8, lw=2, ec='darkred'))
        ax.text(ghost[1], ghost[0], 'G', ha='center', va='center',
                fontsize=20, color='white', fontweight='bold')

        # draw pacman
        ax.add_patch(Circle((pac[1], pac[0]), 0.35, color='yellow',
                             alpha=0.9, lw=2, ec='orange'))
        ax.text(pac[1], pac[0], 'P', ha='center', va='center',
                fontsize=20, color='black', fontweight='bold')

        ax.text(gs[1] / 2, -0.8,
                "scoring: food +10 | last food +100 | caught by ghost -100 | "
                "step -1 | near ghost -5",
                ha='center', va='top', fontsize=9, style='italic')

def plot_convergence(solver, title="Convergence"):
    # show how the algorithm converged
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(solver.convergence_history, linewidth=2)
    ax.set_xlabel("iteration")
    if isinstance(solver, ValueIteration):
        ax.set_ylabel("max value change")
        ax.set_yscale('log')
    else:
        ax.set_ylabel("number of policy changes")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_policy_grid(mdp, policy, ghost_pos=None, food_mask=None):
    # visualize policy as arrows on the grid
    if ghost_pos is None:
        ghost_pos = mdp.initial_ghost_pos
    if food_mask is None:
        food_mask = (1 << mdp.num_food) - 1

    arrow = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→', 'STAY': '●'}
    fig, ax = plt.subplots(figsize=(7, 7))
    gs = mdp.grid_size
    ax.set_xlim(-0.5, gs[1] - 0.5)
    ax.set_ylim(gs[0] - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'learned policy (ghost at {ghost_pos}, food state: {bin(food_mask)})',
                 fontsize=13, fontweight='bold')

    for r in range(gs[0]):
        for c in range(gs[1]):
            pos = (r, c)
            if pos in mdp.wall_positions:
                ax.add_patch(Rectangle((c - 0.4, r - 0.4), 0.8, 0.8,
                                       color='dimgray', ec='black', lw=2))
            else:
                s = mdp.components_to_state(pos, ghost_pos, food_mask)
                act = mdp.actions[policy[s]]
                ax.text(c, r, arrow[act], ha='center', va='center',
                        fontsize=22, color='navy', fontweight='bold')

            if pos in mdp.food_positions:
                idx = mdp.food_positions.index(pos)
                if (food_mask >> idx) & 1:
                    ax.add_patch(Circle((c, r), 0.4, fill=False,
                                        ec='green', lw=2, ls='--'))

    ax.plot(ghost_pos[1], ghost_pos[0], 'rs', markersize=20, alpha=0.5)
    ax.text(ghost_pos[1], ghost_pos[0] - 0.38, 'ghost',
            ha='center', fontsize=9, color='red')
    plt.tight_layout()
    plt.show()


def evaluate_policy(mdp, policy, num_episodes=200, max_steps=50,
                    start_pos=None):
    # run a bunch of episodes and see how well policy does
    if start_pos is None:
        start_pos = mdp.initial_pacman_pos
    rewards, goals, lengths = [], 0, []

    for _ in range(num_episodes):
        state = mdp.reset(start_pos)
        total_r = 0.0
        for t in range(max_steps):
            ns, r, done = mdp.step(state, policy[state])
            total_r += r
            if done:
                _, _, fm = mdp.state_to_components(ns)
                if fm == 0:
                    goals += 1
                lengths.append(t + 1)
                break
            state = ns
        else:
            lengths.append(max_steps)
        rewards.append(total_r)

    print(f"\npolicy evaluation ({num_episodes} episodes starting from {start_pos})")
    print(f"  won (ate all food): {goals}/{num_episodes} "
          f"({100 * goals / num_episodes:.1f}%)")
    print(f"  average reward: {np.mean(rewards):.2f} "
          f"(std: {np.std(rewards):.2f})")
    print(f"  average steps: {np.mean(lengths):.1f}")
    return rewards


def demonstrate_learned_policy(mdp, policy, num_episodes=3,
                               start_positions=None):
    # run some episodes with visualization
    if start_positions is None:
        start_positions = [(0, 0), (0, 4), (4, 0)]
    vis = LiveVisualizer(mdp, policy)
    for i, sp in enumerate(start_positions[:num_episodes]):
        print(f"\n*** episode {i + 1}/{num_episodes} ***")
        vis.run_episode(sp, max_steps=40, delay=0.5)

def main():
    print("pacman MDP with value/policy iteration")
    print("-" * 60)

    mdp = PacmanMDP(grid_size=(5, 5), discount=0.9,
                    action_success_prob=0.8, ghost_chase_prob=0.4)
    print(f"grid has {mdp.num_valid} valid cells, {mdp.num_states} total states, {mdp.num_actions} actions")

    print("\nwhich algorithm do you want to use?")
    print("1. value iteration")
    print("2. policy iteration")
    choice = input("enter 1 or 2: ").strip()

    if choice == '2':
        print("\nrunning policy iteration...")
        solver = PolicyIteration(mdp)
    else:
        print("\nrunning value iteration...")
        solver = ValueIteration(mdp)

    values, policy = solver.solve()

    # plot convergence
    plot_convergence(
        solver,
        title=("value iteration convergence"
               if isinstance(solver, ValueIteration)
               else "policy iteration convergence"))

    # show policy as arrows
    plot_policy_grid(mdp, policy)

    # test policy performance
    evaluate_policy(mdp, policy, num_episodes=500)

    # show some example runs
    print("\n" + "-" * 60)
    print("running demonstration episodes")
    print("-" * 60)
    input("\npress enter to see the policy in action...")
    demonstrate_learned_policy(mdp, policy, num_episodes=3)

    print("\nall done!")


if __name__ == "__main__":
    main()