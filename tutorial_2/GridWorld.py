from pprint import pprint
import random

import numpy as np

import utils

from tutorial_2.Grid import Grid
from tutorial_2.Item import *
from tutorial_2.utils import plot_line_graph

class Environment():
    def __init__(self, terminal_states):
        self.terminal_states = terminal_states
        self.actions = []
    # def is_this_action_possible(self, state, action):
    #     pass
    # def get_random_starting_state(self):
    #     pass

class GridWorld(Environment):
    def __init__(self, w, h, terminal_states, hard_borders=True):
        super().__init__(terminal_states)
        self.width = w
        self.height = h
        self.grid = Grid(w, h, hard_borders=hard_borders)
        # put the terminal state flags into the grid
        for ts in self.terminal_states:
            ts = Flag(self.grid, ts[0], ts[1], "G")
            self.grid.put(ts, *ts.position)

    def is_this_out_of_bounds(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return False

    def is_this_action_possible(self, state, action):
        destination = self.apply_action(state, action)
        return not self.is_this_out_of_bounds(*destination)

    def get_random_starting_state(self):
        s_0 = (random.choice(range(w)), random.choice(range(h)))
        return s_0 if s_0 not in self.terminal_states else self.get_random_starting_state()

    def apply_action(self, state, action):
        return state[0] + action[0], state[1] + action[1]

    def __str__(self):
        return str(self.grid)

    # def episode(self, policy, n, T, gamma, reward, start=None, epsilon=0.0, verbose=False):
    #     # policy: dict mapping from states to actions to probability of picking that action in that state
    #     # n: dict mapping from states to actions to execution counts
    #     # T: int, maximum number of time steps
    #     # A: dict, mapping action names to moves
    #     R = 0
    #     # S_0
    #     # start the agent in a random position if a starting point was not given
    #     s_t = start if start is not None else self.get_random_starting_position()
    #     # create an agent
    #     agent = Agent(grid, *s_t, label="A")
    #     if verbose: grid.draw()
    #
    #     # timestep loop
    #     t = 0
    #     while s_t not in self.terminal_states and t < T:
    #         t += 1
    #         # make a state transition: move and get a reward for it
    #         # a_t
    #         # what actions a are possible?
    #         A_t = self.get_possible_actions(s_t)
    #
    #         # if there is no policy yet ...
    #         if s_t not in policy.keys():
    #             # initialize all possible actions' probabilities to be equal
    #             for a, move in A_t.items():
    #                 policy[s_t] = {a: 1 / len(A_t) for a, move in A_t.items()}
    #                 n[s_t] = {a: 0 for a, move in A_t.items()}
    #
    #         # Picking an action
    #         v = pick_action(A_t, policy, s_t)
    #
    #         if verbose: print(s_t, ":", policy[s_t], "==>", v)
    #         # perform the action to transition to a new state
    #         if verbose: print("Moving", v)
    #
    #         agent.move(*actions[a]) # unpack the action to get the movement values (x and y)
    #         # increase the execution counter for this action
    #         n[s_t][v] += 1
    #         # r_t
    #         r_t1 = reward
    #         R += r_t1
    #         # update the policy on this state s_t and chosen action a
    #         # transition_probs[s_t][a] += r_t1
    #         s_t = agent.position
    #
    #     if verbose: print("From this episode:", R)
    #     if verbose: print("Policy:")
    #     if verbose: print(policy)
    #     if verbose: print()
    #
    #     return R, policy, n, t


if __name__ == "__main__":

    w, h = 4, 4
    grid = Grid(w, h, hard_borders=True)
    # the episode is finished when position == terminal_state
    # the position is a class instance variable of the agent
    terminal_states= [(0, 0), (w-1, h-1)]
    # define possible actions
    actions = { "up": (0, -1),
                "down": (0, 1),
                "left": (-1, 0),
                "right": (1, 0)}
    # max number of time steps
    max_time_steps = 100

    grid_world = GridWorld(
        Grid(w, h, hard_borders=True),
        # the episode is finished when position == terminal_state
        # the position is a class instance variable of the agent
        terminal_states = terminal_states,
        # define possible actions
        actions = actions,
        T = max_time_steps
    )

    # set up a data structure for policy(state, action => probability of picking that action)
    # a dictionary where a state maps to possible actions which each map to the probability of picking that action
    policy_pi = {}
    # q = {
    #   s_a: {
    #         up: ...
    #         down: ...
    #         }
    #   }
    # }
    n = {} # this is the same as the policy, just measuring the number of executed times per action and state



    total_num_episodes = 1000
    all_total_rewards = []
    epsilon = .1
    start = None
    # start = (w//2, h//2)
    inflation_rate_gamma = 1 # discounting of future rewards
    reward = -1 # rewards of each timestep
    T = 100 # maximum number of timesteps
    for i in range(total_num_episodes):
        R, q, n, t = grid_world.episode(policy_pi, n, T, inflation_rate_gamma, reward, start, epsilon, verbose=False)
        all_total_rewards.append(R)

    # print("Total Reward:", R)
    # print("Time of termination:", t)
    utils.plot_line_graph(all_total_rewards)
    print()
    print("Return estimates:")
    pprint(q)
    print("Execution counts:")
    pprint(n)

