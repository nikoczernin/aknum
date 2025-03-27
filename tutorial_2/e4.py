from pprint import pprint
import random

import numpy as np

import utils

from tutorial_2.Grid import Grid
from tutorial_2.Item import *
from tutorial_2.utils import plot_line_graph


# function for picking the best action from a dict of actions, that map to estimated return values
# draws are resolved randomly
def pick_action(actions: dict, explore=False):
    if explore:
        return random.choice(list(actions.keys()))
    else:
        max_value = max(actions.values())
        best_keys = [k for k, v in actions.items() if v == max_value]
        chosen_key = random.choice(best_keys)
        return chosen_key

def get_starting_position(grid: Grid):
    s_0 = (random.choice(range(w)), random.choice(range(h)))
    return s_0 if s_0 not in terminal_states else get_starting_position(grid)

def episode(q, start=None, epsilon=0, verbose=False):
    grid = Grid(w, h)
    R = 0
    # set up a policy for the agent
    policy = {}
    # set an ending flag for each terminal state
    goals = []
    for (x, y) in terminal_states:
        goals.append(Flag(grid, x, y, 'E'))
    # S_0
    # start the agent in a random position
    s_t = start if start is not None else get_starting_position(grid)
    agent = Agent(grid, *s_t, label="A")
    if verbose: grid.draw()

    # timestep loop
    t = 0
    while s_t not in terminal_states and t < T:
        t += 1
        # make a state transition: move and get a reward for it
        # a_t
        # what actions a are possible?
        A_t = [a for a, move in A.items() if agent.can_i_make_this_move(*move)]
        # the q(s, a) is the expected return G_t if action a is picked in state s
        # if there is no estimate yet ...
        if s_t not in q.keys():
            # initialize all possible actions estimates to be 0
            q[s_t] = {a:0 for a in A_t}
        # epsilon greedy: with probability of epsilon
        if np.random.uniform() < epsilon:
            # we explore and pick a random action
            a = pick_action(q[s_t], explore=True)
        else:
            # we exploit & pick the action with the highest estimated return
            a = pick_action(q[s_t])

        if verbose: print(s_t, ":", q[s_t], "==>", a)
        # perform the action to transition to a new state
        if verbose: print("Moving", a)
        agent.move(*A[a]) # unpack the action to get the movement values (x and y)
        # r_t
        r_t1 = reward
        R += r_t1
        # update the estimated reward of the action
        q[s_t][a] += r_t1
        s_t = agent.position

    if verbose: print("From this episode:", R)
    # print("estimated returns:")
    verbose: print(q)
    verbose: print()

    return R, q, t


w, h = 4, 4
gamma = 1 # discounting of future rewards
reward = -1 # rewards of each timestep
# maximum number of timesteps
T = 100

# the episode is finished when position == terminal_state
# the position is a class instance variable of the agent
terminal_states = [(0, 0), (w-1, h-1)]

# define possible actions
A = {"up": (0, -1),
     "down": (0, 1),
     "left": (-1, 0),
     "right": (1, 0)}

# set up a data structure for q(state, action => estimated reward)
# a dictionary where a state maps to possible actions which each map to the estimated reward
# initialize the estimated expected returns per action and state as 0 ... when you get there
q = {}
start = None
start = (w//2, h//2)

total_num_episodes = 1000
all_total_rewards = []

epsilon = .1

for i in range(total_num_episodes):
    R, q, t = episode(q, start, epsilon, verbose=False)
    all_total_rewards.append(R)

# print("Total Reward:", R)
# print("Time of termination:", t)
utils.plot_line_graph(all_total_rewards)
print()
print("Return estimates:")
pprint(q)

