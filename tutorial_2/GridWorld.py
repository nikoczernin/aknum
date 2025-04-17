# ===================================================
# Author: Nikolaus Czernin
# Script: GridWorld Environment for Reinforcement Learning
# Description: Defines a 2D GridWorld environment as a subclass of a generic Environment,
#              including states, actions, terminal states, rewards, and grid visualization.
# ===================================================
import random

from tutorial_2.Grid import Grid
from tutorial_2.utils import plot_line_graph

from Environment import Environment



class GridWorld(Environment):
    action_mappings = { # (y, x)
        "(1, 0)": "↓",
        "(-1, 0)": "↑",
        "(0, 1)": "→",
        "(0, -1)": "←",
        "(2, 0)": "↓↓",
        "(-2, 0)": "↑↑",
        "(0, 2)": "→→",
        "(0, -2)": "←←",
        "None": "o",
        "(0, 0)": "o"
    }

    def __init__(self, h, w, terminal_states, starting_state):
        # the actions are sorted in clockwise order
        actions = [(0, -1), #up
                   (1, 0), #right
                   (0, 1),  #down
                   (-1, 0) #left
                   ]
        # initialize the superclass: Environment
        super().__init__(actions)
        # define custom rewards, r for steps in a grid_world are a constant -1, but you can adjust this manually
        self.height = h
        self.width = w
        self.terminal_states = terminal_states
        self.set_start(starting_state)
        # OPTIONAL
        # Grid is a clas that draws a pretty grid with whatever items you put on them
        self.grid = Grid(self.height, self.width)
        # put the terminal state flags into the grid
        self.put_onto_grid({ts:"G" for ts in terminal_states})
        # also put the starting state in the grid
        self.grid.put("S", *starting_state)

    def state_generator(self):
        for y in range(self.height):
            for x in range(self.width):
                yield y, x

    def state_is_terminal(self, state) -> bool:
        if state in self.terminal_states:
            return True

    def put_onto_grid(self, position_value_mapping:dict):
        for position, value in position_value_mapping.items():
            self.grid.put(value, *position)

    def is_this_out_of_bounds(self, y, x):
        if int(x) < 0 or int(x) >= self.width or int(y) < 0 or int(y) >= self.height:
            return True
        return False

    def is_this_action_possible(self, state, action):
        # for terminal states, just return False
        if state in self.terminal_states: return False
        return True

    def apply_action(self, state, action):
        new_state = state[0] + action[0], state[1] + action[1]
        if self.is_this_out_of_bounds(*new_state):
            new_state = state
        return new_state

    # returns a dict: mapping from new_state to transition probability p(new_state, reward | state, action)
    def get_possible_outcomes(self, state, action):
        if action is None:
            return None
        new_state = self.apply_action(state, action)
        return {new_state: 1}

    def get_reward(self, state:tuple, action:tuple, new_state:tuple=None):
        # if action and new_state are None, that means no action was taken, which means we already terminated
        # in that case return the approprate reward of 0
        if new_state is None and action is None:
            return 0
        return -1

    @staticmethod
    def action_str(a):
        return GridWorld.action_mappings[str(a)]

    def __str__(self):
        return str(self.grid)
