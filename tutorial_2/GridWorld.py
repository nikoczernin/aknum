# ===================================================
# Author: Nikolaus Czernin
# Script: GridWorld Environment for Reinforcement Learning
# Description: Defines a 2D GridWorld environment as a subclass of a generic Environment,
#              including states, actions, terminal states, rewards, and grid visualization.
# ===================================================



from tutorial_2.Grid import Grid
from tutorial_2.utils import plot_line_graph

from Environment import Environment



class GridWorld(Environment):
    action_mappings = {
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

    def __init__(self, w, h, terminal_states):
        self.w, self.h = w, h
        actions = [(0, -1), #up
                   (0, 1),  #down
                   (-1, 0), #left
                   (1, 0)]  #right
        states = [(x, y) for x in range(w) for y in range(h)]
        # initialize the superclass: Environment
        super().__init__(actions, states, terminal_states)
        # define custom rewards, r for steps in a grid_world are a constant -1, but you can adjust this manually
        self.width = w
        self.height = h

        # OPTIONAL
        # Grid is a clas that draws a pretty grid with whatever items you put on them
        self.grid = Grid(w, h)
        # put the terminal state flags into the grid
        for ts in self.terminal_states:
            self.grid.put("G", *ts)

    def put_onto_grid(self, items:dict):
        for position, value in items.items():
            self.grid.put(value, *position)

    def is_this_out_of_bounds(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
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

    def get_state_action_transition(self, state, action):
        # returns a dict: mapping from new_state to transition probability p(new_state, reward | state, action)
        new_state = state[0] + action[0], state[1] + action[1]
        if self.is_this_out_of_bounds(*new_state):
            new_state = state
        return {new_state: 1}

    def get_reward(self, state, action, new_state=None):
        return -1

    @staticmethod
    def action_str(a):
        return GridWorld.action_mappings[str(a)]

    def __str__(self):
        return str(self.grid)
