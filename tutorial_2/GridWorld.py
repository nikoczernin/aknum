# ===================================================
# Author: Nikolaus Czernin
# Script: GridWorld Environment for Reinforcement Learning
# Description: Defines a 2D GridWorld environment as a subclass of a generic Environment,
#              including states, actions, terminal states, rewards, and grid visualization.
# ===================================================


from pprint import pprint
import random

from tutorial_2.Grid import Grid
from tutorial_2.utils import plot_line_graph

class Environment():
    def __init__(self, actions, states, terminal_states=[]):
        if not all(terminal_state in states for terminal_state in terminal_states):
            raise ValueError("Terminal state must be in states list")
        self.actions = actions
        self.states = states
        self.terminal_states = terminal_states
        self.rewards = {a: 0 for a in self.actions}

    def is_this_action_possible(self, state, action):
        pass

    def get_random_state(self):
        pass

    @staticmethod
    def apply_action(state, action):
        return state[0] + action[0], state[1] + action[1]


class GridWorld(Environment):
    def __init__(self, w, h, hard_borders=True):
        self.w, self.h = w, h
        actions = [(0, -1), #up
                   (0, 1),  #down
                   (-1, 0), #left
                   (1, 0)]  #right
        states = [(x, y) for x in range(w) for y in range(h)]
        # the first and last (top left and bottom right) fields are terminal states (fields)
        # you can change them manually though
        terminal_states = [states[0], states[-1]]
        # initialize the superclass: Environment
        super().__init__(actions, states, terminal_states)
        # define custom rewards, r for steps in a grid_world are a constant -1, but you can adjust this manually
        self.rewards = {a: -1 for a in self.actions}
        self.width = w
        self.height = h

        # OPTIONAL
        # Grid is a clas that draws a pretty grid with whatever items you put on them
        self.grid = Grid(w, h, hard_borders=hard_borders)
        # put the terminal state flags into the grid
        for ts in self.terminal_states:
            self.grid.put("G", *ts)

    def is_this_out_of_bounds(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return False

    def is_this_action_possible(self, state, action):
        destination = GridWorld.apply_action(state, action)
        return not self.is_this_out_of_bounds(*destination)

    @staticmethod
    def apply_action(state, action):
        return state[0] + action[0], state[1] + action[1]

    def get_random_state(self, avoid_terminal_states=True):
        s = random.choice(self.states)
        return self.get_random_state(True) if avoid_terminal_states and s in self.terminal_states else s

    def __str__(self):
        return str(self.grid)


if __name__ == "__main__":
    env = GridWorld(4, 4)
    print(env)