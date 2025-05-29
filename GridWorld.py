# ===================================================
# Author: Nikolaus Czernin
# Script: GridWorld Environment for Reinforcement Learning
# Description: Defines a 2D GridWorld environment as a subclass of a generic Environment,
#              including states, actions, terminal states, rewards, and grid visualization.
# ===================================================

from Grid import Grid

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
        # Initializes GridWorld with grid dimensions, terminal states, and starting position.
        # Input: h (int), w (int), terminal_states (list), starting_state (tuple)
        # Output: None
        # the actions are sorted in clockwise order
        actions = [(0, -1),  # up
                   (1, 0),  # right
                   (0, 1),  # down
                   (-1, 0)  # left
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
        self.put_onto_grid({ts: "G" for ts in terminal_states})
        # also put the starting state in the grid
        self.grid.put("S", *starting_state)

    def reset(self):
        self.set_start(self.starting_state)

    def state_generator(self):
        # Generates all possible states as (y, x) coordinates.
        # Input: None
        # Output: generator yielding tuples (y, x)
        for y in range(self.height):
            for x in range(self.width):
                yield y, x

    def state_is_terminal(self, state) -> bool:
        # Checks if a state is terminal.
        # Input: state (tuple)
        # Output: bool (True if terminal, False otherwise)
        if state in self.terminal_states:
            return True

    def put_onto_grid(self, position_value_mapping: dict):
        # Places specified values onto the visualization grid at given positions.
        # Input: position_value_mapping (dict: position tuple → value)
        # Output: None
        for position, value in position_value_mapping.items():
            self.grid.put(value, *position)

    def is_this_out_of_bounds(self, y, x):
        # Checks if given coordinates are outside the grid boundaries.
        # Input: y (int), x (int)
        # Output: bool (True if out-of-bounds, False otherwise)
        if int(x) < 0 or int(x) >= self.width or int(y) < 0 or int(y) >= self.height:
            return True
        return False

    def is_this_action_possible(self, state, action):
        # Determines if an action is allowed in the given state.
        # Input: state (tuple), action (tuple)
        # Output: bool
        # for terminal states, just return False
        if state in self.terminal_states: return False
        return True

    def apply_action(self, state, action):
        # Computes the resulting state from applying an action in the current state.
        # Input: state (tuple), action (tuple)
        # Output: new_state (tuple)
        new_state = state[0] + action[0], state[1] + action[1]
        if self.is_this_out_of_bounds(*new_state):
            new_state = state
        return new_state

    def get_possible_outcomes(self, state, action):
        # Returns possible outcomes and their probabilities from a state-action pair.
        # Input: state (tuple), action (tuple)
        # Output: dict {new_state: probability}
        if action is None:
            return None
        new_state = self.apply_action(state, action)
        return {new_state: 1}

    def get_reward(self, state: tuple, action: tuple, new_state: tuple = None):
        # Computes the reward obtained by transitioning from state to new_state using action.
        # Input: state (tuple), action (tuple), new_state (tuple, optional)
        # Output: int (reward)
        # if action and new_state are None, that means no action was taken, which means we already terminated
        # in that case return the approprate reward of 0
        if new_state is None and action is None:
            return 0
        return -1

    @staticmethod
    def action_str(a):
        # Converts an action tuple into a human-readable arrow representation.
        # Input: a (tuple)
        # Output: str (arrow representation)
        return GridWorld.action_mappings[str(a)]

    def __str__(self):
        # Returns the visual representation of the GridWorld.
        # Input: None
        # Output: str (grid visualization)
        return str(self.grid)