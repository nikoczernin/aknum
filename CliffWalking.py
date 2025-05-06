from GridWorld import GridWorld

class CliffWalking(GridWorld):
    def __init__(self, h, w, terminal_states, starting_state, cliffs, penalty=-100):
        # Initializes CliffWalking environment based on GridWorld, with cliff cells and penalties.
        # Inputs:
        #   h (int): Grid height
        #   w (int): Grid width
        #   terminal_states (list): List of terminal state tuples
        #   starting_state (tuple): Coordinates of the start state
        #   cliffs (list): List of cliff cell coordinates
        #   penalty (int): Penalty for stepping on a cliff (default: -100)
        # Output: None
        super().__init__(h, w, terminal_states, starting_state)
        self.cliffs = cliffs
        # put all cliff cells into the grid for visuals
        self.put_onto_grid({pos: "X" for pos in self.cliffs})
        self.penalty = penalty

    def apply_action(self, state, action):
        # Computes resulting state from an action, sending agent to start if stepping onto a cliff.
        # Inputs:
        #   state (tuple): Current state coordinates
        #   action (tuple): Action to perform
        # Output:
        #   tuple: Resulting state coordinates
        new_state = state[0] + action[0], state[1] + action[1]
        # if you're trying to leave the grid
        if self.is_this_out_of_bounds(*new_state):
            # ...stay
            new_state = state
        # if you're stepping onto a cliff cell
        elif new_state in self.cliffs:
            pass
            new_state = self.starting_state
        return new_state

    def get_reward(self, state, action, new_state=None):
        # Computes reward for transitioning to new state; severe penalty for stepping onto cliff.
        # Inputs:
        #   state (tuple): Current state coordinates
        #   action (tuple): Action taken
        #   new_state (tuple, optional): Resulting state coordinates after action
        # Output:
        #   int: Reward value (negative penalty if cliff, else -1)
        # if the agent steps onto a cliff and gets sent to the start, give him a penalty
        # new_state will never be a cliff, because apply_action instantly sends you to the start
        # this is a bias because this means that circling back ot the start without falling down
        # the cliffs also gets penalized, but who gives a shit
        if new_state == self.starting_state:
            return self.penalty
        return -1
