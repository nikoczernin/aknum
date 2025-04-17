from tutorial_2.GridWorld import GridWorld

class CliffWalking(GridWorld):
    def __init__(self, h, w, terminal_states, starting_state, cliffs, penalty=-100):
        super().__init__(h, w, terminal_states, starting_state)
        self.cliffs = cliffs
        # put all cliff cells into the grid for visuals
        self.put_onto_grid({pos: "X" for pos in self.cliffs})
        self.penalty = penalty

    # Apply action is similar to GridWorld, but stepping on a cliff transports
    # you back to the beginning
    def apply_action(self, state, action):
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
        # if the agent steps onto a cliff and gets sent to the start, give him a penalty
        # new_state will never be a cliff, because apply_action instantly sends you to the start
        # this is a bias because this means that circling back ot the start without falling down
        # the cliffs also gets penalized, but who gives a shit
        if new_state == self.starting_state:
            return self.penalty
        return -1

