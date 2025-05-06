from GridWorld import GridWorld

class WindyGridWorld(GridWorld):
    def __init__(self, h, w, terminal_states, starting_state, forces):
        # Initializes WindyGridWorld, a GridWorld with wind forces influencing movement.
        # Inputs:
        #   h (int): Grid height
        #   w (int): Grid width
        #   terminal_states (list): List of terminal state coordinates
        #   starting_state (tuple): Starting position of the agent
        #   forces (list): Wind force vectors for each column, length should match grid width
        # Output: None
        super().__init__(h, w, terminal_states, starting_state)
        # forces should be an array of length w of integer values
        self.forces = forces
        # put all the forces as arrows onto the grid for visuals
        # use the GridWorld.action_mappings for this
        for x, f in enumerate(forces):
            for y in range(h):
                if (y, x) not in terminal_states + [starting_state]:
                    self.put_onto_grid({(y, x): GridWorld.action_mappings[str(f)]})

    def apply_action(self, state, action):
        # Computes the new state after applying an action and wind force.
        # Inputs:
        #   state (tuple): Current position of the agent
        #   action (tuple): Action the agent wants to perform
        # Output:
        #   tuple: New position after considering action and wind force
        new_state = state[0] + action[0], state[1] + action[1]
        if self.is_this_out_of_bounds(*new_state):
            new_state = state
        y, x = new_state
        # apply wind force
        new_state_shifted = (y + self.forces[x][0], x + self.forces[x][1])
        if self.is_this_out_of_bounds(*new_state_shifted):
            new_state_shifted = new_state
        return new_state_shifted



if __name__ == "__main__":
    from MonteCarlo import test_windy_world
    test_windy_world()