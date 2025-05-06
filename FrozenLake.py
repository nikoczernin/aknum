from GridWorld import GridWorld

class FrozenLake(GridWorld):
    def __init__(self, h:int, w:int, goals:list, holes:list, starting_state:tuple, slippery=True):
        # Initializes the Frozen Lake environment as a subclass of GridWorld.
        # The environment has goals, holes (hazards), and can optionally be slippery.
        # Inputs:
        #   h (int): height of the grid
        #   w (int): width of the grid
        #   goals (list): list of goal states
        #   holes (list): list of hazardous states ("holes")
        #   starting_state (tuple): starting coordinates of the agent
        #   slippery (bool): whether the surface is slippery (stochastic transitions)
        # Output: None (initializes the environment)
        self.holes = holes
        self.goals = goals
        self.slippery = slippery
        terminal_states = holes + goals
        super().__init__(h, w, terminal_states, starting_state)
        # put all cliff cells into the grid for visuals
        self.put_onto_grid({pos: "H" for pos in self.holes})

    def get_reward(self, state:tuple, action:tuple, new_state:tuple=None):
        # Computes the immediate reward for transitioning to a new state.
        # Input:
        #   state (tuple): the current state of the agent
        #   action (tuple): the action taken
        #   new_state (tuple): the resulting state after taking the action
        # Output:
        #   int: reward (1 if goal reached, else 0)
        # when you reach a goal, get a reward of 1
        if new_state in self.goals:
            return 1
        # otherwise the reward is zero
        return 0

    def apply_action(self, state: tuple, action:tuple):
        # Computes the new state from taking an action, considering slipperiness.
        # Input:
        #   state (tuple): current state
        #   action (tuple): action to take
        # Output:
        #   tuple: resulting state after action
        if not self.slippery:
            return super().apply_action(state, action)
        else:
            pass

    def get_possible_outcomes(self, state:tuple, action:tuple):
        # Computes possible state outcomes and their probabilities from a state-action pair.
        # Input:
        #   state (tuple): current state
        #   action (tuple): action to take
        # Output:
        #   dict: mapping of possible resulting states to their transition probabilities
        if not self.slippery:
            return {super().apply_action(state, action): 1}
        else:
            # with a probability of 1/3 each, you go where you want to go or
            # slip to either perpendicular direction
            # to get the perpendicular directions, use smart indexing:
            # actions in GridWorld are defined in clockwise order (up, right, down, left)
            # that means if you have direction of index i,
            # then turning left is the direction of index i - 1
            # and turning right is the direction of index i - 3
            return {
                super().apply_action(state, action): 1/3,
                super().apply_action(state, self.actions[self.actions.index(action) - 1]): 1/3,
                super().apply_action(state, self.actions[self.actions.index(action) - 3]): 1/3
            }
