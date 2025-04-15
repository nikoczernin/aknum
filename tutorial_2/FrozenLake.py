from tutorial_2.GridWorld import GridWorld

class FrozenLake(GridWorld):
    def __init__(self, h:int, w:int, goals:list, holes:list, starting_state:tuple, slippery=True):
        self.holes = holes
        self.goals = goals
        self.slippery = slippery
        terminal_states = holes + goals
        super().__init__(h, w, terminal_states, starting_state)
        # put all cliff cells into the grid for visuals
        self.put_onto_grid({pos: "H" for pos in self.holes})

    def get_reward(self, state:tuple, action:tuple, new_state:tuple):
        # when you reach a goal, get a reward of 1
        if new_state in self.goals:
            return 1
        # otherwise the reward is zero
        return 0

    def apply_action(self, state: tuple, action:tuple):
        if not self.slippery:
            return super().apply_action(state, action)
        else:
            pass

    def get_possible_outcomes(self, state:tuple, action:tuple):
        if not self.slippery:
            return {super().apply_action(state, action):1}
        else:
            # with a probability of 1/3 each, you go where you want to go or
            # slip to either perpendicular direction
            #  to get the perpendicular directions, use smart indexing:
            # actions in GridWorld are defined in clockwise order (up, right, down, left)
            # that means if you have direction of index i,
            # then turning left is the direction of index i - 1
            # and turning right is the direction of index i - 3
            return {
                super().apply_action(state, action): 1/3,
                super().apply_action(state, self.actions[self.actions.index(action) - 1]): 1/3,
                super().apply_action(state, self.actions[self.actions.index(action) - 3]): 1/3
            }
