
from pprint import pprint
import random

class Environment():
    def __init__(self, actions, starting_state):
        self.actions = actions
        self.starting_state = starting_state
        self.rewards = {a: 0 for a in self.actions}

    def state_generator(self):
        pass

    def terminal_state_generator(self) -> bool:
        for state in self.state_generator():
            if self.state_is_terminal(state):
                yield state

    def state_is_terminal(self, state) -> bool:
        pass

    @staticmethod
    def get_random_state(states, states_to_avoid=None):
        s = random.choice(states)
        if states_to_avoid is not None:
            if s in states_to_avoid:
                s = Environment.get_random_state(states, states_to_avoid)
        return s

    def is_this_action_possible(self, state, action) -> bool:
        return True if action in self.actions else False

    def apply_action(self, state, action):
        pass

    # returns a dict: mapping from new_state to transition probability p(new_state, reward | state, action)
    def get_possible_outcomes(self, state, action) -> dict:
        pass

    @staticmethod
    def resolve_outcome(outcomes_dict: dict):
        # outcomes_dict is a dict mapping possible states to transition probabilities
        return random.choices(list(outcomes_dict.keys()), weights=list(outcomes_dict.values()))[0]

    def get_reward(self, state, action, new_state=None):
        pass

    @staticmethod
    def action_str(a) -> str:
        pass