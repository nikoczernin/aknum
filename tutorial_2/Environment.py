
from pprint import pprint
import random

class Environment():
    def __init__(self, actions, states, terminal_states, starting_state):
        if not all(terminal_state in states for terminal_state in terminal_states):
            raise ValueError("Terminal state must be in states list")
        self.actions = actions
        self.states = states
        self.terminal_states = terminal_states
        self.starting_state = starting_state
        self.rewards = {a: 0 for a in self.actions}

    def state_generator(self):
        pass

    def get_random_state(self, avoid_terminal_states=True):
        s = random.choice(self.states)
        return self.get_random_state(True) if avoid_terminal_states and s in self.terminal_states else s

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