from pprint import pprint
import random

from pyparsing import actions

from Environment import Environment

class BlackJack(Environment):
    # 1 can be an 11 if wanted
    cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    def __init__(self):
        actions = ["hit", "stick"]
        # States
        # (
        #   player_value:int,
        #   player_num_aces:int,
        #   player_sticks: bool,
        #   dealer_facing_value:int,
        #   dealer_hidden_value:int,
        #   dealer_num_aces:int
        #   dealer_sticks: bool,
        # )
        # to the agent the dealer_hidden_sum will always be shown as the same, e.g. 0
        # because the agent doesnt see the dealers hidden cards
        # precompute all possible states here
        states = []
        for pv in range(1, 31): # player_value
            for pa in range(0, 2): # player_num_aces
                for ps in range(2):
                    for df in range(1, 12): # dealer_facing_value
                        for dh in range(1, 27): # dealer_hidden_value
                            for da in range(0, 2): # dealer_num_aces
                                for ds in range(2):
                                    states.append((pv, pa, ps, df, dh, da, ds))
        # Terminal States
        terminal_states = []
        for s in states:
            # if either the player or the dealer has a sum > 21, terminate
            if self.get_value(s[0], s[1]) >= 21 or self.get_value(s[3] + s[4], s[5]) >= 21:
                terminal_states.append(s)
            # if both players stick, terminate
            if s[2] and s[6]:
                terminal_states.append(s)
        pprint(terminal_states)
        print(len(terminal_states))
        # starting_state
        starting_state = (0, 0, 0, 0, 0)
        super().__init__(actions, states, terminal_states, starting_state)

    @staticmethod
    def get_value(value_sum, num_usable_aces):
        # for every usable ace
        for i in range(num_usable_aces):
            # any 1 is a usable ace
            # try subtracting 1 and adding 11 to use the ace
            value_sum_2 = value_sum + 10
            # if the result is still < 21, keep it
            if value_sum_2 <= 21:
                value_sum = value_sum_2
        return value_sum

    @staticmethod
    def draw_card():
        return random.choice(BlackJack.cards)

    def apply_action(self, state, action) -> tuple:
        if action == "hit":
            # draw another card
            new_card = BlackJack.draw_card()
            new_state = (state[0] + new_card,
                         state[1] + (new_card == 1),
                         state[2], state[3], state[4])
            return new_state
        elif action == "stick":
            return state


if __name__ == "__main__":
    env = BlackJack()
    print(env)