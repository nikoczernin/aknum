from pprint import pprint
import random

from pyparsing import actions

from Environment import Environment

class BlackJack(Environment):
    # 1 can be an 11 if wanted
    cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    def __init__(self):
        blackjack_actions = ["hit", "stick"]
        # States
        # {
        #   player_value:int,
        #   player_num_aces:int,
        #   player_sticks: int,
        #   dealer_facing_value:int,
        #   dealer_hidden_value:int,
        #   dealer_num_aces:int,
        #   dealer_sticks: int,
        # }
        # to the agent the dealer_hidden_sum will always be shown as the same, e.g. 0
        # because the agent doesnt see the dealers hidden cards
        # we dont precompute all possible states, rather we exploit state generators here
        # Starting_state
        super().__init__(blackjack_actions)

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

    @staticmethod
    def reset_game():
        # generate a single state, set all its values to zero and return it
        return {key: 0 for key in next(BlackJack.state_generator()).keys()}

    @staticmethod
    def start_game():
        # start with a zero-state
        state = BlackJack.reset_game()
        # Player gets his first card
        # Player gets his second card
        for i in range(2):
            new_card = BlackJack.draw_card()
            state["player_value"] += new_card
            if new_card == 1:
                state["player_aces"] += 1
        # Dealer gets his first card, it is the open-faced card
        new_card = BlackJack.draw_card()
        state["dealer_face_value"] += new_card
        if new_card == 1: state["dealer_aces"] += 1
        # Dealer gets his second card, it is hidden
        new_card = BlackJack.draw_card()
        state["dealer_hidden_value"] += new_card
        if new_card == 1: state["dealer_aces"] += 1
        return state

    @staticmethod
    def state_generator():
        for pv in range(1, 31): # player_value
            for pa in range(0, 2): # player_num_aces
                for ps in range(2):
                    for df in range(1, 12): # dealer_facing_value
                        for dh in range(1, 27): # dealer_hidden_value
                            for da in range(0, 2): # dealer_num_aces
                                for ds in range(2):
                                    yield {"player_value": pv, 
                                           "player_aces": pa, 
                                           "player_sticks": ps, 
                                           "dealer_face_value": df, 
                                           "dealer_hidden_value": dh, 
                                           "dealer_aces": da, 
                                           "dealer_sticks": ds}

    @staticmethod
    def state_is_terminal(state):
        # if both players stick, terminate
        if state["player_aces"] and state["dealer_sticks"]:
            return True
        else:
            # otherwise you can keep playing ...
            return False

    @staticmethod
    def dealer_turn(state):
        # state: a state dict that you're allowed to permute
        # the dealer sticks if his value >= 17, otherwise he hits
        if BlackJack.get_value(state["dealer_face_value"] + state["dealer_hidden_value"], state["dealer_aces"]) >= 17:
            state["dealer_sticks"] = 1
        else:
            new_card = BlackJack.draw_card()
            state["dealer_hidden_value"] += new_card
            if new_card == 1: state["dealer_aces"] += 1

    @staticmethod
    def apply_action(state:dict, action:str) -> dict:
        # any episode starts with a s_0 and a a_0
        # TODO: if either or both players starts with 21, you terminate with a win loss or draw

        # states in BlackJack are mutable dicts, so you need to create a copy before changing anything!!
        new_state = state.copy()
        #### PLAYER TURN ####
        if action == "hit":
            # draw another card
            new_card = BlackJack.draw_card()
            new_state["player_value"] = state["player_value"] + new_card
            if new_card == 1: new_state["player_aces"] += 1
            # if the player now has 21, he doesn't automatically win,
            # rather he has to wait his next turn and then stick with 21 to win
            # but for now, since the player hit, it is the dealers turn
            #### PLAYER TURN OVER ####

            #### DEALER TURN ####
            # let the dealer play a single turn, because after him the player gets another turn
            # the dealer only plays if he didnt stick yet though
            if not new_state["dealer_sticks"]:
                BlackJack.dealer_turn(new_state)
            #### DEALER TURN OVER ####
            return new_state

        elif action == "stick":
            new_state["player_sticks"] = 1
            #### PLAYER TURN OVER ####
            # if the player sticks with a 21 the dealer does not get a turn anymore because the player wins
            # otherwise the dealer keeps playing until he also sticks
            if BlackJack.get_value(new_state["player_value"], new_state["player_aces"]) != 21:
                # the dealer can play with a while not dealer_sticks loop
                # that means the state-action sequence will not include all the turns of the dealer
                # but rather only the turns where the player did not stick in the turn before
                # does that make sense??
                # should agents have full search tree w possible actions to consider,
                # although they're paralyzed after sticking?
                while not new_state["dealer_sticks"]:
                    BlackJack.dealer_turn(new_state)
                #### DEALER TURN OVER ####

        return new_state

    @staticmethod
    def get_possible_outcomes(state, action) -> dict:
        # TODO: compute all possible outcomes to a state and its probabilities
        pass

    @staticmethod
    def get_reward(state, action, new_state):
        # if the state is terminal
        # -> both player either stick, whatever the reason
        # the dealer behaviour and policy handle why both stick
        if BlackJack.state_is_terminal(state):
            player_value = BlackJack.get_value(new_state["player_value"], new_state["player_aces"])
            dealer_value = BlackJack.get_value(new_state["dealer_value"] + new_state["dealer_hidden_value"], new_state["dealer_aces"])
            # Player wins if
            # his value is larger than that of the dealer and the did not go bust
            # or if the player did not go bust but the dealer did
            if dealer_value < player_value <= 21 or player_value <= 21 < dealer_value:
                return 1
            # the dealer wins if his value is larger and the dealer did not go bust
            # or if the player went bust but the dealer did not
            if player_value < dealer_value <= 21 or dealer_value <= 21 < player_value:
                return -1
        # in any other case, we are either not terminate
        # or the dealer's and player's values are equal even though they both stick
        # or both went bust
        return 0



if __name__ == "__main__":
    env = BlackJack()
    print(env)