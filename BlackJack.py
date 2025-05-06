# Environment subclass for simulating Blackjack in a reinforcement learning context.
# Implements custom state representation, card drawing, player/dealer logic,
# action transitions, and game termination based on Blackjack rules.


from pprint import pprint
import random

from Environment import Environment

class BlackJack(Environment):
    # 1 can be an 11 if wanted
    cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    def __init__(self):
        # initializes Blackjack environment with action set and resets game
        blackjack_actions = ["hit", "stick"]
        # States
        # {
        #   player_value:int,
        #   player_num_aces:int,
        #   player_sticks: int,
        #   dealer_facing_value:int,
        #   dealer_hidden_cards:int,
        #   dealer_sticks: int,
        # }
        # to the agent the dealer_hidden_sum will always be shown as the same, e.g. 0
        # because the agent doesn't see the dealers hidden cards
        # we don't precompute all possible states, rather we exploit state generators here
        # Starting_state
        super().__init__(blackjack_actions)
        self.dealers_hidden_value = None
        self.dealers_hidden_aces = None
        self.starting_state = self.reset_game()

    @staticmethod
    def get_value(value_sum, num_usable_aces):
        # computes the actual value of a hand considering usable aces
        # input: value_sum (int), num_usable_aces (int); output: int
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
        # randomly draws one card from the Blackjack deck
        return random.choice(BlackJack.cards)

    def reset_game(self):
        # reset the hidden states values
        self.dealers_hidden_value = 0
        self.dealers_hidden_aces = 0
        # generate a single state, set all its values to zero and return it
        return 0, 0, 0, 0, 0, 0

    def set_start(self, starting_state=None):
        # sets the starting state of the environment, draws cards if none given
        print("Setting starting state of BlackJack")
        if starting_state is None:
            # start with a zero-state
            state = list(self.reset_game())
            # Player gets his first card
            # Player gets his second card
            for i in range(2):
                new_card = BlackJack.draw_card()
                state[0] += new_card
                if new_card == 1:
                    state[1] += 1
            # Dealer gets his first card, it is the open-faced card
            new_card = BlackJack.draw_card()
            state[3] += new_card
            # if new_card == 1: state[5] += 1
            # Dealer gets his second card, it is hidden
            new_card = BlackJack.draw_card()
            state[4] += 1#new_card
            # if new_card == 1: state[5] += 1
            self.starting_state = tuple(state)
        else:
            self.starting_state = starting_state

    @staticmethod
    def state_generator():
        # generates all valid state combinations of the Blackjack environment
        for pv in range(0, 50): # player_value
            for pa in range(0, 12): # player_aces
                if pa > pv: continue # you cannot have any aces with a sum of 0
                for ps in range(2): # player_sticks
                    for df in range(0, 12): # dealer_facing_value
                        for dh in range(0, 27): # dealer_hidden_card
                            if not df and dh: continue # dealer gains open card before hidden ones
                            for ds in range(2): # dealer_sticks
                                yield pv, pa, ps, df, dh, ds

    @staticmethod
    def print_state(state):
        # prints individual components of a state in readable format
        print("(0) player value:", state[0])
        print("(1) player aces:", state[1])
        print("(2) player sticks:", state[2])
        print("(3) dealer face_value:", state[3])
        print("(4) dealer hidden cards:", state[4])
        print("(5) dealer sticks:", state[5])
        print()
    
    @staticmethod
    def state_is_terminal(state) -> bool:
        # returns True if both players have chosen to stick
        # print("State:", state)
        if state[2] and state[5]:
            return True
        else:
            # otherwise you can keep playing ...
            return False

    def dealer_turn(self, state):
        # performs one dealer move (hit or stick)
        state = list(state)
        # he sticks if his value >= 17,
        if self.get_dealer_value(state[3]) >= 17:
            # print("The dealer is gonna stick")
            state[5] = 1
        # otherwise he hits (he draws another hidden card)
        else:
            # print("The dealer is gonna hit")
            new_card = BlackJack.draw_card()
            state[4] += 1 # increase the hidden card counter
            self.dealers_hidden_value += new_card
            self.dealers_hidden_aces += new_card == 1
        return tuple(state)

    def is_this_action_possible(self, state, action) -> bool:
        # determines if a given action is legal in the current state
        if action == "hit" and self.get_value(state[0], state[1]) >= 21:
                return False
        return True


    def apply_action(self, state:tuple, action:str) -> tuple:
        # applies a player's action and returns the resulting state
        # # if the game just started, i.e. both players have 0 values, make each of them draw 2 cards
        # # for this there is the function BlackJack.start_game
        # if state == self.starting_state:
        #     state = BlackJack.start_game()
        # states in BlackJack are mutable dicts, so you need to create a copy before changing anything!!
        new_state = list(state)
        #### PLAYER TURN ####
        if action == "hit":
            # print("Entry state", state)
            # draw another card
            new_card = BlackJack.draw_card()
            # print("New card", new_card)
            new_state[0] = state[0] + new_card
            if new_card == 1: new_state[1] += 1
            # if the player now has 21, he doesn't automatically win,
            # rather he has to wait his next turn and then stick with 21 to win
            # but for now, since the player hit, it is the dealers turn
            #### PLAYER TURN OVER ####

            #### DEALER TURN ####
            # let the dealer play a single turn, because after him the player gets another turn
            # the dealer only plays if he didn't stick yet though
            if not new_state[5]:
                new_state = self.dealer_turn(new_state)
            #### DEALER TURN OVER ####

        elif action == "stick":
            new_state[2] = 1
            #### PLAYER TURN OVER ####
            # if the player sticks with a 21 the dealer does not get a turn anymore because the player wins
            # otherwise the dealer keeps playing until he also sticks
            if BlackJack.get_value(new_state[0], new_state[1]) != 21:
                # the dealer can play with a while not dealer_sticks loop
                # that means the state-action sequence will not include all the turns of the dealer
                # but rather only the turns where the player did not stick in the turn before
                # does that make sense??
                # should agents have full search tree w possible actions to consider,
                # although they're paralyzed after sticking?
                while not new_state[5]:
                    new_state = self.dealer_turn(new_state)
                #### DEALER TURN OVER ####
        # print("New state", new_state)
        return tuple(new_state)

    # returns a dict: mapping from new_state to transition probability p(new_state, reward | state, action)
    def get_possible_outcomes(self, state, action) -> dict:
        # compute all possible outcomes to a state and its probabilities
        pass


    def get_reward(self, state, action, new_state):
        # calculates reward based on terminal outcome
        # returns: 1 (win), -1 (loss), or 0 (draw or ongoing)
        # if the state is terminal
        # -> both player either stick, whatever the reason
        # the dealer behaviour and policy handle why both stick
        if BlackJack.state_is_terminal(new_state):
            player_value = BlackJack.get_value(new_state[0], new_state[1])
            dealer_value = self.get_dealer_value(new_state[3])
            # print("Final scores:", player_value, "---", dealer_value)
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

    def get_dealer_value(self, dealer_facing_value):
        # computes the full dealer hand value including hidden cards
        # this is an extra function because the dealers hand is also dependent on the hidden state
        # dealer_facing_value: value of the card face-up
        # dealer_hidden_cards: number of hidden cards the dealer was hit with
        return BlackJack.get_value(dealer_facing_value + self.dealers_hidden_value, self.dealers_hidden_aces)


if __name__ == "__main__":
    env = BlackJack()
    print(env)
    print(BlackJack.get_dealer_value(8, 3))