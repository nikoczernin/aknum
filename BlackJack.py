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
        # TODO: new strategy, the player plays alone, only seeing the dealer's facing value
        # the dealer waits his turn until the player is done, then play his full hand
        # this means that no dealer moves will be done until just before termination
        # and also dealer hidden cards and dealer_sticks are obsolete
        # usable aces are now limited to a single one, and thus player_usable_ace is a boolean,
        # because you cannot use more than a single ace anyway without busting
        # Dealer value is initially just the facing card and later gets bumped up by his hand
        # States
        # {
        #   player_value:int,
        #   player_usable_aces:int,
        #   player_sticks: int,
        #   dealer_value:int,
        # }
        # there is no need for any hidden states variables, because the dealer values are computed at termination
        # we don't precompute all possible states, rather we exploit state generators here
        # Starting_state
        super().__init__(blackjack_actions)
        self.starting_state = self.reset_game()

    def reset_game(self):
        # reset the hidden states values
        # generate a single state, set all its values to zero and return it
        return 0, 0, 0, 0

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
            # Dealer gets the rest of his cards at the end
            self.starting_state = tuple(state)
        else:
            self.starting_state = starting_state

    @staticmethod
    def get_value(value_sum, usable_ace):
        # computes the actual value of a hand considering usable aces
        # input: value_sum (int), usable_ace (boolean)
        if usable_ace and value_sum + 10 <= 21:
            value_sum += 10
        return value_sum

    @staticmethod
    def draw_card():
        # randomly draws one card from the Blackjack deck
        return random.choice(BlackJack.cards)

    @staticmethod
    def state_generator():
        # generates all valid state combinations of the Blackjack environment
        for pv in range(0, 31): # player_value
            for pa in range(0, 12): # player_usable_ace
                if pa > pv: continue # you cannot have any aces with a sum of 0
                for ps in range(2): # player_sticks
                    for dv in range(0, 11): # dealer_facing_value
                                yield pv, pa, ps, dv

    @staticmethod
    def print_state(state):
        # prints individual components of a state in readable format
        print("(0) player value: ", state[0])
        print("(1) player ace:   ", state[1])
        print("(2) player sticks:", state[2])
        print("(3) dealer value: ", state[3])
        print()
    
    @staticmethod
    def state_is_terminal(state, verbose=False) -> bool:
        if verbose: print("is this state terminal?", state)
        # returns True if both players have chosen to stick
        # print("State:", state)
        if state[2] or BlackJack.get_value(state[0], state[1])>=21:
            if verbose: print("\tYes it is")
            return True
        else:
            if verbose: print("\tNo it isnt")
            # otherwise you can keep playing ...
            return False

    def is_this_action_possible(self, state, action) -> bool:
        # determines if a given action is legal in the current state
        # hitting with a value above 21 is illegal
        if action == "hit" and self.get_value(state[0], state[1]) >= 21:
                return False
        return True


    def apply_action(self, state:tuple, action:str) -> tuple:
        # applies a player's action and returns the resulting state
        # # if the game just started, i.e. both players have 0 values, make each of them draw 2 cards
        # states in BlackJack are mutable dicts, so you need to create a copy before changing anything!!
        new_state = list(state)
        #### PLAYER TURN ####
        if action == "hit":
            # print("Entry state", state)
            # draw another card
            new_card = BlackJack.draw_card()
            new_state[0] = state[0] + new_card
            if new_card == 1: new_state[1] += 1
            # if the player now has 21, he doesn't automatically win,
            # rather he has to wait his next turn and then stick with 21 to win
            # but for now, since the player hit, it is the dealers turn
            # there is no dealer turn until termination
        elif action == "stick":
            new_state[2] = 1
        #### PLAYER TURN OVER ####
        #### DEALER TURN ####
        # the dealer gets to play his turn if the state is terminal
        if BlackJack.state_is_terminal(new_state):
            # the dealer keeps playing until he also sticks
            dealer_value = self.dealer_full_turn(new_state[3])
            new_state[3] = dealer_value
        #### DEALER TURN OVER ####
        return tuple(new_state)


    # new strategy: the player plays this full hand and then the dealer also plays his full hand
    # this means the states are independent of the dealer-cards except his starting card
    # therefore: here is a function that computes a full dealer-hand using only the starting card
    @staticmethod
    def dealer_full_turn(facing_value):
        # draw cards until the dealer is > 17 or goes bust
        total_dealer_value = facing_value
        usable_aces = facing_value == 1
        while True:
            new_card = BlackJack.draw_card()
            if new_card == 1: usable_aces = True
            total_dealer_value += new_card
            if BlackJack.get_value(total_dealer_value, usable_aces) >= 17:
                # the dealer sticks when he gets higher than 17 or goes bust
                break
        return total_dealer_value



    def get_reward(self, state, action, new_state):
        # calculates reward based on terminal outcome
        # returns: 1 (win), -1 (loss), or 0 (draw or ongoing)
        # if the state is terminal
        # -> both player either stick, whatever the reason
        # the dealer behaviour and policy handle why both stick
        if BlackJack.state_is_terminal(new_state):
            player_value = BlackJack.get_value(new_state[0], new_state[1])
            dealer_value = new_state[3]
            # print("Final scores:", player_value, "---", dealer_value)
            # Player wins if
            # his value is larger than that of the dealer, and he did not go bust
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

    # def get_dealer_value(self, dealer_facing_value):
    #     # computes the full dealer hand value including hidden cards
    #     # this is an extra function because the dealers hand is also dependent on the hidden state
    #     # dealer_facing_value: value of the card face-up
    #     # dealer_hidden_cards: number of hidden cards the dealer was hit with
    #     return BlackJack.get_value(dealer_facing_value + self.dealers_hidden_value, self.dealers_hidden_aces)


if __name__ == "__main__":
    env = BlackJack()
    print(env)
