from pprint import pprint
import random

from pyparsing import actions

from Environment import Environment

class BlackJack(Environment):
    # 1 can be an 11 if wanted
    cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    def __init__(self):
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
        self.starting_state = BlackJack.reset_game()

    @staticmethod
    def get_value(value_sum, num_usable_aces):
        # print(f"Computing value from a total of {value_sum} and {num_usable_aces} aces")
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
        return 0, 0, 0, 0, 0, 0

    def set_start(self, starting_state=None):
        print("Setting starting state of BlackJack")
        if starting_state is None:
            # start with a zero-state
            state = list(BlackJack.reset_game())
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
        for pv in range(1, 31): # player_value
            for pa in range(0, 2): # player_aces
                for ps in range(2): # player_sticks
                    for df in range(1, 12): # dealer_facing_value
                        for dh in range(1, 27): # dealer_hidden_card
                            for ds in range(2): # dealer_sticks
                                yield pv, pa, ps, df, dh, ds

    @staticmethod
    def print_state(state):
        print("(0) player value:", state[0])
        print("(1) player aces:", state[1])
        print("(2) player sticks:", state[2])
        print("(3) dealer face_value:", state[3])
        print("(4) dealer hidden cards:", state[4])
        print("(5) dealer sticks:", state[5])
        print()
    
    @staticmethod
    def state_is_terminal(state):
        # if both players stick, terminate
        # print("State:", state)
        if state[2] and state[5]:
            return True
        else:
            # otherwise you can keep playing ...
            return False

    @staticmethod
    def dealer_turn(state):
        state = list(state)
        # state: a state dict that you're allowed to permute
        # if BlackJack.get_value(state[3] + state[4], state[5]) >= 17:
        #     state[5] = 1
        # else:
        #     new_card = BlackJack.draw_card()
        #     state[4] += new_card
        #     if new_card == 1: state[5] += 1
        # if the dealer takes a turn, i.e. he did not stick yet
        # compute the dealer value to find out what he has
        # he sticks if his value >= 17,
        if BlackJack.get_dealer_value(state[3], state[4]) >= 17:
            print("The dealer is gonna stick")
            state[5] = 1
        # otherwise he hits (he draws another hidden card)
        else:
            print("The dealer is gonna hit")
            state[4] += 1
        return tuple(state)

    def is_this_action_possible(self, state, action) -> bool:
        if action == "hit" and BlackJack.get_value(state[0], state[1]) >= 21:
                return False
        return True



    def apply_action(self, state:dict, action:str) -> dict:
        # # if the game just started, i.e. both players have 0 values, make each of them draw 2 cards
        # # for this there is the function BlackJack.start_game
        # if state == self.starting_state:
        #     state = BlackJack.start_game()
        # states in BlackJack are mutable dicts, so you need to create a copy before changing anything!!
        new_state = list(state)
        #### PLAYER TURN ####
        if action == "hit":
            # draw another card
            new_card = BlackJack.draw_card()
            new_state[0] = state[0] + new_card
            if new_card == 1: new_state[1] += 1
            # if the player now has 21, he doesn't automatically win,
            # rather he has to wait his next turn and then stick with 21 to win
            # but for now, since the player hit, it is the dealers turn
            #### PLAYER TURN OVER ####

            #### DEALER TURN ####
            # let the dealer play a single turn, because after him the player gets another turn
            # the dealer only plays if he didnt stick yet though
            if not new_state[5]:
                new_state = BlackJack.dealer_turn(new_state)
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
                    new_state = BlackJack.dealer_turn(new_state)
                #### DEALER TURN OVER ####

        return tuple(new_state)

    # # returns a dict: mapping from new_state to transition probability p(new_state, reward | state, action)
    # @staticmethod
    # def get_possible_outcomes(state, action) -> dict:
    #     # TODO: compute all possible outcomes to a state and its probabilities
    #     # to get all possible future states, iterate over ALL possible states S using the state_generator
    #     all_possible_next_states = []
    #     for next_state in BlackJack.state_generator():
    #         # if we stick, all future states have the same first 3 values, the other may increase
    #         if action == "stick":
    #             if state[:3] == next_state[:3]:
    #                 if state[3] <= next_state[3] and state[4] <= next_state[4] and state[5] <= next_state[5]:
    #                     all_possible_next_states.append(next_state)
    #         # if we hit, all state values must increase for future states
    #         if state[0] <= next_state[0] and state[1] <= next_state[1] and state[2] <= next_state[2]:
    #             if state[3] <= next_state[3] and state[4] <= next_state[4] and state[5] <= next_state[5]:
    #                     all_possible_next_states.append(next_state)
    #     # each future state's probability is probably not equal, but i will work with that now though
    #     return {p:1/len(all_possible_next_states) for p in all_possible_next_states}

    @staticmethod
    def get_reward(state, action, new_state):
        # if the state is terminal
        # -> both player either stick, whatever the reason
        # the dealer behaviour and policy handle why both stick
        if BlackJack.state_is_terminal(new_state):
            print("This bitch is terminal!")
            player_value = BlackJack.get_value(new_state[0], new_state[1])
            dealer_value = BlackJack.get_value(new_state[3] + new_state[4], new_state[5])
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


    @staticmethod
    def get_dealer_value(dealer_facing_value, dealer_hidden_cards):
        # dealer_facing_value: value of the card face-up
        # dealer_hidden_cards: number of hidden cards the dealer was hit with
        # generate a random total value using the predetermined open obvervable card
        # plus the value of the known number of unknown unobservable cards values
        # since they are unknown and random, you can resample everytime
        # careful: if you resample the last turns card and the dealer did not stick that turn, he must not stick now
        # initiate the total as the value of the face-up card
        dealer_total = dealer_facing_value
        # print("Start with", dealer_total)
        # print()
        # also count aces
        dealer_aces = dealer_facing_value == 1
        i = 0
        # for every card the dealer has drawn
        while i < dealer_hidden_cards:
            # generate a new card value
            new_card = BlackJack.draw_card()
            # print("New card:", new_card)
            # if this is not the last card and it would case a go bust
            if i+1 < dealer_hidden_cards and BlackJack.get_value(dealer_total + new_card, dealer_aces + new_card==1) > 21:
                # restart this iteration
                # print("\tWe will skip this card")
                continue
            # otherwise keep this card
            # print("\tWe keep this card")
            # print()
            dealer_total = dealer_total + new_card
            dealer_aces += new_card == 1
            i += 1
        return BlackJack.get_value(dealer_total, dealer_aces)


if __name__ == "__main__":
    env = BlackJack()
    print(env)
    print(BlackJack.get_dealer_value(8, 3))