from pprint import pprint
import random

from Environment import Environment

class Mäxchen(Environment):
    """
    State: (previous_value_public, own_value_real, own_value_public)
    Hidden state: previous_value_real
    The agent receives the state, without knowing the hidden state
    He either receives a state where own_value_real are None,
        which means he didn't toss yet
        and still has the option to doubt the previous_value_public
        or accept it and make a toss
    or own_value_real is already determined
        which means he already tossed
        and now has the option to set a own_value_public or toss again blindly
    """
    def __init__(self):
        actions = ["doubt", "accept", "blind_toss"] + [f"bluff_{v}" for v in self.value_generator()]
        super().__init__(actions)

    def toss_dice(self):
        x, y = random.randint(1, 6), random.randint(1, 6)
        print(x, y)
        return self.get_value(x, y)

    def get_value(self, x:int, y:int):
        pair = [x,y]
        pair.sort(reverse=True)
        # Pasch is higher than everything
        if x == y:
            pair.append(0)
        string_pair = "".join(str(x) for x in pair)
        # 21 is the highest
        if string_pair == "21": string_pair = "2100"
        return int(string_pair)

    def value_generator(self):
        for i in range(1, 7):
            for j in range(1, 7):
                yield self.get_value(i, j)


env=Mäxchen()
for i in range(100):
    print(env.toss_dice())
    print()