from pprint import pprint
import random

import numpy as np

from Environment import Environment
from utils import plot_line_graph


# States:
# (price_t, reward_t, previous_action, timestep_t, min_price, max_price)

class AutoregressiveTrendProcess(Environment):
    def __init__(self, T=100, Lambda=3, kappa=0.9, transaction_cost=0.1):
        actions = [1, 0, -1] # "long", "neutral", "short"
        # set the final timestep, i.e. the terminal state
        self.T = T # timesteps are t, in the lecture notes they are k
        self.trends = [] # this is a in the lecture notes
        self.noises = [] # this is b in the lecture notes
        self.Lambda = Lambda
        self.kappa = kappa
        self.transaction_cost = transaction_cost # transaction cost
        self.prices = []
        super().__init__(actions)

    def set_start(self):
        # compute all T trend and noise values
        self.trends = [1]
        self.noises = [np.random.normal()]
        for _ in range(self.T-1):
            self.trends.append(self.trends[-1] + self.noises[-1] + self.Lambda * np.random.normal())
            self.noises.append(self.kappa * self.noises[-1] + np.random.normal())
        # now compute all T prices
        for t in range(self.T):
            self.prices.append(np.exp(self.trends[t] / (max(self.trends) - min(self.trends))))
        # set the starting state
        starting_price = self.prices[0]
        super().set_start((starting_price, 0, 0, 0, starting_price, starting_price))

    def state_is_terminal(self, state) -> bool:
        if state[3] == self.T-1:
            return True
        return False

    def apply_action(self, state, action):
        previous_action = state[2]

        t_new = state[3] + 1
        # get the old and the new prices
        price_old = self.prices[max(0, t_new - 1)]
        price_new  = self.prices[t_new] if t_new < self.T else price_old
        min_new = min(state[4], price_new)
        max_new = max(state[5], price_new)
        # the reward is the price change (if held or bought)
        # - transaction costs if bought or sold
        # transaction cost is the saved ratio times the new price
        reward_new = (price_new - price_old) * (action >= 0) - abs(self.transaction_cost * action * price_new)
        return price_new, reward_new, action, t_new, min_new, max_new

    def get_reward(self, state, action, new_state=None):
        if new_state is None: raise Exception("new_state is None, that's illegal!")
        return new_state[1]

    def print_state(self,  state):
        print("Trend: ", state[0])
        print("Reward: ", state[1])
        print("Action: ", state[2])
        print("Time-step: ", state[3])
        print("Price-min: ", state[4])
        print("Price-max: ", state[5])
        print()

def main():
    T=30
    env = AutoregressiveTrendProcess(T=T)
    env.set_start()
    s = env.starting_state
    print("Starting state:")
    env.print_state(s)
    rewards = []
    print(f"Running {T} time-steps, picking random actions ...")
    for t in range(T):
        # perform a single random action
        a = random.choice(env.actions)
        s = env.apply_action(s, a)
        a=0
        # and collect the reward
        # env.print_state(s)
        rewards.append(s[1])
        pass
    print("Final state:")
    env.print_state(s)
    print("Rewards: ", rewards)
    plot_line_graph(env.prices, rewards, labels=["Prices", "Rewards"], title="")
    plot_line_graph()


if __name__ == "__main__":
    main()
