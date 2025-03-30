import random
from pprint import pprint
import numpy as np
from GridWorld import GridWorld


class Bot():
    def __init__(self, T=10):
        self.r = -1 # # rewards are a constant -1
        self.T = T # max number of time steps
        self.S = [] # all states
        self.A = [] # all actions
        # in the scriptum the policy is often referred to as a vector of length S, containing actions
        # I opted to use a dictionary that maps from states to actions to probabilities of using that action in that state
        # self.policy: pi(a|s) -> [0, 1]
        # we initialize them as equiprobable, which leads to random action-picking
        self.policy = {}
        # the environment can also be saved into a class variable ... I think
        self.env = None

    def init_policy(self, hardline=False):
        # env in this case would be the GridWorld
        # states is a list of possible states
        # actions is also a list
        # give every action the same probability
        for s in self.S:
            possible_actions = [a for a in self.A if (self.env.is_this_action_possible(s, a) if self.env is not None else True)]
            self.policy.update({s: {a: 1 / len(possible_actions) for a in possible_actions}})
        if hardline: self.hardline_policy()

    # policy is typically a probability of all possible actions in a given state
    # when an action is performed, the policy returns each action with a certain probability
    # if you want to max out the probability of the most probable actions to get only probs of either 0 or 1
    # then this function is for you
    def hardline_policy(self):
        for s in self.policy.keys():
            # get a random pick of the most probable actions
            a_max = self.pick_action(s)
            # now set all probabilities to zero, except for the max one
            for a in self.policy[s].keys():
                self.policy[s][a] = 1 if a == a_max else 0

    # with this function you can update a policy to set all actions for a given state to a probability of 0
    # except for the action a_new, which will get a probability of 1
    def policy_set_action(self, s, a_new):
        for a in self.policy[s].keys():
            self.policy[s][a] = 1 if a == a_new else 0

    # function for picking the best action from a policy, draws are resolved randomly
    def pick_action(self, s_t):
        # get the action with the highest probability given the current state from the policy
        max_prob = max(self.policy[s_t].values())
        # get all actions for this state that have the max probability
        best_keys = [k for k, v in self.policy[s_t].items() if v == max_prob]
        # pick a random action from the most likely ones
        chosen_key = random.choice(best_keys)
        return chosen_key

    # define the recursive value function that gets the best value possible for each state
    def value_fun(self, s_t, t, gamma):
        total_reward = 0
        # if this recursion is either at a terminal state or at maximum depth: return without any reward
        if s_t in self.env.terminal_states or t >= self.T:
            return total_reward
        # for every action that is possible from this state
        for a in self.policy[s_t].keys():
            # get the probability of performing the action
            action_prob = self.policy[s_t][a]
            if action_prob: # > 0, otherwise no need to go deeper in the search tree here
                action_reward = self.action_value_fun(s_t, a, t, gamma)
                # print(action_reward, "->", action_reward*action_prob)
                total_reward += action_reward * action_prob
        # return the sum of the expected reward in this state given the policy
        # print("\t"*t, total_reward)
        return total_reward

    def action_value_fun(self, s_t, a, t, gamma):
        # for all states s_t_1 that can result from using this action in this state s_t
        # which is this case is wuascht because the states are deterministic anyway
        s_t_1 = self.env.apply_action(s_t, a)
        # the same thing goes for the rewards, they are always fixed at -1, no matter the s_t_1
        reward = self.r
        # the future possible rewards are computed recursively using the value function, that means
        # any future decisions are made with the policy rather than a fixed function
        return reward + gamma * self.value_fun(s_t_1, t + 1, gamma)

    def iterative_policy_evaluation(self, accuracy_thresh, gamma):
        # initiate v: vector with the value q of the best possible action a in state s
        # initial values are random, except for terminal states, for them pick 0
        v = [1 if s not in self.env.terminal_states else 0 for s in self.S]
        while True:
            Delta = 0
            for i, s in enumerate(self.S):
                # get the value w of the currently best action for state s
                w = v[i]
                # apply the Bellman function to iteratively find a better action's value
                v[i] = self.value_fun(s, 0, gamma)
                # get your improvement
                # if Delta >= abs(w - v[i]): print("picking new best function!")
                Delta = max(Delta, abs(w - v[i]))
            if Delta < accuracy_thresh:
                break
        return v

    def policy_iteration(self, accuracy_thresh, gamma):
        print("Policy Iteration")
        # accuracy threshold. >0
        # loop the whole thing until the policy is stable and thus doesnt get updated anymore
        j = 0
        while True:
            j += 1
            print("Iteration", j)
            # Policy Evaluation
            # initiate v: vector with the value q of the best possible action a in state s
            # initial values are random, except for terminal states, for them pick 0
            v = [1 if s not in self.env.terminal_states else 0 for s in self.S]
            while True:
                Delta = 0
                # for every possible state
                for i, s in enumerate(self.S):
                    # if s in self.env.terminal_states: continue # no need to do anything with the terminal states
                    w = v[i]
                    v[i] = self.value_fun(s, 0, gamma)
                    Delta = max(Delta, abs(w - v[i]))
                if Delta < accuracy_thresh:
                    break
            # Policy Improvement
            policyIsStable = True
            for s in self.S:
                # if s in self.env.terminal_states: continue # no need to do anything with the terminal states
                oldAction = self.pick_action(s)
                # set a new action for the policy
                # pick the action that maximizes the action-value function
                # do that by picking the max value of the policy-keys (i.e. the actions) using the
                # action-value-function as a "key" (i.e. the thing that the max function uses to evaluate the values)
                best_action = max(self.policy[s].keys(), key = lambda a: self.action_value_fun(s, a, 0, gamma))
                # update the policy
                self.policy_set_action(s, best_action)
                if best_action != oldAction:
                    print(f"State {s}: Better action found. {oldAction} ==> {best_action}")
                    policyIsStable = False
            if policyIsStable:
                print("Policy Is Stable. Returning ...")
                print()
                return v
            print()


    def value_iteration(self, accuracy_thresh, gamma):
        print("Value Iteration")
        v = [1 if s not in self.env.terminal_states else 0 for s in self.S]
        while True:
            Delta = 0
            for i, s in enumerate(self.S):
                w = v[i]
                # get the maximum possible action-value function given the state s
                v[i] = max([self.value_fun(s, 0, gamma) for a in self.policy[s].keys()])
                Delta = max(Delta, abs(w - v[i]))
            if Delta < accuracy_thresh:
                break
        # Policy calculation
        for i, s in enumerate(self.S):
            # set a new best action for the policy
            # pick the action that maximizes the action-value function
            # do that by picking the max value of the policy-keys (i.e. the actions) using the
            # action-value-function as a "key" (i.e. the thing that the max function uses to evaluate the values)
            best_action = max(self.policy[s].keys(), key=lambda a: self.action_value_fun(s, a, 0, gamma))
            self.policy_set_action(s, best_action)
            print(f"State {s}: Best action = {best_action}")
        print()
        return v



def main():

    bot = Bot(T = 3 )
    w, h = 4, 4 # grid size
    # create an array of all states s in S
    bot.S = [(x, y) for x in range(w) for y in range(h)]
    # the first and last (top left and bottom right) fields are terminal states (fields)
    terminal_states = [bot.S[0], bot.S[-1]]
    bot.env = GridWorld(w, h, terminal_states=terminal_states)
    print(bot.env)
    # define possible actions
    bot.A = [(0, -1), #up
               (0, 1),  #down
               (-1, 0), #left
               (1, 0)]  #right
    # create a policy for the bot
    bot.init_policy()
    accuracy_thresh = .001
    gamma = 1

    # perform iterative policy evaluation
    v = bot.iterative_policy_evaluation(accuracy_thresh, gamma)
    pprint(bot.policy)
    print(v)
    print()

    # set the probabilities in the policy from uniform probs to all either 0 or 1
    bot.hardline_policy()
    # perform policy iteration to get the optimal policy
    v = bot.policy_iteration(accuracy_thresh, gamma)
    pprint(bot.policy)
    print(v)
    print()

    # perform value iteration for show
    # first reset the policy to random hardline probabilities
    bot.init_policy(hardline=True)
    v = bot.value_iteration(accuracy_thresh, gamma)
    pprint(bot.policy)
    print(v)
    print()
if __name__ == "__main__":
    main()
