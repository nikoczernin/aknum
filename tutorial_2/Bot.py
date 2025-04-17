# ===================================================
# Author: Nikolaus Czernin
# Script: Bot for GridWorld — Policy & Value Iteration (Tutorial 2)
# Description: Implements a reinforcement learning agent (Bot) that interacts
#              with a GridWorld environment. Includes policy evaluation,
#              policy iteration, and value iteration algorithms.
# ===================================================



import random
from pprint import pprint

import numpy as np

from Environment import Environment

class Bot():
    def __init__(self, env, T=10):
        self.T = T # max number of time steps the agent is allowed to take
        # the environment contains states and actions
        self.env = env
        # in the scriptum the policy is often referred to as a vector of length S, containing actions
        # I opted to use a dictionary that maps from states to actions to probabilities of using that action in that state
        # self.policy: pi(a|s) -> [0, 1]
        # we initialize them as equiprobable, which leads to random action-picking
        self.policy = {}
        self.init_policy()

    def init_policy(self, hardline=False):
        # env in this case would be the GridWorld
        # states is a list of possible states
        # actions is also a list
        # give every action the same probability
        for s in self.env.state_generator():
            possible_actions = [a for a in self.env.actions if (self.env.is_this_action_possible(s, a))]
            self.policy[s] = {}
            for a in self.env.actions:
                self.policy[s].update({a: 1 / len(possible_actions) if a in possible_actions else 0})
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
    def policy_set_action(self, s, a_new, policy = None):
        if policy is None: policy = self.policy
        for a in policy[s].keys():
            policy[s][a] = 1 if a == a_new else 0
        # return policy


    # function for picking the best action from a policy, draws are resolved randomly
    # you can supply an epsilon for epsilon greedy exploration-exploitation
    def pick_action(self, s_t, epsilon=-1, policy=None):
        if policy is None: policy = self.policy
        if s_t in self.env.terminal_states:
            return
        # generate a random uniform number
        # if it is smaller than epsilon, we explore, otherwise we exploit normally
        if random.random() < epsilon:
            return random.choice(list(policy[s_t].keys()))
        else:
            # get the action with the highest probability given the current state from the policy
            try:
                max_prob = max(policy[s_t].values())
            except Exception as e:
                pprint(policy)
                raise e
            # get all actions for this state that have the max probability
            best_keys = [k for k, v in policy[s_t].items() if v == max_prob]
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
                total_reward += action_reward * action_prob
        # return the sum of the expected reward in this state given the policy
        return total_reward


    def action_value_fun(self, s_t, a, t, gamma):
        # for all states s_t_1 that can result from using this action in this state s_t
        # which is this case is wuascht because the states are deterministic anyway
        state_transition_probs = self.env.get_possible_outcomes(s_t, a)
        # the future possible rewards are computed recursively using the value function, that means
        # any future decisions are made with the policy rather than a fixed function
        return sum([prob * (self.env.get_reward(s_t, a, s_t_1) + gamma * self.value_fun(s_t_1, t + 1, gamma))
                for s_t_1, prob in state_transition_probs.items()])


    def action_value_fun_star(self, s, a, gamma, v):
        state_transition_probs = self.env.get_possible_outcomes(s, a)
        # if s_t_1 is a terminal state, its reward will be zero because were already at the goal
        # otherwise just return the optimal next action's expected reward discounted by gamma
        return sum([prob * (self.env.get_reward(s, a, s_t_1) + (gamma * v[s_t_1] if s_t_1 not in self.env.terminal_states else 0))
                for s_t_1, prob in state_transition_probs.items()])

    def draw_v(self, v):
        self.env.grid.draw_grid(v)
        print()

    def draw_policy(self):
        print("Current policy")
        pi = {s:self.env.action_str(self.pick_action(s)) for s in self.policy.keys()}
        self.env.grid.draw_grid(pi)
        print()

    def episode(self, policy=None, epsilon=-1, verbose=False):
        # 1 episode should look like this: {S0, A0, R1, S1, A1, R2, ..., ST-1, AT-1, RT}
        if policy is None: policy = self.policy
        s_t = self.env.starting_state
        transitions = []
        if verbose: print("Starting episode at", s_t)
        R, t = 0, 0
        for t in range(self.T):
            a = self.pick_action(s_t, epsilon=epsilon, policy=policy)
            # move into a new state
            outcomes_dict = self.env.get_possible_outcomes(s_t, a)
            # the transition includes
            s_t_1 = self.env.resolve_outcome(outcomes_dict)
            r = self.env.get_reward(s_t, a, s_t_1)
            R += r
            transitions.append((s_t, a, r, s_t_1))
            s_t = s_t_1
            if self.env.state_is_terminal(s_t):
                break
        if verbose:
            print("Finished episode at", s_t)
            print("Total reward:", R)
            print()
        return R, t, transitions

    def make_test_runs(self, k=100, *args, **kwargs):
        print(f"Performing {k} test runs ...")
        results = [self.episode(*args, **kwargs) for _ in range(k)]
        print("Best reward:", np.max([x[0] for x in results]))
        print("Mean reward:", np.mean([x[0] for x in results]))
        print("Mean time-step of termination:", np.mean([x[1] for x in results]))
