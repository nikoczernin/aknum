# ===================================================
# Author: Nikolaus Czernin
# Script:
# Description:
# ===================================================

from pprint import pprint
from GridWorld import GridWorld
from Bot import Bot


class MarkovDecisionProcess():
    @staticmethod
    def iterative_policy_evaluation(bot, accuracy_thresh, gamma):
        # initiate v: vector with the value q of the best possible action a in state s
        # initial values are random, except for terminal states, for them pick 0
        v = {s: (0 if s not in bot.env.terminal_states else 0) for s in bot.env.states}
        while True:
            Delta = 0
            for i, s in enumerate(bot.env.states):
                # get the value w of the currently best action for state s
                w = v[s]
                # apply the Bellman function to iteratively find a better action's value
                v[s] = bot.value_fun(s, 0, gamma)
                # get your improvement
                # if Delta >= abs(w - v[s]): print("picking new best function!")
                Delta = max(Delta, abs(w - v[s]))
            if Delta < accuracy_thresh:
                break
        return v

    @staticmethod
    def policy_iteration(bot, accuracy_thresh, gamma):
        print("Policy Iteration")
        # accuracy threshold. >0
        # loop the whole thing until the policy is stable and thus doesn't get updated anymore
        j = 0
        while True:
            j += 1
            print("Iteration", j)
            # Policy Evaluation
            # initiate v: vector with the value q of the best possible action a in state s
            # initial values are random, except for terminal states, for them pick 0
            v = {s: (0 if s not in bot.env.terminal_states else 0) for s in bot.env.states}
            while True:
                Delta = 0
                # for every possible state
                for i, s in enumerate(bot.env.states):
                    # if s in self.env.terminal_states: continue # no need to do anything with the terminal states
                    w = v[s]
                    v[s] = bot.value_fun(s, 0, gamma)
                    Delta = max(Delta, abs(w - v[s]))
                if Delta < accuracy_thresh:
                    break
            # Policy Improvement
            policyIsStable = True
            for s in bot.env.states:
                # skip terminal states
                if s in bot.env.terminal_states: continue
                # if s in self.env.terminal_states: continue # no need to do anything with the terminal states
                oldAction = bot.pick_action(s)
                # set a new action for the policy
                # pick the action that maximizes the action-value function
                # do that by picking the max value of the policy-keys (i.e. the actions) using the
                # action-value-function as a "key" (i.e. the thing that the max function uses to evaluate the values)
                best_action = max(bot.policy[s].keys(), key=lambda a: bot.action_value_fun(s, a, 0, gamma))
                # update the policy
                bot.policy_set_action(s, best_action)
                if best_action != oldAction:
                    print(f"State {s}: Better action found. {oldAction} ==> {best_action}")
                    policyIsStable = False
            if policyIsStable:
                print("Policy Is Stable. Returning ...")
                return v
            print()

    @staticmethod
    def value_iteration(bot, accuracy_thresh, gamma):
        print("Value Iteration")
        v = {s: (0 if s not in bot.env.terminal_states else 0) for s in bot.env.states}
        for j in range(1000000):
            Delta = 0
            for i, s in enumerate(bot.env.states):
                # skip terminal states
                if s in bot.env.terminal_states: continue
                w = v[s]
                # get the maximum possible action-value function given the state s
                # states are deterministic so no need to get probabilities of s_t_1
                v[s] = max([bot.action_value_fun_star(s, a, gamma, v) for a in bot.policy[s].keys()])
                Delta = max(Delta, abs(w - v[s]))
            if Delta < accuracy_thresh:
                break
        # Policy calculation
        for i, s in enumerate(bot.env.states):
            # skip terminal states
            if s in bot.env.terminal_states: continue
            # set a new best action for the policy
            # pick the action that maximizes the action-value function
            # do that by picking the max value of the policy-keys (i.e. the actions) using the
            # action-value-function as a "key" (i.e. the thing that the max function uses to evaluate the values)
            best_action = max(bot.policy[s].keys(), key=lambda a: bot.action_value_fun_star(s, a, gamma, v))
            bot.policy_set_action(s, best_action)
            print(f"State {s}: Best action = {best_action}")
        return v



def main():
    w, h = 4, 4 # grid size
    env = GridWorld(w, h, terminal_states=[(0, 0), (w-1, h-1)])
    bot = Bot(env=env, T = 4)
    print(bot.env) # draw the grid
    accuracy_thresh = .001
    gamma = 1

    # perform iterative policy evaluation
    v = MarkovDecisionProcess.iterative_policy_evaluation(bot, accuracy_thresh, gamma)
    print()
    print("Initial policy (all actions are equally probable")
    # pprint(bot.policy)
    bot.draw_policy()
    print("Value estimations:")
    bot.draw_v(v)
    print()

    # set the probabilities in the policy from uniform probs to all either 0 or 1
    bot.hardline_policy()
    # perform policy iteration to get the optimal policy
    v = MarkovDecisionProcess.policy_iteration(bot, accuracy_thresh, gamma)
    print()
    print("Policy after policy iteration")
    # pprint(bot.policy)
    bot.draw_policy()
    print("Value estimations:")
    bot.draw_v(v)
    print()
    # print(bot.env) # draw the grid

    # perform value iteration for show
    # first reset the policy to random hardline probabilities
    print("... resetting policy ...")
    bot.init_policy(hardline=True)
    v = MarkovDecisionProcess.value_iteration(bot, accuracy_thresh, gamma)
    print()
    print("Value estimations:")
    bot.draw_v(v)
    print("Final policy after value iteration")
    # pprint(bot.policy)
    bot.draw_policy()
    print()
    # print(bot.env) # draw the grid
    print("All done :)")


if __name__ == "__main__":
    main()
