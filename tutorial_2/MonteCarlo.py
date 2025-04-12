# ===================================================
# Author: Nikolaus Czernin
# Script: Off-Policy Monte Carlo Control for GridWorld
# Description: Implements MC control for a 4×4 grid world using both on-policy and off-policy
#              learning. Uses ε-greedy policies, returns tracking, and value updates to
#              estimate the optimal policy π* via Monte Carlo sampling.
# ===================================================

from pprint import pprint
import random
import numpy as np

from GridWorld import GridWorld
from Bot import Bot

def policy_control(bot: Bot, epsilon=.1, gamma=1, visit="first", off_policy=False, behaviour_policy=None, num_episodes=1000):
    # if we are working off policy, the behaviour_policy is a copy of the bot policy
    # that will not be updated during the episodes
    if off_policy:
        behaviour_policy = bot.policy.copy()
    else: # on-policy: use the Bot's policy for the episodes, while also updating it
        behaviour_policy = bot.policy
    q = {s:{a:0 for a in bot.env.actions} for s in bot.env.states}
    returns = {s: {a: [] for a in bot.env.actions} for s in bot.env.states}
    for k in range(num_episodes):
        # generate an episode
        R_k, t_k, transitions = bot.episode(epsilon=epsilon, policy=behaviour_policy)
        # print("Length of episode:", t_k)
        # transitions looks like this: [(S0, A0, R1, S1), (S1, A1, R2, S2), ..., (ST-1, AT-1, RT, _)]
        # pprint(transitions)
        g = 0
        # iterate over sequence backwards!
        for t in range(len(transitions)-1, -1, -1):
            # compute the return of the current time-step
            g = gamma * g + transitions[t][2]
            s_t = transitions[t][0]
            a_t = transitions[t][1]
            # for first visit MC, check if the state occurred before the current time-step, then save its return
            # for every visit MC, save its return anyway
            if any(previous_transition[0] == s_t for previous_transition in transitions[:t]) or visit == "every":
                # save g to the returns list for the current state and action
                returns[s_t][a_t].append(g)
                # update q, the state-value mapping
                q[s_t][a_t] = np.mean(returns[s_t][a_t])
                # get the action that leads to the maximum value in the current state according to q
                # ties are broken randomly
                a_optimal = random.choice([
                    a for a, v in q[s_t].items()
                    if v == max(q[s_t].values())
                ])
                # update the Bot policy (not necessarily the behaviour_policy) for all actions in the current state
                for a in bot.env.actions:
                    bot.policy[s_t][a] = (1 - epsilon + epsilon/len(q[s_t])) if a_optimal == a else epsilon/len(q[s_t])




def test_control_grid_world():
    w, h = 4, 4 # grid size
    env = GridWorld(w, h, terminal_states=[(0, 0), (w-1, h-1)])
    bot = Bot(env=env, T = 100)
    epsilon = .1
    print("Policy before policy control:")
    pprint(bot.policy)
    print("Now we run some episodes and see what we get:")
    results = [bot.episode() for _ in range(1000)]
    print("Mean reward:", np.mean([x[0] for x in results]))
    print("Mean time-step of termination:", np.mean([x[1] for x in results]))
    print()
    print("##### Performing policy control #####")
    print()
    policy_control(bot, epsilon=epsilon, off_policy=True)
    print("Policy after policy control:")
    pprint(bot.policy)
    print("Now we run some episodes and see what we get:")
    results = [bot.episode() for _ in range(1000)]
    print("Mean reward:", np.mean([x[0] for x in results]))
    print("Mean time-step of termination:", np.mean([x[1] for x in results]))


def main():
    test_control_grid_world()

if __name__ == '__main__':
    main()