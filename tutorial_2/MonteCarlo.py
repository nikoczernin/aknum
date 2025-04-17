# ===================================================
# Author: Nikolaus Czernin
# Script: Off-Policy Monte Carlo Control for GridWorld
# Description: Implements MC control for a 4×4 grid world using both on-policy and off-policy
#              learning. Uses ε-greedy policies, returns tracking, and value updates to
#              estimate the optimal policy π* via Monte Carlo sampling.
# ===================================================
from email import policy
from pprint import pprint
import random
import numpy as np

from GridWorld import GridWorld
from Bot import Bot
from tutorial_2.CliffWalking import CliffWalking
from tutorial_2.FrozenLake import FrozenLake
from tutorial_2.WindyGridWorld import WindyGridWorld


def MC_policy_control(bot: Bot, epsilon=.1, gamma=1, visit="first", off_policy=False, behaviour_policy=None, num_episodes=1000):
    # if we are working off policy, the behaviour_policy is a copy of the bot policy
    # that will not be updated during the episodes
    if off_policy:
        behaviour_policy = bot.policy.copy()
    else: # on-policy: use the Bot's policy for the episodes, while also updating it
        behaviour_policy = bot.policy

    q = {}
    returns = {}
    for s in bot.env.state_generator():
        q[s] = {a: 0 for a in bot.env.actions}
        returns[s] = {a: [] for a in bot.env.actions}

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



def test(env, epsilon=.4, num_episodes=1000, off_policy=True):
    print(env)
    bot = Bot(env=env, T = 100)
    # print("Policy before policy control:")
    # pprint(bot.policy)
    print("Now we run some episodes and see what we get (before optimizing the policy):")
    bot.make_test_runs(k=1000)
    print()
    print("##### Performing policy control #####")
    print()
    MC_policy_control(bot, epsilon=epsilon, off_policy=off_policy, num_episodes=num_episodes)
    print("Policy after policy control:")
    bot.draw_policy()
    pprint(bot.policy)
    bot.make_test_runs(k=1000)



def test_grid_world():
    h, w = 4, 4 # grid size
    env = GridWorld(h, w, terminal_states=[(0, 0), (w-1, h-1)], starting_state=(3, 2))
    epsilon = .1
    test(env, epsilon)


def test_windy_world():
    h, w = 4, 4
    wind_forces = [ # only vertical please
        (0, 0),
        (-2, 0),
        (-1, 0),
        (0, 0)
    ]
    env = WindyGridWorld(h, w, terminal_states=[(2, 2)], starting_state=(0, 0), forces=wind_forces)
    epsilon = .5
    test(env, epsilon)


def test_cliff_walking():
    h, w = 4, 5
    cliffs = [(1, 1), (1, 2), (1, 3)]
    env = CliffWalking(h, w, terminal_states=[(1, 4)], starting_state=(1, 0), cliffs=cliffs)
    epsilon = .6
    test(env, epsilon)


def test_frozen_lake():
    h, w = 4, 4
    holes = [(3, 0), (1, 1), (1, 3), (2, 3)]
    goals = [(3, 3)]
    env = FrozenLake(h, w, goals, holes, (0, 0), slippery=True)

    # test(env, epsilon=.5)
    epsilon = .3
    test(env, epsilon, num_episodes=10000)




def main():
    pass
    # test_grid_world()
    # test_windy_world()
    test_cliff_walking()
    # test_frozen_lake()

if __name__ == '__main__':
    main()
