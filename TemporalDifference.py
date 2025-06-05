from pprint import pprint
from random import random

from matplotlib import pyplot as plt

from Bot import Bot
from CliffWalking import CliffWalking
from GridWorld import GridWorld
from utils import plot_blockwise_mean_rewards_line_graph, plot_line_graph


def SARSA(bot:Bot, alpha=.5, epsilon=.1, gamma=1.0, num_episodes=1000, expected=False):
    if expected: print("Performing Expected-SARSA...")
    else: print("Performing SARSA...")
    # init action-value function (tabular, finite)
    # should we consider all states beforehand?
    Q = {s:{a:0 for a in bot.env.actions} for s in bot.env.state_generator()}
    total_rewards = [] # keep a list of total rewards for each episode
    for k in range(num_episodes):
        episode_reward = 0
        # reset the env
        bot.env.reset()
        # before every episode: set a policy according to Q
        for s in Q.keys():
            # update the policy of the bot
            bot.policy_set_action(s, max(Q[s], key=Q[s].get))

        # initialize s
        s_t = bot.env.starting_state
        # choose action a from s_t using e-greedy policy
        a_t = bot.pick_action(s_t, epsilon)
        for t in range(bot.T):
            # if s is terminal, terminate the episode
            if bot.env.state_is_terminal(s_t):
                break
            # take action a_t and receive s_t_1 and reward r
            s_t_1 = bot.env.apply_action(s_t, a_t)
            r = bot.env.get_reward(s_t, a_t, s_t_1)
            episode_reward += r
            # choose action a_t_1 from s_t_1 using e-greedy-policy
            a_t_1 = bot.pick_action(s_t_1, epsilon)
            # perform update of Q using SARSA update formula
            # if the next state is terminal, its future reward is zero
            if bot.env.state_is_terminal(s_t_1):
                Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r - Q[s_t][a_t])
            else:
                if not expected:
                    Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r + gamma * Q[s_t_1][a_t_1] - Q[s_t][a_t])
                else:
                    Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r - Q[s_t][a_t] + sum([
                        bot.policy[s_t_1][a_t_1] * Q[s_t_1][a_t_1] for a_t_1 in bot.policy[s_t_1]
                    ]))

            # set s_t and a_t to the new state and action (we are doing on-policy control)
            s_t, a_t = s_t_1, a_t_1
        total_rewards.append(episode_reward)

    return Q, total_rewards

def expected_SARSA(bot:Bot, alpha=.5, epsilon=.1, gamma=1.0, num_episodes=1000):
    return SARSA(bot, alpha, epsilon, gamma, num_episodes, expected=True)


def Q_Learning(bot:Bot, alpha=.5, epsilon=.1, gamma=1.0, num_episodes=1000):
    print("Performing Q-Learning...")
    # init action-value function (tabular, finite)
    # should we consider all states beforehand?
    # Q = {}
    Q = {s: {a: 0 for a in bot.env.actions} for s in bot.env.state_generator()}
    total_rewards = [] # keep a list of total rewards for each episode

    for k in range(num_episodes):
        # reset the env
        bot.env.reset()
        episode_reward = 0
        for s in Q.keys():
            # update the policy of the bot
            bot.policy_set_action(s, max(Q[s], key=Q[s].get))

        # initialize s
        s_t = bot.env.starting_state
        for t in range(bot.T):
            # if s is terminal, terminate the episode
            if bot.env.state_is_terminal(s_t):
                break
            # choose action a from s_t using e-greedy policy
            a_t = bot.pick_action(s_t, epsilon)
            # take action a_t and receive s_t_1 and reward r
            s_t_1 = bot.env.apply_action(s_t, a_t)
            r = bot.env.get_reward(s_t, a_t, s_t_1)
            episode_reward += r
            # choose action a_t_1 from s_t_1 that maximizes Q[s_t_1]
            a_t_1 = max(Q[s_t], key=Q[s_t].get)
            # perform update of Q using SARSA update formula
            # if the next state is terminal, its future reward is zero
            if bot.env.state_is_terminal(s_t_1):
                Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r - Q[s_t][a_t])
            else:
                Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r + gamma * Q[s_t_1][a_t_1] - Q[s_t][a_t])


            # set s_t to the new state (off-policy control so a_t_1 does not get used)
            s_t = s_t_1
        total_rewards.append(episode_reward)
    return Q, total_rewards



def Double_Q_Learning(bot:Bot, alpha=.5, epsilon=.1, gamma=1.0, num_episodes=1000):
    print("Performing Double-Q-Learning...")
    # init 2 action-value functions (tabular, finite)
    Q1 = {s: {a: 0 for a in bot.env.actions} for s in bot.env.state_generator()}
    Q2 = {s: {a: 0 for a in bot.env.actions} for s in bot.env.state_generator()}
    total_rewards = []
    for k in range(num_episodes):
        # reset the env
        bot.env.reset()
        episode_reward = 0
        for s in Q1.keys():
            # update the policy of the bot
            # get the action with the highest mean action-value
            avg_action_values = {a:(Q1[s][a] + Q2[s][a])/2 for a in Q1[s].keys()}
            bot.policy_set_action(s, max(avg_action_values, key=avg_action_values.get))

        # initialize s
        s_t = bot.env.starting_state
        for t in range(bot.T):
            # if s is terminal, terminate the episode
            if bot.env.state_is_terminal(s_t):
                break
            # choose action a from s_t using e-greedy policy
            a_t = bot.pick_action(s_t, epsilon)
            # take action a_t and receive s_t_1 and reward r
            s_t_1 = bot.env.apply_action(s_t, a_t)
            r = bot.env.get_reward(s_t, a_t, s_t_1)
            episode_reward += r
            # generate random uniform number, with a 50/50 chance swap them
            u = random()
            if u < .5:
                Q1, Q2 = Q2, Q1
            # choose action a_t_1 from s_t_1 that maximizes Q[s_t_1]
            a_t_1 = max(Q1[s_t], key=Q1[s_t].get)
            # perform update of Q using SARSA update formula
            # if the next state is terminal, its future reward is zero
            if bot.env.state_is_terminal(s_t_1):
                Q1[s_t][a_t] = Q1[s_t][a_t] + alpha * (r - Q1[s_t][a_t])
            else:
                Q1[s_t][a_t] = Q1[s_t][a_t] + alpha * (r + gamma * Q2[s_t_1][a_t_1] - Q1[s_t][a_t])


            # set s_t to the new state (off-policy control so a_t_1 does not get used)
            s_t = s_t_1
        total_rewards.append(episode_reward)
    # compute an average Q
    Q = {s:{a:(Q1[s][a] + Q2[s][a])/2 for a in Q1[s].keys()} for s in Q1.keys()}
    return Q, total_rewards


def Speedy_Q_Learning(bot:Bot, alpha=None, gamma=1.0, epsilon=.1,  num_episodes=1000):
    print("Performing Speedy-Q-Learning...")
    # init action-value function (tabular, finite)
    Q = {s: {a: 0 for a in bot.env.actions} for s in bot.env.state_generator()}
    total_rewards = []
    for k in range(num_episodes):
        # reset the env
        bot.env.reset()
        episode_reward = 0
        for s in Q.keys():
            # update the policy of the bot
            bot.policy_set_action(s, max(Q[s], key=Q[s].get))

        # initialize s
        s_t = bot.env.starting_state
        for t in range(bot.T):
            # if s is terminal, terminate the episode
            if bot.env.state_is_terminal(s_t):
                break
            # set the learning_rate (dep on t)
            alpha_t = 1 / (t + 1)
            # choose action a from s_t using e-greedy policy
            a_t = bot.pick_action(s_t, epsilon)
            # take action a_t and receive s_t_1 and reward r
            s_t_1 = bot.env.apply_action(s_t, a_t)
            r = bot.env.get_reward(s_t, a_t, s_t_1)
            episode_reward += r
            # choose action a_t_1 from s_t_1 that maximizes Q[s_t_1]
            a_t_1 = max(Q[s_t], key=Q[s_t].get)
            # perform update of Q using SARSA update formula
            # before updating, save Q,r as Q_prev,r_prev
            Q_prev = Q.copy()
            r_prev = r
            # if the next state is terminal, its future reward is zero
            if bot.env.state_is_terminal(s_t_1):
                BQt = r
                BQt_1 = r_prev
                Q[s_t][a_t] = Q[s_t][a_t] + alpha_t * (BQt_1 - Q[s_t][a_t]) + (1 - alpha_t) * (BQt - BQt_1)

            else:
                BQt = r + gamma * Q[s_t_1][a_t_1]
                BQt_1 = r_prev + gamma * Q_prev[s_t_1][max(Q_prev[s_t_1], key=Q_prev[s_t_1].get)]
                Q[s_t][a_t] = Q[s_t][a_t] + alpha_t * (BQt_1 - Q[s_t][a_t]) + (1-alpha_t) * (BQt - BQt_1)

            # set s_t to the new state (off-policy control so a_t_1 does not get used)
            s_t = s_t_1
        total_rewards.append(episode_reward)
    return Q, total_rewards


def test_grid_world(algo=SARSA, print_q=False, alpha = .2, epsilon = .3, gamma = .9, num_episodes=1000):
    print(f"Testing {algo.__name__} on Grid-World...")
    # init env
    h, w = 4, 4  # grid size
    env = GridWorld(h, w, terminal_states=[(0, 0), (w - 1, h - 1)], starting_state=(2, 1))
    print(env)
    # init bot
    bot = Bot(env)
    Q, total_rewards = algo(bot, alpha=alpha, epsilon=epsilon, gamma=gamma, num_episodes=num_episodes)
    if print_q: pprint(Q)
    bot.draw_policy()
    # pprint(bot.policy)
    plot_blockwise_mean_rewards_line_graph(total_rewards, title=f"Total Rewards {algo.__name__}", xlabel="episodes", ylabel="reward")
    bot.make_test_runs(1000)
    print("-"*50)


def test_cliff_walking(algo=SARSA, print_q=False, alpha = .2, epsilon = .3, gamma = .9, num_episodes=10000):
    print(f"Testing {algo.__name__} on Cliff-Walking...")
    # tests Monte Carlo control on cliff walking scenario
    h, w = 4, 5
    cliffs = [(1, 1), (1, 2), (1, 3)]
    env = CliffWalking(h, w, terminal_states=[(1, 4)], starting_state=(1, 0), cliffs=cliffs)
    print(env)
    bot = Bot(env)
    alpha = .2
    epsilon = .3
    gamma = .9
    Q, total_rewards = algo(bot, alpha=alpha, epsilon=epsilon, gamma=gamma, num_episodes=num_episodes)
    if print_q:
        print("Q")
        pprint(Q)
    if print_q:
        pprint(bot.policy)
        # print("policy")
    bot.draw_policy()
    plot_blockwise_mean_rewards_line_graph(total_rewards, title=f"Total Rewards {algo.__name__}", xlabel="episodes", ylabel="reward")
    bot.make_test_runs(1000)
    print("-"*50)


if __name__ == '__main__':
    test_grid_world(SARSA)
    # test_grid_world(expected_SARSA)
    # test_grid_world(Q_Learning)
    # test_cliff_walking(SARSA)
    # test_cliff_walking(expected_SARSA)
    # test_cliff_walking(Q_Learning)
    # test_cliff_walking(Double_Q_Learning)
    # test_cliff_walking(Speedy_Q_Learning)
