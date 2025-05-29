from pprint import pprint

from Bot import Bot
from CliffWalking import CliffWalking
from GridWorld import GridWorld


def SARSA(bot:Bot, alpha=.5, epsilon=.1, gamma=1, num_episodes=1000, expected=False):
    if expected: ("Performing Expected-SARSA...")
    else: ("Performing SARSA...")
    # init action-value function (tabular, finite)
    # should we consider all states beforehand?
    # Q = {}
    Q = {s:{a:0 for a in bot.env.actions} for s in bot.env.state_generator()}

    for k in range(num_episodes):
        # reset the env
        bot.env.reset()
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

            # update the policy of the bot
            # TODO: was setten wir als policy?
            bot.policy_set_action(s_t, max(Q[s_t], key=Q[s_t].get))

            # set s_t and a_t to the new state and action (we are doing on-policy control)
            s_t, a_t = s_t_1, a_t_1
    return Q

def expected_SARSA(bot:Bot, alpha=.5, epsilon=.1, gamma=1, num_episodes=1000):
    return SARSA(bot, alpha, epsilon, gamma, num_episodes, expected=True)


def Q_Learning(bot:Bot, alpha=.5, epsilon=.1, gamma=1, num_episodes=1000):
    print("Performing Q-Learning...")
    # init action-value function (tabular, finite)
    # should we consider all states beforehand?
    # Q = {}
    Q = {s: {a: 0 for a in bot.env.actions} for s in bot.env.state_generator()}

    for k in range(num_episodes):
        # reset the env
        bot.env.reset()
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
            # choose action a_t_1 from s_t_1 that maximizes Q[s_t_1]
            a_t_1 = max(Q[s_t], key=Q[s_t].get)
            # perform update of Q using SARSA update formula
            # if the next state is terminal, its future reward is zero
            if bot.env.state_is_terminal(s_t_1):
                Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r - Q[s_t][a_t])
            else:
                Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r + gamma * Q[s_t_1][a_t_1] - Q[s_t][a_t])

            # update the policy of the bot
            # TODO: was setten wir als policy?
            bot.policy_set_action(s_t, max(Q[s_t], key=Q[s_t].get))

            # set s_t to the new state (off-policy control so a_t_1 does not get used)
            s_t = s_t_1
    return Q


def test_grid_world(algo=SARSA):
    # init env
    h, w = 4, 4  # grid size
    env = GridWorld(h, w, terminal_states=[(0, 0), (w - 1, h - 1)], starting_state=(2, 1))
    print(env)
    # init bot
    bot = Bot(env)
    Q = algo(bot, alpha=.1, epsilon=.1, gamma=1, num_episodes=10000)
    pprint(Q)
    bot.draw_policy()
    # pprint(bot.policy)
    bot.make_test_runs(1000)
    print("-"*50)


def test_cliff_walking(algo=SARSA):
    # tests Monte Carlo control on cliff walking scenario
    h, w = 4, 5
    cliffs = [(1, 1), (1, 2), (1, 3)]
    env = CliffWalking(h, w, terminal_states=[(1, 4)], starting_state=(1, 0), cliffs=cliffs)
    print(env)
    bot = Bot(env)
    alpha = .2
    epsilon = .3
    gamma = .9
    Q = algo(bot, alpha=alpha, epsilon=epsilon, gamma=gamma, num_episodes=10000)
    pprint("Q")
    pprint(Q)
    print("policy")
    pprint(bot.policy)
    bot.draw_policy()
    bot.make_test_runs(1000)
    print("-"*50)


if __name__ == '__main__':
    # test_grid_world(SARSA)
    # test_grid_world(expected_SARSA)
    # test_grid_world(Q_Learning)
    test_cliff_walking(Q_Learning)
