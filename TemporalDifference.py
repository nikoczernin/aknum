from pprint import pprint

from Bot import Bot
from CliffWalking import CliffWalking
from GridWorld import GridWorld


def SARSA(bot:Bot, alpha=.5, epsilon=.1, gamma=1, num_episodes=1000):

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
            Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r + gamma * (Q[s_t_1][a_t_1] if not bot.env.state_is_terminal(s_t_1) else 0) - Q[s_t][a_t])
            # set s_t and a_t to the new state and action (we are doing on-policy control)
            s_t, a_t = s_t_1, a_t_1
            # update the policy of the bot
            bot.policy_set_action(s_t, a_t)
    return Q

def test_grid_world():
    # init env
    h, w = 4, 4  # grid size
    env = GridWorld(h, w, terminal_states=[(0, 0), (w - 1, h - 1)], starting_state=(2, 1))
    # init bot
    bot = Bot(env)
    Q = SARSA(bot, alpha=.5, epsilon=.1, gamma=1, num_episodes=1000)
    pprint(Q)
    bot.draw_policy()

if __name__ == '__main__':
    test_grid_world()