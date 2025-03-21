from statistics import mean
from pprint import pprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def play(i, Q):
    return np.random.normal(Q[i], 1, 1)[0]


# apply a single episode of Algorithm 1
def episode(e):
    k = 10 # number of arms
    Q = np.random.normal(0, 1, k)  # True value of each arm
    optimal_a = np.argmax(Q)

    # Agent data
    n = np.zeros(k)  # number of executions per action
    q = np.zeros(k)  # estimated gain per action
    R = []  # all received rewards

    T = 1000  # number of time steps
    for i in range(T):
        # do we explore or exploit?
        if e > np.random.uniform():
            # explore
            # pick a random action from 1 to k
            a = np.random.choice(range(k))
        else:
            # exploit
            # perform the action with the highest estimated return
            a = np.argmax(q)
        # perform the action and get a reward
        r = play(a, Q)
        # save the reward for statistics later
        R.append(r)
        # update the reward estimation for the chosen action
        q[a] = q[a] + (r - q[a]) / (n[a] if n[a] else 1)
        # increase its play-counter
        n[a] += 1

    # compute the average reward
    mean_reward = mean(R)
    # compute the percentage of optimal actions taken
    perc_optimal_a_picks = n[optimal_a] / T * 100
    return mean_reward, perc_optimal_a_picks


mean_rewards = []
percentages_optimal_a_picks = []
# get "all" possible epsilon values between 0 and 1
step=0.01
epsilons = np.arange(0, 1+step, step)
for e in epsilons:
    mr, pop = episode(e)
    mean_rewards.append(mr)
    percentages_optimal_a_picks.append(pop)

# visualize the results
# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

ax1.scatter(epsilons, mean_rewards)
ax1.set_xlabel("epsilon")
ax1.set_ylabel("mean reward")

ax2.scatter(epsilons, percentages_optimal_a_picks)
ax2.set_xlabel("epsilon")
ax2.set_ylabel("% of optimal actions performed")

plt.show()