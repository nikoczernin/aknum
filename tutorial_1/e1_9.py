
R=[0, 1, -1, 2, -1, 2, 0, 0, 0]
T=5
gamma = 0.5

def get_G(r, t):
    return sum([r * gamma ** (k - t - 1) for k in range(t+1, T+1)])


print([get_G(R[t+1], t) for t in range(T+1)])







R=[0, -1, 2] + [1]*1000
T=1000
gamma = 0.9

def get_G(r, t):
    return sum([r * gamma ** (k - t - 1) for k in range(t+1, T+1)])


print([get_G(R[t+1], t) for t in range(4)])