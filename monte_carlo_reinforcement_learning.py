import gymnasium

# environment = gym.make("CartPole-v1", render_mode="human")
# import gymnasium as gym
environment = gymnasium.make(
    'FrozenLake-v1',
    render_mode="human",
    desc=None,
    map_name="4x4",
    is_slippery=False,
    success_rate=1/3,
    reward_schedule=(1, 0, 0)
)


def suffix_sum_tuples(data):
    out = [None] * len(data)
    running = 0
    for i in range(len(data) - 1, -1, -1):
        s, a, r = data[i]
        running += r
        out[i] = (s, a, running)
    return out


def monte_carlo_update(q, episode_data, alpha=0.9):
    for s, a, g in episode_data:
        q[s][a] = q[s][a] + alpha * (g - q[s][a])
    return q


# initialize Q-matrix
q_matrix = [[0 for _ in range(environment.action_space.n)] for _ in range(environment.observation_space.n)]
q_matrix = [[9.899999999999798e-71, 0.0008999999999999998, 9.899999999999619e-140, 9.89999999999976e-93], [9.899999999999827e-58, 0.0, 9.899999999999824e-64, 9.899999999999806e-70], [9.89999999999991e-32, 8.99999999999984e-62, 8.99999999999995e-21, 9.899999999999672e-121], [8.999999999999975e-09, 0.0, 8.99999999999995e-16, 8.999999999999419e-261], [9.998999999999876e-43, 0.008999999999999994, 0.0, 9.899999999999873e-46], [0, 0, 0, 0], [0.0, 0.9000000000009, 0.0, 9.000000089999886e-39], [0, 0, 0, 0], [8.999999999999872e-52, 0.0, 0.9000000090900009, 9.989999999999911e-25], [9.899999999999716e-104, 0.009098999999999996, 0.9009090000009, 0.0], [8.999999999999959e-11, 0.999000099000999, 0.0, 0.9], [0, 0, 0, 0], [0, 0, 0, 0], [0.0, 0.09000909000000001, 0.000999909999899998, 8.999999999999938e-25], [0.09009099899999995, 0.90999999000001, 1.0, 9.000000000009072e-08], [0, 0, 0, 0]]

for epoch in range(10):
    observation, info = environment.reset()
    episode_over = False
    total_reward = 0
    trajectory = []
    while not episode_over:
        current_state = observation
        action = q_matrix[current_state].index(max(q_matrix[current_state]))
        # action = environment.action_space.sample()
        observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        trajectory.append((current_state, action, reward))
        episode_over = terminated or truncated

    data = suffix_sum_tuples(trajectory)
    q_matrix = monte_carlo_update(q_matrix, data)

print(q_matrix)

environment.close()
