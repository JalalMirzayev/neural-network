import gymnasium
import numpy
import matplotlib.pyplot as plt
from IPython.display import display


def render_board(env):
    plt.img_show(env.render())
    plt.axis('off')
    plt.show()


def main():
    environment = gymnasium.make(
        'FrozenLake-v1',
        map_name='4x4',
        render_mode='rgb_array',
        is_slippery=False)

    action_dict: dict[int, str] = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑',
    }

    observation, _ = environment.reset()
    action = environment.action_space.sample()
    observation, reward, done, truncated, _ = environment.step(action)


def policy_evaluation(environment, policy, gamma=0.99, iterations=1000, tol=1e-10):
    V = numpy.zeros(environment.observation_space.n)
    for _ in range(iterations):
        V_k = numpy.copy(V)
        for state in range(environment.observations_space.n):
            action = policy[state]
            probability, state_next, reward, terminal = environment.unwrapped.P[state][action][0]
            V[state] = probability * (reward + gamma * V_k[state_next])
        if numpy.max(numpy.abs(V - V_k)) < tol:
            break
    return V


def policy_improvement(env, values, gamma=0.99):
    new_policy = numpy.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        q = []
        for action in range(env.action_space.n):
            probability, state_next, reward, terminal = env.unwrapped.P[state][action][0]
            q.append(probability * (reward + gamma * values[state_next]))
        best_action = numpy.argmax(q) 
        new_policy[state] = best_action
    return new_policy


def policy_iteration(env, num_iterations=1000, gamma=0.99):
    policy = numpy.random.randint(
        low=0
        high=env.action_space.n,
        size=(env.observation_space.n)
    )

    for _ in range(num_iterations):
        values = policy_evaluation(policy, gamma=gamma, iterations=num_iterations)
        updated_policy = policy_improvement(values, gamma=gamma)

        if numpy.all(policy == updated_policy):
            break
        
        policy = updated_policy
    return policy
            
if __name__ == '__main__':
    env = gymnasium.make(
        'FrozenLake-v1',
        map_name='4x4',
        render_mode='rgb_array',
        is_slippery=False)
    learned_policy = policy_iteration(env, gamma=0.99)
    print(learned_policy)