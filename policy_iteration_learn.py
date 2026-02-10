import gymnasium

"""
action space
    0: left
    1: down
    2: right
    3: up

observation space
    0 to 15 to fill a 4x4 field
    | 0| 1| 2| 3|
    | 4| 5| 6| 7|
    | 8| 9|10|11|
    |12|13|14|15|
"""

def main():
    env = gymnasium.make(
        'FrozenLake-v1',
        desc=None,
        render_mode="human",
        map_name="4x4",
        is_slippery=False,
        success_rate=1.0/3.0,
        reward_schedule=(1, 0, 0) 
    )

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


    print(env.action_space)


if __name__ == "__main__":
    main()