# %%
import numpy as np
import gym
import time


# %%
def play_episodes(enviorment, n_episodes, policy, random=False):
    # intialize wins and total reward
    wins = 0
    total_reward = 0

    # loop over number of episodes to play
    for episode in range(n_episodes):

        # flag to check if the game is finished
        terminated = False

        # reset the enviorment every time when playing a new episode
        state = enviorment.reset()

        while not terminated:

            # check if the random flag is not true then follow the given policy other wise take random action
            if random:
                action = enviorment.action_space.sample()
            else:
                action = policy[state]

            # take the next step
            next_state, reward, terminated, info = enviorment.step(action)

            enviorment.render()

            # accumalate total reward
            total_reward += reward

            # change the state
            state = next_state

        # if game is over with positive reward then add 1.0 in wins
        if terminated and reward == 1.0:
            wins += 1

    # calculate average reward
    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward


# %%
def value_eval(env, state, state_value, discount=0.99):
    action_value = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, info in env.P[state][action]:
            action_value[action] += prob * (reward + (discount * state_value[next_state]))
    return action_value


def policy_improve(env, state_value, discount):
    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_value = value_eval(env, state, state_value, discount)
        # take the action with max action_value
        policy[state] = np.argmax(action_value)
    return policy


# Policy Iteration
def value_iteration(env, discount=0.99, theta=1e-8, episodes=1000):
    state_value = np.zeros(env.nS)
    state_value_prev = np.copy(state_value)
    iter_count_i = 0
    for i in range(episodes):
        iter_count_i += 1
        delta = 0

        for state in range(env.nS):
            # state_value_prev[state] = state_value[state]
            state_value[state] = np.max(value_eval(env, state, state_value, discount))
            delta = np.maximum(delta, np.abs(state_value[state] - state_value_prev[state]))
        if delta < theta:
            break

        state_value_prev = np.copy(state_value)

    policy = policy_improve(env, state_value, discount)
    return state_value, policy


# %%
# Frozen Lake stochastisch
# env = gym.make('FrozenLake8x8-v0')
# env = gym.make('FrozenLake-v0')

# Frozen Lake not slippery (-> deterministisch)
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

# To delete any new environment
# del gym.envs.registry.env_specs['FrozenLakeNotSlippery-v0']

# Make the environment based on deterministic policy
env = gym.make('FrozenLakeNotSlippery-v0')

'''
ForzenLake: https://github.com/openai/gym/wiki/FrozenLake-v0
Actions: 
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
'''
action_mapping = {
    3: '\u2191',  # UP
    2: '\u2192',  # RIGHT
    1: '\u2193',  # DOWN
    0: '\u2190'  # LEFT
}

# %%
# Parameters
discount = 0.999
episodes = 10000
theta = 1e-8

tic = time.time()
opt_state_value, opt_policy = value_iteration(env, discount, theta, episodes)
toc = time.time()
elapsed_time = (toc - tic) * 1000
print(f"Time to converge: {elapsed_time: 0.3} ms")
print('Optimal Value function: ')
print(opt_state_value.reshape((4, 4)))
print('Final Policy: ')
print(opt_policy)
print(' '.join([action_mapping[int(action)] for action in opt_policy]))

for i in range(5):
    j = i * 4
    print(' '.join([action_mapping[int(action)] for action in opt_policy[j - 4:j]]))

# wins, total_reward, avg_reward = play_episodes(env, 1, opt_policy, random=False)
