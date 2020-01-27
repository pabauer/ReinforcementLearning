# %%
import numpy as np
import gym
import time


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
        # Aktion mit maximalen Aktionswert
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
            state_value[state] = np.max(value_eval(env, state, state_value, discount))
            delta = np.maximum(delta, np.abs(state_value[state] - state_value_prev[state]))
        if delta < theta:
            print('policy converged at iteration %d' % (i + 1))
            break

        state_value_prev = np.copy(state_value)

    policy = policy_improve(env, state_value, discount)
    return state_value, policy


# %%
# Frozen Lake stochastisch
# env_stoch = gym.make('FrozenLake8x8-v0')
# env_stoch = gym.make('FrozenLake-v0')

# Frozen Lake deterministisch
# from gym.envs.registration import register
#
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name': '4x4', 'is_slippery': False},
# )

# Löschen einer Umgebung
# del gym.envs.registry.env_specs['FrozenLakeNotSlippery-v0']

# deterministische Umgebung
env_det = gym.make('FrozenLakeNotSlippery-v0')

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


opt_state_value, opt_policy = value_iteration(env, discount, theta, episodes)
print('Optimal Value function: ')
print(opt_state_value.reshape((4, 4)))
print('Final Policy: ')
print(opt_policy)
print(' '.join([action_mapping[int(action)] for action in opt_policy]))

for i in range(5):
    j = i * 4
    print(' '.join([action_mapping[int(action)] for action in opt_policy[j - 4:j]]))
