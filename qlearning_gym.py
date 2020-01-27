import gym
import numpy as np
import random

# %%
def q_Learning(env, discount=0.99, epsilon=0.1, alpha=0.85, episodes=1000):
    action_table = np.zeros((env.nS, env.nA))
    policy = [0 for i in range(env.nS)]

    for episode in range(episodes):
        state = env.reset()
        done = False  # done = true -> Spiel zu Ende


        for i in range(max_iter):
            # Wahrscheinlichkeit (1 - epsilon): wähle die Greedy-Aktion aus, sonst eine zufällige aus den Aktionen
            if random.uniform(0, 1) > epsilon:
                action = np.argmax(action_table[state, :])
            else:
                action = env.action_space.sample()

            next_state, reward, done, info = env.step(action)

            # Update Aktionswertfunktion q
            action_table[state, action] = (1 - alpha) * action_table[state, action] + alpha * (
                        reward + discount * np.max(action_table[next_state, :]))
            state = next_state

            if done:
                break

        # Jedes Spiel (Episode) Epsilon verändern - Exploreation vs Exploitation
        if epsilon >= epsilon_min:
            epsilon *= epsilon_faktor

    for state in range(env.nS):
        policy[state] = np.argmax(action_table[state, :])

    return policy


# %%
# Frozen Lake stochastisch
# env_stoch = gym.make('FrozenLake8x8-v0')
env_stoch = gym.make('FrozenLake-v0')

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
alpha = 0.65
discount = 0.999
episodes = 5000
max_iter = 100

epsilon = 1.0
epsilon_min = 0.005
epsilon_faktor = 0.99993

opt_policy = q_Learning(env_stoch, discount, epsilon, alpha, episodes)
print('Final Policy: ')
print(opt_policy)
print(' '.join([action_mapping[int(action)] for action in opt_policy]))

for i in range(5):
    j = i*4
    print(' '.join([action_mapping[int(action)] for action in opt_policy[j-4:j]]))


