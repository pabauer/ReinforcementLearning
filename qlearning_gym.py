import gym
import numpy as np
import random
import time

# %%
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


def q_Learning(env, discount=0.999, epsilon=0.1, alpha=0.85, episodes=1000):
    action_table = np.zeros((env.nS, env.nA))
    scores = []
    policy = [0 for i in range(env.nS)]

    for episode in range(episodes):
        state = env.reset()
        done = False  # done = true -> Spiel zu Ende
        # score = 0

        for i in range(max_iter):
            # Wahrscheinlichkeit (1 - epsilon): wähle die Greedy-Aktion aus, sonst eine zufällige aus den Aktionen
            if random.uniform(0, 1) > epsilon:
                action = np.argmax(action_table[state, :])
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)

            # score += reward

            # Update Aktionswertfunktion q
            action_table[state, action] = (1 - alpha) * action_table[state, action] + alpha * (
                        reward + discount * np.max(action_table[next_state, :]))
            state = next_state

            if done:
                break

        # Jedes Spiel (Episode) Epsilon verändern - Exploreation vs Exploitation
        if epsilon >= epsilon_min:
            epsilon *= epsilon_faktor

        # scores.append(score)
    for state in range(env.nS):
        policy[state] = np.argmax(action_table[state, :])

    return policy


# %%
# Frozen Lake stochastisch
# env = gym.make('FrozenLake8x8-v0')
# env = gym.make('FrozenLake-v0')

# Frozen Lake not slippery (-> deterministisch)
# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
#     max_episode_steps=100,
#     reward_threshold=0.78, # optimum = .8196
# )

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

# Frozen Lake stochastisch
#env = gym.make('FrozenLake8x8-v0')
env_stoch = gym.make('FrozenLake-v0')

# # Frozen Lake not slippery (-> deterministisch)
# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
#     max_episode_steps=100,
#     reward_threshold=0.78, # optimum = .8196
# )

# To delete any new environment
# del gym.envs.registry.env_specs['FrozenLakeNotSlippery-v0']

# Make the environment based on deterministic policy
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
    3: '\u2191', # UP
    2: '\u2192', # RIGHT
    1: '\u2193', # DOWN
    0: '\u2190' # LEFT
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

tic = time.time()
opt_policy = q_Learning(env_stoch, discount, epsilon, alpha, episodes)
toc = time.time()
elapsed_time = (toc - tic) * 1000
print (f"Time to converge: {elapsed_time: 0.3} ms")
print('Final Policy: ')
print(opt_policy)
print(' '.join([action_mapping[int(action)] for action in opt_policy]))

for i in range(5):
    j = i*4
    print(' '.join([action_mapping[int(action)] for action in opt_policy[j-4:j]]))

wins, total_reward, avg_reward = play_episodes(env_stoch, 100, opt_policy, random=False)

