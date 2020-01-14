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


def value_eval(env, state, state_value, discount = 0.99):
    action_value = np.zeros(env.nA)
    # loop over the actions we can take in an enviorment
    for action in range(env.nA):
        # loop over the P_sa distribution.
        for prob, next_state, reward, info in env.P[state][action]:
            action_value[action] += prob*(reward + (discount*state_value[next_state]))
    return action_value

def policy_eval(env, policy, state_value, discount = 0.99):
    policy_value = np.zeros(env.nS)
    for state, action in enumerate(policy):
        # state-value
        for prob, next_state, reward, info in env.P[state][action]:
            policy_value[state] += prob*(reward + (discount*state_value[next_state]))
    return policy_value


def policy_improve(env, policy, state_value, discount):
    for state in range(env.nS):
        # state-action-value
        action_value = value_eval(env, state, state_value, discount)
        # take the action with max action_value
        policy[state] = np.argmax(action_value)
    return policy

# Policy Iteration
def policy_iteration(env, discount = 0.99, episodes = 1000, max_iter_eval = 1000, theta = 1e-8):
   state_value = np.zeros(env.nS)
   state_value_prev = np.copy(state_value)
   policy = [0 for i in range(env.nS)]
   policy_prev = np.copy(policy)
   iter_count_i = 0
   iter_count_j = 0
   for i in range(episodes):
       iter_count_i += 1

       # Policy Evaluation
       for j in range(max_iter_eval):
           iter_count_j += 1
           delta = 0
           state_value = policy_eval(env, policy, state_value, discount)
           delta = np.maximum(delta, np.max(np.abs(state_value_prev - state_value)))
           if delta < theta:
               break
           state_value_prev = np.copy(state_value)

       #  # Policy Evaluation
       # for j in range(max_iter_eval):
       #     iter_count_j += 1
       #     state_value = policy_eval(env, policy, state_value, discount)
       #     if j % 5 == 0:
       #         if (np.all(np.equal(state_value, state_value_prev))):
       #             break
       #     state_value_prev = np.copy(state_value)

       # Policy Improvement
       policy = policy_improve(env, policy, state_value, discount)

       # Convergence of the policy, if the policy does not change over min 5 iterations
       if i % 5 == 0:
           if (np.all(np.equal(policy, policy_prev))):
               print('policy converged at iteration %d' % (i + 1))
               break
           policy_prev = np.copy(policy)
   return state_value, policy

# %%
# Frozen Lake stochastisch
#env = gym.make('FrozenLake8x8-v0')
#env = gym.make('FrozenLake-v0')

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
    3: '\u2191', # UP
    2: '\u2192', # RIGHT
    1: '\u2193', # DOWN
    0: '\u2190' # LEFT
}

# %%
# Parameters
discount = 0.999
episodes = 10000

tic = time.time()
opt_state_value, opt_policy = policy_iteration(env, discount, episodes)
toc = time.time()
elapsed_time = (toc - tic) * 1000
print (f"Time to converge: {elapsed_time: 0.3} ms")
print('Optimal Value function: ')
print(opt_state_value.reshape((4, 4)))
print('Final Policy: ')
print(opt_policy)
print(' '.join([action_mapping[int(action)] for action in opt_policy]))

for i in range(5):
    j = i*4
    print(' '.join([action_mapping[int(action)] for action in opt_policy[j-4:j]]))

wins, total_reward, avg_reward = play_episodes(env, 1, opt_policy, random=False)
