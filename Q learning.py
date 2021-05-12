import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

print("Starting Point:", env.reset())

print("Action number:", env.action_space.n)
print("High Observation:", env.observation_space.high)
print("Low Observation:", env.observation_space.low)

discrete_size = [20, 10]
discrete_win_size = (env.observation_space.high - env.observation_space.low) / discrete_size

#print(discrete_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(discrete_size + [env.action_space.n]))
print(q_table)

def discretify(state):
    d_s = (state - env.observation_space.low) / discrete_win_size
    return tuple(d_s.astype(np.int))

#print(discrete_state)

#print(q_table[discrete_state])
#print(q_table[discrete_state].argmax())

learning_rate = 0.1
discount = 0.95
episodes = 2000
every = 500

ep_rewards = []
ep_rewards_dict = {"ep": [], "avg": [], "min": [], "max": []}
for i in range(episodes+1):
    episode_reward = 0
    print("Episode:", i)
    if not i%every:
        render = True
    else:
        render = False
    discrete_state = discretify(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        #print(new_state)
        new_discrete_state = discretify(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
    
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action,)] = new_q
    
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
    
        discrete_state = new_discrete_state

    ep_rewards.append(episode_reward)
    if not i % every:
        average_reward = sum(ep_rewards[-every:]) / every
        ep_rewards_dict["ep"].append(i)
        ep_rewards_dict["avg"].append(average_reward)
        ep_rewards_dict["min"].append(min(ep_rewards[-every:]))
        ep_rewards_dict["max"].append(max(ep_rewards[-every:]))
        print(f'Episode: {i:>5d}, average reward: {average_reward:>4.1f}')

env.close()

plt.plot(ep_rewards_dict['ep'], ep_rewards_dict['avg'], label="average rewards")
plt.plot(ep_rewards_dict['ep'], ep_rewards_dict['max'], label="max rewards")
plt.plot(ep_rewards_dict['ep'], ep_rewards_dict['min'], label="min rewards")
plt.legend(loc=4)
plt.show()

