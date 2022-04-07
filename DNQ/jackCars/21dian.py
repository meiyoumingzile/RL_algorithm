import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Blackjack-v0")
observation = env.reset()
print("观测={}".format(observation))
while True:
    print("玩家={}，庄家={}，".format(env.player, env.dealer))
    action = np.random.choice(env.action_space.n)
    print("动作={}".format(action))
    observation, reward, done, _ = env.step(action)
    print("观测={}，奖励={}，结束指示={}".format(observation, reward, done))
    if done:
        break