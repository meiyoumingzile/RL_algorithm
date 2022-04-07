import gym
env = gym.make('Blackjack-v0')
from collections import defaultdict
import numpy as np
import sys

MAX=1000
inf=0.01
STICK = 0#代表要牌
HIT = 1#代表停牌

γ=1.0
powγ=[γ]*1010#γ数组避免重复计算

def initData():
    powγ[0]=1.0
    for i in range(1,MAX):
        powγ[i]=powγ[i-1]*γ

def getCardtoEnd1(env,n):#带概率的要牌
    li = []
    state = env.reset()#重新开始游戏，21点游戏的初始牌是随机的
    is_terminal=False
    while(not is_terminal):
        probs = [0.8, 0.2] if state[0] >=n else [0.2, 0.8]#我方点数大于等于n，有很大概率停牌
        action = np.random.choice(np.arange(2), p=probs)#np.arange(2)代表生成[0,1],随机取出一个行动
        next_state, reward, is_terminal, info = env.step(action)
        li.append((state, action, reward))
        state = next_state
    return li

def mc_contril(env, episodesCnt):#动作值函数Q(S,A)的预测,用迭代法
    N = defaultdict(lambda:np.zeros(env.action_space.n))#初始化字典,若没有key则赋值为0矩阵
    Q = defaultdict(lambda:np.zeros(env.action_space.n))
    while(episodesCnt):#训练
        episodesCnt-=1
        episode = getCardtoEnd1(env,20)#一个自己推演的动作序列,来模拟接下来要牌
        states, actions, rewards = zip(*episode)#zip(*)是个语法糖,代表unzip函数
        for i, state in enumerate(states):
            N[state][actions[i]] += 1.0
            mul=np.array(rewards[i:len(rewards)])*np.array(powγ[0:len(rewards) - i])
            Q[state][actions[i]] += (sum(mul)-Q[state][actions[i]]) / N[state][actions[i]]
    return Q

initData()
Q=mc_contril(env,100000)
print(Q)

