import gym
env = gym.make('Blackjack-v0')
from collections import defaultdict
import numpy as np
import sys

MAX=1000
inf=0.001
STICK = 1#代表要牌
HIT = 0#代表停牌

γ=1.0
α=0.1#学习率

def ϵ_greedy(Q,state,ϵ=0.01):  # 按照当前状态，用ϵ_greedy选取策略
    actList = Q[state]
    n = len(actList)
    if np.random.uniform(0,1)>ϵ:#较大概率为贪心
        return np.argmax(actList)
    else:
        return np.random.randint(0,n)#随机选取动作
def qlearning():
    Q = defaultdict(lambda: np.zeros(env.action_space.n))#Q[i][j]代表状态i下选取动作j的价值，类似背包问题的dp数组
    for episode_i in range(100000):#循环若干次
        print(episode_i)
        state = env.reset()  # 重新开始游戏，21点游戏的初始牌是随机的
        is_terminal=False
        action=1#初始永远要牌
        while(not is_terminal):
            next_state, reward, is_terminal, info = env.step(action)
            Q[state][action]+=α*(reward+γ*Q[next_state].max()-Q[state][action])
            action = ϵ_greedy(Q, state)
            state=next_state
    return Q
Q=qlearning()
print(Q)


