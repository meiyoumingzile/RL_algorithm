import math
import random
import time
import platform
import torchvision
import cv2
import gym
import os
from collections import defaultdict
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.optim as optim
import torch.nn.functional as F#激活函数
from torchvision import transforms#图像处理的工具类
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BATCH_SZ=10
EPOCHS_CNT=50000
OVERLAY_CNT=1#针对此游戏的叠帧操作
env=gym.make('CartPole-v0')#吃豆人#MsPacman-ram-v0
MAX=210
inf=0.0001
INF=1000000
ENVMAX=[env.observation_space.shape[0],2]
# print(env.spaces())
γ=0.90#reward_decay
α=0.0002#学习率learning_rate,也叫lr
EXP_ALPHA=1
powγ=np.array([γ]*MAX)#γ数组避免重复计算

if platform.system() == "Linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if platform.system() =="Windows":
    BATCH_SZ = 1

def cv_show( mat):
    cv2.imshow("name", mat)
    cv2.waitKey(0)
class QNET(nn.Module):#拟合Q的神经网络
    def __init__(self):
        super().__init__()
        # self.lstm=nn.LSTM(input_size=OVERLAY_CNT,hidden_size=32,num_layers=3,bias=True,batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, ENVMAX[1]),
        )
        for con in self.fc:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)

    def forward(self,x):#重写前向传播算法，x是张量
        x=x.cuda()
        # print(x.shape)
        x = self.fc(x)
        return x
class Agent(object):#智能体，蒙特卡洛策略梯度算法

    def __init__(self,beginShape):
        self.eval_net= QNET().cuda()#初始化两个网络，target and training
        self.eval_net.train()
        #eval_net是新网络，self.target_net是旧网络，target_net相当于学习的成果
        self.beginShape=beginShape
        self.lossFun=nn.CrossEntropyLoss(reduction='none')#交叉熵
        self.learn_step_i = 0  # count the steps of learning process
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(), lr=α)

    def choose_action(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        x=np.expand_dims(x, axis=0)#加一个维度
        x = self.eval_net.forward(torch.FloatTensor(x))
        prob = F.softmax(x, dim=1).cpu()

        return np.random.choice(a=2, p=prob[0].detach().numpy())

    def learn(self,stateList,actList,rewardList):#学习以往经验,frame_i代表当前是游戏哪一帧
        output = self.eval_net(torch.FloatTensor(stateList))
        # print(actList)
        output=F.log_softmax(output, dim=1)
        loss = self.lossFun(output, torch.LongTensor(actList).cuda())
        g = [0]*len(rewardList) # powγ从0到len(rewards)-i，rewards从i到结尾，做点乘，g[i]代表蒙特卡洛Gt
        for i in range(len(rewardList)):
            g[i]=powγ[0:len(rewardList) - i] * rewardList[i:]
            g[i] = g[i].sum()
        g=torch.FloatTensor(g).cuda()
        g-= g.mean()
        g/= g.std()

        loss = (loss * g).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def toGameEnd(self):#随机游戏到结束，同蒙特卡洛,返回(state,action,reward)
        is_terminal = False
        ans=([],[],[])
        state = env.reset()
        sumreward=0
        while(not is_terminal):
            # env.render()
            action = self.choose_action(state)
            next_state, reward, is_terminal, info = env.step(action)
            sumreward+=reward
            reward=setReward(next_state)
            ans[0].append(state)
            ans[1].append(action)
            ans[2].append(reward)#上一个state下的reward
            state=next_state
        return np.array(ans[0]),np.array(ans[1]),np.array(ans[2]),sumreward

    def save(self):
        torch.save(self.eval_net, "mod/eval_net.pt")  #
    def read(self):
        if os.path.exists("mod/eval_net.pt"):
            self.eval_net=torch.load("mod/eval_net.pt")  #
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)

def setReward(reward):#设置奖励函数函数
    x, x_dot, theta, theta_dot = reward
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    return r1 + r2

def train(beginShape):
    agent=Agent(beginShape)
    # net.read()
    sumreward = 0
    powγ[0] = 1.0
    for i in range(1, MAX):#预处理
        powγ[i] = powγ[i - 1] * γ
    for episode_i in range(EPOCHS_CNT):#循环若干次
        stateList,actList,rewardList,__sumreward = agent.toGameEnd()
        agent.eval_net.train()
        agent.learn(stateList,actList,rewardList)
        sumreward+=__sumreward
        if episode_i%50==0 :
            print(str(episode_i)+"reward:"+str(sumreward/50))
            sumreward = 0
            # if episode_i%2000==0:
            #     print("episode_i:"+str(episode_i))
            #     net.save()
    return agent
def test_mod(beginShape):
    agent = Agent(beginShape)
    agent.read()
    for episode_i in range(EPOCHS_CNT):  # 循环若干次
        state=env.reset()
        is_terminal = False
        while (not is_terminal):
            env.render()
            action = agent.choose_action(state)
            state, reward, is_terminal, info = env.step(action)
            time.sleep(0.01)
shape = env.reset().shape# 重新开始游戏,(210, 160, 3)

agent=train(shape)
agent.save()
test_mod(shape)

