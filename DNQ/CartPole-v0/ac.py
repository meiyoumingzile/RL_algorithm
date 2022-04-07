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
MEMORY_CAPACITY=10
OVERLAY_CNT=1#针对此游戏的叠帧操作
env=gym.make('CartPole-v0')#吃豆人#MsPacman-ram-v0
MAX=210
inf=0.0001
INF=1000000
ENVMAX=[env.observation_space.shape[0],2]
# print(env.spaces())
γ=0.90#reward_decay
α=0.0002#学习率learning_rate,也叫lr

if platform.system() == "Linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if platform.system() =="Windows":
    BATCH_SZ = 1

def cv_show( mat):
    cv2.imshow("name", mat)
    cv2.waitKey(0)
class ANET(nn.Module):#演员网络
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
class CNET(nn.Module):#评论家
    def __init__(self):
        super().__init__()
        # self.lstm=nn.LSTM(input_size=OVERLAY_CNT,hidden_size=32,num_layers=3,bias=True,batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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
        self.actor_net = ANET().cuda()#初始化两个网络，target and training
        self.critic_net = CNET().cuda()
        self.net=(self.actor_net,self.critic_net)
        self.actor_net.train()
        self.critic_net.train()

        self.beginShape=beginShape

        self.learn_step_i = 0  # count the steps of learning process
        self.optimizer=(torch.optim.Adam(self.actor_net.parameters(), lr=α),torch.optim.Adam(self.critic_net.parameters(), lr=α))


        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1))]

    def choose_action(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        x=np.expand_dims(x, axis=0)#加一个维度
        x = self.actor_net.forward(torch.FloatTensor(x))
        prob = F.softmax(x, dim=1).cpu()
        return np.random.choice(a=2, p=prob[0].detach().numpy())
    def pushRemember(self,state,next_state,act,reward):#原本是DQN的方法的结构，但这里用它当作缓存数据
        t=self.memory_i%MEMORY_CAPACITY
        self.memoryQue[0][t] = np.expand_dims(state, axis=0)
        self.memoryQue[1][t] = np.expand_dims(next_state, axis=0)
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memory_i=(self.memory_i+1)%MEMORY_CAPACITY
    def learn(self):#学习,训练ac算法2个网络
        #          得TD_error = r + γV(s’)-V(s)
        #           顺便用TD_error的均方误差训练Q_Network
        #       ④ TD_error反馈给Actor，Policy
        # Gradient公式
        # 训练Actor
        t=(self.memory_i-1+MEMORY_CAPACITY)%MEMORY_CAPACITY
        state = torch.FloatTensor(self.memoryQue[0][t])  # state
        next_state = torch.FloatTensor(self.memoryQue[1][t])
        act = torch.LongTensor(self.memoryQue[2][t].astype(int)).cuda()
        reward = torch.FloatTensor(self.memoryQue[3][t]).cuda()

        v=(self.critic_net.forward(state).squeeze(dim=1),self.critic_net.forward(next_state).squeeze(dim=1))
        #用cri网络计算
        loss=F.mse_loss(reward+γ*v[1],v[0])#用tderror的方差来算，代表减少两次状态评分的差值，而actor网络负责保证这个价值会提高，而非下降
        self.optimizer[1].zero_grad()
        loss.backward()
        self.optimizer[1].step()

        #更新actor网络
        output = self.actor_net.forward(state)
        acloss = F.cross_entropy(output, act)#loss相当于算交叉熵-output*log(p),p是act算出来的
        tderror = (reward + γ * v[1] - v[0]).detach()  # td error
        acloss = tderror *acloss#这里要反向传播acloss，tderror去掉梯度
        # print(acloss)
        self.optimizer[0].zero_grad()
        acloss.backward()
        self.optimizer[0].step()

    def save(self):
        torch.save(self.actor_net, "mod/actor_net.pt")  #
        torch.save(self.critic_net, "mod/critic_net.pt")  #
    def read(self):
        if os.path.exists("mod/actor_net.pt") and os.path.exists("mod/critic_net.pt") :
            self.actor_net=torch.load("mod/actor_net.pt")  #
            self.critic_net = torch.load("mod/critic_net.pt")  #
            self.optimizer=(torch.optim.Adam(self.actor_net.parameters(), lr=α),torch.optim.Adam(self.critic_net.parameters(), lr=α))

def setReward(reward):#设置奖励函数函数
    x, x_dot, theta, theta_dot = reward
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    return r1 + r2

def train(beginShape):
    agent=Agent(beginShape)
    # net.read()
    sumreward = 0
    for episode_i in range(EPOCHS_CNT):#循环若干次
        is_terminal=False
        state = env.reset()
        while (not is_terminal):
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, is_terminal, info = env.step(action)
            sumreward += reward
            reward = setReward(next_state)
            agent.pushRemember(state,next_state,action,reward)
            agent.learn()
            state = next_state
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

