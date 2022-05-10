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
import cartPole_util

BATCH_SZ=1
EPOCHS_CNT=50000
MEMORY_CAPACITY=10#在GAE-PPO里它用来存储步数
TARGET_REPLACE_ITER=1000#隔多少论更新参数
OVERLAY_CNT=1#针对此游戏的叠帧操作
CLIP_EPSL=0.1
env=gym.make('CartPole-v0')#吃豆人#MsPacman-ram-v0
ENVMAX=[env.observation_space.shape[0], 2]
# print(env.spaces())
γ=0.90#reward_decay
α=0.001#学习率learning_rate,也叫lr
GAEλ=0.95

if platform.system() == "Linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
            nn.Linear(64, ENVMAX[1]),#算分布,0是均值，1是标准差
        )
        for con in self.fc:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)

    def forward(self,x):#重写前向传播算法，x是张量
        x=x.cuda()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
class CNET(nn.Module):#评论家，代表Q值函数，所以输出也是ENVMAX[1]维度
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
class Agent(object):#智能体，蒙特卡洛策略梯度算法
    GAE_advantage=0
    def initData(self):
        self.net = (self.actor_net, self.critic_net)
        self.optimizer = (torch.optim.Adam(self.actor_net.parameters(), lr=α),torch.optim.Adam(self.critic_net.parameters(), lr=α))
        for net in self.net:
            net.train()

    def __init__(self,beginShape):
        self.actor_net = ANET().cuda()#初始化两个网络，target and training
        self.critic_net = CNET().cuda()

        self.beginShape=beginShape
        self.learn_step_i = 0  # count the steps of learning process
        self.initData()
        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1))]

    def initState(self, env):  #
        state = cartPole_util.resetPic(env.reset())  # 重新开始游戏,(210, 160)
        stList = np.zeros((OVERLAY_CNT + 1,) + state.shape)
        # li[OVERLAY_CNT]=state
        self.GAE_advantage=torch.Tensor([[0]]).cuda()
        self.GAE_xi =torch.Tensor([[1]]).cuda() #λ*γ
        for i in range(0, OVERLAY_CNT):
            action = np.random.randint(0, ENVMAX[1])
            state, reward, is_terminal, info = env.step(action)
            stList[i] = cartPole_util.resetPic(state)
        return stList

    def nextState(self, stList, state):
        for i in range(OVERLAY_CNT, 0, -1):
            stList[i] = stList[i - 1]
        stList[0] = state
        return stList

    def choose_action(self, x,epsl=0):  # 按照当前状态，选取概率最大的动作
        if np.random.uniform(0,1)>epsl:
            x=np.expand_dims(x, axis=0)#加一个维度
            # print(x)
            x = self.actor_net.forward(torch.FloatTensor(x))
            prob = F.softmax(x, dim=1).cpu()
            # print(x,prob[0].detach().numpy())
            act=np.random.choice(a=ENVMAX[1], p=prob[0].detach().numpy())

            return act,prob[0][act]
        return np.random.randint(0,ENVMAX[1]),epsl/ENVMAX[1]#随机选取动作

    def pushRemember(self, state, act, reward,preRate,is_terminal):  # 原本是DQN的方法的结构，但这里用它当作缓存数据
        t = self.memory_i % MEMORY_CAPACITY
        self.memoryQue[0][t] = state[1:OVERLAY_CNT + 1]
        self.memoryQue[1][t] = state[0:OVERLAY_CNT]
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(preRate.detach().numpy())
        self.memoryQue[5][t] = np.array(is_terminal)
        self.memory_i = self.memory_i + 1
    def learn(self):#学习,训练ac算法2个网络
        self.learn_step_i += 1

        t=(self.memory_i-1) % MEMORY_CAPACITY
        state = torch.FloatTensor(self.memoryQue[0][t])  # state
        next_state = torch.FloatTensor(self.memoryQue[1][t])
        act = torch.LongTensor(self.memoryQue[2][t].astype(int)).cuda().unsqueeze(0)
        reward = torch.FloatTensor(self.memoryQue[3][t]).cuda()
        preRate = torch.FloatTensor(self.memoryQue[4][t]).cuda()
        is_ter=self.memoryQue[5][t]

        v = (self.critic_net.forward(state).squeeze(dim=1), self.critic_net.forward(next_state).squeeze(dim=1))
        cri_loss = F.mse_loss(reward + γ * v[1], v[0])  # 用tderror的方差来算


        advantage = reward + γ * v[1]-v[0] #tderro也是优势函数,根据具体问题可以在is_ter==True时阻断
        advantage = torch.Tensor([advantage]).cuda()
        self.GAE_advantage+=advantage*self.GAE_xi
        self.GAE_xi*=GAEλ*γ
        nowRate = self.actor_net.forward(state)
        nowRate= F.softmax(nowRate, dim=1)

        nowRate=   nowRate.gather(1, act)
        ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))#计算p1/p2防止精度问题
        surr1 = ratio * self.GAE_advantage
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSL, 1 + CLIP_EPSL) * self.GAE_advantage#clamp(a,l,r)代表把x限定在[l,r]之间
        actor_loss=-torch.min(surr1, surr2).mean()
        self.optimizer[0].zero_grad()#0代表演员，1代表评论家
        actor_loss.backward()
        self.optimizer[0].step()

        self.optimizer[1].zero_grad()  # 0代表演员，1代表评论家
        cri_loss.backward()
        self.optimizer[1].step()

    def save(self):
        torch.save(self.acTar_net, "mod/acTar_net.pt")  #
        torch.save(self.acEval_net, "mod/acEval_net.pt")  #
        torch.save(self.crTar_net, "mod/crTar_net.pt")  #
        torch.save(self.crEval_net, "mod/crEval_net.pt")  #
    def read(self):
        if os.path.exists("mod/actor_net.pt") and os.path.exists("mod/critic_net.pt") :
            self.acTar_net = torch.load("mod/acTar_net.pt")  #
            self.acEval_net= torch.load("mod/acEval_net.pt")  #
            self.crTar_net = torch.load("mod/crTar_net.pt")  #
            self.crEval_net= torch.load("mod/crEval_net.pt")  #
            self.initData()
    def fullExperience(self,env):#充满经验池
        is_terminal = True
        for i in range(MEMORY_CAPACITY):#先随机取动作，填满经验池
            if is_terminal:
                stList=self.initState(env)  # 用stList储存前几步
            action = np.random.randint(0, ENVMAX[1])
            next_state, reward, is_terminal, info = env.step(action)
            stList = self.nextState(stList, cartPole_util.resetPic(next_state))  # 用stList储存前几步
            reward = cartPole_util.setReward(env,next_state)
            self.pushRemember(stList, action, reward,1/ENVMAX[1],is_terminal)  # 记下状态，动作，回报
        print("go!")
        return stList

def train(beginShape):
    agent=Agent(beginShape)
    # net.read()
    sumreward = 0
    for episode_i in range(EPOCHS_CNT):#循环若干次
        is_terminal=False
        stList = agent.initState(env)
        while (not is_terminal):
            # env.render()
            action,rate = agent.choose_action(stList[0:OVERLAY_CNT])
            next_state, reward, is_terminal, info = env.step(action)
            stList = agent.nextState(stList, next_state)  # 用stList储存前几步
            sumreward += reward
            reward = cartPole_util.setReward(env, next_state)
            agent.pushRemember(stList,action,reward,rate,is_terminal)
            agent.learn()
        if episode_i%10==0 :
            print(str(episode_i)+"reward:"+str(sumreward/10))
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
            action,rate = agent.choose_action(state)
            state, reward, is_terminal, info = env.step(action)
            time.sleep(0.01)
shape = env.reset().shape# 重新开始游戏,(210, 160, 3)
print(shape)
net=train(shape)
net.save()
test_mod(shape)