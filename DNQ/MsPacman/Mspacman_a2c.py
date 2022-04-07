import math
import random
import time
import platform

import pynvml
import torchvision
import cv2
import gym
import os
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.nn.functional as F#激活函数
from attention import MultiHeadAttention

BATCH_SZ=10
EPOCHS_CNT=50000
MEMORY_CAPACITY=10
OVERLAY_CNT=4#针对此游戏的叠帧操作
env=gym.make('MsPacman-v0')#吃豆人#MsPacman-ram-v0
ENVMAX=[env.observation_space.shape[0],9]
# print(env.spaces())
γ=0.90#reward_decay
α=0.01#学习率learning_rate,也叫lr
PATH="/home/hz3/A_SYB/RL/DNQ/"
pynvml.nvmlInit()
print(platform.system())
if platform.system() == "Linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if platform.system() =="Windows":
    BATCH_SZ = 1
    MEMORY_CAPACITY =10
    PATH = "D:\code\python\ReinforcementLearning1\DNQ/"
def initInfo(s,f=PATH+"MsPacman/mod.txt"):
    if os.path.exists(f):
        os.remove(f)
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def wInfo(s,f=PATH+"MsPacman/mod.txt"):
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def cv_show( mat):
    cv2.imshow("name", mat)
    cv2.waitKey(0)
def checkWD():
    gpu_device1 = pynvml.nvmlDeviceGetHandleByIndex(0)
    temperature1 = pynvml.nvmlDeviceGetTemperature(gpu_device1, pynvml.NVML_TEMPERATURE_GPU)
    gpu_device2 = pynvml.nvmlDeviceGetHandleByIndex(1)
    temperature2 = pynvml.nvmlDeviceGetTemperature(gpu_device2, pynvml.NVML_TEMPERATURE_GPU)
    if temperature1>76 or temperature2>76:
        print(str(temperature1)+"  "+str(temperature2)+" exit!!!")
        # sys.exit(0)
def resetPic(pic):
    pic=cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic=pic[0:172, 0:160]
    pic=cv2.resize(pic,(84,84))
    dst = np.zeros(pic.shape, dtype=np.float32)
    pic=cv2.normalize(pic, dst=dst, alpha=1.0, beta=0,norm_type=cv2.NORM_L1)
    return pic
class NET(nn.Module):#演员网络
    def __init__(self,kind):
        super().__init__()
        prenet = torchvision.models.resnet18(pretrained=False)
        # print(prenet)
        # for parma in prenet.parameters():
        #     parma.requires_grad = False
        prenet.conv1 = nn.Conv2d(in_channels=OVERLAY_CNT, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                                 padding=(3, 3), bias=False)
        prenet.fc=nn.Sequential()
        self.prenet = prenet

        hdim=128
        self.fc1 =nn.Sequential(
            nn.Linear(in_features=512, out_features=hdim),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=ENVMAX[1])
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=hdim),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=1)
        )
        self.fc=[self.fc1,self.fc2]

        self.att = [MultiHeadAttention(1, hdim, hdim, hdim).cuda(),MultiHeadAttention(1, hdim, hdim, hdim).cuda()]

    def forward(self,x,k):#重写前向传播算法，x是张量
        x=x.cuda()
        # x=self.conv(x)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        x=self.prenet(x)
        x=self.fc[k](x)
        return x
    def addAtten(self,x,id):
        x = x.unsqueeze(1)
        x=self.att[id](x,x,x)
        x = x.squeeze(1)
        return x
class Agent(object):#智能体，蒙特卡洛策略梯度算法

    def __init__(self,beginShape):
        self.actor_net = NET(0).cuda()#初始化两个网络，target and training
        self.critic_net =self.actor_net
        self.net=(self.actor_net,self.critic_net)
        self.actor_net.train()
        self.critic_net.train()

        self.beginShape=beginShape

        self.learn_step_i = 0  # count the steps of learning process
        self.optimizer=torch.optim.Adam(self.actor_net.parameters(), lr=α)


        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1))]

    def initState(self,env):#
        state = resetPic(env.reset())  # 重新开始游戏,(210, 160)
        stList=np.zeros((OVERLAY_CNT+1,)+state.shape)
        # li[OVERLAY_CNT]=state
        for i in range(0,OVERLAY_CNT):
            action = np.random.randint(0,ENVMAX[1])
            state, reward, is_terminal, info = env.step(action)
            stList[i] = resetPic(state)
        return stList
    def nextState(self,stList,state):
        for i in range(OVERLAY_CNT,0,-1):
            stList[i] = stList[i-1]
        stList[0] = state
        return stList
    def choose_action(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        x=np.expand_dims(x, axis=0)#加一个维度
        x = self.actor_net.forward(torch.FloatTensor(x),0)#0代表演员网络，1代表评论家网络
        prob = F.softmax(x, dim=1).cpu()
        # print(prob)
        return np.random.choice(a=ENVMAX[1], p=prob[0].detach().numpy())
    def pushRemember(self,state,act,reward):#原本是DQN的方法的结构，但这里用它当作缓存数据
        t=self.memory_i%MEMORY_CAPACITY

        self.memoryQue[0][t] = state[1:OVERLAY_CNT+1]
        self.memoryQue[1][t] = state[0:OVERLAY_CNT]
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
        state = torch.FloatTensor(self.memoryQue[0][t]).unsqueeze(dim=0)  # state
        next_state = torch.FloatTensor(self.memoryQue[1][t]).unsqueeze(dim=0)
        act = torch.LongTensor(self.memoryQue[2][t].astype(int)).cuda()
        reward = torch.FloatTensor(self.memoryQue[3][t]).cuda()

        v=(self.critic_net.forward(state,1).squeeze(dim=1),self.critic_net.forward(next_state,1).squeeze(dim=1))
        #用cri网络计算
        loss=F.mse_loss(reward+γ*v[1],v[0])#用tderror的方差来算，代表减少两次状态评分的差值，而actor网络负责保证这个价值会提高，而非下降

        #更新actor网络
        output = self.actor_net.forward(state,0)
        acloss = F.cross_entropy(output, act)#loss相当于算交叉熵-output*log(p),p是act算出来的
        tderror = (reward + γ * v[1] - v[0]).detach()  # td error
        # print(reward,γ * v[1] - v[0],acloss)

        acloss =tderror *acloss+loss#这里要反向传播acloss，tderror去掉梯度
        # print(acloss)
        self.optimizer.zero_grad()
        acloss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.actor_net, PATH+"MsPacman/mod/actor_net.pt")  #
        torch.save(self.critic_net, PATH+"MsPacman/mod/critic_net.pt")  #
    def read(self):
        if os.path.exists(PATH+"MsPacman/mod/actor_net.pt") and os.path.exists(PATH+"MsPacman/mod/critic_net.pt") :
            self.actor_net=torch.load(PATH+"MsPacman/mod/actor_net.pt")  #
            self.critic_net = torch.load(PATH+"MsPacman/mod/critic_net.pt")  #
            self.optimizer=torch.optim.Adam(self.actor_net.parameters(), lr=α)
rewardInd={0:-0.05,10:0.5,50:0.6,100:0.7,200:0.8,300:0.9}
def setReward(reward,is_terminal,lives,preLives,sum):#设置奖励函数函数
    if is_terminal and lives>0:
        return 1
    if preLives > lives:  # 死了命
        return -0.9999
    if reward == 0:
        return 0
    return 1 / (1 + math.exp(-reward / 100))  # +1/(1+math.exp(-step/100))

def train(beginShape, cartPole_util=None):
    agent=Agent(beginShape)
    # agent.read()
    sumreward = 0
    for episode_i in range(EPOCHS_CNT):#循环若干次
        is_terminal=False
        stList = agent.initState(env)
        preLives=3
        frame_i=0
        while (not is_terminal):
            # env.render()
            action = agent.choose_action(stList[0:OVERLAY_CNT])
            # print(action)
            next_state, reward, is_terminal, info = env.step(action)
            stList = agent.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            sumreward += reward
            frame_i+=1
            reward = setReward(reward,is_terminal,info["lives"],preLives)
            agent.pushRemember(stList,action,reward)
            agent.learn()
            preLives=info["lives"]
        # if episode_i%10==0:
        #     # checkWD()
        if episode_i%50==0 :
            s=str(episode_i) + "reward:" + str(sumreward / 50)
            wInfo(s)
            sumreward = 0
            if episode_i%1000==0:
                wInfo("episode_i:"+str(episode_i))
                agent.save()
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
shape = resetPic(env.reset()).shape# 重新开始游戏,(210, 160, 3)
initInfo("begin!!")
if platform.system() =="Linux":
    agent=train(shape)
    agent.save()
if platform.system() == "Windows":
    test_mod(shape)
agent=train(shape)
agent.save()
