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
MEMORY_CAPACITY=1000
TARGET_REPLACE_ITER=1000#隔多少论更新参数
OVERLAY_CNT=1#针对此游戏的叠帧操作
env=gym.make('CartPole-v0')#吃豆人#MsPacman-ram-v0
MAX=210
inf=0.0001
INF=1000000
ENVMAX=[env.observation_space.shape[0],2]
# print(env.spaces())
γ=0.90#reward_decay
α=0.001#学习率learning_rate,也叫lr
RUAN=0.1#软更新

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
            nn.Linear(64, ENVMAX[1]),
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
    def initData(self):
        self.net = (self.acEval_net, self.acTar_net, self.crEval_net, self.crTar_net)
        self.optimizer = (torch.optim.Adam(self.acEval_net.parameters(), lr=α),torch.optim.Adam(self.crEval_net.parameters(), lr=α))
        for net in self.net:
            net.train()

    def __init__(self,beginShape):
        self.acEval_net , self.acTar_net = ANET().cuda(),ANET().cuda()#初始化两个网络，target and training
        self.crEval_net , self.crTar_net = CNET().cuda(),CNET().cuda()

        self.beginShape=beginShape
        self.learn_step_i = 0  # count the steps of learning process
        self.initData()
        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1))]

    def initState(self, env):  #
        state = resetPic(env.reset())  # 重新开始游戏,(210, 160)
        stList = np.zeros((OVERLAY_CNT + 1,) + state.shape)
        # li[OVERLAY_CNT]=state
        for i in range(0, OVERLAY_CNT):
            action = np.random.randint(0, ENVMAX[1])
            state, reward, is_terminal, info = env.step(action)
            stList[i] = resetPic(state)
        return stList

    def nextState(self, stList, state):
        for i in range(OVERLAY_CNT, 0, -1):
            stList[i] = stList[i - 1]
        stList[0] = state
        return stList
    def __choose_action(self,x,net):#利用演员网络选取动作
        x = net.forward(torch.FloatTensor(x))
        prob = F.softmax(x, dim=1).cpu()
        return [[np.random.choice(a=ENVMAX[1], p=prob[i].detach().numpy())] for i in range(BATCH_SZ)]
    def choose_action(self, x,epsl=0):  # 按照当前状态，可以用ϵ_greedy选取策略
        if np.random.uniform(0,1)>epsl:
            x=np.expand_dims(x, axis=0)#加一个维度
            x = self.acEval_net.forward(torch.FloatTensor(x))
            prob = F.softmax(x, dim=1).cpu()
            return np.random.choice(a=ENVMAX[1], p=prob[0].detach().numpy())
        return np.random.randint(0,ENVMAX[1])#随机选取动作

    def pushRemember(self, state, act, reward,is_terminal):  # 原本是DQN的方法的结构，但这里用它当作缓存数据
        t = self.memory_i % MEMORY_CAPACITY
        self.memoryQue[0][t] = state[1:OVERLAY_CNT + 1]
        self.memoryQue[1][t] = state[0:OVERLAY_CNT]
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(is_terminal)
        self.memory_i = self.memory_i + 1
    def updateNet(self,tarDir,evalDir):
        for key in tarDir.keys():
            if 'weight' in key:
                tarDir[key] = (1 - RUAN) * tarDir[key] + RUAN * evalDir[key]
    def learn(self):#学习,训练ac算法2个网络
        self.learn_step_i += 1
        if self.learn_step_i % TARGET_REPLACE_ITER == 0:  # 它和DQN不同，由于RUAN存在，每次更新Tar网络的比例很小，也就是所谓"软"更新
            print("dsfdsf")
            self.updateNet(self.acTar_net.state_dict(), self.acEval_net.state_dict())
            self.updateNet(self.crTar_net.state_dict(), self.crEval_net.state_dict())
        # print("hhhhhhh")
        sampleList=np.random.choice(MEMORY_CAPACITY, BATCH_SZ)#从[0,MEMORY_CAPACITY-1]区间随机选取BATCH_SZ个数
        state = torch.FloatTensor(self.memoryQue[0][sampleList])  # state
        next_state = torch.FloatTensor(self.memoryQue[1][sampleList])
        act = torch.LongTensor(self.memoryQue[2][sampleList].astype(int)).cuda()
        reward = torch.FloatTensor(self.memoryQue[3][sampleList]).cuda()
        is_ter=self.memoryQue[4][sampleList]

        act_tar=torch.LongTensor(self.__choose_action(next_state,self.acTar_net)).cuda()
        act_eval = torch.LongTensor(self.__choose_action(state, self.acEval_net)).cuda()
        Q_tar = self.crTar_net.forward(next_state).gather(1,act_tar)
        Q = self.crEval_net.forward(state)  # 利用python语法糖，对每个批次的 Q数据选择动作

        #用cri网络计算
        y=reward
        for i in range(len(is_ter)):
            if is_ter[i]==0:
                y[i]+=γ * Q_tar[i]
        loss=F.mse_loss(y,Q.gather(1,act))#用tderror的方差来算，代表减少两次状态评分的差值，而actor网络负责保证这个价值会提高，而非下降

        self.optimizer[1].zero_grad()#0代表演员，1代表评论家
        loss.backward(retain_graph=True)
        self.optimizer[1].step()

        Q = self.crEval_net.forward(state)
        loss=-Q.gather(1,act_eval).mean()#梯度上升更新
        # print(loss)
        self.optimizer[0].zero_grad()#0代表演员，1代表评论家
        loss.backward()
        self.optimizer[0].step()
        #
        # #更新actor网络
        # output = self.actor_net.forward(state)
        # acloss = F.cross_entropy(output, act)#loss相当于算交叉熵-output*log(p),p是act算出来的
        # tderror = (reward + γ * v[1] - v[0]).detach()  # td error
        # acloss = tderror *acloss#这里要反向传播acloss，tderror去掉梯度
        # # print(acloss)
        # self.optimizer[0].zero_grad()
        # acloss.backward()
        # self.optimizer[0].step()

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
            stList = self.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            reward = setReward(next_state)
            self.pushRemember(stList, action, reward,is_terminal)  # 记下状态，动作，回报
        print("go!")
        return stList

def resetPic(pic):
    return pic
def setReward(state):#设置奖励函数函数
    x, x_dot, theta, theta_dot = state
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    return r1 + r2

def train(beginShape):
    agent=Agent(beginShape)
    # net.read()
    agent.fullExperience(env)
    sumreward = 0
    for episode_i in range(EPOCHS_CNT):#循环若干次
        is_terminal=False
        stList = agent.initState(env)
        while (not is_terminal):
            # env.render()
            action = agent.choose_action(stList[0:OVERLAY_CNT])
            next_state, reward, is_terminal, info = env.step(action)
            stList = agent.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            sumreward += reward
            reward = setReward(next_state)
            agent.pushRemember(stList,action,reward,is_terminal)
            if agent.memory_i%5==0:
                agent.learn()
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

