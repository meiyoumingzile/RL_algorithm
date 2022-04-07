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
import sys
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
OVERLAY_CNT=4#针对此游戏的叠帧操作
TARGET_REPLACE_ITER=1000#隔多少论更新参数
MEMORY_CAPACITY=10000#记忆容量,不能很小，否则学不到东西
env=gym.make('MsPacman-v0')#吃豆人#MsPacman-ram-v0
mspacman_color = np.array([210, 164, 74]).mean()
MAX=10000
inf=0.0001
INF=1000000
ENVMAX=[env.observation_space.shape[0],9]
# print(env.spaces())
γ=0.90#reward_decay
α=0.001#学习率learning_rate,也叫lr

if platform.system() == "Linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if platform.system() =="Windows":
    BATCH_SZ = 1
    MEMORY_CAPACITY =10

def cv_show( mat):
    cv2.imshow("name", mat)
    cv2.waitKey(0)
def resetPic(pic):
    pic=cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic=pic[0:172, 0:160]
    dst = np.zeros(pic.shape, dtype=np.float32)
    # pic=cv2.resize(pic,(90,90))
    pic=cv2.normalize(pic, dst=dst, alpha=1.0, beta=0,norm_type=cv2.NORM_L1)
    return pic
class QNET(nn.Module):#拟合Q的神经网络
    def __init__(self):
        super().__init__()
        # self.lstm=nn.LSTM(input_size=OVERLAY_CNT,hidden_size=32,num_layers=3,bias=True,batch_first=True)

        self.conv=nn.Sequential(
            nn.Conv2d(OVERLAY_CNT, 32, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=4),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, ENVMAX[1]),
        )
        self.advantage=self.fc#优势函数可以直接等于fc的结构,神经网络自己会调整。
        self.value = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        for con in self.conv:
            if con=="<class 'torch.nn.modules.conv.Conv2d'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)
        for con in self.fc:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)

    def forward(self,x):#重写前向传播算法，x是张量
        x=x.cuda()

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # x = self.lstm(x)
        # print(x.shape)
        # x = self.fc(x)
        # return x
        a = self.advantage(x)
        v = self.value(x)
        return v + a - a.mean()#决斗DQN,这里相加使用语法糖，实际上v 与a 维度不同

class Agent(object):#拟合Q的神经网络
    def __init__(self,beginShape):
        self.eval_net, self.target_net = QNET().cuda(), QNET().cuda()#初始化两个网络，target and training
        #eval_net是新网络，self.target_net是旧网络，target_net相当于学习的成果
        self.beginShape=beginShape
        self.learn_step_i = 0  # count the steps of learning process
        self.memory_i = 0  # counter used for experience replay buffer
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1))]
        #self.memoryQue为循环队列
        self.lossFun=torch.nn.SmoothL1Loss()
        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)
        self.optimizer=torch.optim.RMSprop(self.eval_net.parameters(), lr=α, alpha=0.95, eps=0.01)

    def greedy(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 加1维
        actVal = self.eval_net.forward(x.cuda())  # 从eval_net里选取结果
        act = torch.max(actVal, 1)[1].data.cpu().numpy()
        return act[0]
    def choose_action(self, x,epsl):  # 按照当前状态，用ϵ_greedy选取策略
        if np.random.uniform(0,1)>epsl:#较大概率为贪心
            x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 加1维
            actVal=self.eval_net.forward(x.cuda())#从eval_net里选取结果
            act=torch.max(actVal,1)[1].data.cpu().numpy()
            return act[0]
        else:
            return np.random.randint(0,ENVMAX[1])#随机选取动作

    def pushRemember(self,state,act,reward,is_ter):#把记忆存储起来，记忆库是个循环队列
        t=self.memory_i%MEMORY_CAPACITY#不能在原地址取模，因为训练时候要用memory_i做逻辑判断
        self.memoryQue[0][t] = state[1:OVERLAY_CNT+1]#存状态
        self.memoryQue[1][t] = state[0:OVERLAY_CNT]
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(is_ter)
        self.memory_i+=1
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
    def learn(self,rand_up):#学习以往经验,frame_i代表当前是游戏哪一帧
        if  self.learn_step_i % TARGET_REPLACE_ITER ==0:#每隔TARGET_REPLACE_ITER步，更新一次旧网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_i+=1

        sampleList=np.random.choice(min(rand_up,MEMORY_CAPACITY), BATCH_SZ)#从[0,MEMORY_CAPACITY-5]区间随机选取BATCH_SZ个数
        # randomMenory=self.memoryQue[np.random.choice(MEMORY_CAPACITY, BATCH_SZ)]#语法糖，传入一个列表，代表执行for(a in sampleList)self.memoryQue[a]
        #上述代表随机去除BATCH_SZ数量的记忆，记忆按照(state,act,reward,next_state)格式存储
        state=torch.FloatTensor(self.memoryQue[0][sampleList])#state
        next_state = torch.FloatTensor(self.memoryQue[1][sampleList])
        act = torch.LongTensor(self.memoryQue[2][sampleList].astype(int)).cuda()
        reward = torch.FloatTensor(self.memoryQue[3][sampleList]).cuda()
        is_ter = self.memoryQue[4][sampleList]

        q_eval = self.eval_net(state).gather(1, act)  # 采取最新学习成果
        maxInd=q_eval.argmax(1)
        # print(maxInd)
        q_next = self.target_net(next_state).detach()  # 按照旧经验回忆.detach()为返回一个深拷贝，它没有梯度
        q_target = reward
        for i in range(BATCH_SZ):
            if is_ter[i]==0:#如果不是终点
                q_target[i]+=γ*q_next[i][maxInd[i]]#double DQN
        loss=self.lossFun(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # if self.memory_i%10000==0:
        #     print([x.grad for x in self.optimizer.param_groups[0]['params']])
        self.optimizer.step()

    def save(self):
        torch.save(self.target_net, "mod/target_net.pt")#保存模型pt || pth
        torch.save(self.eval_net, "mod/eval_net.pt")  #
    def read(self):
        self.target_net=torch.load( "mod/target_net.pt")#加载模型
        self.eval_net=torch.load("mod/eval_net.pt")  #
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)

def setReward(reward,is_terminal,lives,preLives,sum):#设置奖励函数函数
    if preLives>lives :#死了命
        return -100
    return reward
def getepsl(frame_i):
    if frame_i>1000000:
        return 0.05
    return max(1-frame_i//1000*0.01,0.1)

def train(beginShape):
    net=Agent(beginShape)
    # net.read()
    sumreward = 0
    frame_i=0
    net.eval_net.train()
    net.target_net.eval()
    for episode_i in range(EPOCHS_CNT):#循环若干次
        stList=net.initState(env)#用stList储存前几步
        is_terminal=False
        preLives=3
        while(not is_terminal):
            # env.render()
            action = net.choose_action(stList[0:OVERLAY_CNT],0.1)  # 采用ϵ贪婪策略选取一个动作
            next_state,reward,is_terminal,info = env.step(action)
            stList = net.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            frame_i += 1
            sumreward+=reward
            reward=setReward(reward,is_terminal,info["lives"],preLives,sumreward)
            net.pushRemember(stList, action, reward,is_terminal)#记下状态，动作，回报
            if net.memory_i > 300 and net.memory_i%5==0:  # 当步数大于脑容量的时候才开始学习，每五步学习一次
                net.learn(frame_i)
            preLives=info["lives"]

        if episode_i%50==0 :
            print(str(episode_i)+"reward:"+str(sumreward/50))
            sumreward = 0
            if episode_i%500==0:
                print("episode_i:"+str(episode_i))
                net.save()
    return net
def test(beginShape):
    net = Agent(beginShape)
    net.read()
    net.eval_net.train()
    net.target_net.eval()
    for episode_i in range(EPOCHS_CNT):  # 循环若干次
        stList = net.initState(env)  # 用stList储存前几步
        is_terminal = False
        while (not is_terminal):
            env.render()
            action = net.greedy(stList[0:OVERLAY_CNT])  # 采用ϵ贪婪策略选取一个动作
            next_state, reward, is_terminal, info = env.step(action)
            stList = net.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            time.sleep(0.01)
shape = resetPic(env.reset()).shape# 重新开始游戏,(210, 160, 3)

if platform.system() =="Linux":
    net=train(shape)
    net.save()
if platform.system() == "Windows":
    test(shape)

