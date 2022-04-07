import random
import time
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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
BATCH_SZ=100
EPOCHS_CNT=50

TARGET_REPLACE_ITER=200#隔多少论更新参数
MEMORY_CAPACITY=200#记忆容量

DEVICE = torch.device("cuda") #设置训练设备，torch.device是判断电脑设备
env=gym.make('MsPacman-v0')#吃豆人#MsPacman-ram-v0
MAX=1000
inf=0.001
INF=1000000
ENVMAX=[env.observation_space.shape[0],env.action_space.n]
# print(env.spaces())
γ=0.9#reward_decay
ϵ=0.1#ϵ贪婪策略
α=0.001#学习率learning_rate,也叫lr
class QNET(nn.Module):#拟合Q的神经网络
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 6, 3, stride=2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2),
            nn.ReLU(),
        )
        self.fc=nn.Sequential(
            nn.Linear(1728, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, ENVMAX[1]),
        )

    def forward(self,x):#重写前向传播算法，x是张量
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        # x=F.softmax(x,dim=1)#平方差损失函数不压缩
        return x
class DQN(object):#拟合Q的神经网络
    def __init__(self,beginShape):
        self.eval_net, self.target_net = QNET(), QNET()#初始化两个网络，target and training
        #eval_net是新网络，self.target_net是旧网络，target_net相当于学习的成果

        self.learn_step_i = 0  # count the steps of learning process
        self.memory_i = 0  # counter used for experience replay buffer
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,)+beginShape),np.zeros((MEMORY_CAPACITY, 1)),
                          np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY,)+beginShape)]
        #self.memoryQue为循环队列

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)

    def greedy(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        x=torch.unsqueeze(torch.FloatTensor(x),0)#加1维
        actVal = self.eval_net.forward(x)  # 从eval_net里选取结果
        act = torch.max(actVal, 1)[1].data.numpy()
        return act[0]
    def choose_action(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        x=torch.unsqueeze(torch.FloatTensor(x),0)#加1维
        if np.random.uniform(0,1)>ϵ:#较大概率为贪心
            actVal=self.eval_net.forward(x)#从eval_net里选取结果
            act=torch.max(actVal,1)[1].data.numpy()
            return act[0]
        else:
            return np.random.randint(0,ENVMAX[1])#随机选取动作

    def store_transition(self,state,act,reward,next_state):#把记忆存储起来
        # print(state.shape,self.memoryQue[0][self.memory_i].shape)
        t=self.memory_i%MEMORY_CAPACITY#不能在原地址取模，因为训练时候要用memory_i做逻辑判断
        self.memoryQue[0][t] = state
        self.memoryQue[1][t] = np.array(act)
        self.memoryQue[2][t] = np.array(reward)
        self.memoryQue[3][t] = next_state
        self.memory_i+=1
    # def myloss(self,q_eval,q_target):
    def learn(self):#学习以往经验
        if  self.learn_step_i % TARGET_REPLACE_ITER ==0:#每隔TARGET_REPLACE_ITER步，更新一次旧网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_i+=1

        sampleList=np.random.choice(MEMORY_CAPACITY, BATCH_SZ)#从[0,MEMORY_CAPACITY-1]区间随机选取BATCH_SZ个数
        # randomMenory=self.memoryQue[np.random.choice(MEMORY_CAPACITY, BATCH_SZ)]#语法糖，传入一个列表，代表执行for(a in sampleList)self.memoryQue[a]
        #上述代表随机去除BATCH_SZ数量的记忆，记忆按照(state,act,reward,next_state)格式存储
        state=torch.FloatTensor(self.memoryQue[0][sampleList])#state
        act = torch.LongTensor(self.memoryQue[1][sampleList].astype(int))
        reward = torch.FloatTensor(self.memoryQue[2][sampleList])
        next_state = torch.FloatTensor(self.memoryQue[3][sampleList])

        q_eval=self.eval_net(state).gather(1, act)#采取最新学习成果
        q_next = self.target_net(next_state).detach()#按照旧经验回忆.detach()为返回一个深拷贝，它没有梯度
        q_target=reward+γ*q_next.max(1,keepdim=True)[0]
        # print(q_eval)
        loss=F.mse_loss(q_eval,q_target)
        # if self.memory_i%10==0:
        #     print(loss)
        self.optimizer.zero_grad()
        # sub=q_eval-q_target
        loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.target_net, "target_net.pt")#保存模型pt || pth
        torch.save(self.eval_net, "eval_net.pt")  #
    def read(self):
        self.target_net=torch.load( "target_net.pt")#加载模型
        self.eval_net=torch.load("eval_net.pt")  #
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)

def train(beginShape):
    net=DQN(beginShape)
    for episode_i in range(EPOCHS_CNT):#循环若干次
        # print(episode_i)
        state = env.reset().transpose((2,0,1))  # 重新开始游戏,(210, 160, 3)
        is_terminal=False
        # action=random.randint(0,env.action_space.n-1)
        while(not is_terminal):
            # env.render()
            action = net.choose_action(state)  # 采用ϵ贪婪策略选取一个动作
            next_state,reward,is_terminal,info = env.step(action)
            next_state=next_state.transpose((2, 0, 1))
            # print(next_state,state)
            net.store_transition(state, action, reward, next_state)#记下状态，动作，回报
            if net.memory_i > MEMORY_CAPACITY and net.memory_i%5==0:  # 当步数大于脑容量的时候才开始学习，每五步学习一次
                net.learn()
            # Q[state][action]+=α*(reward+γ*Q[next_state].max()-Q[state][action])
            state=next_state
        if episode_i%10==0:
            print(episode_i)
    return net
def test(beginShape):
    net = DQN(beginShape)
    net.read()
    for episode_i in range(EPOCHS_CNT):  # 循环若干次
        state = env.reset().transpose((2,0,1))  # 重新开始游戏
        is_terminal = False
        while (not is_terminal):
            env.render()
            action = net.choose_action(state)  # 采用ϵ贪婪策略选取一个动作
            next_state, reward, is_terminal, info = env.step(action)
            next_state = next_state.transpose((2, 0, 1))
            # net.store_transition(state, action, reward, next_state)  # 记下状态，动作，回报
            state = next_state
            time.sleep(0.01)

shape=env.reset().transpose((2,0,1)).shape
Q=train(shape)
Q.save()
# test(shape)

