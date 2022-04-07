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

if platform.system() == "Linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

BATCH_SZ=100
EPOCHS_CNT=30000
OVERLAY_CNT=10#针对此游戏的叠帧操作
TARGET_REPLACE_ITER=200#隔多少论更新参数
MEMORY_CAPACITY=5000#记忆容量,不能很小，否则学不到东西

env=gym.make('MsPacman-v0')#吃豆人#MsPacman-ram-v0
MAX=10000
inf=0.001
INF=1000000
ENVMAX=[env.observation_space.shape[0],env.action_space.n]
# print(env.spaces())
γ=0.9#reward_decay
ϵ=0.1#ϵ贪婪策略
α=0.001#学习率learning_rate,也叫lr
def cv_show(name, mat):
    cv2.imshow(name, mat)
    cv2.waitKey(0)
class QNET(nn.Module):#拟合Q的神经网络
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(OVERLAY_CNT, 32, kernel_size=8, stride=4),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, ENVMAX[1]),
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
        # print(x.shape)
        x = self.fc(x)
        # x=F.softmax(x,dim=1)#平方差损失函数不压缩
        return x
class DQN(object):#拟合Q的神经网络
    def __init__(self,beginShape):
        self.eval_net, self.target_net = QNET().cuda(), QNET().cuda()#初始化两个网络，target and training
        #eval_net是新网络，self.target_net是旧网络，target_net相当于学习的成果
        self.beginShape=beginShape
        self.learn_step_i = 0  # count the steps of learning process
        self.memory_i = 0  # counter used for experience replay buffer
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,)+beginShape),np.zeros((MEMORY_CAPACITY, 1)),
                          np.zeros((MEMORY_CAPACITY, 1))]
        self.memoryInd=[0]*self.memoryQue#记忆队列里有帧的索引
        self.memoryInd_i=0
        #self.memoryQue为循环队列

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)

    def greedy(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        x=torch.unsqueeze(torch.FloatTensor(x),0)#加1维
        actVal = self.eval_net.forward(x)  # 从eval_net里选取结果
        act = torch.max(actVal, 1)[1].data.numpy()
        return act[0]
    def choose_action(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        if np.random.uniform(0,1)>ϵ:#较大概率为贪心
            x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 加1维
            actVal=self.eval_net.forward(x.cuda())#从eval_net里选取结果
            act=torch.max(actVal,1)[1].data.cpu().numpy()
            return act[0]
        else:
            return np.random.randint(0,ENVMAX[1])#随机选取动作

    def store_transition(self,state,act,reward,isRandIndex=True):#把记忆存储起来，记忆库是个循环队列
        # print(state.shape,self.memoryQue[0][self.memory_i].shape)
        t=self.memory_i%MEMORY_CAPACITY#不能在原地址取模，因为训练时候要用memory_i做逻辑判断
        self.memoryQue[0][t] = state#存状态
        self.memoryQue[1][t] = np.array(act)
        self.memoryQue[2][t] = np.array(reward)
        if isRandIndex:
            self.memoryInd[self.memoryInd_i]=t
            self.memoryInd_i+=1
        self.memory_i+=1
    def getState(self,i=0):#从循环队列取出第i个状态,
        return self.memoryQue[0][(self.memory_i-i-1)%MEMORY_CAPACITY]
    # def myloss(self,q_eval,q_target):
    def initState(self,env):#从循环队列取出前两个状态
        state = cv2.cvtColor(env.reset(), cv2.COLOR_BGR2GRAY)  # 重新开始游戏,(210, 160)
        for i in range(0,OVERLAY_CNT):
            action = np.random.randint(0,ENVMAX[1])
            state, reward, is_terminal, info = env.step(action)
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            t = self.memory_i % MEMORY_CAPACITY
            self.memoryQue[0][t] = state
            self.memory_i += 1

    def seqPic(self,k,sampleList):#叠帧,分批次取前OVERLAY_CNT帧
        li=np.zeros((len(sampleList),OVERLAY_CNT)+self.beginShape)
        t=self.memory_i%MEMORY_CAPACITY#
        for i in range(len(sampleList)):
            for j in range(OVERLAY_CNT):
                li[i][j]=self.memoryQue[0][(t-1-sampleList[i]-j-k+MEMORY_CAPACITY)%MEMORY_CAPACITY]
        return li
    def getSeqState(self):#叠帧得到前几步
        li=np.zeros((OVERLAY_CNT,)+self.beginShape)
        for i in range(OVERLAY_CNT):
            li[i]=self.memoryQue[0][(self.memory_i-i-1)%MEMORY_CAPACITY]
        return li
    def learn(self,frame_i):#学习以往经验,frame_i代表当前是游戏哪一帧
        if  self.learn_step_i % TARGET_REPLACE_ITER ==0:#每隔TARGET_REPLACE_ITER步，更新一次旧网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_i+=1

        sampleList=np.random.choice(MEMORY_CAPACITY-OVERLAY_CNT-1, BATCH_SZ)#从[0,MEMORY_CAPACITY-5]区间随机选取BATCH_SZ个数
        # randomMenory=self.memoryQue[np.random.choice(MEMORY_CAPACITY, BATCH_SZ)]#语法糖，传入一个列表，代表执行for(a in sampleList)self.memoryQue[a]
        #上述代表随机去除BATCH_SZ数量的记忆，记忆按照(state,act,reward,next_state)格式存储
        state=torch.FloatTensor(self.seqPic(1, sampleList,frame_i))#state
        act = torch.LongTensor(self.memoryQue[1][sampleList].astype(int)).cuda()
        reward = torch.FloatTensor(self.memoryQue[2][sampleList]).cuda()
        next_state = torch.FloatTensor(self.seqPic(0,sampleList))

        q_eval=self.eval_net(state).gather(1, act)#采取最新学习成果
        q_next = self.target_net(next_state).detach()#按照旧经验回忆.detach()为返回一个深拷贝，它没有梯度
        q_target=reward+γ*q_next.max(1,keepdim=True)[0]
        # print(reward,q_next.max(1,keepdim=True)[0])
        # print(q_eval)
        loss=F.mse_loss(q_eval,q_target)
        # if self.memory_i%10==0:
        #      print(loss)
        self.optimizer.zero_grad()
        # sub=q_eval-q_target
        loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.target_net, "mod/target_net.pt")#保存模型pt || pth
        torch.save(self.eval_net, "mod/eval_net.pt")  #
    def read(self):
        self.target_net=torch.load( "mod/target_net.pt")#加载模型
        self.eval_net=torch.load("mod/eval_net.pt")  #
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)
def setReward(reward,is_terminal,lives,preLives):#设置奖励函数函数
    if is_terminal:#结束
        return 100000 * (lives == 0 and -1 or 1)
    elif preLives>lives :#死了命
        return -100
    elif reward==0:
        return -1
    return reward
def train(beginShape):
    net=DQN(beginShape)
    net.read()
    # is_terminal = True
    # for i in range(MEMORY_CAPACITY):#先填满经验池
    #     if is_terminal:
    #         net.initState(env)  # 用stList储存前几步
    #         is_terminal = False
    #     action = net.choose_action(net.getSeqState())  # 采用ϵ贪婪策略选取一个动作
    #     next_state, reward, is_terminal, mod = env.step(action)
    #     next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
    #     net.store_transition(next_state, action, reward)  # 记下状态，动作，回报


    for episode_i in range(EPOCHS_CNT):#循环若干次
        # print(episode_i)
        net.initState(env)#用stList储存前几步
        is_terminal=False
        sumreward=0
        frame_i=OVERLAY_CNT#帧数
        preLives=3
        while(not is_terminal):
            # env.render()
            action = net.choose_action(net.getSeqState())  # 采用ϵ贪婪策略选取一个动作
            next_state,reward,is_terminal,info = env.step(action)
            sumreward+=reward
            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            reward=setReward(reward,is_terminal,info["lives"],preLives)

            net.store_transition(next_state, action, reward)#记下状态，动作，回报
            if net.memory_i > MEMORY_CAPACITY and net.memory_i%5==0:  # 当步数大于脑容量的时候才开始学习，每五步学习一次
                net.learn(frame_i)
            frame_i+=1
            preLives=info["lives"]
            # Q[state][action]+=α*(reward+γ*Q[next_state].max()-Q[state][action])
        print(str(episode_i)+"reward:"+str(sumreward))
        if episode_i%100==0:
            print("episode_i:"+str(episode_i))
            net.save()
    return net
def test(beginShape):
    net = DQN(beginShape)
    net.read()
    for episode_i in range(10):  # 循环若干次
        # print(episode_i)
        net.initState(env)  # 用stList储存前几步
        is_terminal = False
        while (not is_terminal):
            env.render()
            action = net.choose_action(net.getSeqState())  # 采用ϵ贪婪策略选取一个动作
            next_state, reward, is_terminal, info = env.step(action)
            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            net.store_transition(next_state, action, reward)  # 记下状态，动作，回报
            time.sleep(0.01)



shape = cv2.cvtColor(env.reset() , cv2.COLOR_BGR2GRAY).shape# 重新开始游戏,(210, 160, 3)
net=train(shape)
net.save()
# test(shape)

