import math
import random
import sys
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
import pynvml
import torch.optim as optim
import torch.nn.functional as F#激活函数
from torchvision import transforms#图像处理的工具类
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#带上全部优化,double,dueling,Priority,noisenet
from torchvision.transforms import Compose

# from DNQ.MsPacman.attention import MultiHeadAttention
# from DNQ.myutil import initInfo, wInfo
from attention import MultiHeadAttention

BATCH_SZ=10
EPOCHS_CNT=50000
OVERLAY_CNT=4#针对此游戏的叠帧操作
TARGET_REPLACE_ITER=1000#隔多少论更新参数
MEMORY_CAPACITY=10000#记忆容量,不能很小，否则学不到东西
env=gym.make('MsPacman-v0').unwrapped#吃豆人#MsPacman-ram-v0
MAX=10000
inf=0.0001
INF=1000000
ENVMAX=[env.observation_space.shape[0],9]
# print(env.spaces())
γ=0.99#reward_decay
α=0.1#学习率learning_rate,也叫lr
EXP_ALPHA=1
pynvml.nvmlInit()

if platform.system() == "Linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if platform.system() =="Windows":
    BATCH_SZ = 1
    MEMORY_CAPACITY =10
def initInfo(s,f="mod.txt"):
    if os.path.exists(f):
        os.remove(f)
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def checkWD():
    gpu_device1 = pynvml.nvmlDeviceGetHandleByIndex(0)
    temperature1 = pynvml.nvmlDeviceGetTemperature(gpu_device1, pynvml.NVML_TEMPERATURE_GPU)
    gpu_device2 = pynvml.nvmlDeviceGetHandleByIndex(1)
    temperature2 = pynvml.nvmlDeviceGetTemperature(gpu_device2, pynvml.NVML_TEMPERATURE_GPU)
    if temperature1>76 or temperature2>76:
        print(str(temperature1)+"  "+str(temperature2)+" exit!!!")
        sys.exit(0)
def wInfo(s,f="mod.txt"):
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def cv_show( mat):
    cv2.imshow("name", mat)
    cv2.waitKey(0)
def resetPic(pic):
    pic=cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic=pic[0:172, 0:160]
    pic=cv2.resize(pic,(84,84))
    dst = np.zeros(pic.shape, dtype=np.float32)
    pic=cv2.normalize(pic, dst=dst, alpha=1.0, beta=0,norm_type=cv2.NORM_L1)
    return pic
class QNET(nn.Module):#拟合Q的神经网络
    def initWeight(self,convs):#
        for con in convs :
            if con=="<class 'torch.nn.modules.linear.Linear'>" or con=="<class 'torch.nn.modules.conv.Conv2d'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)
    def __init__(self):
        super().__init__()
        prenet = torchvision.models.resnet18()
        # print(prenet)
        # for parma in prenet.parameters():
        #     parma.requires_grad = False
        prenet.conv1=nn.Conv2d(in_channels=OVERLAY_CNT, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                                 padding=(3, 3), bias=False)
        self.prenet = prenet
        prenet.fc = nn.Sequential()
        hdim = 128
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=hdim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=ENVMAX[1], bias=True),
        )
        self.advantage=self.fc2
        self.value=nn.Sequential(
            nn.Linear(in_features=512, out_features=hdim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=1, bias=True),
        )
        # self.initWeight(self.advantage)
        # self.initWeight(self.value)

        self.att = MultiHeadAttention(1, hdim, hdim, hdim)
    def forward(self,x):#重写前向传播算法，x是张量
        x=x.cuda()
        x = self.prenet(x)

        a = self.advantage(x)
        v = self.value(x)
        return v + a - a.mean()  # 决斗DQN,这里相加使用语法糖，实际上v 与a 维度不同
    def addAtten(self,x):
        x = x.unsqueeze(1)
        x=self.att(x,x,x)
        x = x.squeeze(1)
        return x
class SumTree(object):
    def __init__(self,n):
        self.sumtree = [[0,0,0] for i in range((n * 4))]  # 初始化线段树：数组存储，乘法索引,[和，最大，最小]
        self.totreeInd = [0] * n  # 把memoryQue的位置索引到线段树的叶子
        self.tomemInd = [0] * (n * 4)  # 把线段树的叶子索引到memoryQue的位置
        self.buildTree(1, 0, n)  # 建树
        self.n=n
    def buildTree(self,now,l,r):
        if(r-l==1):
            self.totreeInd[l]=now
            self.tomemInd[now]=l
        else:
            mid = (l + r) // 2
            self.buildTree(now*2,l,mid)
            self.buildTree(now*2+1, mid, r)
    def print(self):
        def prString(now,l,r):
            if(r-l==1):
                return str(self.sumtree[now][0])+" "
            else:
                mid = (l + r) // 2
                return prString(now*2,l,mid)+prString(now*2+1, mid, r)
        print(prString(1,0,self.n))
    def update(self, pos, x):#单点修改把pos变成x
        now=self.totreeInd[pos]
        add=x-self.sumtree[now][0]
        self.sumtree[now][1]=self.sumtree[now][2]=x
        # print(pos,self.sumtree[now][0])
        while(now>0):
            self.sumtree[now][0]+=add
            now=now//2
            self.sumtree[now][1] = max(self.sumtree[now*2][1], self.sumtree[now*2+1][1])#更新
            self.sumtree[now][2] = max(self.sumtree[now * 2][2], self.sumtree[now * 2 + 1][2])  # 更新
    def topMax(self):
        return self.sumtree[1][1]
    def topMin(self):
        return self.sumtree[1][2]
    def queryPred(self):#单点查询，按照权重随机取个batch_sz个数返回位置
        l=0
        r=self.n
        now=1
        while(r-l>1):
            mid = (l + r) // 2
            now *= 2
            if np.random.uniform(0, 1)<self.sumtree[now][0]/(self.sumtree[now][0]+self.sumtree[now+1][0]):#如果随机数小于左边
                r=mid
            else:
                l=mid
                now+=1
        return self.tomemInd[now]
class Agent(object):#拟合Q的神经网络
    readMod = False
    def __init__(self,beginShape):
        self.eval_net, self.target_net = QNET().cuda(), QNET().cuda()#初始化两个网络，target and training
        self.eval_net.train()
        #eval_net是新网络，self.target_net是旧网络，target_net相当于学习的成果
        self.beginShape=beginShape
        self.learn_step_i = 0  # count the steps of learning process
        self.memory_i = 0  # counter used for experience replay buffer
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1))]
        self.sumtree=SumTree(MEMORY_CAPACITY)

        #self.memoryQue为循环队列
        self.lossFun=torch.nn.MSELoss()
        self.beta=0
        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(), lr=α)

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

    def pushRemember(self,state,act,reward,is_ter,val):#把记忆存储起来，记忆库是个循环队列
        t=self.memory_i%MEMORY_CAPACITY#不能在原地址取模，因为训练时候要用memory_i做逻辑判断
        self.memoryQue[0][t] = state[1:OVERLAY_CNT+1]#存状态
        self.memoryQue[1][t] = state[0:OVERLAY_CNT]
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(is_ter)
        self.sumtree.update(t, val)  # 初始化时tderro为最大值
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
    def getsqrtwi(self,p):
        return math.pow(self.sumtree.topMin()/p,self.beta/2)
    def learn(self,turn_i):#学习以往经验,frame_i代表当前是游戏哪一帧
        if  self.learn_step_i % TARGET_REPLACE_ITER ==0:#每隔TARGET_REPLACE_ITER步，更新一次旧网络
            checkWD()
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_i+=1
        sampleList=np.array([self.sumtree.queryPred() for i in range(BATCH_SZ)])
        # randomMenory=self.memoryQue[np.random.choice(MEMORY_CAPACITY, BATCH_SZ)]#语法糖，传入一个列表，代表执行for(a in sampleList)self.memoryQue[a]
        #上述代表随机去除BATCH_SZ数量的记忆，记忆按照(state,act,reward,next_state)格式存储
        state=torch.FloatTensor(self.memoryQue[0][sampleList])#state
        next_state = torch.FloatTensor(self.memoryQue[1][sampleList])
        act = torch.LongTensor(self.memoryQue[2][sampleList].astype(int)).cuda()
        reward = torch.FloatTensor(self.memoryQue[3][sampleList]).cuda()
        is_ter = self.memoryQue[4][sampleList]
        # print(act)
        q_eval = self.eval_net(state).gather(1, act)  # 采取最新学习成果
        # maxInd=q_eval.argmax(1)
        # print(maxInd)
        q_next = self.target_net(next_state).detach()  # 按照旧经验回忆.detach()为返回一个深拷贝，它没有梯度
        q_target = reward.clone()
        for i in range(BATCH_SZ):
            k=sampleList[i]
            add=γ*q_next[i].max()
            if is_ter[i]==0:
                q_target[i]+=add#double DQN,is_ter[i]==0:#如果不是终点
            ppi=math.fabs(reward[i] + add - q_eval[i]) + 0.1
            self.sumtree.update(k,math.pow(ppi, EXP_ALPHA))
            wi=self.getsqrtwi(ppi)
            q_target[i]*=wi
            q_eval[i]*=wi
        if self.learn_step_i%10000==0 and turn_i>1000:
            self.beta=min(self.beta+0.001,1)
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
        if os.path.exists("mod/target_net.pt") and os.path.exists("mod/eval_net.pt"):
            self.readMod=True
            self.target_net=torch.load( "mod/target_net.pt")#加载模型
            self.eval_net=torch.load("mod/eval_net.pt")  #
            self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=α)
rewardInd={0:-0.001,10:0.5,50:0.6,100:0.8}
def setReward(reward,is_terminal,lives,preLives,sum):#设置奖励函数函数
    if is_terminal and lives>0:
        return 1
    if preLives > lives:  # 死了命
        return -0.9999
    if reward == 0:
        return 0
    return 1 / (1 + math.exp(-reward / 100))  # +1/(1+math.exp(-step/100))
def getepsl(frame_i):
    if frame_i>1000000:
        return 0.051
    return max(1-frame_i//1000*0.01,0.1)

def train(beginShape):
    net=Agent(beginShape)
    #net.read()
    sumreward = 0
    frame_i=0
    is_terminal = True
    preLives=3
    for i in range(MEMORY_CAPACITY):#先随机取动作，填满经验池
        if is_terminal:
            stList=net.initState(env)  # 用stList储存前几步
            is_terminal = False
        action = net.readMod and net.choose_action(stList[0:OVERLAY_CNT],0) or np.random.randint(0, ENVMAX[1])
        next_state, reward, is_terminal, info = env.step(action)
        stList = net.nextState(stList, resetPic(next_state))  # 用stList储存前几步
        reward = setReward(reward, is_terminal, info["lives"], preLives, sumreward)
        net.pushRemember(stList, action, reward, is_terminal,0.1)  # 记下状态，动作，回报
        preLives = info["lives"]
    print("ready go!\n")
    # net.sumtree.print()
    # return
    presumreward=0
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
            net.pushRemember(stList, action, reward,is_terminal,net.sumtree.topMax())#记下状态，动作，回报
            if net.memory_i%5==0:  # 当步数大于脑容量的时候才开始学习，每五步学习一次
                net.learn(frame_i)
            preLives=info["lives"]

        if episode_i%10==0 :
            wInfo(str(episode_i)+"reward:"+str(sumreward/10))
            sumreward = 0
            if episode_i>0 and episode_i%1000==0:
                print("episode_i:"+str(episode_i))
                net.save()
    return net
def test(beginShape):
    net = Agent(beginShape)
    net.read()
    for episode_i in range(EPOCHS_CNT):  # 循环若干次
        stList = net.initState(env)  # 用stList储存前几步
        is_terminal = False
        while (not is_terminal):
            env.render()
            action = net.choose_action(stList[0:OVERLAY_CNT],0.1)  # 采用ϵ贪婪策略选取一个动作
            next_state, reward, is_terminal, info = env.step(action)
            stList = net.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            time.sleep(0.01)
shape = resetPic(env.reset()).shape# 重新开始游戏,(210, 160, 3)
initInfo("begin!!")
if platform.system() =="Linux":
    net=train(shape)
    net.save()
if platform.system() == "Windows":
    test(shape)

