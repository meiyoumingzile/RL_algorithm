import math
import random
import time
import platform

# import pynvml
import torchvision
import cv2
import gym
import os
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.nn.functional as F#激活函数
from snake_v0 import CC
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

def initDuoka():
    # rank: 多机多卡时代表某一台机器，单机多卡时代表某一块GPU
    # world_size: 多机多卡时代表有几台机器，单机多卡时代表有几块GPU
    # local_rank: 多机多卡时代表某一块GPU， 单机多卡时代表某一块GPU
    torch.multiprocessing.set_start_method('spawn')
    rank = 0
    local_rank = 0
    world_size = 4

    torch.distributed.init_process_group(backend="nccl",  rank=rank, world_size=world_size) # 注：这里的backend表示通信后端，非常多，如NCCL、MPI、Gloo，最快的应该还是nccl
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()
    device = torch.device('cuda:{}'.format(local_rank))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", )  # 初始化进程组
    torch.dist.barrier()



BATCH_SZ=10
EPOCHS_CNT=1000000
MEMORY_CAPACITY=10
CLIP_EPSL=0.1
OVERLAY_CNT=1#针对此游戏的叠帧操作
env=CC()#吃豆人#MsPacman-ram-v0
ENVMAX=[1,4]
# print(env.spaces())
γ=0.97#reward_decay
α=0.00001#学习率learning_rate,也叫lr
PATH="/home/hz/Z_BING333/RL/DNQ/mygame/snake_v0/"
# PATH="/home/hw1/syb/rl/DNQ/mygame/snake_v0/"
# pynvml.nvmlInit()
print(platform.system())
parser = argparse.ArgumentParser()
parser.add_argument('--cudaID', type=int, default=0)

if platform.system() == "Linux":
    args = parser.parse_args()
    print("显卡"+str(args.cudaID))
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"#str(args.cudaID)
if platform.system() =="Windows":
    BATCH_SZ = 1
    MEMORY_CAPACITY =10
    PATH = "D:\code\python\ReinforcementLearning1\DNQ\mygame\snake_v0/"
def initInfo(s,f=PATH+"info.txt"):
    if os.path.exists(f):
        os.remove(f)
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def wInfo(s,f=PATH+"info.txt"):
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")

def resetPic(pic):
    # pic=np.array(pic).flatten()
    pic=np.zeros((24))
    headPos=env.head.pos
    k=0
    for i in range(-2,3):
        for j in range(-2, 3):
            x = headPos[0] + i
            y = headPos[1] + j
            if (i!=0 or j!=0) and x>0 and y>0 and x<=env.SZ+1 and y<=env.SZ+1:
                pic[k]=(env.board[x][y]==0 or env.board[x][y]==-1)
                k+=1

    a=np.array([abs(headPos[0]-env.foodList[0][0]),abs(headPos[1]-env.foodList[0][1]),abs(headPos[0]-env.foodList[1][0]),abs(headPos[1]-env.foodList[1][1]),
                env.stepCnt,env.stepCntUp-env.nowstepCnt
                ])
    pic=np.concatenate((pic, a), axis=0)
    return np.expand_dims(pic,axis=0)
def resetPic1(pic):
    # pic=np.array(pic).flatten()
    pic=np.zeros((24))
    headPos=env.head.pos
    k=0
    for i in range(-2,3):
        for j in range(-2, 3):
            x = headPos[0] + i
            y = headPos[1] + j
            if (i!=0 or j!=0) and x>0 and y>0 and x<=env.SZ+1 and y<=env.SZ+1:
                pic[k]=(env.board[x][y]==0 or env.board[x][y]==-1)
                k+=1

    a=np.array([env.head.pos[0],env.head.pos[1],env.foodList[0][0],env.foodList[0][1],env.foodList[1][0],env.foodList[1][1],
                env.stepCnt
                ])
    pic=np.concatenate((pic, a), axis=0)
    return np.expand_dims(pic,axis=0)
class NET(nn.Module):#演员网络
    def __init__(self,kind):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=OVERLAY_CNT, out_channels=1, kernel_size=(1,1), stride=(1,1)),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(2, 2),padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),

        )
        in_fea=30
        hdim=128
        # self.lstm=nn.LSTM(3,64,2)
        self.fc1 =nn.Sequential(
            nn.Linear(in_features=in_fea, out_features=512,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=hdim,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hdim, out_features=ENVMAX[1],bias=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=in_fea, out_features=512, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=hdim, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hdim, out_features=1, bias=True)
        )
        self.fc=[self.fc1,self.fc2]
        for con in self.fc1:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)
        for con in self.fc2:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)
    def forward(self,x,k):#重写前向传播算法，x是张量
        x=x.cuda()
        # x=self.conv(x)
        # x = x.view(x.size(0), -1)
        # x=x[0][0]
        # print(x.shape)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        x=self.fc[k](x)
        return x
class Agent(object):#智能体，蒙特卡洛策略梯度算法
    saveMask=[]
    def __init__(self,beginShape):
        self.actor_net = NET(0)#初始化两个网络，target and training
        self.actor_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.actor_net)
        self.critic_net =self.actor_net

        self.net=(self.actor_net,self.critic_net)
        self.actor_net.train()
        self.critic_net.train()

        self.beginShape=beginShape

        self.learn_step_i = 0  # count the steps of learning process
        self.optimizer=torch.optim.Adam(self.actor_net.parameters(), lr=α)


        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),np.zeros((MEMORY_CAPACITY,OVERLAY_CNT)+beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1))]

    def initState(self,env):#
        state = resetPic(env.reset())  # 重新开始游戏,(210, 160)
        stList=np.zeros((OVERLAY_CNT+1,)+state.shape)
        # li[OVERLAY_CNT]=state
        for i in range(0,OVERLAY_CNT):
            action = np.random.randint(0,ENVMAX[1])
            state, reward, is_terminal = env.step(action)
            stList[i] = resetPic(state)
        return stList
    def nextState(self,stList,state):
        for i in range(OVERLAY_CNT,0,-1):
            stList[i] = stList[i-1]
        stList[0] = state
        return stList
    def choose_action(self, x,epsl=0):  # 按照当前状态，选取概率最大的动作
        if np.random.uniform(0,1)>epsl:
            x=np.expand_dims(x, axis=0)#加一个维度
            # print(x)
            x = self.actor_net.forward(torch.FloatTensor(x),0)
            # infcnt = 0
            # for i in range(len(env.fp)):
            #     act = env.fp[i]
            #     pos = (env.head.pos[0] + act[0], env.head.pos[1] + act[1])
            #     if env.board[pos[0]][pos[1]] > 0:
            #         x[0][i] = -math.inf
            #         infcnt += 1
            # if infcnt == len(env.fp):
            #     for i in range(len(env.fp)):
            #         x[0][i] = 1
            # self.saveMask=x
            prob = F.softmax(x, dim=1).cpu()
            # print(x,prob[0].detach().numpy())
            act=np.random.choice(a=ENVMAX[1], p=prob[0].detach().numpy())
            return act,prob[0][act]
        return np.random.randint(0,ENVMAX[1]),epsl/ENVMAX[1]#随机选取动作
    def pushRemember(self,state,act,reward,preRate,ister):#原本是DQN的方法的结构，但这里用它当作缓存数据
        t=self.memory_i%MEMORY_CAPACITY

        self.memoryQue[0][t] = state[1:OVERLAY_CNT+1]
        self.memoryQue[1][t] = state[0:OVERLAY_CNT]
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(preRate.detach().numpy())
        self.memoryQue[5][t] = np.array(ister)
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
        act = torch.LongTensor(self.memoryQue[2][t].astype(int)).cuda().unsqueeze(0)
        reward = torch.FloatTensor(self.memoryQue[3][t]).cuda()
        preRate = torch.FloatTensor(self.memoryQue[4][t]).cuda()
        ister = torch.FloatTensor(self.memoryQue[5][t]).cuda()

        v = (self.critic_net.forward(state,1).squeeze(dim=1), self.critic_net.forward(next_state,1).squeeze(dim=1))
        cri_loss = F.mse_loss(reward + γ * v[1], v[0])  # 用tderror的方差来算


        delta = reward + γ * v[1]-v[0] #tderro也是优势函数,根据具体问题可以在is_ter==True时阻断
        delta = delta.cpu().detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = γ * advantage + delta_t
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.Tensor(advantage_list).cuda()

        nowRate = self.actor_net.forward(state,0)
        nowRate= F.softmax(nowRate, dim=1)
        nowRate=   nowRate.gather(1, act)
        ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))#计算p1/p2防止精度问题
        # print(ratio)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSL, 1 + CLIP_EPSL) * advantage#clamp(a,l,r)代表把x限定在[l,r]之间
        actor_loss=-torch.min(surr1, surr2).mean()
        # self.optimizer.zero_grad()#0代表演员，1代表评论家
        # actor_loss.backward()
        # self.optimizer.step()
        cri_loss=cri_loss+actor_loss
        self.optimizer.zero_grad()  # 0代表演员，1代表评论家
        cri_loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.actor_net, PATH+"mod/actor_net.pt")  #
        torch.save(self.critic_net, PATH+"mod/critic_net.pt")  #
    def read(self):
        if os.path.exists(PATH+"mod/actor_net.pt") and os.path.exists(PATH+"mod/critic_net.pt") :
            self.actor_net=torch.load(PATH+"mod/actor_net.pt")  #
            self.critic_net = torch.load(PATH+"mod/critic_net.pt")  #
            self.optimizer=torch.optim.Adam(self.actor_net.parameters(), lr=α)
rewardInd={0:-0.05,10:0.5,50:0.6,100:0.7,200:0.8,300:0.9}
preDis=100000
def setReward(reward,is_terminal):#设置奖励函数函数
    global preDis
    if is_terminal :
        if env.emptyCnt>0:
            return -33
        return 33
    if reward==0:
        dis = env.getDisFromFood()
        ans=dis/40
        return ans
    return 33

def train(beginShape, cartPole_util=None):
    agent=Agent(beginShape)
    # agent.read()
    sumreward = 0
    for episode_i in range(EPOCHS_CNT):#循环若干次
        is_terminal=False
        stList = agent.initState(env)
        frame_i=0
        while (not is_terminal):
            # env.render()
            action,rate = agent.choose_action(stList[0:OVERLAY_CNT])
            # print(action)
            next_state, reward, is_terminal = env.step(action)
            stList = agent.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            sumreward += reward
            frame_i+=1
            reward = setReward(reward,is_terminal)
            agent.pushRemember(stList,action,reward,rate,1-is_terminal)
            agent.learn()
        # if episode_i%10==0:
        #     # checkWD()
        if episode_i>0 and episode_i%100==0 :
            s=str(episode_i) + "reward:" + str(sumreward / 100)
            wInfo(s)
            sumreward = 0
            if episode_i%20000==0:
                wInfo("episode_i:"+str(episode_i))
                agent.save()
    return agent
shape = resetPic(env.reset()).shape# 重新开始游戏,(210, 160, 3)
initInfo("begin!!")
if platform.system() =="Linux":
    agent=train(shape)
    agent.save()
# agent=train(shape)
# agent.save()
