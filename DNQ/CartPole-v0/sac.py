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
from torch.distributions import Normal
from torchvision import transforms#图像处理的工具类
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BATCH_SZ=128
EPOCHS_CNT=50000
MEMORY_CAPACITY=200
OVERLAY_CNT=1#针对此游戏的叠帧操作
env=gym.make('CartPole-v0').unwrapped#吃豆人#MsPacman-ram-v0
ENVMAX=[env.observation_space.shape[0],2]#状态维度和动作维度


class MyPar():#充当静态类
    log_std_range=[-20,2]
    gamma = 0.90  # reward_decay
    lr = 0.001  # 学习率learning_rate,也叫lr
    halpha=0.2
    tau = 0.01
    inf =1e-8
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
par=MyPar()


if platform.system() == "Linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if platform.system() =="Windows":
    BATCH_SZ = 1

def cv_show( mat):
    cv2.imshow("name", mat)
    cv2.waitKey(0)
class ANetwork(nn.Module):#演员网络,actor
    def __init__(self,state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self,x):#重写前向传播算法，x是张量
        x=x.cuda()
        # print(x.shape)
        x = self.fc(x)
        return x
    # def sample(self, x):#如果是连续动作空间就输出分布
    #     mean, log_std = self.forward(x)
    #     normal = Normal(mean, log_std.exp())
    #     x = normal.rsample()#表示现在N(0,1)上采样，在减去mean处于std
    #     action = torch.tanh(x)
    #     log_prob = normal.log_prob(x)#计算正在分布对应的概率，然后取对数
    #     log_prob -= torch.log(1 - action.pow(2) + 1e-6)
    #     log_prob = log_prob.sum(1, keepdim=True)
    #     return action, log_prob, mean, log_std
class QNetwork(nn.Module):#相当于评论家网络,估计Q
    def __init__(self,state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Sequential(#Qnetwork有两个网络，它们完全一样，思路是sac2论文的,如果是连续空间，把状态和动作合并输入输出1个值，如果离散空间，输出action_dim个值
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,action_dim),
        )
        self.fc2 = nn.Sequential(  # Qnetwork有两个网络，它们完全一样，思路是sac2论文的
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.apply(self.initw)
    def initw(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    def forward(self,x):#
        # x = torch.cat([state.cuda(), action.cuda()], 1)
        x=x.cuda()
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1,x2


class ReplayBuffer:
    def __init__(self, beginShape):
        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.memoryQue = [np.zeros((MEMORY_CAPACITY, OVERLAY_CNT) + beginShape),
                          np.zeros((MEMORY_CAPACITY, OVERLAY_CNT) + beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)), np.zeros((MEMORY_CAPACITY, 1)),
                          np.zeros((MEMORY_CAPACITY, 1))]

    def pushRemember(self, state, next_state, act, reward):  # 原本是DQN的方法的结构，但这里用它当作缓存数据
        t = self.memory_i % MEMORY_CAPACITY
        self.memoryQue[0][t] = np.expand_dims(state, axis=0)
        self.memoryQue[1][t] = np.expand_dims(next_state, axis=0)
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memory_i = self.memory_i + 1

    def __len__(self):
        return len(self.memoryQue)
beginShape = env.reset().shape# 重新开始游戏,(210, 160, 3)
buffer=ReplayBuffer(beginShape)
class Agent(object):#智能体，蒙特卡洛策略梯度算法
    def __init__(self,beginShape):
        self.actor_net = ANetwork(ENVMAX[0],ENVMAX[1]).cuda()#初始化两个网络，target and training
        self.critic_net,self.critic_net_target = QNetwork(ENVMAX[0],ENVMAX[1]).cuda(),QNetwork(ENVMAX[0],ENVMAX[1]).cuda()

        self.net=(self.actor_net,self.critic_net)
        self.actor_net.train()
        self.critic_net.train()
        self.beginShape=beginShape
        self.learn_step_i = 0  # count the steps of learning process
        self.optimizer=(torch.optim.Adam(self.actor_net.parameters(), lr=par.lr),torch.optim.Adam(self.critic_net.parameters(), lr=par.lr))
        
        #动态
        self.target_entropy = -torch.prod(torch.Tensor([ENVMAX[1]]).cuda()).item()
        print('entropy：', self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True,device=par.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=par.lr)#利用优化器动态调整log_alpha
        self.halpha=par.halpha

    def soft_update(self,net, source):
        for target_param, param in zip(net.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - par.tau) + param.data * par.tau)

    def hard_update(self,net, source):
        for target_param, param in zip(net.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    def choose_action(self, x):  # 按照当前状态，选取概率最大的动作
        prob = self.actor_net.forward(x)  # 采样一次
        prob = F.softmax(prob, dim=1)
        act = torch.multinomial(prob, 1, replacement=False, out=None)  # 选取动作
        prob = prob.gather(1, act)
        return act, prob
    def choose_actTensor(self, x):  # 按照当前状态，选取概率最大的动作
        prob = self.actor_net.forward(x)  # 采样一次
        prob = F.softmax(prob, dim=1)
        return prob, torch.log(prob+par.inf)
    def learn(self):#学习,训练ac算法2个网络
        self.learn_step_i=(self.learn_step_i+1)%10000
        t = np.random.choice(MEMORY_CAPACITY, BATCH_SZ)
        # t[0] = (buffer.memory_i - 1) % MEMORY_CAPACITY
        state = torch.FloatTensor( buffer.memoryQue[0][t]).squeeze(dim=1)  # state
        next_state = torch.FloatTensor( buffer.memoryQue[1][t]).squeeze(dim=1)
        act = torch.LongTensor( buffer.memoryQue[2][t].astype(int)).cuda()
        reward = torch.FloatTensor( buffer.memoryQue[3][t]).cuda()
        with torch.no_grad():
            next_prob,next_logprob = self.choose_actTensor(next_state)  # 输入batch_size*shape
            # print(next_state.shape,next_act.shape)
            # x = torch.cat([next_state.cuda(), next_act.cuda()], 1)
            # print(x.shape)
            next_qx1,next_qx2=self.critic_net_target.forward(next_state)

            min_nextq = next_prob*(torch.min(next_qx1, next_qx2) - self.halpha * next_logprob)#Q(next_state,next_act)-halpha*logp
            min_nextq=min_nextq.sum(dim=1).unsqueeze(1)
            # print(reward.shape, min_nextq.shape)
            q_value = reward +  par.gamma * min_nextq#由贝尔曼期望方程：Q(s,a)=r+gamma*Q(s(t+1)，a(t+1))
        # print(act.shape, next_prob.shape, next_act.shape)
        qx1, qx2 = self.critic_net(state)
        qx1=qx1.gather(1,act)
        qx2 = qx2.gather(1, act)

        q1_loss = F.mse_loss(qx1, q_value)  #
        q2_loss = F.mse_loss(qx2, q_value)  #
        q_loss = q1_loss + q2_loss
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        prob,logprob = self.choose_actTensor(state)#
        qx1,qx2=self.critic_net(state)
        minqx = torch.min(qx1,qx2)
        actor_loss = (self.halpha * logprob - minqx)*prob#根据KL散度化简得来
        actor_loss=actor_loss.sum(dim=1).mean()
        self.optimizer[0].zero_grad()
        actor_loss.backward()
        self.optimizer[0].step()


        alpha_loss = -(self.log_alpha * (torch.log(prob) + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.halpha = self.log_alpha.exp()

        # if self.learn_step_i % 5 == 0:
        self.soft_update(self.critic_net_target, self.critic_net)

    def save(self):
        torch.save(self.actor_net, "mod/actor_net.pt")  #
        torch.save(self.critic_net, "mod/critic_net.pt")  #
        torch.save(self.critic_net_target, "mod/critic_net_target.pt")  #
    def read(self):
        if os.path.exists("mod/actor_net.pt") and os.path.exists("mod/critic_net.pt") :
            self.actor_net=torch.load("mod/actor_net.pt")  #
            self.critic_net = torch.load("mod/critic_net.pt")  #
            self.critic_net_target = torch.load("mod/critic_net_target.pt")  #
            self.optimizer=(torch.optim.Adam(self.actor_net.parameters(), lr=par.lr),torch.optim.Adam(self.critic_net.parameters(), lr=par.lr))

def setReward(reward):#设置奖励函数函数
    x, x_dot, theta, theta_dot = reward
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    return r1 + r2

def train(beginShape):
    agent=Agent(beginShape)
    # net.read()
    sumreward = 0
    sumreward1=0
    for episode_i in range(EPOCHS_CNT):#循环若干次
        is_terminal=False
        state = env.reset()
        while (not is_terminal):
            # env.render()
            x = torch.FloatTensor(state).unsqueeze(0).cuda()  # 加一个维度
            action,prob = agent.choose_action(x)
            action=action[0][0].cpu().detach().numpy()

            next_state, reward, is_terminal, info = env.step(action)
            sumreward += reward
            reward = setReward(next_state)
            sumreward1+=reward
            buffer.pushRemember(state,next_state,action,reward)
            if buffer.memory_i>MEMORY_CAPACITY and buffer.memory_i%1==0:
                agent.learn()
            state = next_state
        if episode_i%10==0 :
            print(str(episode_i)+"reward:"+str(sumreward/10)+"  "+str(sumreward1/10))
            sumreward = 0
            sumreward1=0
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
            print(setReward(state))
            time.sleep(1)

agent=train(beginShape)
# agent.save()
# test_mod(shape)

