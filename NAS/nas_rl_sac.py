import math
import random
import time
import platform

from torch.distributions import Normal

import guideNet
import os
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.nn.functional as F#激活函数
import glo_unit
from glo_unit import geoDir
from glo_unit import subNetPar
from glo_unit import convKindInd
from glo_unit import rlPar
import nas_env
import argparse

PATH="/home/hz/Z_BING333/RL/NAS/"


parser = argparse.ArgumentParser()
parser.add_argument('--cudaID', type=int, default=rlPar.cudaID)

if platform.system() == "Linux":
    args = parser.parse_args()
    print("显卡"+str(args.cudaID))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cudaID)#str(args.cudaID)
# if platform.system() =="Windows":
#     BATCH_SZ = 1
#     MEMORY_CAPACITY =10
#     PATH = "D:\code\python\ReinforcementLearning1\DNQ\mygame\snake_v0/"
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
    return pic

class ReplayBuffer:
    def __init__(self, beginShape):
        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.rangeList=torch.IntTensor([i for i in range(rlPar.MEMORY_CAPACITY)])
        # print(self.rangeList)
        self.memoryQue = [np.zeros((rlPar.MEMORY_CAPACITY, ) + beginShape),
                          np.zeros((rlPar.MEMORY_CAPACITY, ) + beginShape),
                          np.zeros((rlPar.MEMORY_CAPACITY, 1)),
                          np.zeros((rlPar.MEMORY_CAPACITY, 1)),
                          np.zeros((rlPar.MEMORY_CAPACITY, rlPar.act_dim)),
                          np.zeros((rlPar.MEMORY_CAPACITY, rlPar.act_dim)),
                          np.zeros((rlPar.MEMORY_CAPACITY, 1))]

    def pushRemember(self, state, next_state,act, reward, probList,nextpList, ister):  # 原本是DQN的方法的结构，但这里用它当作缓存数据
        t = self.memory_i % rlPar.MEMORY_CAPACITY
        self.memoryQue[0][t] = state
        self.memoryQue[1][t] = next_state
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(probList.detach().numpy())
        self.memoryQue[5][t] = np.array(nextpList.detach().numpy())
        self.memoryQue[6][t] = np.array(ister)
        self.memory_i = (self.memory_i + 1)

    def __len__(self):
        return len(self.memoryQue)
beginShape = resetPic(nas_env.reset()).shape# 重新开始游戏,(210, 160, 3)
buffer=ReplayBuffer(beginShape)

class Public_Network(nn.Module):#演员网络,actor
    def __init__(self, state_dim, out_dim):
        super().__init__()
        # self.preModule=torch.load("mod_RL/lenet.pt")#加载神经网络
        self.fc = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=out_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):  # 重写前向传播算法，x是张量
        return self.fc(x)

class ANetwork(nn.Module):#演员网络,actor
    def __init__(self,preNet,in_dim, action_dim):#preNet就是Public_Network
        super().__init__()
        hdim = 128
        self.preNet=preNet
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hdim),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(hdim, action_dim)
        self.mean_linear = nn.Linear(hdim, 1)
        self.log_std_linear = nn.Linear(hdim, 1)
    def forward(self,x):#重写前向传播算法，x是张量
        x = x.cuda(rlPar.cudaID)
        # print(x.shape)
        x = self.preNet(x)
        # print(x.shape)
        x = self.fc1(x)
        y = self.fc2(x)
        return y,x

    def sample(self, x, epsilon=1e-6):
        mean=self.mean_linear(x)
        log_std=self.log_std_linear(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action*(rlPar.feaClip[1]-rlPar.feaClip[0])/2+(rlPar.feaClip[1]+rlPar.feaClip[0])/2, log_prob, mean, log_std
class QNetwork(nn.Module):#相当于评论家网络,估计Q
    def __init__(self,state_dim, action_dim):
        super().__init__()
        hdim=128
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=hdim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=action_dim, bias=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=hdim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=action_dim, bias=True)
        )
        self.apply(self.initw)
    def initw(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    def forward(self,x):#
        x=x.cuda(rlPar.cudaID)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1,x2

class Agent(object):#智能体，蒙特卡洛策略梯度算法
    saveMask=[]
    def __init__(self,beginShape):
        self.public_net=Public_Network(rlPar.state_dim, 128).cuda(rlPar.cudaID)
        self.actor_net = ANetwork(self.public_net,128, rlPar.act_dim).cuda(rlPar.cudaID)  # 初始化两个网络，target and training
        self.critic_net, self.critic_net_target = QNetwork(rlPar.state_dim, rlPar.act_dim).cuda(rlPar.cudaID), QNetwork(rlPar.state_dim,
                                                                                                  rlPar.act_dim).cuda(rlPar.cudaID)

        self.net = (self.actor_net, self.critic_net)
        self.actor_net.train()
        self.critic_net.train()
        self.beginShape = beginShape
        self.learn_step_i = 0
        self.optimizer = (torch.optim.Adam(self.actor_net.parameters(), lr=rlPar.lr),
                          torch.optim.Adam(self.critic_net.parameters(), lr=rlPar.lr))

        # 动态
        self.target_entropy = -torch.prod(torch.Tensor([rlPar.act_dim]).cuda(rlPar.cudaID)).item()
        # print('entropy：', self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=torch.device(type='cuda', index=rlPar.cudaID))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=rlPar.lr)  # 利用优化器动态调整log_alpha
        self.halpha = rlPar.halpha


    # def nextState(self,stList,state):
    #     for i in range(OVERLAY_CNT,0,-1):
    #         stList[i] = stList[i-1]
    #     stList[0] = state
    #     return stList
    def soft_update(self,net, source):
        for target_param, param in zip(net.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - rlPar.tau) + param.data * rlPar.tau)

    def hard_update(self,net, source):
        for target_param, param in zip(net.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    def saveNowMask(self,x):

        for i in range(len(x[0])):
            if convKindInd[i]==convKindInd[nas_env.net.preAct] and i>1:
                x[0][i] = -math.inf
        return F.softmax(x, dim=1)
    def choose_action(self, x):  # 按照当前状态，选取概率最大的动作
        x = torch.FloatTensor(x).unsqueeze(dim=0).cuda(rlPar.cudaID)  # 加一个维度
        # print(x.shape)
        x,fea_x = self.actor_net.forward(x)
        # print(x)
        prob = self.saveNowMask(x).cpu()
        # print(prob)
        # print(x,prob[0].detach().numpy())
        act = np.random.choice(a=rlPar.act_dim, p=prob[0].detach().numpy())
        fea=0
        if convKindInd[act]==1:
            fea,log_prob, mean, log_std=self.actor_net.sample(fea_x)
        return act,fea, prob[0],
    def choose_actTensor(self, x,maskList):  # 按照当前状态，选取概率最大的动作
        prob,fea_x = self.actor_net.forward(x)  # 采样一次
        for i in range(len(maskList)):
            for j in range(len(maskList[i])):
                if maskList[i][j]==0:
                    prob[i][j]=-math.inf
        prob = F.softmax(prob, dim=1)

        # prob=torch.max(prob,par.inf)
        return prob, torch.log(prob+rlPar.inf)
    def learn(self):#学习,训练ac算法2个网络
        # t = np.random.choice(MEMORY_CAPACITY, BATCH_SZ)
        t=buffer.rangeList
        state = torch.FloatTensor(buffer.memoryQue[0][t]).cuda(rlPar.cudaID)  # state
        next_state = torch.FloatTensor(buffer.memoryQue[1][t]).cuda(rlPar.cudaID)
        act = torch.LongTensor(buffer.memoryQue[2][t].astype(int)).cuda(rlPar.cudaID)
        reward = torch.FloatTensor(buffer.memoryQue[3][t]).cuda(rlPar.cudaID)
        pMask = torch.FloatTensor(buffer.memoryQue[4][t]).cuda(rlPar.cudaID)
        nextpMask = torch.FloatTensor(buffer.memoryQue[5][t]).cuda(rlPar.cudaID)
        # ister = torch.FloatTensor(buffer.memoryQue[6][t]).cuda(par.cudaID)
        # print(state.shape,act.shape,pMask)

        with torch.no_grad():
            next_prob, next_logprob = self.choose_actTensor(next_state,nextpMask)  # 输入batch_size*shape
            # print(next_prob,next_logprob)
            # print(next_state.shape,next_act.shape)
            # x = torch.cat([next_state.cuda(par.cudaID), next_act.cuda(par.cudaID)], 1)
            # print(x.shape)
            next_qx1, next_qx2 = self.critic_net_target.forward(next_state)

            min_nextq = next_prob * (
                        torch.min(next_qx1, next_qx2) - self.halpha * next_logprob)  # Q(next_state,next_act)-halpha*logp
            min_nextq = min_nextq.sum(dim=1).unsqueeze(1)
            # print(reward.shape, min_nextq.shape)
            q_value = reward + rlPar.gamma * min_nextq  # 由贝尔曼期望方程：Q(s,a)=r+gamma*Q(s(t+1)，a(t+1))
            # print(act.shape, next_prob.shape, next_act.shape)
        qx1, qx2 = self.critic_net(state)
        # print(qx1,act)
        qx1 = qx1.gather(1, act)
        qx2 = qx2.gather(1, act)
        # print(qx1)
        q1_loss = F.mse_loss(qx1, q_value)  #
        q2_loss = F.mse_loss(qx2, q_value)  #
        q_loss = q1_loss + q2_loss
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        prob, logprob = self.choose_actTensor(state,pMask)  #
        # print(pMask)
        qx1, qx2 = self.critic_net(state)
        minqx = torch.min(qx1, qx2)
        actor_loss = (self.halpha * logprob - minqx) * prob  # 根据KL散度化简得来
        # print(self.halpha , logprob , minqx)
        actor_loss = actor_loss.sum(dim=1).mean()
        # print(actor_loss)
        self.optimizer[0].zero_grad()
        actor_loss.backward()
        self.optimizer[0].step()

        alpha_loss = -(self.log_alpha * (logprob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.halpha = self.log_alpha.exp()

        # if self.learn_step_i % 5 == 0:
        self.soft_update(self.critic_net_target, self.critic_net)

    def save(self):
        torch.save(self.actor_net, "mod_RL/sac_actor_net.pt")  #
        torch.save(self.critic_net, "mod_RL/sac_critic_net.pt")  #
        torch.save(self.critic_net_target, "mod_RL/sac_critic_net_target.pt")  #

    def read(self):
        if os.path.exists("mod_RL/sac_actor_net.pt") and os.path.exists("mod_RL/sac_critic_net.pt"):
            self.actor_net = torch.load("mod_RL/sac_actor_net.pt")  #
            self.critic_net = torch.load("mod_RL/sac_critic_net.pt")  #
            self.critic_net_target = torch.load("mod_RL/sac_critic_net_target.pt")  #
            self.optimizer = (torch.optim.Adam(self.actor_net.parameters(), lr=par.lr),
                              torch.optim.Adam(self.critic_net.parameters(), lr=par.lr))


def setReward(reward,is_terminal):#设置奖励函数函数

    return reward

def train(beginShape):
    agent=Agent(beginShape)
    # agent.read()
    sumreward = 0
    for episode_i in range(rlPar.EPOCHS_CNT):#循环若干次
        is_terminal=False
        state=nas_env.reset()

        frame_i=0
        while (not is_terminal):
            # env.render()
            action,fea,prob_mask = agent.choose_action(state)
            # print(action,fea,len(prob_mask))
            # print(action)
            next_state, reward, is_terminal = nas_env.step(action,int(fea))
            print(reward)
            sumreward += reward
            frame_i+=1
            next_prob_mask=agent.saveNowMask(torch.ones([1,rlPar.act_dim]))
            reward = setReward(reward,is_terminal)
            buffer.pushRemember(state,next_state,action,reward,prob_mask,next_prob_mask,1-is_terminal)
            if buffer.memory_i%1==0 and buffer.memory_i>rlPar.MEMORY_CAPACITY:
                agent.learn()
            state=next_state
        # if episode_i%10==0:
        #     # checkWD()
        if episode_i>0 and episode_i%1==0 :
            s=str(episode_i) + "reward:" + str(sumreward / 1)
            wInfo(s)
            sumreward = 0
            if episode_i%2000==0:
                wInfo("episode_i:"+str(episode_i))
                agent.save()
    return agent

initInfo("begin!!")
if platform.system() =="Linux":
    agent=train(beginShape)
    agent.save()
# agent=train(shape)
# agent.save()
