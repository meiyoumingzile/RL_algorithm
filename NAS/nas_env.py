import math
import random
import torch #直接包含整个包
import torch.nn as nn #
import glo_unit
from glo_unit import geoDir
from glo_unit import subNetPar
from glo_unit import rlPar
import numpy as np
import pygame
import pygame.locals
from pygame.locals import *
import sys
import time
import torch.nn.functional as F#激活函数
from torchvision import transforms
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载
import torch.optim as optim
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class NET(nn.Module):
    def reset(self):
        self.conv = nn.Sequential()
        self.convIDList=np.zeros(subNetPar.layerCnt)#conv的每个模块编号
        self.convFeaList = np.zeros(subNetPar.layerCnt)  #conv的每个模块输出的维度
        self.mat=np.zeros((subNetPar.layerCnt,subNetPar.layerCnt))#conv的每个模块编号
        self.now_fea=1
        self.preAct=rlPar.act_dim-1#记录动作id
        return self.getNetCode()

    def __init__(self,trainloader,in_fea=1):  #
        super(NET, self).__init__()
        self.trainloader = trainloader
        self.linermid_fea=128
        self.reset()
        self.example_pic,_=iter(self.trainloader).next()
        self.example_pic=self.example_pic.cuda()
        in_ch=1
        out_ch=6
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=5, stride=1, padding=1, groups=in_ch),
        #     nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,stride=1, padding=0, groups=1),
        #
        #     geoDir["pool_max2*2"],
        #     geoDir["actfun_relu"],
        #     geoDir["batchNorm2d"](6),
        #     geoDir["conv_5*5"](6, 16),
        #     geoDir["pool_max2*2"],
        # )
        # self.conv1 = nn.Sequential(
        #     geoDir["actfun_relu"],
        #     geoDir["conv_1*1"](16, 16),
        # )

        self.fc = nn.Sequential(#默认
            nn.Linear(0, self.linermid_fea),
            nn.ReLU(),
            nn.Linear(self.linermid_fea, 10)
        )


    #self.conv.add_module(name, geoDir["conv_5*5"]())

    def updateFc(self,in_fea=0):
        if in_fea==0:
            x = self.forwardConv(self.example_pic)
            in_fea=x.shape[1]
        self.fc[0] = nn.Linear(in_fea, self.linermid_fea).cuda()
    def addModule(self,act,out_fea):
        name = glo_unit.geokeys[act]
        i = len(net.conv)
        # print( name[0:4],self.now_fea, out_fea)
        if name[0:4]=="conv":
            self.conv.add_module(name, geoDir[name](self.now_fea, out_fea).cuda())
            self.now_fea = out_fea
        elif name=="batchNorm2d":
            self.conv.add_module(name, geoDir[name](self.now_fea).cuda())
        else:
            self.conv.add_module(name, geoDir[name].cuda())
        self.convFeaList[i] = self.now_fea
        self.mat[i][i] = 1
        if i > 0:
            self.mat[i - 1][i] = 1
        self.preAct=act
        self.convIDList[i] = act + 1
    def forwardConv(self,x):
        x = self.conv(x)
        return x.view(x.size(0), -1)
    def forward(self,x):#重写前向传播算法，x是张量
        x=self.forwardConv(x)
        # print(x.shape)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    def getNetCode(self):
        return np.concatenate((self.mat.flatten(), self.convIDList,self.convFeaList), axis=0)

def __train_net(model,train_loader,optimizer,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.train()
    correct=0#正确率
    avgLoss=0

    for batch_index, (d,t) in enumerate(train_loader):
        data, target = d.cuda(), t.cuda()  # 部署到device
        optimizer.zero_grad()#初始化梯度为0
        output=model(data)#训练后结果
        loss=F.cross_entropy(output,target)#计算损失,用交叉熵,默认是累计的
        loss.backward()#反向传播
        avgLoss +=loss.item()
        optimizer.step()#参数优化
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /=len(train_loader.dataset)  # 计算平均数
    Accuracy=100 * correct / len(train_loader.dataset)
    print("第{}，正确率{:.6f}  loss：{:.6f}".format(epoch,Accuracy,avgLoss))
    return Accuracy,avgLoss
    # writer.add_scalar('Accuracy', Accuracy, epoch)
    # writer.add_scalar('avgLoss', avgLoss, epoch)
def train_net(net,train_loader,optimizer):
    for epoch in range(0, subNetPar.EPOCHS_CNT):
        __train_net(net, train_loader, optimizer, epoch)


def test_net(net,test_loader):##model是模型，device是设备，test_loader是测试数据集
    net.eval()#模型验证
    correct=0#正确率
    sumLoss=0#测试损失
    with torch.no_grad():#，不进行训练时
        for batch_index, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()  # 部署到device
            output=net(data)#训练后结果
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
    return correct/len(test_loader.dataset)




pipeline =transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))])

train_set=datasets.MNIST(glo_unit.rootPath+"NAS/data/",train=True,download=False,transform=pipeline)
test_set=datasets.MNIST(glo_unit.rootPath+"NAS/data/",train=False,download=False,transform=pipeline)
train_loader = DataLoader(train_set,batch_size=subNetPar.BATCH_SZ,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
test_loader = DataLoader(test_set,batch_size=subNetPar.BATCH_SZ,shuffle=False)
net=NET(train_loader).cuda()
optimizer = optim.Adam(net.parameters(), lr = subNetPar.lr)

# net.updateFc()
# train_net(net,train_loader,optimizer)
def step(act,out_fea):
    net.addModule(act,out_fea)
    net.updateFc()
    print(net.conv)
    # train_net(net,train_loader,optimizer)
    return net.getNetCode(),test_net(net,test_loader),len( net.conv)==subNetPar.layerCnt
reset=net.reset#强化学习环境接口

# next_state,reward,is_ter=step(2,111)
# print(next_state.shape)
# print(reward)
# next_state,reward,is_ter=step(10)
