import os
import platform
import glo_unit
from glo_unit import geoDir
from glo_unit import subNetPar
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.optim as optim
import torch.nn.functional as F#激活函数
from sklearn import decomposition
from torchvision import transforms
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载
#1.以上是加载库

BATCH_SZ=100#由于数据量很大，要分批读入内存，代表一批有多少数据
EPOCHS_CNT=20 #训练的轮次

class NET(nn.Module):
    def __init__(self):  #
        super(NET, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(6),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(#默认
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self,x):#重写前向传播算法，x是张量
        x=self.conv(x)
        x=x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_model(model,train_loader,optimizer,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.train()
    correct=0#正确率
    avgLoss=0

    for batch_index, (d,t) in enumerate(train_loader):
        data,target=d.cuda(),t.cuda()#部署到device
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
    # writer.add_scalar('Accuracy', Accuracy, epoch)
    # writer.add_scalar('avgLoss', avgLoss, epoch)

# 6.以上是定义训练方法
def test_model(model,test_loader):##model是模型，device是设备，test_loader是测试数据集
    model.eval()#模型验证
    correct=0#正确率
    sumLoss=0#测试损失
    with torch.no_grad():#，不进行训练时
        for batch_index, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()  # 部署到device
            output=model(data)#训练后结果
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
        print("正确率：{:.6f}".format(100*correct/len(test_loader.dataset)))            #输出


pipeline =transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))])
# print(torch.cuda.is_available())

# train_set=datasets.MNIST(glo_unit.rootPath+"NAS/data/",train=True,download=False,transform=pipeline)
# test_set=datasets.MNIST(glo_unit.rootPath+"NAS/data/",train=False,download=False,transform=pipeline)
# train_loader = DataLoader(train_set,batch_size=BATCH_SZ,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
# test_loader = DataLoader(test_set,batch_size=BATCH_SZ,shuffle=False)
# net=NET().cuda()
#
# optimizer = optim.Adam(net.parameters(), lr = 0.01)
#
# for epoch in range(0,EPOCHS_CNT):
#     train_model(net,train_loader,optimizer,epoch)
# torch.save(net, "mod_RL/lenet.pt")#保存模型pt || pth
# model = torch.load("mod_RL/lenet.pt")#加载神经网络
# for epoch in range(0, EPOCHS_CNT):
#     test_model(model,test_loader)
# #97,1精度