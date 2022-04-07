import os
import platform

from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
from PIL import Image
import torchvision
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.optim as optim
import torch.nn.functional as F#激活函数

from torchvision import transforms
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载
#1.以上是加载库

BATCH_SZ=20#由于数据量很大，要分批读入内存，代表一批有多少数据
DEVICE = torch.device("cuda" if  torch.cuda.is_available() else "cpu") #设置训练设备，torch.device是判断电脑设备
EPOCHS_CNT=40 #训练的轮次
# writer=SummaryWriter(log_dir="logs",flush_secs=60)#指定存储目录和刷新间隔
class MyDataset(torch.utils.data.Dataset):#加载自己数据集，以标注而非文本
    def getDataFile(self):
        sysstr = platform.system()
        if (sysstr == "Windows"):
            return "../data/FMD3/"
        elif (sysstr == "Linux"):
            return "/home/hz/YZ/data/FMD3/"
        else:
            return "/home/hz/YZ/data/FMD3/"
    def getModelFile(self):
        if (platform.system() == "Windows"):
            return "../myModel/resnet152.pth"
        return '/home/hz/YZ/models/resnet152.pth'#"Linux"下
    def __init__(self, train=True,transform=None,target_transform=None):  #
        super(MyDataset, self).__init__()
        self.transform = transform
        lines = []
        self.rootPath = self.getDataFile()
        self.imgPath = self.rootPath+"train/"
        self.testPath = self.rootPath+"test/"
        if not os.path.isdir(self.rootPath):
            print(self.rootPath + ' does not exist!')
        self.fileList = []
        self.kindList = []
        self.kindInd = {}
        self.nowPath=train and self.imgPath or self.testPath
        for kind in os.listdir(self.nowPath):
            self.kindList.append(kind)
            self.kindInd[kind] = len(self.kindList) - 1
            file = self.nowPath + kind + "/"
            for pic in os.listdir(file):
                if os.path.splitext(pic)[-1]==".jpg":
                    self.fileList.append((pic, kind))
        # print(len(self.fileList))

    def __getitem__(self, idx):  # 这里应该返回正确数据和其标注。在遍历data_loader时需要用
        image =Image.open(self.nowPath + self.fileList[idx][1] + "/" + self.fileList[idx][0]).convert('RGB')  # use skitimage
        label = self.kindInd[self.fileList[idx][1]]
        if not self.transform[0]:
            return None
        image= self.transform[0](image)
        if self.transform[1]:
            image = self.transform[1](image)
        return image,label

    def __len__(self):#返回长度
        return len(self.fileList)

class BasicBlock(nn.Module):
    def __init__(self, insz, outsz, stride=1,downsampling=False):#
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=insz, out_channels=outsz, kernel_size=3, stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(outsz),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=outsz, out_channels=outsz, kernel_size=3, stride=1,padding=1,bias=False),
        )
        self.shortcut = nn.Sequential()
        if downsampling:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=insz, out_channels=outsz, kernel_size=1, stride=stride,bias=False),
            )
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        out = self.conv(x)
        out+=self.shortcut(x)
        out = self.relu(out)
        return out
class RESNET(nn.Module):#必须继承nn.Module
    def __init__(self,blocks):#其中，numblocks是一个列表代表每层网络的块数
        super(RESNET,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer=nn.Sequential(
            self.make_layer(64, 64, block=blocks[0], stride=2),
            self.make_layer(64, 128, block=blocks[1], stride=2),
            self.make_layer(128, 256, block=blocks[2], stride=2),
            self.make_layer(256, 512, block=blocks[3], stride=2),
            nn.AvgPool2d(4)
        )
        self.fc = nn.Linear(4608, 4)

    def forward(self,x):#重写前向传播算法，x是张量
        #3*224*224
        x=self.conv1(x)
        x = self.layer(x)
        x=x.view(x.size(0),-1)
        # print(x.shape)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def make_layer(self, insz, outsz, block, stride):
        layers = [BasicBlock(insz, outsz,stride, True)]
        for i in range(1,block):
            layers.append(BasicBlock(outsz, outsz, 1,False))#输出维度为expansion*outsz
        return nn.Sequential(*layers)
def adjustOpt(optimizer,epoch):
    p = 0.9
    if epoch%4==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= p

def train_model(model,device,train_loader,optimizer,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.train()
    correct=0#正确率
    avgLoss=0
    adjustOpt(optimizer, epoch)
    for batch_index, (d,t) in enumerate(train_loader):
        data,target=d.to(device),t.to(device)#部署到device
        optimizer.zero_grad()#初始化梯度为0
        print(data.shape)
        output=model(data)#训练后结果
        # print(output.shape)
        loss=F.cross_entropy(output,target)#计算损失,用交叉熵,默认是累计的
        loss.backward()#反向传播
        avgLoss +=loss.item()
        optimizer.step()#参数优化
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /=len(train_loader.dataset)  # 计算平均数
    Accuracy=100 * correct / len(train_loader.dataset)
    print("第{}，正确率{:.6f}  loss：{:.6f}".format(epoch,Accuracy,avgLoss))

# 6.以上是定义训练方法
def test_model(model,device,test_loader):##model是模型，device是设备，test_loader是测试数据集
    model.eval()#模型验证
    correct=0#正确率
    sumLoss=0#测试损失
    with torch.no_grad():#，不进行训练时
        for batch_index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)  # 部署到device
            output=model(data)#训练后结果
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
        print("正确率：{:.6f}".format(100*correct/len(test_loader.dataset)))            #输出


pipeline = (
transforms.Compose([#两个预处理
    transforms.Resize([224,224]),
    transforms.ToTensor(),#把图像转变成张量
]),
transforms.Compose([#定义一个图像处理的方法,做变换，类似opencv
    transforms.Normalize([ 0.61517555 , 0.34357786, -0.12583818], [0.20161158, 0.2345495 , 0.19751817])
]),
)
# print(torch.cuda.is_available())
train_set=MyDataset(train=True,transform=pipeline)
test_set=MyDataset(train=False,transform=pipeline)
train_loader = DataLoader(train_set,batch_size=BATCH_SZ,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
test_loader = DataLoader(test_set,batch_size=BATCH_SZ,shuffle=False)

model=RESNET([3, 4, 6, 3]).to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

for epoch in range(0,EPOCHS_CNT):
    train_model(model,DEVICE,train_loader,optimizer,epoch)

for epoch in range(0, EPOCHS_CNT):
    test_model(model,DEVICE,test_loader)
