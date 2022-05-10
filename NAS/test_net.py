import os
import platform

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

BATCH_SZ=30#由于数据量很大，要分批读入内存，代表一批有多少数据
DEVICE = torch.device("cuda" if  torch.cuda.is_available() else "cpu") #设置训练设备，torch.device是判断电脑设备
EPOCHS_CNT=40 #训练的轮次
# writer=SummaryWriter(log_dir="logs", flush_secs=60)#指定存储目录和刷新间隔

x=torch.ones([1,1,10,10])
print(x.shape)
conv=nn.Conv2d(in_channels=1, out_channels=6, kernel_size=7, stride=1, padding=3, groups=1)
x=conv(x)
print(x.shape)