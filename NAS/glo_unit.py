import platform
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.optim as optim
import torch.nn.functional as F#激活函数

from sklearn import decomposition
from torchvision import transforms
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载

rootPath="/home/hz/Z_BING333/RL/"
print(platform.system())
if platform.system() == "Windows":
    rootPath = "D:/code/python/ReinforcementLearning1/"
elif platform.system() == "Linux":
    rootPath="/home/hz/Z_BING333/RL/"

class SubNetPar():
    BATCH_SZ=200
    EPOCHS_CNT=5
    lr=0.01
    layerCnt=5
    # maxDeep=6
subNetPar=SubNetPar()

class RLPar():
    BATCH_SZ = 32
    EPOCHS_CNT = 10000
    MEMORY_CAPACITY = 2
    OVERLAY_CNT = 1  # 针对此游戏的叠帧个数

    feaClip=[60,600]
    log_std_range = [-20, 2]
    gamma = 0.97  # reward_decay
    lr = 0.00001  # 学习率learning_rate,也叫lr
    halpha = 0.4
    tau = 0.1
    inf = 1e-9
    CLIP_EPSL = 0.1
    state_dim = 35
    act_dim = 0#由len(geokeys)得到
    cudaID = 0

rlPar=RLPar()

geoDir={
    "conv_1*1":lambda in_fea, out_fea: nn.Conv2d(in_channels=in_fea,out_channels=out_fea,kernel_size=1),
    "conv_3*3":lambda in_fea, out_fea: nn.Conv2d(in_channels=in_fea,out_channels=out_fea,kernel_size=3),
    "conv_5*5":lambda in_fea, out_fea: nn.Conv2d(in_channels=in_fea,out_channels=out_fea,kernel_size=5),
    "conv_7*7":lambda in_fea, out_fea: nn.Conv2d(in_channels=in_fea,out_channels=out_fea,kernel_size=7),
    "conv_sqp3*3":lambda in_fea,out_fea: nn.Sequential(
            nn.Conv2d(in_channels=in_fea, out_channels=in_fea, kernel_size=3, stride=1, padding=1, groups=in_fea),
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=1,stride=1, padding=0, groups=1),
    ),
    "conv_sqp5*5":lambda in_fea,out_fea: nn.Sequential(
            nn.Conv2d(in_channels=in_fea, out_channels=in_fea, kernel_size=5, stride=1, padding=2, groups=in_fea),
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=1,stride=1, padding=0, groups=1),
    ),
    "conv_sqp7*7":lambda in_fea,out_fea: nn.Sequential(
            nn.Conv2d(in_channels=in_fea, out_channels=in_fea, kernel_size=7, stride=1, padding=3, groups=in_fea),
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=1,stride=1, padding=0, groups=1),
    ),
    "pool_max2*2":nn.MaxPool2d(2, 2),
    "pool_nor2*2":nn.AvgPool2d(2,2),
    "actfun_relu":nn.ReLU(),
    "actfun_tanh":nn.Tanh(),
    "actfun_sig":nn.Sigmoid(),
    "batchNorm2d":lambda in_fea: nn.BatchNorm2d(in_fea),
    "dropout":nn.Dropout(0.3)
}
geokeys=list(geoDir.keys())
rlPar.act_dim=len(geokeys)
print(geokeys)
def init_kind():#对模块分类
    convKindInd=np.zeros(len(geokeys))
    dir={}
    cnt=0
    for i in range(len(geokeys)):#按照模块把前缀分类
        k=geokeys[i][0:4]
        if not (k in dir):
            cnt+=1
            dir[k]=cnt
        convKindInd[i]=cnt
    return convKindInd
convKindInd=init_kind()
print(convKindInd)
# 卷积：1*1,3*3,5*5,7*7的cnn或者seq_cnn,
# 池化：最大池化3x3,平均池化3*3,
# 激活：tanh,relu, sigmoid
# 跳跃：跳跃0层，1层，2层，3层
# 卷积：1*1,3*3,5*5,7*7的cnn或者seq_cnn,
# 池化：最大池化3x3,平均池化3*3,
# 激活：tanh,relu, sigmoid
# 跳跃：跳跃0层，1层，2层，3层

