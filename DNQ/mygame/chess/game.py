import numpy as np
import pygame
import pygame.locals
import sys
import time



###中国象棋环境


class ChineseChess():
    init_borad = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 0, 0, 11, 0],
        [12, 0, 13, 0, 14, 0, 15, 0, 16],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [17, 0, 18, 0, 19, 0, 20, 0, 21],
        [0, 22, 0, 0, 0, 0, 0, 23, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [24, 25, 26, 27, 28,29, 30,31, 32],
    ])
    now_borad = init_borad.copy()
    #棋子id索引到名称
    pieces_toname = [' ','车','马','相','士','帅','士','相','马','车','炮','炮','兵','兵','兵','兵','兵',
                     '卒', '卒', '卒', '卒', '卒','炮','炮','车','马','相','士','帅','士','相','马','车',]
    #棋子id索引棋子种类
    pieces_toKind = [' ', 1, 2, 3, 4, 5,4, 3, 2, 1, 6, 6, 7,7,7,7,7,
                     -7,-7,-7,-7,-7,-6, -6,  -1, -2, -3, -4, -5,-4, -3, -2, -1, ]
    pieces_pos=[]
    actions_dir={1: 1, 2: '马', 3: '相', 4: '仕', 5: '帅', 6: '炮', 7: '兵',}
    kindActionsList=[]#种类动作空间
    actionsList = []  # 动作空间
    pre_eatStep=0#上一次吃子
    step_i=0
    def __init__(self):
        self.now__borad = self.init_borad.copy()
        self.now_place=[(0,0)]#棋子id索引到当前位置
        for row in range(len(self.now_borad)):#给每个棋子确定位置
            for col in range(len(self.now_borad[row])):
                if self.now_borad[row][col]!=0:
                    self.now_place.append((row,col))


        self.kindActionsList=[]
        self.kindActionsList.append()
        for i in range(-9,10):#车的动作空间
            if i!=0:
                self.kindActionsList.append((1,(0,i)))#(种类id,移动距离)
                self.kindActionsList.append((1, (i, 0)))  # (种类id,移动距离)
        #马的动作空间
        self.kindActionsList.append((2, (1, 2)))  # (种类id,移动距离)
        self.kindActionsList.append((2, (1, -2)))  # (种类id,移动距离
        self.kindActionsList.append((2, (-1, 2)))  # (种类id,移动距离)
        self.kindActionsList.append((2, (-1, -2)))  # (种类id,移动距离
        self.kindActionsList.append((2, (2, 1)))  # (种类id,移动距离)
        self.kindActionsList.append((2, (2, -1)))  # (种类id,移动距离
        self.kindActionsList.append((2, (-2, 1)))  # (种类id,移动距离)
        self.kindActionsList.append((2, (-2, -1)))  # (种类id,移动距离
        # 象的动作空间
        self.kindActionsList.append((3, (2, 2)))  # (种类id,移动距离)
        self.kindActionsList.append((3, (2, -2)))  # (种类id,移动距离
        self.kindActionsList.append((3, (-2, 2)))  # (种类id,移动距离)
        self.kindActionsList.append((3, (-2, -2)))  # (种类id,移动距离
        #士的动作空间
        self.kindActionsList.append((4, (1, 1)))  # (种类id,移动距离)
        self.kindActionsList.append((4, (1, -1)))  # (种类id,移动距离
        self.kindActionsList.append((4, (-1, 1)))  # (种类id,移动距离)
        self.kindActionsList.append((4, (-1, -1)))  # (种类id,移动距离
        #帅的动作空间
        self.kindActionsList.append((5, (0, 1)))  # (种类id,移动距离)
        self.kindActionsList.append((5, (0, -1)))  # (种类id,移动距离
        self.kindActionsList.append((5, (1, 0)))  # (种类id,移动距离)
        self.kindActionsList.append((5, (-1, 0)))  #
        #炮的动作空间
        for i in range(-9,10):#车的动作空间
            if i!=0:
                self.kindActionsList.append((6,(0,i)))#(种类id,移动距离)
                self.kindActionsList.append((6, (i, 0)))  # (种类id,移动距离)
        #兵的动作空间
        self.kindActionsList.append((7, (0, 1)))  # (种类id,移动距离)
        self.kindActionsList.append((7, (0, -1)))  # (种类id,移动距离
        self.kindActionsList.append((7, (1, 0)))  # (种类id,移动距离)
        self.kindActionsList.append((7, (-1, 0)))  #


    def reset(self):
        self.__init__()
        return self.now_borad

    def nowActionMask(self,state):
        mask=[True]*len(self.kindActionsList)

        return mask
    def render(self):
        #画出界面
        pass
    def step(self,action_id):#act_id代表第几个动作
        act=self.actionsList[action_id]
        pos=self.pieces_pos[act[0]]
        d=act[1]
        self.now_borad[pos[0]][pos[1]] = 0
        eated_piece = self.now_borad[pos[0] + d[0]][pos[1] + d[1]]
        self.now_borad[pos[0]+d[0]][pos[1]+d[1]] = act[0]
        ans=self.judgeState(eated_piece)

        if eated_piece>0:
            self.pre_eatStep=self.step_i
        self.step_i+=1
        return self.now_borad.copy(),eated_piece,ans#返回状态，吃了哪个棋子,有没有结束
    def judgeState(self,eated_piece):#判断状态1,代表红方胜，-1代表黑方胜，0代表和棋，其它代表没有分出胜负
        if eated_piece==5:#是帅
            return -1
        if eated_piece==28:#是将
            return 1
        if self.step_i-self.pre_eatStep>60:#60步不吃子，和棋
            return 0
        #判断白脸杀
        pos=self.pieces_pos[5]
        for i in range(1,10):
            if pos[0] + i >= 10 or (self.now_borad[pos[0]+i][pos[1]]>0 and self.now_borad[pos[0]+i][pos[1]]!=28):
                break
            if self.now_borad[pos[0]+i][pos[1]]==28:
                return self.step_i%2==0 and -1 or 1#self.step_i是偶数，则红方失败，黑方赢.
        #困毙，不用判断，因为软件以吃到帅为目的

        return 2

    def judgeJiang(self):#判断将军

        return False

    def judgeLongJiang(self):  # 判断长将,连续10步存在相同局面算长将

        return False
    def judgeLongZhuo(self):  # 判断长捉，大子(车马炮)连续10步存在相同局面算长捉

        return False