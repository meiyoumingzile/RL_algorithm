import math
import random

import numpy as np
import pygame
import pygame.locals
from pygame.locals import *
import sys
import time

RED_COLOR = pygame.Color(255, 0, 0)
BLACK_COLOR = pygame.Color(0, 0, 0)
WITHER_COLOR = pygame.Color(255, 255, 255)
GRE_COLOR = pygame.Color(50, 50, 50)
LIGHT_GRE = pygame.Color(220, 220, 220)
MAP_FOOD=-1
MAP_SNAKE=1
MAP_SNAKE_HEAD=2
MAP_OB=3
MAP_EMPTY=0

class Snake_Node():
    pos=(0,0)
    nex=None
    def __init__(self, x,y):
        self.pos=(x,y)
class CC():
    SZ=20
    nowstepCnt=0
    stepCnt=0
    saveAction=[-1,-1]
    snakeLen = 3
    emptyCnt = SZ * SZ - snakeLen
    ini_borad=np.zeros((SZ+2,SZ+2))#0和SZ+1代表墙
    board=np.zeros((SZ,SZ))#0代表空，1代表蛇身体，2代表头，3代表墙壁，-1代表食物
    head=None
    rear = None
    fp=[(0,1),(0,-1),(1,0),(-1,0)]
    foodList=[(1,1),(0,0)]
    def randomOb(self):
        pass
    def __init__(self,randomOb=False):#randomOb代表随机生成墙
        for i in range(self.SZ+2):
            self.ini_borad[i][0] = MAP_OB
            self.ini_borad[0][i] = MAP_OB
            self.ini_borad[self.SZ+1][i] = MAP_OB
            self.ini_borad[i][self.SZ+1] = MAP_OB
        if randomOb:
            self.randomOb()
        self.reset()

    def reset(self):
        self.stepCntUp=self.SZ*self.SZ+1
        self.nowstepCnt=0
        self.stepCnt=0
        self.snakeLen=3
        self.emptyCnt=self.SZ * self.SZ - self.snakeLen
        self.board = self.ini_borad.copy()
        self.head= Snake_Node(1,3)
        self.rear = Snake_Node(1, 1)
        self.rear.nex=Snake_Node(1, 2)
        self.saveAction=[1,1]
        self.board[1][1]=self.board[1][2]=self.board[1][3]=MAP_SNAKE
        # self.board[1][3] =MAP_SNAKE_HEAD
        self.rear.nex.nex=self.head
        self.mkFood(0)
        self.mkFood(1)
        return self.board
    def mkFood(self,p):
        k = random.randint(0, self.emptyCnt) + 1
        for i in range(1,self.SZ+2):
            for j in range(1,self.SZ+2):
                if self.board[i][j] == MAP_EMPTY:
                    k -= 1
                    if k == 0:
                        self.board[i][j] = MAP_FOOD
                        self.emptyCnt -= 1
                        self.foodList[p]=(i,j)
                        # print((i,j))
                        return


    def step(self, action):#4种动作
        self.stepCnt+=1
        self.nowstepCnt+=1
        fooded = 0
        self.saveAction=[action,self.saveAction[0]]#保存的动作队列
        nextPos = (self.head.pos[0] + self.fp[action][0], self.head.pos[1] + self.fp[action][1])
        # if nextPos[0]<0 or nextPos[0]>=self.SZ or nextPos[1]<0 or nextPos[1]>=self.SZ :
        #     return self.board, fooded, True
        nextOb=self.board[nextPos[0]][nextPos[1]]
        if nextOb==MAP_EMPTY or nextPos[0]==self.rear.pos[0] and nextPos[1]==self.rear.pos[1]:
            self.board[self.rear.pos[0]][self.rear.pos[1]] = MAP_EMPTY
            self.rear = self.rear.nex
            self.head.nex=Snake_Node(nextPos[0],nextPos[1])
            self.head=self.head.nex
            self.board[nextPos[0]][nextPos[1]] = MAP_SNAKE
        elif nextOb==MAP_OB or nextOb==MAP_SNAKE:
            return self.board, fooded,True
        elif nextOb == MAP_FOOD:#食物
            self.nowstepCnt=0
            self.snakeLen+=1
            self.head.nex = Snake_Node(nextPos[0], nextPos[1])
            self.head = self.head.nex
            self.board[nextPos[0]][nextPos[1]] = MAP_SNAKE
            if self.foodList[0][0]==nextPos[0] and self.foodList[0][1]==nextPos[1]:
                self.mkFood(0)
            else:
                self.mkFood(1)
            fooded=1
        if self.nowstepCnt>=self.stepCntUp:
            self.board, fooded, True
        return self.board, fooded,self.emptyCnt==0

    def getDisFromFood(self):
        mind=10000
        for a in self.foodList:
            d=abs(self.head.pos[0]-a[0])+abs(self.head.pos[1]-a[1])
            mind=min(mind,d)
        return mind

    play_sur_face = None
    PIX=10
    def render(self):
        if self.play_sur_face==None:
            pygame.init()
            self.play_sur_face = pygame.display.set_mode(((self.SZ+2)*self.PIX, (self.SZ+2)*self.PIX))
            # 设置标题
            pygame.display.set_caption("小蛇冲冲冲！！！")
        self.play_sur_face.fill(WITHER_COLOR)
        # for position in snake_h[1:]:  # 蛇身为白色
        nowNode=self.rear
        while(nowNode!=None):
            pos=nowNode.pos
            pygame.draw.rect(self.play_sur_face, GRE_COLOR, Rect(pos[0]*self.PIX, pos[1]*self.PIX, self.PIX,  self.PIX))
            nowNode=nowNode.nex
        pygame.draw.rect(self.play_sur_face, BLACK_COLOR, Rect(self.head.pos[0]*self.PIX, self.head.pos[1]*self.PIX, self.PIX, self.PIX))  # 蛇头为灰色
        for foodPos in self.foodList:
            pygame.draw.rect(self.play_sur_face, RED_COLOR, Rect(foodPos[0]*self.PIX, foodPos[1]*self.PIX, self.PIX, self.PIX))
        for i in range(self.SZ+2):
            pygame.draw.rect(self.play_sur_face, BLACK_COLOR,
                             Rect(i * self.PIX, 0 * self.PIX, self.PIX, self.PIX))
            pygame.draw.rect(self.play_sur_face, BLACK_COLOR,
                             Rect(0 * self.PIX, i * self.PIX, self.PIX, self.PIX))
            pygame.draw.rect(self.play_sur_face, BLACK_COLOR,
                             Rect((self.SZ+1) * self.PIX, i * self.PIX, self.PIX, self.PIX))
            pygame.draw.rect(self.play_sur_face, BLACK_COLOR,
                             Rect(i * self.PIX, (self.SZ+1) * self.PIX, self.PIX, self.PIX))
        # 刷新显示层
        pygame.display.flip()
    def printf(self):
        print(self.board)

def mkdemo():
    env = CC()
    isTer = False
    env.printf()
    env.render()
    fps_clock = pygame.time.Clock()
    while(True):
        env.reset()
        isTer = False
        while (not isTer):
            act = -1
            for event in pygame.event.get():
                # print(event.type)
                act = -1
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    # 判断键盘事件
                    if event.key == K_RIGHT or event.key == ord('d'):
                        act = 2
                    if event.key == K_LEFT or event.key == ord('a'):
                        act = 3
                    if event.key == K_UP or event.key == ord('w'):
                        act = 1
                    if event.key == K_DOWN or event.key == ord('s'):
                        act = 0
                    if event.key == K_ESCAPE:  # 按esc键
                        pygame.event.post(pygame.event.Event(QUIT))  # 退出游戏
            if act > -1:
                st, reward, isTer = env.step(act)
            else:
                st, reward, isTer = env.board, 0, False
            if isTer:
                break
            # env.printf()
            env.render()
            fps_clock.tick(16)
# mkdemo()