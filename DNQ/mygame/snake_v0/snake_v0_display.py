import pygame
import sys
import time
import random
from pygame.locals import *

# 初始化pygame
pygame.init()
# 这是游戏框
fps_clock = pygame.time.Clock()
play_sur_face = pygame.display.set_mode((640, 480))
# 设置标题
pygame.display.set_caption("小蛇冲冲冲！！！")
# 加载图标
# image = pygame.image.load("game.jpg")
# pygame.display.set_icon(image)

# 需要自定义设置一些颜色

RED_COLOR = pygame.Color(255, 0, 0)
BLACK_COLOR = pygame.Color(0, 0, 0)
WITHER_COLOR = pygame.Color(255, 255, 255)
GRE_COLOR = pygame.Color(150, 150, 150)
LIGHT_GRE = pygame.Color(220, 220, 220)


# 游戏结束
def game_over(play_sur_face, score):
    # 显示GAME OVER 并定义字体以及大小
    # game_over_font = pygame.font.Font('arial.tff', 72)
    game_over_font = pygame.font.Font('res/arialbd.ttf', 72)
    game_over_surf = game_over_font.render("GAME OVER", True, GRE_COLOR)
    game_over_rect = game_over_surf.get_rect()
    game_over_rect.midtop = (320, 125)
    play_sur_face.blit(game_over_surf, game_over_rect)
    # 显示分数并定义字体大小
    score_font = pygame.font.Font('res/arialbd.ttf', 48)
    score_surf = score_font.render('score ' + str(score), True, GRE_COLOR)
    score_rect = score_surf.get_rect()
    score_rect.midtop = (320, 255)
    play_sur_face.blit(score_surf, score_rect)
    # 刷新页面
    pygame.display.flip()
    time.sleep(5)
    pygame.quit()
    sys.exit()



snake_position = [100, 100]  # 蛇头位置
snake_h = [[100, 100], [80, 100], [60, 100]]  # 初始长度，三个单位
tree_position = [300, 300]
# 初始化树莓的数量
tree = 1
direction = 'right'  # 初始化方向
change_direction = direction
score = 0
# 检测例如按键等pygame事件
while True:
    for event in pygame.event.get():
        # print(event.type)
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            # 判断键盘事件
            if event.key == K_RIGHT or event.key == ord('d'):
                change_direction = 'right'
            if event.key == K_LEFT or event.key == ord('a'):
                change_direction = 'left'
            if event.key == K_UP or event.key == ord('w'):
                change_direction = "up"
            if event.key == K_DOWN or event.key == ord('s'):
                change_direction = 'down'
            if event.key == K_ESCAPE:  # 按esc键
                pygame.event.post(pygame.event.Event(QUIT))  # 退出游戏

    if change_direction == "right" and not direction == 'left':
        direction = change_direction
    if change_direction == "left" and not direction == 'right':
        direction = change_direction
    if change_direction == "up" and not direction == 'down':
        direction = change_direction
    if change_direction == "down" and not direction == 'up':
        direction = change_direction

    # 根据放下移动蛇头坐标
    if direction == 'right':
        snake_position[0] += 20
    if direction == 'left':
        snake_position[0] -= 20
    if direction == 'up':
        snake_position[1] -= 20
    if direction == 'down':
        snake_position[1] += 20
    snake_h.insert(0, list(snake_position))

    # 判断是否吃到树莓
    if snake_position[0] == tree_position[0] and snake_position[1] == tree_position[1]:
        tree = 0
    else:
        snake_h.pop()  # 每次将最后一单位蛇身剔除列表

    # 重新生成树莓
    if tree == 0:
        x = random.randrange(1, 32)
        y = random.randrange(1, 24)
        tree_position = [int(20 * x), int(20 * y)]
        tree = 1
        score += 1

    # 刷新显示层
    # def sx_face():
    #     # 绘制pygame显示层
    play_sur_face.fill(BLACK_COLOR)
    for position in snake_h[1:]:  # 蛇身为白色
        pygame.draw.rect(play_sur_face, WITHER_COLOR, Rect(position[0], position[1], 20, 20))
    pygame.draw.rect(play_sur_face, LIGHT_GRE, Rect(snake_position[0], snake_position[1], 20, 20))  # 蛇头为灰色
    pygame.draw.rect(play_sur_face, RED_COLOR, Rect(tree_position[0], tree_position[1], 20, 20))
    # 刷新显示层
    pygame.display.flip()

    # def check_is_alive():
    #     """
    #     判断蛇是否死亡
    #     :return:
    #     """
    if snake_position[0] > 620 or snake_position[0] < 0:  # 超出左右边界
        game_over(play_sur_face, score)
    if snake_position[1] > 460 or snake_position[1] < 0:  # 超出上下边界
        game_over(play_sur_face, score)
    for snack_body in snake_h[1:]:
        if snake_position[0] == snack_body[0] == snack_body[1] == snake_position[1]:
            game_over(play_sur_face, score)
    if len(snake_h) < 40:
        speed = 6 + len(snake_h) // 4
    else:
        speed = 16
    fps_clock.tick(speed)