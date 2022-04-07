from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.stats import poisson  # 统计学的包，用于生成泊松分布

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正确显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示正负号


#初始化
poisson_cache = dict()
x = np.arange(0, 21)
y = np.arange(0, 21)
x, y = np.meshgrid(x, y)

value = np.zeros((21, 21))
max_value_change = 100
policy = np.zeros(value.shape, dtype=np.int)  # 存储策略【0，1，2……20】

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

def init():                                     # 画布坐标轴范围初始化

    ax1.set_xlim([0, 20])
    ax1.set_ylim([0, 20])
    ax1.set_zlim([-6, 6])

    ax1.set_xlabel("场地A的车辆状态")
    ax1.set_ylabel("场地B的车辆状态")
    ax1.set_zlabel("决策中的动作分布")

    ax2.set_xlim([0, 20])
    ax2.set_ylim([0, 20])
    ax2.set_zlim([0, 1000])

    ax2.set_xlabel("场地A的车辆状态")
    ax2.set_ylabel("场地B的车辆状态")
    ax2.set_xlabel("场地A的车辆状态")
    ax2.set_ylabel("场地B的车辆状态")
    ax2.set_zlabel("状态价值估计")

    plt.tight_layout()
    return ax1, ax2,

def poisson_probability(n, lam):
    """
    输出n个车辆在参数为lam下的泊松分布概率
    @n: 车辆数
    @lam: 期望值
    """
    global poisson_cache
    key = n * 10 + lam  # 给每个状态设置一个标签，防止出现重叠
    # 例如，第二个地点借车期望为4，还车期望为2，如果直接用n+λ表示，“还2辆车”“借0辆车”时间重合了。

    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)  # 储存n个车辆，发生的概率
    return poisson_cache[key]


def expected_return(state, action, state_value):
    """
    函数功能【策略评估】：在某一个状态下，根据一个策略，估计新的状态价值。
    @state：一个数组，state[0]存储A场地的车辆数；state[1]存储B场地的车辆数
    @action：动作
    @state_value:状态价值矩阵
    """

    returns = 0.0  # 初始化

    # step1  减去移车费
    returns -= 2 * abs(action)

    # 遍历所有可能的租车请求
    for rental_request_first_loc in range(10):
        for rental_request_second_loc in range(10):
            p1 = poisson_probability(rental_request_first_loc, 3)
            p2 = poisson_probability(rental_request_second_loc, 4)
            prob = p1 * p2  # 当前租车请求发生的概率

            # 更新状态——晚上运完以后的车辆状态
            num_of_cars_first_loc = min(state[0] - action, 20)
            num_of_cars_second_loc = min(state[1] + action, 20)

            # 有效租车数量     min（已有车辆和租车数量）
            valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
            valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

            # 获取盈利值
            reward = (valid_rental_first_loc + valid_rental_second_loc) * 10

            # 更新状态——减去租出去的车数量
            num_of_cars_first_loc -= valid_rental_first_loc
            num_of_cars_second_loc -= valid_rental_second_loc

            returned_cars_first_loc = 3
            returned_cars_second_loc = 2
            num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, 20)
            num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, 20)
            returns += prob * (reward + 0.9 * state_value[num_of_cars_first_loc, num_of_cars_second_loc])

    return returns


def value_iterative(value):  # 策略优化过程

    actions = np.arange(-5, 6)
    old_value = value.copy()

    for i in range(21):
        for j in range(21):
            action_returns = []
            for action in actions:
                if (0 <= action <= i) or (-j <= action <= 0):  # 遍历每个动作取值
                    action_returns.append(expected_return([i, j], action, value))
                else:
                    action_returns.append(-np.inf)  # 挪车数目不够了，加上一个很大的负值

            max_action = actions[np.argmax(action_returns)]
            value[i, j] = expected_return([i, j], max_action, value)  # 状态、动作、价值————》新的状态价值

    max_value_change = abs(old_value - value).max()

    return value, max_value_change

def choose_action(value):

    actions = np.arange(-5, 6)

    for i in range(21):
        for j in range(21):   # 遍历每个状态位置
            action_returns = []

            for action in actions:

                if (0 <= action <= i) or (-j <= action <= 0): # 遍历每个动作取值
                    action_returns.append(expected_return([i, j], action, value))
                else:
                    action_returns.append(-np.inf)  # 挪车数目不够了，加上一个很大的负值

            new_action = actions[np.argmax(action_returns)]
            policy[i, j] = new_action

    return policy


def update_point(n):
    global value
    global policy
    global max_value_change

    if n != 0:
        if max_value_change < 3:
            policy = choose_action(value)
            ax1.scatter(x, y, policy, marker='o', c='r')
            plt.suptitle('第%d次价值迭代收敛，选择策略' % n, fontsize=20)

        else:
            value, max_value_change = value_iterative(value)
            ax2.scatter(x, y, value, marker='o', c='g')
            plt.suptitle('第%d次价值迭代，状态价值最大变化为%f' % (n, max_value_change), fontsize=20)


    else:
        ax1.scatter(x, y, policy, marker='o', c='r')
        ax2.scatter(x, y, value, marker='o', c='g')
        plt.suptitle('初始化', fontsize=20)

    return ax1, ax2,


ani = animation.FuncAnimation(fig=fig,
                              func=update_point,
                              frames=23,
                              interval=300,
                              repeat=False,
                              blit=False,
                              init_func=init)

ani.save('test4.gif', writer='pillow')
#plt.show()
