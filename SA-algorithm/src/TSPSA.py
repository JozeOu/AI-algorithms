#使用模拟退火算法解决 TSP 问题
import numpy
import math
import tsplib95
import random
import copy
import matplotlib.pyplot as plt

global problem # TSP问题
global dimension # problem的规模，即城市的数量
global DisMatrix # 存储城市间距离的矩阵
global path #存储路径
global distance # 存储路径长度

#用于绘制分析图的变量
global temps #记录退火过程中的每个温度值
global dists #记录生成的每个状态的distance值
global dists_accepted #记录被接受状态的distance值
global index_accepted #记录被接收状态的横坐标

# 设置参数
INIT_TEMP = 100 # 初温
RATE = 0.98 # 降温速率
ITERATION_NUM = 500 # 每轮迭代次数
OPTIMAL_SOLUTION = 15780 # TSP问题最优解

#计算路径中一个城市与相邻两个城市的距离之和，为了方便交换的时候的计算
#index 要计算的城市在路径中的索引
#aPath 一条路径
def neighbor_dis(index, aPath):
    dis = DisMatrix[aPath[index]][aPath[(index + 1) % dimension]]
    dis += DisMatrix[aPath[index]][aPath[(index - 1 + dimension) % dimension]]
    return dis

#生成方式一：交换两个城市在路径上的位置
def method1(x_index, y_index):
    newPath = copy.deepcopy(path)
    newPath[x_index], newPath[y_index] = newPath[y_index], newPath[x_index]
    newDis = distance + neighbor_dis(x_index, newPath) + neighbor_dis(y_index, newPath) - neighbor_dis(x_index, path) - neighbor_dis(y_index, path)
    return newDis, newPath

#生成方式二：两个城市之间的路径进行逆序
def method2(x_index, y_index):
    newPath = copy.deepcopy(path)
    num_min = min(x_index, y_index)
    num_max = max(x_index, y_index)
    for i in range(num_max - num_min + 1):
        newPath[num_min + i] = path[num_max - i]
    newDis = distance + neighbor_dis(x_index, newPath) + neighbor_dis(y_index, newPath) - neighbor_dis(x_index, path) - neighbor_dis(y_index, path)
    return newDis, newPath

#生成方式三，一个城市移到另一个城市前面
def method3(x_index, y_index):
    newPath = copy.deepcopy(path)
    num_min = min(x_index, y_index)
    num_max = max(x_index, y_index)
    temp = path[num_max]
    for i in range(num_max - num_min):
        newPath[num_max - i] = path[num_max - i - 1]
    newPath[num_min] = temp
    newDis = distance - neighbor_dis(num_max, path) - DisMatrix[path[num_min]][path[(num_min - 1 +dimension) % dimension]] + neighbor_dis(num_min, newPath) + DisMatrix[newPath[num_max]][newPath[(num_max + 1) % dimension]]
    return newDis, newPath

#产生新的状态，三种生成方式取最优（局部贪心）
def generate_new_state():
    x_index = random.randint(1, dimension - 1)
    y_index = random.randint(1, dimension - 1)
    while x_index == y_index:
        y_index = random.randint(1, dimension - 1)
    state1 = method1(x_index, y_index)
    state2 = method2(x_index, y_index)
    state3 = method3(x_index, y_index)
    newDis = state1[0]
    newPath = state1[1]
    if newDis > state2[0]:
        newDis = state2[0]
        newPath = state2[1]
    if newDis > state3[0]:
        newDis = state3[0]
        newPath = state3[1]
    return newDis, newPath

#计算两个城市之间的距离
def cal_distance(i, j):
    #获取城市(i+1)和(j+1)的坐标
    x1, y1 = problem.get_display(i + 1)
    x2, y2 = problem.get_display(j + 1)
    return math.sqrt((x1-x2)**2 + (y1 - y2)**2)

#生成存储任意两个城市间距离的矩阵
def generate_matrix():
    global DisMatrix
    DisMatrix = numpy.zeros([dimension, dimension]) #创建了一个dim*dim的矩阵，其元素值均为0.0
    for i in range(dimension):
        DisMatrix[i][i] = float("inf") #每个城市到自身的距离设置为正无穷大
        for j in range(i):
            dis = cal_distance(i, j)
            DisMatrix[i][j] = dis #将城市(i+1)和(j+1)之间的距离记入矩阵
            DisMatrix[j][i] = dis #两个城市之间的距离是对称的，得到对称矩阵

def init():
    global problem
    global dimension
    global path
    global distance
    # 载入TSP问题数据（load_problem 需要指定一个 distance function）
    problem = tsplib95.load_problem("../data/d198.tsp", special=cal_distance)
    # 初始化变量
    dimension = problem.dimension
    generate_matrix()
    path = [i for i in range(dimension)]
    distance = 0
    for i in range(dimension - 1):
        distance += DisMatrix[path[i]][path[i+1]]
    distance += DisMatrix[path[dimension - 1]][path[0]]
    print("初始路径长度：{0}".format(distance))
    # 打开交互模式，非阻塞画图
    plt.figure(figsize=(8, 6), dpi=80)
    plt.ion()

#模拟退火算法求解（注释掉if条件中的 or 部分就是贪心算法）
def tsp_sa():
    global path
    global distance

    global temps
    global dists
    global dists_accepted
    global index_accepted
    temps = []
    dists = []
    dists_accepted = []
    index_accepted = []

    epoch = 0
    temp = INIT_TEMP
    while temp > 0.1:
        temps.append(temp)
        for i in range(ITERATION_NUM): # 迭代
            new_state = generate_new_state()
            dists.append(new_state[0])
            if new_state[0] < distance or random.random() < math.exp(-1*((new_state[0]-distance)/temp)):
                distance = new_state[0]
                path = copy.deepcopy(new_state[1])
                dists_accepted.append(distance)
                index_accepted.append(ITERATION_NUM * epoch + i)
        temp *= RATE #降温
        epoch += 1
        generate_picture_dyn(epoch, temp)

def generate_picture_dyn(epoch, temp): 
    position_x = numpy.zeros([dimension + 1])
    position_y = numpy.zeros([dimension + 1])
    for i in range(dimension):
        position_x[i], position_y[i] = problem.get_display(path[i] + 1)
    position_x[dimension], position_y[dimension] = problem.get_display(1)
    plt.cla() # 清除原有图像
    precision = 100 * (distance - OPTIMAL_SOLUTION) / OPTIMAL_SOLUTION
    title = 'No.' + str(epoch) + '   temperature:' + str(temp) + '\n distance:' + str(distance) + '    precision:' + str(precision) + '%'
    plt.title(title)
    plt.plot(position_x, position_y, 'b.-')
    plt.pause(0.1)

#主函数
def main():
    init()
    tsp_sa()
    print("最短路径长度：{0}".format(distance))
    print("最短路径：{0}".format(path))
    print('该问题已找到的最优解：{0}'.format(OPTIMAL_SOLUTION))
    print('误差：%.2f%%' % (100 * (distance - OPTIMAL_SOLUTION) / OPTIMAL_SOLUTION))
    #提供可视化
    generate_picture()

'''
# 生成分析图图
def generate_picture():
    global temps
    global dists
    global dists_accepted
    global index_accepted

    #生成最短路径图
    position_x = numpy.zeros([dimension + 1])
    position_y = numpy.zeros([dimension + 1])
    for i in range(dimension):
        position_x[i], position_y[i] = problem.get_display(path[i] + 1)
    position_x[dimension], position_y[dimension] = problem.get_display(path[0] + 1)
    #绘制曲线图，横坐标数组 position_x，纵坐标数组 position_y，linestyle='-'（线类型），marker='*'（曲线上标记的特殊符号）
    plt.figure(1)
    plt.title('模拟退火算法得到的最短路径图')
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.plot(position_x, position_y, '-*')

    #生成过程分析图
    temps_x = [i * ITERATION_NUM for i in range(len(temps))]
    dists_x = [i for i in range(len(dists))]
    plt.figure(2)
    plt.title('模拟退火算法迭代过程')
    #温度变化曲线
    plt.plot(temps_x, temps, '-*', color='blue', label='温度')
    #生成的所有状态的曲线
    plt.plot(dists_x, dists, '-*', color='green', label='所有生成解的路径总长度')
    #被接受状态的曲线
    plt.plot(index_accepted, dists_accepted, '-*', color='yellow', label='被接受解的路径总长度')
    plt.xlabel('迭代次数')
    plt.legend() #显示图例
    plt.show()
'''

if __name__ == "__main__":
    main()