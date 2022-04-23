# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# https://blog.csdn.net/qq_27806947/article/details/106315337
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def test():
    # 扩展卡尔曼滤波
    # 状态转移方程为：x(k) = sin(3 * x(k-1)), 注意这里没有控制量
    # 观测方程为：y(k) = x(k)^2
    # 注意似然概率是多峰分布，具有强烈的非线性，当y=4的时候，我们无法判断x=2还是-2
    
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import random
    
    # 生成真实的信号与观测
    t = [0.01 * i for i in range(1, 100)]
    n = len(t)
    
    x = []
    for i in range(n):
        x.append(0)
    
    y = []
    for i in range(n):
        y.append(0)
    
    x[0] = 0.1
    y[0] = 0.1 ** 2
    
    for i in range(1, 99):
        print(i)
        x[i] = math.sin(3 * x[i - 1])
        y[i] = x[i] ** 2 + (random.random() - 0.5)
    
    plt.figure()
    plt.plot(t, y)
    plt.plot(t, x)
    plt.show()
    
    # 下面开始扩展卡尔曼滤波
    # 状态分量
    xPlus = []
    for i in range(n):
        xPlus.append(0)
    
    pPlus = []
    for i in range(n):
        pPlus.append(0)

    # 设置初值
    pPlus[0] = 0.1
    xPlus[0] = 0.1  # 状态分类初值
    Q = 0.1  # 观测过程中的噪声方差
    R = 0.1  # 状态转移过程中的噪声方差
    
    for i in range(1, 98):
        # 预测过程
        G = 3 * math.cos(3 * xPlus[i - 1])  # 对应于雅克比矩阵
        Xminus = math.sin(3 * xPlus[i - 1])  # 对应于预测状态分量
        Pminus = G * pPlus[i - 1] * np.transpose(G) + Q  # 对应于预测状态矩阵的方差

        # 更新过程
        H = 2 * Xminus  # 对应于观测过程的雅克比矩阵
        K = Pminus * H * np.mat(H * Pminus * np.transpose(H) + R).I  # 卡尔曼增益
        xPlus[i] = Xminus + K * (y[i] - Xminus ** 2)
        pPlus[i] = (1 - K * H) * Pminus
    
    plt.figure()
    # plt.plot(t, x, "r-")
    plt.plot(t, y, "b-")
    plt.plot(t, xPlus, "g-")
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
