# 扩展卡尔曼滤波
[引用原文](https://blog.csdn.net/qq_27806947/article/details/106315337)
# 1 简述

[卡尔曼](https://so.csdn.net/so/search?q=%E5%8D%A1%E5%B0%94%E6%9B%BC&spm=1001.2101.3001.7020) 滤波适用于线性高斯系统，然而这是一个强假设；对于大部分机器人系统而言，非线性系统才是常态，如此卡尔曼滤波就不太适用了，那么该如何解决这个问题？这引出了扩展卡尔曼滤波。


# 2 扩展卡尔曼滤波的思想

扩展卡尔曼滤波的基本思想来自于线性化，也就是说对一个[非线性](https://so.csdn.net/so/search?q=%E9%9D%9E%E7%BA%BF%E6%80%A7&spm=1001.2101.3001.7020) 系统进行一阶泰勒展开，从而把一个非线性系统转化为线性系统，这样卡尔曼滤波能够处理的了；不过要是该系统非常非线性化，那么扩展卡尔曼算法的效果很有限。

# 3 扩展卡尔曼滤波的核心公式

以下公式来自于《概率机器人》，为了方便与卡尔曼滤波算法进行对比，我们这里先放卡尔曼滤波算法的核心公式

![](https://img-blog.csdnimg.cn/20200524151652433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3ODA2OTQ3,size_16,color_FFFFFF,t_70)

然后是扩展卡尔曼滤波算法的核心公式：

![](https://img-blog.csdnimg.cn/20200524151403674.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3ODA2OTQ3,size_16,color_FFFFFF,t_70)

        经过对比发现，实际上变化并不是不很大，主要是观测方程与状态预测方程从线性变为非线性。注意：输入是上一时刻计算的最佳状态分量及其方差，输出是当前时刻计算的最佳状态 分量及其方差。

接下来，我们对扩展卡尔曼滤波公式中的符号进行解释：

&emsp;&emsp;${\bar\mu _{t}}$ 机器人在t时刻预测的状态分量；

&emsp;&emsp;$\mu _{t-1}$  机器人在t-1时刻估计的最优状态分量；

&emsp;&emsp;$u{_{t}}$ 机器人的控制输入矩阵；

&emsp;&emsp;$g()$ 是一个状态转移函数，非线性函数；

&emsp;&emsp;$\bar{\sum}_{t}$  机器人在t时刻预测状态分量的协方差矩阵；

&emsp;&emsp;${\sum}_{t-1}$ 机器人在t-1时刻估计最优状态分量的协方差矩阵；

&emsp;&emsp;$R{_{t}}$ 状态转移过程中高斯噪声的协方差矩阵；

&emsp;&emsp;$G_{t}$ 处的雅克比矩阵；

&emsp;&emsp;$Q{_{t}}$ 观测过程中高斯噪声的协方差矩阵；

&emsp;&emsp;$K_{t}$ 卡尔曼增益；

&emsp;&emsp;$H{_{t}}$ 观测方程$h(\bar{\mu _{t}})$ 处的雅克比矩阵；

&emsp;&emsp;$z{_{t}}$ 机器人的观测数据；

&emsp;&emsp;${\mu _{t}}$ 机器人在t时刻估计的最优状态分量；

&emsp;&emsp;${\sum}_{t}$ 机器人在t时刻估计最优状态分量的协方差矩阵；

        关于以上这些公式的推导和证明，资料挺多的，我觉得了解就好；会用上面这些公式即可。

# 4 两个例子

## 4.1 正弦运动

        状态转移方程为：$x_{k}=sin(3 * x{_{k-1}})$

        观测方程为：$y_{k} = x_{k}^{2}$

        观测噪声的方差为0.1；状态转移过程的方差为0.1。

```python
#扩展卡尔曼滤波
#状态转移方程为：x(k) = sin(3 * x(k-1)), 注意这里没有控制量
#观测方程为：y(k) = x(k)^2
#注意似然概率是多峰分布，具有强烈的非线性，当y=4的时候，我们无法判断x=2还是-2
 
import matplotlib.pyplot as plt
import numpy as np
import math
import random
 
 
#生成真实的信号与观测
t = [0.01 * i for i in range(1, 100)]
n = len(t)
 
x = []
for i in range(n):
    x.append(0)
    
y = []
for i in range(n):
    y.append(0)
    
x[0] = 0.1
y[0] = 0.1**2
 
for i in range(1, 99):
    print(i)
    x[i] = math.sin(3 * x[i-1])
    y[i] = x[i]**2 + (random.random() - 0.5)
    
plt.figure()
plt.plot(t, y)
plt.plot(t, x)
plt.show()
 
 
#下面开始扩展卡尔曼滤波
#状态分量
xPlus = []
for i in range(n):
    xPlus.append(0)
    
pPlus = []
for i in range(n):
    pPlus.append(0)
    
#设置初值
pPlus[0] = 0.1
xPlus[0] = 0.1 #状态分类初值
Q = 0.1  #观测过程中的噪声方差
R = 0.1  #状态转移过程中的噪声方差
 
for i in range(1, 98):
    #预测过程
    G = 3 * math.cos(3 * xPlus[i-1]) #对应于雅克比矩阵
    Xminus = math.sin(3 * xPlus[i-1]) #对应于预测状态分量
    Pminus = G * pPlus[i-1] * np.transpose(G) + Q #对应于预测状态矩阵的方差
    
    #更新过程
    H = 2 * Xminus  #对应于观测过程的雅克比矩阵
    K = Pminus * H * np.mat(H * Pminus * np.transpose(H) + R).I #卡尔曼增益
    xPlus[i] = Xminus + K * (y[i] - Xminus**2)
    pPlus[i] = (1 - K * H) * Pminus
    
plt.figure()
plt.plot(t, x)
plt.plot(t, xPlus)
plt.show()
```

结果如下图所示：

![](https://img-blog.csdnimg.cn/20200524174331411.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3ODA2OTQ3,size_16,color_FFFFFF,t_70)

效果不咋样！

## 4.2 圆周运动

        状态转移方程：

&emsp;&emsp;![](https://img-blog.csdnimg.cn/20210103102755414.png)

        观测方程：GPS数据

```python
"""
Extended kalman filter (EKF) localization sample
author: Atsushi Sakai (@Atsushi_twi)
"""
 
import math
 
import matplotlib.pyplot as plt
import numpy as np
 
# EKF 状态方程的协方差矩阵 
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
 
# 观测方程的协方差矩阵
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance
 
# 仿真参数
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2  
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
 
DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
 
show_animation = True
 
 
def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u
 
def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)
 
    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
 
    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)
 
    xd = motion_model(xd, ud)
 
    return xTrue, z, xd, ud
 
#运动学模型
def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])
 
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])
 
    x = F @ x + B @ u
 
    return x
 
#观测模型
def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
 
    z = H @ x
 
    return z
 
#运动学方程中的雅克比矩阵
def jacob_f(x, u):
    """
    Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
 
    return jF
 
#观测方程中的雅克比矩阵
def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
 
    return jH
 
#扩展卡尔曼滤波过程
def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q #预测方差
 
    #  Update
    jH = jacob_h() 
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R 
    K = PPred @ jH.T @ np.linalg.inv(S) #卡尔曼增益
    xEst = xPred + K @ y #最优估计
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred #最优估计的方差
    return xEst, PEst
 
#绘制协方差的椭圆
def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)
 
    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0
 
    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")
 
 
def main():
    print(__file__ + " start!!")
 
    time = 0.0
 
    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)
 
    xDR = np.zeros((4, 1))  # Dead reckoning
 
    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))
 
    while SIM_TIME >= time:
        time += DT
        u = calc_input()
 
        xTrue, z, xDR, ud = observation(xTrue, xDR, u)
 
        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)
 
        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))
 
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g") #观测数据
            plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-y") #真实
            plt.plot(hxDR[0, :].flatten(), hxDR[1, :].flatten(), "-k") #黑色
            plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-r") #红色
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
 
 
if __name__ == '__main__':
    main()
```

效果如下图所示：

![](https://img-blog.csdnimg.cn/20200526231753856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3ODA2OTQ3,size_16,color_FFFFFF,t_70)

还不错！！！