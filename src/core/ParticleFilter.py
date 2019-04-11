from numpy.random import uniform, randn, random, seed
from filterpy.monte_carlo import multinomial_resample
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
def create_particles(x_range, y_range, v_mean, v_std, N):
    """这里的粒子状态设置为（坐标x，坐标y，运动方向，运动速度）"""
    particles = np.empty((N, 4))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(0, 2 * np.pi, size=N)
    particles[:, 3] = v_mean + (randn(N) * v_std)
    return particles


def predict_particles(particles, std_heading, std_v, x_range, y_range):
    """这里的预测规则设置为：粒子根据各自的速度和方向（加噪声）进行运动，如果超出边界则随机改变方向再次尝试，"""
    idx = np.array([True] * len(particles))
    particles_last = np.copy(particles)
    for i in range(100):  # 最多尝试100次
        if i == 0:
            particles[idx, 2] = particles_last[idx, 2] + (randn(np.sum(idx)) * std_heading)
        else:
            particles[idx, 2] = uniform(0, 2 * np.pi, size=np.sum(idx))  # 随机改变方向
        particles[idx, 3] = particles_last[idx, 3] + (randn(np.sum(idx)) * std_v)
        particles[idx, 0] = particles_last[idx, 0] + np.cos(particles[idx, 2]) * particles[idx, 3]
        particles[idx, 1] = particles_last[idx, 1] + np.sin(particles[idx, 2]) * particles[idx, 3]
        # 判断超出边界的粒子
        idx = ((particles[:, 0] < x_range[0])
               | (particles[:, 0] > x_range[1])
               | (particles[:, 1] < y_range[0])
               | (particles[:, 1] > y_range[1]))
        if np.sum(idx) == 0:
            break


def update_particles(particles, weights, z, d_std):
    """粒子更新，根据观测结果中得到的位置pdf信息来更新权重，这里简单地假设是真实位置到观测位置的距离为高斯分布"""
    # weights.fill(1.)
    print(len(z))
    if z.shape[0] > 2:
        #weights = np.ones(n_particles) / n_particles
        weights = weights
    else:
        distances = np.linalg.norm(particles[:, 0:2] - z, axis=1)
        weights *= scipy.stats.norm(0, d_std).pdf(distances)
        weights += 1.e-300
        weights /= sum(weights)


def estimate(particles, weights):
    """估计位置"""
    return np.average(particles, weights=weights, axis=0)


def neff(weights):
    """用来判断当前要不要进行重采样"""
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
    """根据指定的样本进行重采样"""
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)


def run_pf(particles, weights, z, x_range, y_range):
    """迭代一次粒子滤波，返回状态估计"""
    predict_particles(particles, 6, 0.1, x_range, y_range)  # 1. 预测
    update_particles(particles, weights, z, 10)  # 2. 更新
    if neff(weights) < len(particles) / 2:  # 3. 重采样
        indexes = multinomial_resample(weights)
        resample_from_index(particles, weights, indexes)
    return estimate(particles, weights)  # 4. 状态估计


index = np.loadtxt(r'E:\Data21\2019-01-15-20-39-07-3845317\index.txt').astype(int)
obpos = np.loadtxt(r'C:\torcheye\TorchEye-master\a.txt')
ta  = np.loadtxt(r'E:\Data21\2019-01-15-20-39-07-3845317\target.txt')
ii = np.where(ta[:,1] != 10000)[0]
ta = ta[ii]
print(max(index))
x_range, y_range = [0, 640], [0, 480]
n_particles = 100
particles = create_particles(x_range, y_range, 6, 0.1, n_particles) # 初始化粒子
weights = np.ones(n_particles) / n_particles # 初始化权重
t=0
knn_pf_predictions = np.empty((np.max(index),2))
presult = np.empty((n_particles*np.max(index), 4))
for i in range(np.max(index)):
    if i == index[t]:
        print(i)
        t = t + 1
        state = run_pf(particles, weights, obpos[t,:], x_range, y_range)
    else:
        state = run_pf(particles, weights, np.zeros((3,1)), x_range, y_range)
    presult[(i)*n_particles:(i+1)*n_particles,:] = particles
    knn_pf_predictions[i, :] = state[0:2]
    # if(i%10 == 0):
    #     plt.scatter(particles[:, 0], particles[:, 1])
    #     plt.show()

plt.plot(knn_pf_predictions[10:,0],knn_pf_predictions[10:,1])
plt.plot(obpos[10:,0],obpos[10:,1])
plt.plot(ta[10:,1],ta[10:,2])
plt.show()
np.savetxt('knn_pf_predictions.txt',knn_pf_predictions)
np.savetxt('knn_pf_particles.txt',presult)
