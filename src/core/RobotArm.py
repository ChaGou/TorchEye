import torch
import math
from src.core import ConstantTerm as c

N_ant = 64

r = 0.16


def get_fake_data(batch_size=1000):
    al = torch.rand(batch_size, 1) * 2 * math.pi
    be = torch.rand(batch_size, 1) * (0.5-1/6) * math.pi+math.pi/6
    dis = torch.rand(batch_size, 1) * 5 + 0.5
    return al, be, dis


def get_fake_measure(alpha, beta, d, N_samples=1000):
    theta_ant = torch.linspace(0, 2 * math.pi * (1 - 1 / N_ant), N_ant)
    theta_ant = theta_ant.view(N_ant, 1)
    k1 = 2
    k2 = torch.matmul(torch.ones(N_ant, 1), torch.rand(1, N_samples))
    k1 = 0
    k2 = 0
    theta_ant = torch.matmul(theta_ant, torch.ones(1, N_samples))

    alpha_ = torch.matmul(torch.ones(N_ant, 1), torch.transpose(alpha, 0, 1))
    beta_ = torch.matmul(torch.ones(N_ant, 1), torch.transpose(beta, 0, 1))
    d = torch.matmul(torch.ones(N_ant, 1), torch.transpose(d, 0, 1))
    d = 0
    theta_m = torch.fmod(2 * math.pi / c.lamb * 2 * (d - r * torch.cos(alpha_ - theta_ant) * torch.cos(beta_) + k1 * theta_ant + k2 *
                                        torch.cos(2 * theta_ant)), 2 * math.pi)
    #torch.fmod(

        #, 2 * math.pi)

    return theta_m


alpha, beta, d = get_fake_data(1000)
y = torch.cat((alpha, beta), 1)
y = torch.transpose(y, 0, 1)
theta_m = get_fake_measure(alpha, beta, d, 1000)

alpha, beta, d = get_fake_data(10)
y2 = torch.cat((alpha, beta), 1)
y2 = torch.transpose(y2, 0, 1)
theta_m2 = get_fake_measure(alpha, beta, d, 10)

# y = y.contiguous()
# y=y.view(50,40)
# torchvision.utils.save_image(y, 'a.jpg')
#torch.save(theta_m,'theta_m')

#the = torch.load('theta_m')
# plt.scatter(alpha.squeeze().numpy(), beta.squeeze().numpy())
# plt.plot(torch.cos(theta_ant).squeeze().numpy())
# plt.show()

