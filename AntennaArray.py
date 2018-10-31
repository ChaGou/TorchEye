import torch
import math
import ConstantTerm as C
import CenterCamera
import torchvision

class SquareArray:
    la = 0.16
    n_row = 8
    n_antenna = n_row * n_row
    antenna_cor = torch.Tensor(n_antenna, 2)
    antenna_cor_r = torch.Tensor(n_antenna, 2)#极坐标

    def __init__(self):
        for i in range(self.n_antenna):
            x = (i % self.n_row) * self.la - (self.n_row - 1)/2 * self.la
            y = (i // self.n_row) * self.la - (self.n_row - 1)/2 * self.la
            self.antenna_cor[i, 0] = x
            self.antenna_cor[i, 1] = y
            self.antenna_cor_r[i, 0] = math.sqrt(x*x + y*y)
            self.antenna_cor_r[i, 1] = math.atan2(y, x)

    def theta_theory(self, al, be):  # 1XN
        theta_k = self.antenna_cor_r[:, 1].contiguous().view(self.n_antenna,1)
        r = self.antenna_cor_r[:, 0].contiguous().view(self.n_antenna,1)
        theta_t = 2 * math.pi / C.lamb *r*torch.cos(al - theta_k)*torch.cos(be)
        return -theta_t

    def theta_simu(self, al, be, d):
        return 2 * math.pi / C.lamb * d + self.theta_theory(al, be)+torch.randn(al.size())*0.1

    def p0(self, al, be, theta_m):
        theta_t = self.theta_theory(al, be)
        k = theta_t.size(0)
        delta = theta_m - theta_t
        cosd = torch.cos(delta).sum(0)
        sind = torch.sin(delta).sum(0)
        p = torch.sqrt(cosd*cosd + sind*sind) / k
        print(k)
        return p



test = SquareArray()
cam = CenterCamera.Camera()
w = int(cam.s[0,0])
h = int(cam.s[1,0])
pixalX=torch.linspace(0,w-1,w).view(1,w)
pixalX = torch.matmul(torch.ones(h,1),pixalX).view(1,w*h)
pixalY=torch.linspace(0,h-1,h).view(h,1)
pixalY = torch.matmul(pixalY,torch.ones(1,w),).view(1,w*h)
cam = CenterCamera.Camera()
al,be = cam.getAlBefromPixal(pixalX,pixalY)
# theta_m = test.theta_simu(torch.Tensor([math.pi*0/180]),torch.Tensor([math.pi*70/180]),torch.Tensor([3]))
# p = test.p0(al,be,theta_m).view(h, w)
# print(cam.getAlBefromPixal(torch.Tensor([82]),torch.Tensor([200])))
# a,b = cam.getAlBefromPixal(torch.Tensor([82]),torch.Tensor([200]))
# print(test.p0(a,b,theta_m))
# print(p)
# import matplotlib.pyplot as plt
# plt.imshow(p)
# plt.figure()
# plt.imshow(theta_m.view(8,8))
# plt.show()

#print(torch.max(be))
#p=p*p
#torchvision.utils.save_image(torch.abs(be/math.pi*2).view(h,w), 'images/a.jpg')
# file = open('testfile55.txt', 'w')
#
# for i in range(0,10000):
#     als=torch.rand(1, 1) * 2 * math.pi
#     bes=torch.rand(1, 1) * (0.5-1/6) * math.pi+math.pi/6
#     d = torch.rand(1, 1) * 5
#     theta_m = test.theta_simu(als, bes, d)
#     p = test.p0(al, be, theta_m)
#     torchvision.utils.save_image(p.view(h, w), '../images5/'+str(i)+'.jpg')
#     file.write(str(als[0,0])+' '+str(bes[0,0])+'\n')
# file.close()

