import torch
import math
import ConstantTerm as C
import CenterCamera
import torchvision
import numpy
import matplotlib.pyplot as plt
import FileDataSet
class SquareArray:
    la = 16.5
    n_row = 8
    n_antenna = n_row * n_row
    antenna_cor = torch.Tensor(n_antenna, 2)
    antenna_cor_r = torch.Tensor(n_antenna, 2)#极坐标

    def __init__(self):
        for i in range(self.n_antenna):
            y = (i % self.n_row) * self.la - (self.n_row - 1)/2 * self.la
            x = (i // self.n_row) * self.la - (self.n_row - 1)/2 * self.la
            y = -y
            x = -x
            self.antenna_cor[i, 0] = x
            self.antenna_cor[i, 1] = y
            self.antenna_cor_r[i, 0] = math.sqrt(x*x + y*y)
            self.antenna_cor_r[i, 1] = math.atan2(y, x)

    def theta_theory(self, al, be):  # 1XN
        theta_k = self.antenna_cor_r[:, 1].contiguous().view(self.n_antenna,1)
        r = self.antenna_cor_r[:, 0].contiguous().view(self.n_antenna,1)
        theta_t = 2 * math.pi / C.lamb *r*torch.cos(al - theta_k)*torch.cos(be)
        return -theta_t
    def theta_theory(self,X,Y,Z):
        dx = self.antenna_cor[:,0]-X
        dy = self.antenna_cor[:,1]-Y
        d = torch.sqrt(dx*dx+dy*dy+Z*Z)
        return d/C.lamb*2*math.pi

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
    def p0(self,X,Y,Z,theta_m):
        theta_t = self.theta_theory(X, Y, Z)
        data = torch.zeros((theta_t.shape[0], 56 * 2))
        data[:, 0:56] = theta_t[:, 8:64] - theta_t[:, 0:56]
        for i in range(8):
            data[:, 7 * i + 56:7 * i + 56 + 7] = theta_t[:, 8 * i + 1:8 * i + 8] - theta_t[:, 8 * i:8 * i + 7]
        theta_t = data
        # theta_m=theta_m.view(1,64)
        # data = torch.zeros((theta_m.shape[0], 56 * 2))
        # data[:, 0:56] = theta_m[:, 8:64] - theta_m[:, 0:56]
        # for i in range(8):
        #     data[:, 7 * i + 56:7 * i + 56 + 7] = theta_m[:, 8 * i + 1:8 * i + 8] - theta_m[:, 8 * i:8 * i + 7]
        # theta_m = data
        delta = theta_m + theta_t-math.pi
        delta = (delta+5*math.pi) %(2*math.pi)-math.pi
        #delta = torch.cos(delta)
        #cosd = torch.cos(delta).sum(1)
        #sind = torch.sin(delta).sum(1)
        #p = torch.sqrt(cosd * cosd + sind * sind)
        p =torch.abs(delta[:,0:56]).sum(1)
        return p


test = SquareArray()
#print(test.antenna_cor)
w = 400
h=400
X = torch.linspace(0, w - 1, w).view(1, w) - w / 2
X = torch.matmul(torch.ones(h, 1), X).view( w * h,1)
Y = torch.linspace(0, h - 1, h).view(h, 1) - h / 2
Y = torch.matmul(Y, torch.ones(1, w), ).view( w * h,1)
Z = torch.ones(w*h,1)*250
testPath = r'E:\DataTest'
testDataset = FileDataSet.FileDataset(testPath+r'\traindata.txt',
                                          testPath+r'\trainlabel.txt')
testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
data,label = testDataset[100]

p=(test.p0(X,Y,Z,data))
a = torch.max(p)
print(a)
print(torch.min(p))
print(label)
data2 = data[0,0:56]
print(data2)

# theta_m = test.theta_simu(torch.Tensor([math.pi*0/180]),torch.Tensor([math.pi*70/180]),torch.Tensor([3]))
# p = test.p0(al,be,theta_m).view(h, w)
# print(cam.getAlBefromPixal(torch.Tensor([82]),torch.Tensor([200])))
# a,b = cam.getAlBefromPixal(torch.Tensor([82]),torch.Tensor([200]))
# print(test.p0(a,b,theta_m))
# print(p)
import matplotlib.pyplot as plt
plt.imshow(p.view(w,h))
#plt.figure()

#plt.imshow(data2.view(-1,8))
plt.show()

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

