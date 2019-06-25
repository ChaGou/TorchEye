import torch
import math
from src.core import ConstantTerm as C
from src.dataprocess import FileDataSet
import numpy
import matplotlib.pyplot as plt

class SquareArray:
    la = 16.5
    n_row = 8
    n_antenna = n_row * n_row
    antenna_cor = torch.Tensor(n_antenna, 2)
    antenna_cor_r = torch.Tensor(n_antenna, 2)  # 极坐标
    center_y = -9.5-57.75
    center_x = 0


    def __init__(self,n_row,center_y,center_x):
        self.center_y = center_y
        self.center_x = center_x
        self.n_row = n_row
        self.n_antenna = n_row*n_row
        self.antenna_cor = torch.Tensor(self.n_antenna, 2)
        self.antenna_cor_r = torch.Tensor(self.n_antenna,2)
        for i in range(self.n_antenna):
            y = (i % self.n_row) * self.la - (self.n_row - 1)/2.0 * self.la
            x = (i // self.n_row) * self.la - (self.n_row - 1)/2.0 * self.la
            y = -y+center_y
            x = x+center_x
            self.antenna_cor[i, 0] = x
            self.antenna_cor[i, 1] = y
            self.antenna_cor_r[i, 0] = math.sqrt(( x-center_x) * ( x-center_x) + (y-center_y) * (y-center_y))
            self.antenna_cor_r[i, 1] = math.atan2(y-center_y, x-center_x)

    def theta_theory_XYZ(self, X, Y, Z):
        dx = self.antenna_cor[:,0].view(1,-1)-X
        dy = self.antenna_cor[:,1].view(1,-1)-Y
        d = torch.sqrt(dx*dx+dy*dy+(Z+2)*(Z+2))
        return -d/C.lamb*2*math.pi

    def theta_theory(self, al, be):  # 1XN
        theta_k = self.antenna_cor_r[:, 1].contiguous().view(self.n_antenna,1)
        r = self.antenna_cor_r[:, 0].contiguous().view(self.n_antenna,1)
        theta_t = 2 * math.pi / C.lamb *r*torch.cos(al - theta_k)*torch.cos(be)
        return theta_t

    def p0(self, al, be, theta_m):
        theta_t = self.theta_theory(al, be)
        k = theta_t.size(1)
        delta = theta_m - theta_t
        cosd = torch.cos(delta).sum(1)
        sind = torch.sin(delta).sum(1)
        p = torch.sqrt(cosd * cosd + sind * sind) / k
        return p


def cal_offset():
    center_x = -57.75 + (0 * 2 + 8 - 1) / 2.0 * 16.5
    center_y = -9.5 - (0 * 2 + 8 - 1) / 2.0 * 16.5
    test = SquareArray(8,center_y,center_x)
    trainPath = r'D:\dataset\datastatic'
    trainDataset = FileDataSet.FileDataset(trainPath + r'\traindata.txt',
                                           trainPath + r'\trainlabel.txt')
    inputs, labels = trainDataset[:]
    d = test.theta_theory_XYZ(torch.Tensor(labels[:, 0] / 10).view(-1, 1), torch.Tensor(labels[:, 1] / 10).view(-1, 1),
                              torch.Tensor(labels[:, 2] / 10).view(-1, 1))
    print(d.shape)
    print(inputs.shape)

    offset = numpy.zeros(shape=(64,1))
    for i in range(64):
        offset[i,0] = numpy.median(numpy.mod((inputs.numpy()[:, i] - (d.numpy()[:, i] - d.numpy()[:, 0])), numpy.pi * 2))
    plt.plot(offset)
    plt.show()
    numpy.savetxt('offsetfromstatic.txt', offset)

def test_sub_anttena(row,colomn,length):
    center_x = -57.75+(row*2+length-1)/2.0*16.5
    center_y = -9.5-(colomn*2+length-1)/2.0*16.5
    subAntenna = SquareArray(length, center_y, center_x)
    print(subAntenna.antenna_cor)
    subAntennaIndex = numpy.zeros(shape=length*length)
    for i in range(length*length):
        subAntennaIndex[i] = (i//length+colomn)*8+ i%length+row
    offset = numpy.loadtxt(r'offsetfromstatic.txt')
    offset = torch.Tensor(offset).view(1,-1)
    trainPath = r'D:\dataset\data2'
    trainDataset = FileDataSet.FileDataset(trainPath + r'\traindata.txt',
                                           trainPath + r'\trainlabel.txt')
    inputs, labels = trainDataset[:]
    inputs = inputs[:,subAntennaIndex]
    offset = offset[:,subAntennaIndex]
    inputs = inputs-offset
    labels = labels/10

    inputs = inputs.view(-1,9,1)
    w = 360
    h = 90
    Al = torch.linspace(0, w - 1, w).view(1, w)/180.0*numpy.pi
    Al = torch.matmul(torch.ones(h, 1), Al).view(1,1,w * h)
    Be = torch.linspace(0, h - 1, h).view(h, 1)/180.0*numpy.pi
    Be = torch.matmul(Be, torch.ones(1, w), ).view(1,1,w * h)

    altruth = numpy.arctan2(labels[:,1].numpy()-center_y,labels[:,0].numpy()-center_x)/numpy.pi*180
    betruth = numpy.arcsin((labels[:,2].numpy()+2)/numpy.sqrt((labels[:,1].numpy()-center_y)*(labels[:,1].numpy()-center_y)+(labels[:,0].numpy()-center_x)*(labels[:,0].numpy()-center_x)+(labels[:,2].numpy()+2)*(labels[:,2].numpy()+2)))/numpy.pi*180
    index = 10
    print(betruth[index])
    print(altruth[index])



    #plt.imshow(p0[index,:].view(h,w))
    for i in range(labels.shape[0]):
        index =  i
        a = subAntenna.p0(Al,Be,inputs[index,:,:].view(1,9,1)).view(h, w).numpy()
        print(str(betruth[index])+" "+str(altruth[index])+" "+str(numpy.unravel_index(a.argmax(), a.shape)[0])+" "+str(numpy.unravel_index(a.argmax(), a.shape)[1]))
    #plt.show()





#cal_offset()
#offset = numpy.loadtxt(r'offsetfromstatic.txt')
#test_sub_anttena(3,3,3)


