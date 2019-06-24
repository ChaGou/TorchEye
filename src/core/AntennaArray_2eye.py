import torch
import math
from src.core import ConstantTerm as C
from src.dataprocess import FileDataSet
import numpy

class SquareArray:
    la = 16.5
    n_row = 8
    n_antenna = n_row * n_row
    antenna_cor = torch.Tensor(n_antenna, 2)


    def __init__(self):
        for i in range(self.n_antenna):
            y = (i % self.n_row) * self.la - (self.n_row - 1)/2 * self.la
            x = (i // self.n_row) * self.la - (self.n_row - 1)/2 * self.la
            y = -y-9.5-57.75
            x = x
            self.antenna_cor[i, 0] = x
            self.antenna_cor[i, 1] = y



    def theta_theory(self,X,Y,Z):
        dx = self.antenna_cor[:,0].view(1,-1)-X
        dy = self.antenna_cor[:,1].view(1,-1)-Y
        d = torch.sqrt(dx*dx+dy*dy+(Z+2)*(Z+2))
        return -d/C.lamb*2*math.pi




test = SquareArray()
trainPath = r'D:\dataset\data1'
trainDataset = FileDataSet.FileDataset(trainPath + r'\traindata.txt',
                                           trainPath + r'\trainlabel.txt')
trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
inputs, labels = trainDataset[:]
d=test.theta_theory(torch.Tensor(labels[:,0]/10).view(-1,1),torch.Tensor(labels[:,1]/10).view(-1,1),torch.Tensor(labels[:,2]/10).view(-1,1))
print(d.shape)
print(inputs.shape)
import matplotlib.pyplot as plt
dd= d- inputs
plt.plot(inputs.numpy()[:,16]-inputs.numpy()[:,0])
plt.plot(numpy.mod(d.numpy()[:,16]-d.numpy()[:,0],numpy.pi*2))
plt.figure()
plt.hist(numpy.mod((inputs.numpy()[:,20]-(d.numpy()[:,20]-d.numpy()[:,0])),numpy.pi*2),200)
plt.show()