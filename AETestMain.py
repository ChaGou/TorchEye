import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import CHAModule
import CenterCamera
import numpy as np
import matplotlib.pyplot as plt
import Parameters as pm
import torch.nn.functional as F
from scipy import signal


import FileDataSet
TestMoade = pm.Mode.Classification2LabelsOneHot
if __name__ == '__main__':
    #import TrainTestbed
    model = torch.load('c.model')
    model.eval()
    testPath = r'E:\Data7'
    testDataset = FileDataSet.FileDataset(testPath+r'\traindata.txt',
                                          testPath+r'\trainlabel.txt')
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    criterion = nn.MSELoss()
    inputs, labels = testDataset[0:10000]
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    outputs = model((Variable(inputs)))
    a = outputs.cpu().detach().numpy().transpose()
    # values, indices = torch.max(x, 0)
    # y = F.softmax(torch.Tensor(a[pm.OutputShape[0]:, :]), 0)
    plt.imshow(inputs.cpu().detach().numpy().transpose())
    plt.figure()
    plt.imshow(outputs.cpu().detach().numpy().transpose())
    plt.figure()
    plt.plot(inputs.cpu().detach().numpy().transpose()[1,:]-outputs.cpu().detach().numpy().transpose()[1,:])
    plt.show()
    print(np.median(np.abs(inputs.cpu().detach().numpy().transpose()[100,:]-outputs.cpu().detach().numpy().transpose()[100,:])))






    #plt.figure()



    plt.show()

    #x=x.numpy()
    # print(np.where(x<0.9))
    #x[np.where(x<0.99)[0],np.where(x<0.99)[1]] = 0
    #

    #plt.contour(x)
    # print(a.shape)
    # a = labels.cpu().numpy()
    # print(a.shape)
    # #plt.figure()
    #
    # plt.figure()
    # #plt.plot(index.numpy()[:,0],index.numpy()[:,1])
    # plt.plot(index.numpy()[:, 0])

