import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from src.core import Parameters as pm

from src.dataprocess import FileDataSet

TestMoade = pm.Mode.Regression
if __name__ == '__main__':
    #import TrainTestbed
    model = torch.load('c.core')
    model.eval()
    trainPath = r'E:\DataTest'
    trainDataset = FileDataSet.FileDataset(trainPath + r'\traindata.txt',
                                           trainPath + r'\trainlabel.txt')
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    testPath = r'E:\Data7'
    testDataset = FileDataSet.FileDataset(testPath + r'\traindata.txt',
                                          testPath +r'\trainlabel.txt')
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)

    inputs, labels = testDataset[:]
    inputs2, labels2 = trainDataset[:]
    inputs = model(Variable(inputs.cuda())).detach().cpu()
    inputs2 = model(Variable(inputs2.cuda())).detach().cpu()
    result=[]
    result_1=[]
    result_2 = []
    result_3 = []
    result2=[]
    for i in range(500):
        delta = inputs[i,:]-inputs2
        #delta = delta[:,0:56]
        #cosd = torch.cos(delta).sum(1)
        #sind = torch.sin(delta).sum(1)
        #p = torch.sqrt(cosd * cosd + sind * sind)
        p = torch.abs(delta).sum(1)
        sorted,ind = torch.sort(p)
        print(ind)
        p = p / p.max()
        p = 1 - p
        result.append(labels2[torch.argmax(p),:].numpy())
        result_1.append(labels2[ind[0],:].numpy())
        result_2.append(labels2[ind[1], :].numpy())
        result_3.append(labels2[ind[2], :].numpy())
        result2.append(p.numpy())

    result = np.asarray(result)
    result2 = np.asarray(result2)
    result_1 = np.asarray(result_1)
    result_2 = np.asarray(result_2)
    result_3 = np.asarray(result_3)
    np.savetxt('dd.txt',result2)
    ind = np.argsort(labels2.numpy()[:,0])
    result2 = result2[:,ind]
    plt.plot(result[:,0])
    plt.plot(result_1[:, 0])
    plt.plot(result_2[:, 0])
    plt.plot(result_3[:, 0])
    plt.plot(labels.numpy()[:,0])
    plt.figure()
    plt.contour(np.transpose(result2))
    #plt.xticks(labels2.numpy()[ind,0])
    plt.show()

