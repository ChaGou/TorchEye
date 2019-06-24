import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from src.core import Parameters as pm

from src.dataprocess import FileDataSet

TestMoade = pm.LearningMode.Regression
if __name__ == '__main__':
    #import TrainTestbed
   # model = torch.load('c.core')
    #model.eval()
    trainPath = r'D:\dataset\data3'
    trainDataset = FileDataSet.FileDataset(trainPath + r'\traindata.txt',
                                           trainPath + r'\trainlabel.txt')
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    testPath = r'D:\dataset\data1'
    testDataset = FileDataSet.FileDataset(testPath + r'\traindata.txt',
                                          testPath +r'\trainlabel.txt')
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)

    inputs, labels = testDataset[:]
    inputs2, labels2 = trainDataset[:]
    #inputs = model(Variable(inputs.cuda())).detach().cpu()
    #inputs2 = model(Variable(inputs2.cuda())).detach().cpu()
    result=[]
    result_1=[]
    result_2 = []
    result_3 = []
    result2=[]
    for i in range(1000):
        delta = inputs[i,:]-inputs2
        delta2 = labels[i,:] - labels2
        #delta = delta[:,0:56]
        cosd = torch.cos(delta).sum(1)
        sind = torch.sin(delta).sum(1)
        p = torch.sqrt(cosd * cosd + sind * sind)
        sorted,ind = torch.sort(p,descending=True)
        print(ind)
        p = p / p.max()
        #p = 1 - p
        result.append(labels2[torch.argmax(p),:].numpy())
        result_1.append(labels2[ind[0],:].numpy())
        result_2.append(labels2[ind[1], :].numpy())
        result_3.append(labels2[ind[2], :].numpy())
        p = torch.abs(delta2).sum(1)
        sorted, ind = torch.sort(p, descending=False)
        result2.append(sorted[0])

    result = np.asarray(result)
    result2 = np.asarray(result2)
    result_1 = np.asarray(result_1)
    result_2 = np.asarray(result_2)
    result_3 = np.asarray(result_3)
    np.savetxt('dd.txt',result2)

    plt.plot(result[:,0])
    plt.plot(result_1[:, 0])
    plt.plot(result_2[:, 0])
    plt.plot(result_3[:, 0])
    plt.plot(labels.numpy()[:,0])
    plt.plot(labels2.numpy()[:, 0])
    plt.plot(result2)
    #plt.xticks(labels2.numpy()[ind,0])
    plt.show()

