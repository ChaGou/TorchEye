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
TestMoade = pm.Mode.Regression
if __name__ == '__main__':
    #import TrainTestbed
    model = torch.load('b.model')
    model.eval()
    testPath = r'E:\Data7'
    testDataset = FileDataSet.FileDatasetRNN(testPath+r'\traindata.txt',
                                          testPath+r'\trainlabel.txt',10)
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    for i, data in enumerate(testloader, 0):
        if i % 100 != 0:
            continue
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(Variable(inputs))
        a = outputs.view(-1,3072).cpu().detach().numpy()
        b = labels.cpu().numpy()
        print(b)
        #plt.plot(b / 10)
        #plt.figure()
        # for i in range(100):
        #     plt.imshow(a[:, i*10].reshape(48, 64))
        #     plt.show()
        plt.scatter(b[:,0],b[:,1])
        plt.imshow(a[:,: ].reshape(48, 64))
        #plt.imshow(np.sum(a.reshape(48, 64, -1), 0))
        plt.show()





    plt.show()


