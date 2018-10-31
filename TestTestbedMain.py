import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import CHAModule
import CenterCamera
import numpy as np
import matplotlib.pyplot as plt



import FileDataSet
if __name__ == '__main__':
    #import TrainTestbed
    model = torch.load('a.model')
    model.eval()

    testDataset = FileDataSet.FileDataset(r'..\VSWorkSpace\RadarEye\RadarEye\bin\Debug\data2\testdata2.txt',
                                          r'..\VSWorkSpace\RadarEye\RadarEye\bin\Debug\data2\testlabel.txt')
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    criterion = nn.MSELoss()

    randindex = torch.linspace(1,100,100)#np.random.randint(0, 80, size=[10])
    for i in randindex:
        inputs, labels = testDataset[int(i)]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(Variable(inputs ))
        scal = (torch.Tensor([640, 480]).view(1, 2)).cuda()
        print(outputs.data * scal)
        print(labels)
        print('========')

    inputs, labels = testDataset[:]
    outputs=model(Variable(inputs.cuda()))
    scal = (torch.Tensor([640, 480]).view(1, 2)).cuda()
    np.savetxt('a.txt', (outputs.data*scal).cpu().numpy(), fmt='%.6f')
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs ), Variable(labels /torch.Tensor([640,480]).view(1,2))
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()


            outputs = model(inputs)
            loss = criterion(outputs, labels)


            # print statistics
            running_loss += loss.data[0]
            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
                # break

    print('Finished Training')
    inputs, labels = testDataset[:]
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    outputs = model(Variable(inputs))
    a=labels.cpu().numpy()
    print(a.shape)
    plt.plot(a[:,0])
    a = (outputs.data*scal).cpu().numpy()
    plt.plot(a[:,0])
    print(a.shape)
    plt.show()