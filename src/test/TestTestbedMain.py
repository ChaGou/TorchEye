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
    model = torch.load('a.model')
    model.eval()
    testPath = r'E:\Data21'
    testDataset = FileDataSet.FileDataset(testPath+r'\traindata.txt',
                                          testPath+r'\trainlabel.txt')
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    criterion = nn.MSELoss()
    modelAE = torch.load('c.model')

    # randindex = torch.linspace(1,100,100)#np.random.randint(0, 80, size=[10])
    # for i in randindex:
    #     inputs, labels = testDataset[int(i)]
    #     labels=labels.view(1,1,2)
    #     one_hot = torch.zeros(1, 640).scatter_(1, labels.data[:,:,0],1)
    #     inputs, labels = Variable(inputs), Variable(one_hot.view(-1, 640))
    #     if torch.cuda.is_available():
    #         inputs = inputs.cuda()
    #         labels = labels.cuda()
    #     outputs = model(Variable(inputs ))
    #     scal = (torch.Tensor([640, 480]).view(1, 2)).cuda()
    #
    #     #print(outputs.data * scal)
    #     print(outputs.data)
    #     print(labels)
    #     print('========')

    # inputs, labels = testDataset[:]
    # outputs=model(Variable(inputs.cuda()))
    # #scal = (torch.Tensor([640, 480]).view(1, 2)).cuda()
    # #np.savetxt('a.txt', (outputs.data*scal).cpu().numpy(), fmt='%.6f')
    # np.savetxt('a.txt', (outputs.data).cpu().numpy(), fmt='%.6f')
    # for epoch in range(1):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(testloader, 0):
    #         # get the inputs
    #         inputs, labels = data
    #         one_hot = torch.zeros(labels.size(0), 640).scatter_(1, labels.data[:, :, 0], 1)
    #         inputs, labels = Variable(inputs), Variable(one_hot.view(-1, 1, 640))
    #
    #         # wrap them in Variable
    #         #inputs, labels = Variable(inputs ), Variable(labels /torch.Tensor([640,480]).view(1,2))
    #         if torch.cuda.is_available():
    #             inputs = inputs.cuda()
    #             labels = labels.cuda()
    #
    #
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #
    #
    #         # print statistics
    #         running_loss += loss.data
    #         if i % 20 == 19:  # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.5f' %
    #                   (epoch + 1, i + 1, running_loss / 20))
    #             running_loss = 0.0
    #             # break
    #
    # print('Finished Training')
    if TestMoade == pm.Mode.Classification1LabelHeatMap:
        inputs, labels = testDataset[:]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
       #outputs = model(Variable(modelAE.encoder(inputs)))
        outputs = model(Variable((inputs)))
        a = outputs.cpu().detach().numpy().transpose()
        # x = F.softmax(torch.Tensor(a[:pm.OutputShape[0],:]),0)
        # values, indices = torch.max(x, 0)
        # y = F.softmax(torch.Tensor(a[pm.OutputShape[0]:, :]), 0)
        # values2, indices2 = torch.max(y, 0)
        # index =torch.cat((indices.view(-1,1),indices2.view(-1,1)),1)
        # np.savetxt('a.txt',index.numpy().astype(int))
        # np.savetxt('b.txt', x.numpy())
        b = labels.cpu().numpy()[0::10, 0]
        plt.plot(b / 10)
        # plt.figure()
        # for i in range(100):
        #     plt.imshow(a[:, i * 10].reshape(48, 64))
        #     plt.show()
        #plt.imshow(a[:, 1].reshape(48, 64))
        plt.imshow(np.sum(a[:, 0::10].reshape(48, 64, -1), 0))
        temp = np.unravel_index(np.argmax(a[:, 0::10], 0), (48, 64))
        plt.plot(temp[1])
        plt.figure()
        b = labels.cpu().numpy()[0::10, 1]
        plt.plot(b / 10)
        # plt.figure()
        # for i in range(100):
        #     plt.imshow(a[:, i * 10].reshape(48, 64))
        #     plt.show()
        # plt.imshow(a[:, 1].reshape(48, 64))
        plt.imshow(np.sum(a[:, 0::10].reshape(48, 64, -1), 1))
        temp = np.unravel_index(np.argmax(a[:, 0::10], 0), (48, 64))
        plt.plot(temp[0])
        plt.figure()
        r1=labels.cpu().numpy()[0::10, 1]/10-temp[0]
        r2=labels.cpu().numpy()[0::10, 0]/10-temp[1]
        plt.hist(np.abs(r1),100)
        plt.figure()
        plt.hist(np.abs(r2),100)
        plt.figure()
        plt.hist(np.abs(np.sqrt(r1*r1+r2*r2)), 100)
        plt.figure()
        plt.plot(np.sqrt(r1*r1+r2*r2))
        print(np.mean(np.abs(r1)))
        print(np.mean(np.abs(r2)))

        temp = np.unravel_index(np.argmax(a[:,:], 0), (48, 64))
        temp2 = np.hstack((temp[1].reshape(-1,1)*10,temp[0].reshape(-1,1)*10))
        np.savetxt('a.txt', temp2)
        # for i, data in enumerate(testloader, 0):
        #     if i % 100 != 0:
        #         continue
        #     inputs, labels = data
        #     if torch.cuda.is_available():
        #         inputs = inputs.cuda()
        #         labels = labels.cuda()
        #     outputs = model(Variable(inputs))
        #     a = outputs.view(-1, 3072).cpu().detach().numpy()
        #     b = labels.cpu().numpy()
        #     print(b)
        #     plt.imshow(a[:, :].reshape(48, 64))
        #     plt.show()
    elif TestMoade == pm.Mode.Regression:
        inputs, labels = testDataset[:]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(Variable((inputs)))
        b = labels.cpu().numpy()[:, 0]
        plt.plot(b)
        plt.plot(outputs.detach().cpu().numpy()[:,0]*640)
        r1 = b - outputs.detach().cpu().numpy()[:,0]*640
        plt.figure()
        b = labels.cpu().numpy()[:, 1]
        plt.plot(b)
        plt.plot((outputs.detach().cpu().numpy()[:, 1]) * 240+240)
        r2 = b - ((outputs.detach().cpu().numpy()[:, 1]) * 240+240)
        np.savetxt('error.txt',np.sqrt(r1*r1+r2*r2))
        plt.show()
        temp2 = np.hstack((outputs.detach().cpu().numpy()[:,0].reshape(-1, 1) * 640, outputs.detach().cpu().numpy()[:,1].reshape(-1, 1) * 240+240))
        np.savetxt('a.txt', temp2)
    elif TestMoade == pm.Mode.Classification2LabelsOneHot:
        inputs, labels = testDataset[:]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(Variable(inputs))
        a = outputs.cpu().detach().numpy().transpose()
        x = F.softmax(torch.Tensor(a[pm.OutputShape[0]:,:]),0)
        #values, indices = torch.max(x, 0)
        #y = F.softmax(torch.Tensor(a[pm.OutputShape[0]:, :]), 0)
        plt.imshow(x*x)
        b = labels.cpu().numpy()[:, 1]
        plt.plot(b)
        aa=np.argmax(x[:, 0::1], 0)
        plt.plot(aa.numpy())
        plt.show()




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

