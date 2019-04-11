import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import CHAModule
import CenterCamera
import numpy as np
import FileDataSet
import  Parameters as pm
import math
import os
import matplotlib.pyplot as plt


testPath = r'E:\Data8'
fileDataset = FileDataSet.FileDataset(testPath+r'\traindata.txt',
                                      testPath+r'\trainlabel.txt')
fileDataset.Uniform()
#fileDataset.make_more(2,0.01)
trainloader = torch.utils.data.DataLoader(fileDataset, batch_size=10,
                                          shuffle=True, num_workers=0)

model = CHAModule.MyAutoEncoder(56*2, 10)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
model.train()
criterion = nn.MSELoss()
if torch.cuda.is_available():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model.cuda()
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        #

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        #plt.imshow(labels[0,:].view(48,64).cpu().numpy(),plt.cm.gray)
        #plt.plot(labels[0,0,:].cpu().numpy())
        #plt.show()
        # forward + backward + optimize
        outputs = model(inputs)


        loss = criterion(outputs,inputs)
            #loss = -outputs*labels
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.20f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
            torch.save(model, 'c.core')
            #break
    # if running_loss <= 0.01:
    #     break
print('Finished Training')

torch.save(model,'c.core')