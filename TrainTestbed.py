import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import CHAModule
import CenterCamera
import numpy as np
import FileDataSet

fileDataset = FileDataSet.FileDataset(r'D:\VSWorkSpace\RadarEye\RadarEye\bin\Debug\data2\traindataUni.txt',
                                      r'D:\VSWorkSpace\RadarEye\RadarEye\bin\Debug\data2\trainlabelUni.txt')
fileDataset.make_more(5,0.01)
trainloader = torch.utils.data.DataLoader(fileDataset, batch_size=50,
                                          shuffle=True, num_workers=0)

model = CHAModule.MyNet1(120, 2)
#model = CHAModule.MyNet3()
model.train()
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data


        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels/torch.Tensor([640,480]).view(1,2))
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            #break

print('Finished Training')

torch.save(model,'a.model')