import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from src.core import CenterCamera, CHAModule

from src.dataprocess import ImageDataset

if __name__ == '__main__':
    imageDataset = ImageDataset.ImageDataset('../images5/', 'testfile55.txt')
    trainloader = torch.utils.data.DataLoader(imageDataset, batch_size=1,
                                              shuffle=True, num_workers=0)
    model = CHAModule.MyNet2()
    camera = CenterCamera.Camera()
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    inputs, labels = imageDataset[0]
    print(torch.max(inputs))
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            ux = camera.getUniformPixalFromAlBe(labels[0, 0, 0], labels[0, 1, 0])
            labels = ux.view(1,2,1)

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
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

    inputs, labels = imageDataset[0]
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    outputs = model(Variable(inputs.view(1,1,400,400)))
    print(outputs)
    ux = camera.getUniformPixalFromAlBe(labels[ 0, 0], labels[ 1, 0])
    labels = ux.view(1, 2, 1)
    print(labels)

