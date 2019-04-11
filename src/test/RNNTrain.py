import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from src.core import CHAModule
from src.dataprocess import FileDataSet
import math

N_STEPS = 10
N_INPUTS = 56*2
N_NEURONS = 500
N_OUTPUTS = 64*48
N_EPHOCS = 10
BATCH_SIZE = 5


testPath = r'E:\Data6'
fileDataset = FileDataSet.FileDatasetRNN(testPath + r'\traindata.txt',
                                         testPath +r'\trainlabel.txt', N_STEPS)

#fileDataset.make_more(2,0.01)
trainloader = torch.utils.data.DataLoader(fileDataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

model = CHAModule.MyRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
criterion = nn.MSELoss()
#core = CHAModule.MyNet3()


w = 64
h = 48
pixalX = torch.linspace(0, w - 1, w).view(1, w)
pixalX = torch.matmul(torch.ones(h, 1), pixalX).view(1, w * h)
pixalY = torch.linspace(0, h - 1, h).view(h, 1)
pixalY = torch.matmul(pixalY, torch.ones(1, w), ).view(1, w * h)
def HeatMapFromLabel(labels,delta):
    dx = torch.floor(labels.view(-1,2)[:,0] / 10).view(-1,1) - pixalX
    dy = torch.floor(labels.view(-1,2)[:,1] / 10).view(-1,1) - pixalY
    g = dx * dx + dy*dy
    g = torch.exp(-g/2/delta/delta)*20#/2/math.pi/delta/delta
    #print(torch.sum(g))
    return g
def HeatMapFromLabel2(labels,delta):
    dx = torch.floor(labels.view(-1,2)[:,0] / 10).view(-1,1) - torch.linspace(0, w - 1, w).view(1, w)
    g = dx * dx
    g = torch.exp(-g/2/delta/delta)/math.sqrt(2*math.pi)/delta*12
    #print(torch.sum(g))
    #print(g[0,:])
    return g

optimizer = optim.SGD(model.parameters(), lr=1e-2)
model.train()
if torch.cuda.is_available():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model.cuda()
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(HeatMapFromLabel(labels,1).view(-1,N_STEPS,64*48))
        # wrap them in Variable
        #

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        #core.hidden = core.init_hidden()
        #plt.imshow(labels[0,:].view(48,64).cpu().numpy(),plt.cm.gray)
        #plt.plot(labels[0,0,:].cpu().numpy())
        #plt.show()
        # forward + backward + optimize
        outputs = model(inputs)

        #bb=sum(outputs[0,0,:])
        #labels = Variable(torch.LongTensor(labels.view(-1,640)))
        #loss = criterion(outputs.view(-1,640),l[:,0,0].cuda())
        #loss = criterion(outputs,labels)
        loss = criterion(outputs,labels[:,N_STEPS-1:].view(1,-1,3072))
            #loss = -outputs*labels
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.20f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
            torch.save(model, 'b.core')
            #break
    # if running_loss <= 0.01:
    #     break
print('Finished Training')

torch.save(model,'b.core')