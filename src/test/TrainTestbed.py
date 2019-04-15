import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from src.dataprocess import FileDataSet
from src.core import Parameters as pm, CHAModule
import math


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.cro = nn.CrossEntropyLoss()

    def forward(self, pred, truth):
        return self.cro(pred[:,:pm.OutputShape[0]],truth[:,0,0].cuda())+self.cro(pred[:,pm.OutputShape[0]:],truth[:,0,1].cuda())

workMode = pm.learnMode
dataMode = pm.dataMode
testPath = r'E:\Data8'
fileDataset = FileDataSet.FileDataset(testPath + r'\traindata.txt',
                                      testPath +r'\trainlabel.txt')
#fileDataset.Uniform()
#fileDataset.make_more(2,0.01)
trainloader = torch.utils.data.DataLoader(fileDataset, batch_size=10,
                                          shuffle=True, num_workers=0)

model = CHAModule.MyNet1(64, 640)
modelAE = torch.load('c.core')
#core = CHAModule.MyNet3()

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss(size_average=True)
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

if workMode == pm.LearningMode.Classification2LabelsOneHot:
    criterion = MyLoss()
    model = CHAModule.MyNet1(56 * 2, pm.OutputShape[0] + pm.OutputShape[1])
elif workMode == pm.LearningMode.Regression:
    criterion = nn.MSELoss()
    if dataMode == pm.DataMode.DeltaMode:
        model = CHAModule.MyNet1(112, 2)
    elif dataMode == pm.DataMode.SquareMode:
        model = CHAModule.CNN_Up()
elif workMode == pm.LearningMode.Classification1LabelHeatMap:
    criterion = nn.MSELoss()
    model = CHAModule.MyNet1(112, 64 * 48)
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
        l=labels
        if workMode == pm.LearningMode.Classification1Label:
            one_hot = torch.zeros(labels.size(0), 640).scatter_(1, labels.data[:,:,0], 1)
            inputs, labels = Variable(inputs), Variable(one_hot.view(-1, 1, 640))
        elif workMode == pm.LearningMode.Classification2LabelsOneHot:
            one_hot = torch.zeros(labels.size(0), pm.OutputShape[0]).scatter_(1, labels.data[:, :, 0], 1)
            one_hot2 = torch.zeros(labels.size(0), pm.OutputShape[1]).scatter_(1, labels.data[:, :, 1], 1)
            inputs, labels = Variable(inputs), Variable(torch.cat((
                one_hot.view(-1, 1, pm.OutputShape[0]),one_hot2.view(-1, 1, pm.OutputShape[1])),2))
        elif workMode == pm.LearningMode.Regression:
            inputs, labels = Variable(inputs), Variable((labels -torch.Tensor([0, 240]).view(1, 2))/ torch.Tensor([640, 240]).view(1, 2))
        elif workMode == pm.LearningMode.Classification1LabelHeatMap:
            inputs, labels = Variable(inputs), Variable(HeatMapFromLabel(labels,1.5).view(-1,1,64*48))
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
        #outputs = core(modelAE.encoder(inputs))
        outputs = model((inputs))

        #bb=sum(outputs[0,0,:])
        #labels = Variable(torch.LongTensor(labels.view(-1,640)))
        #loss = criterion(outputs.view(-1,640),l[:,0,0].cuda())
        if workMode == pm.LearningMode.Classification2LabelsOneHot:
            loss = criterion(outputs.view(-1,1120),l)
        elif workMode == pm.LearningMode.Regression or workMode == pm.LearningMode.Classification1LabelHeatMap:
            #criterion = nn.BCELoss(weight=(labels)+0.1,size_average=True)
            loss = criterion(outputs,labels)
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