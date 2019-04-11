import torch
from src.core import CenterCamera, Parameters as pm, AntennaArray as aa
import torchvision
import math

from src.dataprocess import FileDataSet

TestMoade = pm.Mode.Regression
if __name__ == '__main__':
    #import TrainTestbed
    model = torch.load('a.core')
    model.eval()
    testPath = r'E:\DataTest'
    testDataset = FileDataSet.FileDataset(testPath + r'\testdata.txt',
                                          testPath +r'\testlabel.txt')
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    inputs, labels = testDataset[:]
    square_array = aa.SquareArray()
    cam = CenterCamera.Camera()
    w = int(cam.s[0, 0])
    h = int(cam.s[1, 0])
    pixalX = torch.linspace(0, w - 1, w).view(1, w)
    pixalX = torch.matmul(torch.ones(h, 1), pixalX).view(1, w * h)
    pixalY = torch.linspace(0, h - 1, h).view(h, 1)
    pixalY = torch.matmul(pixalY, torch.ones(1, w), ).view(1, w * h)
    al, be = cam.getAlBefromPixal(pixalX, pixalY)
    # w = 360
    # h = 90
    # al = torch.linspace(0, np.pi*2, w).view(1, w)
    # al = torch.matmul(torch.ones(h, 1), al).view(1, w * h)
    # be = torch.linspace(0, np.pi/2,h).view(h, 1)
    # be = torch.matmul(be, torch.ones(1, w), ).view(1, w * h)
    # cam = CenterCamera.Camera()

    theta_m = inputs[1,0:64].view(64,1)
    for i in range(0,1000):
        als=torch.rand(1, 1) * 2 * math.pi
        bes=torch.rand(1, 1) * (0.5-1/6) * math.pi+math.pi/6
        d = torch.rand(1, 1) * 5
        theta_m = inputs[i,0:64].view(64,1)
        p = square_array.p0(al,be,theta_m).view(h, w)
        p = p /p.max()
        torchvision.utils.save_image(p.view(h, w), 'images/'+str(i)+'.jpg')


