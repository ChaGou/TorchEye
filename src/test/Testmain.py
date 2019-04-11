import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
if __name__ == '__main__':
    from src.core import CHAModule, RobotArm

    x_train = torch.transpose(RobotArm.theta_m, 0, 1)
    y_train = torch.transpose(RobotArm.y, 0, 1)
    model = CHAModule.MyNet1(RobotArm.N_ant, 2)
    if torch.cuda.is_available():
        model.cuda()
    num_epochs = 10000
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        target = Variable(y_train)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()

        # forward
        out = model(inputs)  # 前向传播
        loss = criterion(out, target)  # 计算loss
        # backward
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 方向传播
        optimizer.step()  # 更新参数

        if (epoch + 1) % 200 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, num_epochs, loss.data[0]))
    test = Variable(torch.transpose(RobotArm.theta_m2, 0, 1))
    if torch.cuda.is_available():
        test = test.cuda()
    out = model(test)
    print(out)
    print(RobotArm.y2)

    #print(GenData.alpha)