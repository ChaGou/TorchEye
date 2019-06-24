import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
#from graphviz import Digraph

class MyNet1(nn.Module):
    def __init__(self, n_in, n_out):
        super(MyNet1, self).__init__() #调用父类的初始化函数
        self.f1 = nn.Linear(n_in, 1000)
        self.f2 = nn.Linear(1000, 500)
        self.f3 = nn.Linear(500, 200)
        self.f4 = nn.Linear(400, 200)
        self.f5 = nn.Linear(200, n_out)
        self.dropout = nn.Dropout(p=0)

    def forward(self, x):

        x = self.f1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f3(x)
        x = F.relu(x)
        x = self.dropout(x)
       # x = self.f4(x)
        #x = F.relu(x)
        #x = self.f4(x)
        #x = F.relu(x)
        x = self.f5(x)
        #x = F.sigmoid(x)
        #x = F.relu(x)
        #x = F.tanh(x)
        #x=F.softmax(x,len(x.shape)-1)

        return x

class MyRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(MyRNN, self).__init__()

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons,batch_first=True)
        self.basic_rnn2 = nn.RNN(self.n_neurons,self.n_outputs,batch_first=True)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self, ):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1,self.batch_size, self.n_neurons)).cuda()

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        #X = X.permute(1, 0, 2)

        self.batch_size = X.size(0)
        self.hidden = self.init_hidden()
        self.hidden2 = (torch.zeros(1,self.batch_size, self.n_outputs)).cuda()

        lstm_out, self.hidden = self.basic_rnn(X, self.hidden)
        lstm_out2,out = self.basic_rnn2(lstm_out,self.hidden2)
        #out = F.relu(self.FC(self.hidden))

        return out
class MyAutoEncoder(nn.Module):
    def __init__(self,n_in,n_out):
        super(MyAutoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(n_in,200),
            nn.ReLU(True),
            nn.Linear(200,100),
            nn.ReLU(True),
            nn.Linear(100,n_out),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(n_out,100),
            nn.ReLU(True),
            nn.Linear(100,200),
            nn.ReLU(True),
            nn.Linear(200,n_in),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__() #调用父类的初始化函数
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.fc1 = nn.Linear(6 * 42 * 42, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 42 * 42)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class MyNet3(nn.Module):
    def __init__(self):
        super(MyNet3, self).__init__() #调用父类的初始化函数
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.fc1 = nn.Linear(6 * 2 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x=x.view(-1,2,6,10)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 6 * 2 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class MyNet4(nn.Module):
    def __init__(self, n_in, n_out):
        super(MyNet4, self).__init__() #调用父类的初始化函数
        self.f1 = nn.Linear(n_in, 960)
        self.f2 = nn.Linear(960, 480)
        self.f3 = nn.Linear(480, n_out)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        x = self.f1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f3(x)
        return x


class MyNet_Hot(nn.Module):
    def __init__(self, n_in, n_out):
        super(MyNet1, self).__init__() #调用父类的初始化函数
        self.f1 = nn.Linear(n_in, 1000)
        self.f2 = nn.Linear(1000, 500)
        self.f3 = nn.Linear(500, 100)
        self.f4 = nn.Linear(100, 100)
        self.f5 = nn.Linear(100, n_out)
        self.dropout = nn.Dropout(p=0)

    def forward(self, x):

        x = self.f1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f4(x)
        x = F.relu(x)
        x = self.f5(x)

        return x
class CNN_Up(nn.Module):
    def __init__(self):
        super(CNN_Up, self).__init__() #调用父类的初始化函数
        self.conv1 = nn.Conv2d(1, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.conv5 = nn.Conv2d(256, 256, 2)
        self.fc1 = nn.Linear(2304, 3)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        #print(x.shape)
        x = x.view(x.size(0),1, -1)
        x = F.relu(self.fc1(x))
        return x

# def make_dot(var, params=None):
#     """ Produces Graphviz representation of PyTorch autograd graph
#     Blue nodes are the Variables that require grad, orange are Tensors
#     saved for backward in torch.autograd.Function
#     Args:
#         var: output Variable
#         params: dict of (name, Variable) to add names to node that
#             require grad (TODO: make optional)
#     """
#     if params is not None:
#         assert isinstance(params.values()[0], Variable)
#         param_map = {id(v): k for k, v in params.items()}
#
#     node_attr = dict(style='filled',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()
#
#     def size_to_str(size):
#         return '(' + (', ').join(['%d' % v for v in size]) + ')'
#
#     def add_nodes(var):
#         if var not in seen:
#             if torch.is_tensor(var):
#                 dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
#             elif hasattr(var, 'variable'):
#                 u = var.variable
#                 name = param_map[id(u)] if params is not None else ''
#                 node_name = '%s\n %s' % (name, size_to_str(u.size()))
#                 dot.node(str(id(var)), node_name, fillcolor='lightblue')
#             else:
#                 dot.node(str(id(var)), str(type(var).__name__))
#             seen.add(var)
#             if hasattr(var, 'next_functions'):
#                 for u in var.next_functions:
#                     if u[0] is not None:
#                         dot.edge(str(id(u[0])), str(id(var)))
#                         add_nodes(u[0])
#             if hasattr(var, 'saved_tensors'):
#                 for t in var.saved_tensors:
#                     dot.edge(str(id(t)), str(id(var)))
#                     add_nodes(t)
#
#     add_nodes(var.grad_fn)
#     return dot

#
# if __name__ == '__main__':
#     net = MyAutoEncoder(56*2, 10)
#     x = Variable(torch.randn(1, 112))
#     y = net(x)
#     g = make_dot(y)
#     g.view()
#
#     params = list(net.parameters())
#     k = 0
#     for i in params:
#         l = 1
#         print("该层的结构：" + str(list(i.size())))
#         for j in i.size():
#             l *= j
#         print("该层参数和：" + str(l))
#         k = k + l
#     print("总参数数量和：" + str(k))

