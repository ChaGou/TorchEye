import  torch
from  torch.autograd  import Variable
from torch import nn
from torchviz import make_dot, make_dot_from_trace
import matplotlib.pyplot as plt
import CHAModule
model = CHAModule.MyNet1(120,2)

x = Variable(torch.randn(1,120))
y = model(x)

g=make_dot(y.mean(), params=dict(model.named_parameters()))
g.view()
plt.show()