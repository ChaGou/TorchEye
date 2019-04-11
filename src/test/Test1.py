import  torch
from  torch.autograd  import Variable
from torchviz import make_dot
import matplotlib.pyplot as plt
from src.core import CHAModule

model = CHAModule.MyNet1(120, 2)

x = Variable(torch.randn(1,120))
y = model(x)

g=make_dot(y.mean(), params=dict(model.named_parameters()))
g.view()
plt.show()