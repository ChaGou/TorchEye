import numpy
import torch
import matplotlib.pyplot as plt



a=numpy.loadtxt(r'..\VSWorkSpace\RadarEye\RadarEye\bin\Debug\data2\trainlabel.txt')

plt.hist(a[:,0], bins=10, color='steelblue', normed=True )
plt.figure()
plt.plot(a[:,0])
print(a.shape)
a=numpy.loadtxt(r'..\VSWorkSpace\RadarEye\RadarEye\bin\Debug\data2\testlabel.txt')
plt.scatter(a[:,0],a[:,1])
print(a.shape)

plt.show()
# x= torch.Tensor([1,2]).cuda()
# y = torch.Tensor([3,4]).cuda()
# numpy.savetxt('a.txt',x.cpu().numpy())
# print((x*y))
#
# bb = torch.Tensor([1,2,3,4,5,6])
# bb=bb.view(2,3)
# print(bb)
# bbb=torch.linspace(0, 119, 120)
# bbb=bbb.view(6,10,2)
# print(bbb[0,:,0])