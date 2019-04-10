import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import FileDataSet


a=np.loadtxt('a.txt')
a=a.transpose()
b=F.softmax(torch.Tensor(a),0)
plt.contour(F.softmax(torch.Tensor(a),0))
print(a.shape)
np.savetxt('b.txt',b.numpy())
testPath = r'E:\Data2\2018-11-09-19-19-33-1266271'
testDataset = FileDataSet.FileDataset(testPath+r'\data.txt',
                                          testPath+r'\label.txt')
inputs, labels = testDataset[:]
a=labels.numpy()
print(a.shape)
#plt.figure()
plt.plot(a[:,0])
plt.show()