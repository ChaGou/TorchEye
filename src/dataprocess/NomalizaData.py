import numpy
from mpl_toolkits.mplot3d import Axes3D
testPath = r'D:\dataset\data3'
data_path = testPath+r'\traindata.txt'
label_path = testPath+ r'\trainlabel.txt'
data = numpy.loadtxt(data_path)
label = numpy.loadtxt(label_path)
temp = numpy.mean(data,axis=1)
for i in range(1,data.shape[0]):
    datatemp = data[i,:]
    data[i,numpy.where(datatemp == 0)[0]] = temp[numpy.where(datatemp == 0)[0]]
    temp[numpy.where(datatemp != 0)[0]] = datatemp[numpy.where(datatemp != 0)[0]]
data2 = numpy.zeros(data.shape)
data2[:,0:63] = data[:,1:64]
data2[:,63] = data[:,0]
#data2[:,0:64] = data[:,0:64]
data = numpy.zeros((data2.shape[0],56*2))
data[:,0:56]=data2[:,8:64]-data2[:,0:56]
for i in range(8):
    data[:,7*i+56:7*i+56+7] = data2[:,8*i+1:8*i+8] - data2[:,8*i:8*i+7]
print(data[0,:])
data = numpy.mod(data+numpy.pi,2*numpy.pi)-numpy.pi
import matplotlib.pyplot as plt
for i in range(56*2):
    plt.subplot(14,8,i+1)
    plt.hist(data[:,i],100)
plt.show()
numpy.savetxt('befornormalize.txt',data)
result = numpy.zeros((1,56*2))
for i in range(56*2):
    temp,_ = numpy.histogram(data[:,i],360)
    minn = 100000
    minindex = 0
    for j in range(360):
        sumtemp = temp[j]
        for t in range(10):
            sumtemp += temp[(j + t) % 360]
            sumtemp += temp[(j - t) % 360]
        if sumtemp < minn:
            minn = sumtemp
            minindex = j
    result[0,i] = minindex
print(result)
result = numpy.mod(result,numpy.pi*2)
numpy.savetxt('offset.txt',result)
data = data + (360 -result)/180.0*numpy.pi
data = numpy.mod(data+numpy.pi,2*numpy.pi)-numpy.pi
for i in range(56*2):
    plt.subplot(14,8,i+1)
    plt.hist(data[:,i],100)
plt.show()
numpy.savetxt('afternormalize.txt',data)
