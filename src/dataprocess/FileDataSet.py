import numpy
import torch
import torch.utils.data
from src.core import ConstantTerm
import random
import Parameters as pm
import os

dataMode = pm.dataMode
class FileDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        self.data = numpy.loadtxt(data_path)
        self.data = self.fill_zero_data(self.data)
        data2 = self.CorrectAntNum(self.data)
        # data2[:,0:64] = data[:,0:64]
        if dataMode == pm.DataMode.DeltaMode:
            data = self.GenDeltaModeData(data2)
        else:
            data = self.GenOriginModeData(data2)
        self.data = numpy.mod(data, 2 * numpy.pi)
        #self.data = self.remove_pi_offset(self.data)
        #self.data = self.data / numpy.pi / 2
        numpy.savetxt('hehe2.txt', self.data)
        self.label = numpy.loadtxt(label_path)
        self.label[:,0] = (self.label[:,0]-pm.cx)/pm.fx*self.label[:,2]
        self.label[:, 1] = (self.label[:, 1] - pm.cy) / pm.fy * self.label[:, 2]
        #self.data = numpy.concatenate((self.data,self.data),1)

        #self.label = self.label[1:,:]
    def Uniform(self):
        data = self.data
        la = self.label
        dd = {}
        interval = 10
        for i in range(data.shape[0]):
            axixs = str(numpy.floor(la[i,0]/10)),',',str(numpy.floor(la[i,1]/10))
            if(axixs not in dd.keys()):
                ll = list()
                ll.append(list())
                ll.append(list())
                dd[axixs] = ll
            dd[axixs][0].append(la[i, :])
            dd[axixs][1].append(data[i, :])
        maxcount = 0
        for key,value in dd.items():
            if(maxcount < len(value[0])):
                maxcount = len(value[0])
        print(maxcount)
        outputlabel = numpy.zeros((maxcount*len(dd.items()),2))
        outputdata = numpy.zeros((maxcount*len(dd.items()),data.shape[1]))
        t = 0
        for key, value in dd.items():
            for i in range(len(value[0])):
                outputlabel[t,:] = value[0][i]
                outputdata[t, :] = value[1][i]
                t = t + 1
            for i in range(maxcount - len(value[0])):
                index = random.randint(0,len(value[0])-1)
                outputdata[t, :]=value[1][index]
                outputlabel[t, :] = value[0][index]
                t = t + 1
        self.data = outputdata
        self.label = outputlabel


    def CorrectAntNum(self,data):
        data2 = numpy.zeros(data.shape)
        data2[:, 0:63] = data[:, 1:64]
        data2[:, 63] = data[:, 0]
        return data2
    def GenDeltaModeData(self,data2):
        data = numpy.zeros((data2.shape[0], 56 * 2))
        data[:, 0:56] = data2[:, 8:64] - data2[:, 0:56]
        for i in range(8):
            data[:, 7 * i + 56:7 * i + 56 + 7] = data2[:, 8 * i + 1:8 * i + 8] - data2[:, 8 * i:8 * i + 7]
        print(os.path.abspath('.'))
        offset = numpy.loadtxt(r'../../output/offset.txt').reshape(-1, 112)
        data = data + (360 - offset) / 180.0 * numpy.pi
        return data

    def GenOriginModeData(self,data2):
        #return data2
        return (data2.transpose()-data2[:,0].transpose()).transpose()

    def make_more(self, n, de):
        data_copy = self.data
        label_copy = self.label
        for i in range(n-1):
            tempdata = data_copy
            templabel = label_copy
            noise = numpy.random.normal(0, de, (tempdata.shape[0], ConstantTerm.antennaNum))
            ze = numpy.zeros((tempdata.shape[0], tempdata.shape[1] - ConstantTerm.antennaNum))
            noise = numpy.c_[noise,ze]
            tempdata += noise
            self.data = numpy.r_[self.data,tempdata]
            self.label = numpy.r_[self.label,templabel]

    def __getitem__(self, index):
        label = torch.Tensor(self.label[index,:])
        #data = torch.Tensor(self.data[index, :]-self.data[index,0:1])
        #data = numpy.cos(data*2)
        data = torch.Tensor(self.data[index, :])
        if dataMode == pm.DataMode.DeltaMode:
            data = data.view(-1,56*2)
        elif dataMode == pm.DataMode.SquareMode:
            data = data.view(-1, 8,8)
        else:
            data = data.view(-1,64)
        return data, label.view(-1,3)

    def __len__(self):
        return self.label.shape[0]

    def fill_zero_data(self, data):
        temp = numpy.mean(data,axis=0)
        for i in range(1,data.shape[0]):
            datatemp = data[i,:]
            data[i,numpy.where(datatemp == 0)[0]] = temp[numpy.where(datatemp == 0)[0]]
            temp[numpy.where(datatemp != 0)[0]] = datatemp[numpy.where(datatemp != 0)[0]]
        # data = numpy.mod(numpy.diff(data,axis=0)+2.5*numpy.pi,numpy.pi)-0.5*numpy.pi
        # data = numpy.diff(data,axis=0)
        # data = numpy.mod(numpy.cumsum(data,axis=0),numpy.pi*2)-numpy.pi
        result = numpy.zeros((data.shape[0],data.shape[1]))
        #result[:,1:64] = data[:,1:64] - data[:,0:63]
        #result[:,8:8:64] = data[:,8:8:64]-data[:,0:8:56]
        result[:,8:64] = data[:,8:64] - data[:,0:56]
        result[:,1:8] = data[:,1:8]-data[:,0:7]
        numpy.savetxt('refine_data.txt', data)
        #result = numpy.mod(result+numpy.pi*3,numpy.pi*2)-numpy.pi
        return data
    def remove_pi_offset(self,data):
        dd=numpy.diff(data, axis=0)
        temp = numpy.where(numpy.abs(dd-numpy.pi)< 1.5)
        for i in range(len(temp[0])):
            if(temp[0][i]+1 >= dd.shape[0]):
                continue
            if(numpy.abs(dd[temp[0][i]+1,temp[1][i]]+numpy.pi)<1.5):
                data[temp[0][i]+1,temp[1][i]] = data[temp[0][i]+1,temp[1][i]]-numpy.pi

        dd = numpy.diff(data, axis=0)
        temp = numpy.where(numpy.abs(dd + numpy.pi) < 1.5)
        for i in range(len(temp[0])):
            if(temp[0][i]+1 >= dd.shape[0]):
                continue
            if(numpy.abs(dd[temp[0][i]+1,temp[1][i]]-numpy.pi)<1.5):
                data[temp[0][i]+1,temp[1][i]] = data[temp[0][i]+1,temp[1][i]]+numpy.pi
        return data
class FileDatasetRNN(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path,N_STEPS):
        self.data = numpy.loadtxt(data_path)
        self.n_step = N_STEPS
        self.data = self.refine_data(self.data)
        data2 = numpy.zeros(self.data.shape)
        data2[:, 0:63] = self.data[:, 1:64]
        data2[:, 63] = self.data[:, 0]
        # data2[:,0:64] = data[:,0:64]
        data = numpy.zeros((data2.shape[0], 56 * 2))
        data[:, 0:56] = data2[:, 8:64] - data2[:, 0:56]
        for i in range(8):
            data[:, 7 * i + 56:7 * i + 56 + 7] = data2[:, 8 * i + 1:8 * i + 8] - data2[:, 8 * i:8 * i + 7]
        offset = numpy.loadtxt('offset.txt').reshape(-1, 112)
        data = data + (360 - offset) / 180.0 * numpy.pi
        self.data = numpy.mod(data + numpy.pi, 2 * numpy.pi)
        #self.data = numpy.concatenate((self.data,self.data),1)
        self.label = numpy.loadtxt(label_path)
        #self.label = self.label[1:,:]

    def make_more(self, n, de):
        data_copy = self.data
        label_copy = self.label
        for i in range(n-1):
            tempdata = data_copy
            templabel = label_copy
            noise = numpy.random.normal(0, de, (tempdata.shape[0], ConstantTerm.antennaNum))
            ze = numpy.zeros((tempdata.shape[0], tempdata.shape[1] - ConstantTerm.antennaNum))
            noise = numpy.c_[noise,ze]
            tempdata += noise
            self.data = numpy.r_[self.data,tempdata]
            self.label = numpy.r_[self.label,templabel]

    def __getitem__(self, index):
        label = torch.Tensor(self.label[index:index+self.n_step,:])
        #data = torch.Tensor(self.data[index, :]-self.data[index,0:1])
        #data = numpy.cos(data*2)
        data = torch.Tensor(self.data[index:index+self.n_step,:])
        data = data.view(-1,56*2)
        #data[:,0:64] = ((data[:,0:64]-data[:,0:1]))
        #data[:, 64:128] = ((data[:, 64:128] - data[:, 64:65]))
        return data, label.view(-1,2)

    def __len__(self):
        return self.label.shape[0]-self.n_step

    def refine_data(self,data):
        temp = numpy.mean(data,axis=1)
        for i in range(1,data.shape[0]):
            datatemp = data[i,:]
            data[i,numpy.where(datatemp == 0)[0]] = temp[numpy.where(datatemp == 0)[0]]
            temp[numpy.where(datatemp != 0)[0]] = datatemp[numpy.where(datatemp != 0)[0]]

        return data