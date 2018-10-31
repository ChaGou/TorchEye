import numpy
import torch
import torch.utils.data
import ConstantTerm
from PIL import Image


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        self.data = numpy.loadtxt(data_path)
        self.label = numpy.loadtxt(label_path)

    def make_more(self, n, de):
        data_copy = self.data
        label_copy = self.label
        for i in range(n-1):
            tempdata = data_copy
            templabel = label_copy
            noise = numpy.random.normal(0,de,(tempdata.shape[0],ConstantTerm.antennaNum))
            ze = numpy.zeros((tempdata.shape[0],tempdata.shape[1]-ConstantTerm.antennaNum))
            noise = numpy.c_[noise,ze]
            tempdata += noise
            self.data = numpy.r_[self.data,tempdata]
            self.label = numpy.r_[self.label,templabel]

    def __getitem__(self, index):
        label = torch.Tensor(self.label[index,:])
        data = torch.Tensor(self.data[index, :])
        return data.view(-1,120), label.view(-1,2)

    def __len__(self):
        return self.label.shape[0]

