import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time
import math
#import cupy as np
from scipy import signal

from numba import jit
from numba import vectorize


@jit
def Add(a, b):
  return a + 1j*b


start = time.clock()
#x=B[2::4]+1j*B[3::4]


RN16_length_min = 8000


def CalTagPhase(data):
    data_abs = np.abs((data))
    diff_data = np.diff(data_abs)
    diff_abs = np.abs(np.diff(data_abs))
    diff_mean = np.mean(diff_abs)
    diff_max = np.max(diff_abs)
    #plt.figure()
    #plt.plot(np.abs(data))
    #plt.show()
    edge_index = np.where((diff_abs > diff_max * 0.8) & (diff_abs > 5 * diff_mean))[0]
    tagphase = []
    if(len(edge_index) == 0 or len(edge_index) > 2):
        return tagphase
    for i in edge_index:
        diff_edge = diff_data[i]
        IQedge = data[i + 1] - data[i]
        if(diff_edge < 0):
            IQedge = -IQedge
        tagphase.append(np.angle(IQedge))
    return tagphase


def CalPhaseFromFile(filename,outputfile):
    B = np.fromfile(filename, dtype='float64')
    antennaArray = Add(B[0::4], B[1::4])
    antennaSingle = Add(B[2::4], B[3::4])
    diff_array = np.abs(np.diff(np.abs(antennaArray)))
    #print(np.max(diff_array))
    edge_index = np.where(diff_array > np.max(diff_array) * 0.8)[0]
    #print(diff_array[edge_index])
    array_offset = Counter(np.mod(edge_index, 180)).most_common(1)[0][0]

    x_abs = np.abs(antennaSingle)
    index_low = np.where(x_abs <= np.max(x_abs) / 2)[0]
    onelength = np.diff(index_low) - 1

    indexNzero = np.array(np.where((onelength > 0)))[0]
    NoneLength = onelength[indexNzero]
    OneSegStart = index_low[indexNzero]
    OneSegEnd = index_low[indexNzero + 1]
    #print(Counter(NoneLength))
    phaseList = list()
    for i in range(0, 64):
        phaseList.append(list())
    RN16_index = np.where(NoneLength > RN16_length_min)[0]
    for index in RN16_index:
        start_index = OneSegStart[index]+1500
        end_index = OneSegEnd[index]
        end_index = OneSegStart[index]+4500
        #plt.figure()
        #plt.plot(np.abs(antennaArray[start_index:end_index]))
        #plt.show()
        start_seg = int((start_index - array_offset) / 180) + 1
        end_seg = int((end_index - array_offset) / 180)
        for j in range(start_seg, end_seg):
            tagphase = CalTagPhase(antennaArray[j * 180 + array_offset + 20:j * 180 + array_offset + 160])
            #if(j%64 == 24):
             #    plt.figure()
              #   plt.plot(np.abs(antennaArray[j * 180 + array_offset + 20:j * 180 + array_offset + 160]))
               #  plt.show()

            if (len(tagphase) != 0):
                phaseList[j % 64].extend(tagphase)
    a = np.zeros([64, 1])
    for i in range(0, 64):
        a[i] = np.median(phaseList[i])
        print(str(i)+' '+str(np.std((phaseList[i]))))
    print(str(i) + ' ' + str(np.std((phaseList[40]))))
    np.savetxt(outputfile, a)


import os
indd = 1
folder = r'D:\data\20181015113847_2ch'
start = time.clock()
for filename in os.listdir(folder):
    print(filename)
    CalPhaseFromFile(folder+r'\\'+filename,'mm'+str(indd)+'.txt')
    indd = indd + 1
    end = time.clock()
print(end - start)




