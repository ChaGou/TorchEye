import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time
import math
#import cupy as np
from scipy import signal

from numba import jit
from numba import vectorize

def butter_highpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=10):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


B = np.fromfile(r'G:\TagCamera\Codes\impinj\20180908162020_RF0.bin',dtype='float64')

@jit
def Add(a, b):
  return a + 1j*b

@jit
def TagCodeM4(bitlength):
    highbits=np.ones(shape=[bitlength,1])
    lowbits=-np.ones(shape=[bitlength,1])
    code0_1 = np.vstack((highbits,lowbits,highbits,lowbits,highbits,lowbits,highbits,lowbits))
    code0_2 = -code0_1
    code1_1 = np.vstack((highbits,lowbits,highbits,lowbits,lowbits,highbits,lowbits,highbits))
    code1_2 = - code1_1
    code_preamble = np.vstack((code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code0_1,code1_1,code0_2,code1_2,code1_1,code1_2))
    code_preamble = np.vstack((code0_1,code0_1,code0_1,code0_1,code0_1, code1_1, code0_2, code1_2,code1_1, code1_2))
    return  code_preamble,code0_1,code0_2,code1_1,code1_2


start = time.clock()
#x=B[2::4]+1j*B[3::4]
x = Add(B[0::2],B[1::2])
print (x.shape)
x_abs = np.abs(x)
filtered_x = butter_highpass_filter(x,100e3,6e6)
index_high = np.where(x_abs > np.max(x_abs)/2)[0]
zerolength = np.diff(index_high) - 1
indexNzero = np.array(np.where(zerolength>0))[0]
NzeroLength=zerolength[indexNzero]
#print(Counter(NzeroLength))

index_low = np.where(x_abs <= np.max(x_abs)/2)[0]
onelength = np.diff(index_low) - 1
#print(B.shape[0])

#indexNzero = np.array(np.where((onelength>0) & (onelength <1000) ))[0]
indexNzero = np.array(np.where((onelength>0) ))[0]
NoneLength=onelength[indexNzero]
OneSegStart=index_low[indexNzero]+40
OneSegEnd=index_low[indexNzero+1]-40
print(Counter(NoneLength))
#plt.hist(NoneLength,bins=256)
#plt.show()

PW = 53
data0_1length = 97
data1_1length = 172
RTcal_1length = 320
TRcal_1length = 455

CodePreamble, code0_1, code0_2, code1_1, code1_2 = TagCodeM4(12)
#plt.plot(CodePreamble)
print(len(CodePreamble))
@jit
def FindACK(NoneLength):
    ACK=[]
    RTCalIndex = np.where(np.abs(NoneLength-RTcal_1length)<20)[0]
    for i in RTCalIndex:
        if np.abs(NoneLength[i-1]-data0_1length)<20 and np.abs(NoneLength[i+1]-data0_1length)<20 and np.abs(NoneLength[i+2]-data1_1length)<20 and NoneLength[i+19] > 10000:
            ACK.append(i-1)
    return np.array(ACK)


@jit
def FindQuery(NoneLength):
    Query=[]
    RTCalIndex = np.where(np.abs(NoneLength-TRcal_1length)<20)[0]
    for i in RTCalIndex:
        if np.abs(NoneLength[i-1]-RTcal_1length)<20 and np.abs(NoneLength[i+1]-data1_1length)<20 and np.abs(NoneLength[i+2]-data0_1length)<20 :
            Query.append(i-1)
    return np.array(Query)


def DecodeTag(x_tag_abs,x_tag):
    x_mid = x_tag_abs[(int)(x_tag_abs.shape[0] / 4):(int)(x_tag_abs.shape[0] / 4 * 3)]
    x_mid_diff = np.diff(x_mid)
    tag_thredhold = np.max(np.abs(x_mid_diff))*0.4
    print(tag_thredhold)
    x_tag_bi = np.zeros(shape=[x_tag_abs.shape[0], 1])
    nowstatus = 0
    x_tag_bi[0] = 0
    for i in range (1, x_tag_abs.shape[0]):
        if(np.abs(x_tag_abs[i])-np.abs(x_tag_abs[i - 1]) > tag_thredhold):
            nowstatus = 1
        elif (np.abs(x_tag_abs[i]) - np.abs(x_tag_abs[i - 1]) < -tag_thredhold):
            nowstatus = -1
        x_tag_bi[i] = nowstatus

    cor = np.correlate(x_tag_bi.flatten(),CodePreamble.flatten(),mode='full')


    index = np.argmax(cor)
    print(index)
    Code = np.zeros((128,1))
    l=len(code0_1)
    plt.figure()
    plt.plot(x_tag_bi)
    #plt.plot(np.angle(butter_highpass_filter(x_tag[index:index + l * 128], 200e3, 6e6)))
    if(index+128*l > len(x_tag_bi)):
        return Code
    print(index)


    for i in range(128):
        tag_code = x_tag_bi[index+i*l:index+i*l+l]
        max_cor = [np.correlate(tag_code.flatten(),code0_1.flatten(),mode='full'),np.correlate(tag_code.flatten(),code0_2.flatten(),mode='full'),np.correlate(tag_code.flatten(),code1_1.flatten(),mode='full'),np.correlate(tag_code.flatten(),code1_2.flatten(),mode='full')]
        offset = np.argmax(max_cor)%(2*l)
        #print(offset)
        index = index +offset - l
        Code[i] = (int)(np.argmax(max_cor)/l/2/2)

    return Code




ACK=FindQuery(NoneLength)
print(NoneLength[ACK[1]-1:ACK[1]+30])
ind = ACK[10]-1
ind2 = ACK[10]+25
np.savetxt('Start.txt',OneSegStart[ACK[0:len(ACK)-1]+20])
plt.figure()
plt.plot(x_abs[OneSegStart[ind]:OneSegEnd[ind2]])
M = 4
DR = 64 /3
TRext = 1
#plt.plot((x_abs[OneSegStart[ACK[20]+20]:OneSegEnd[ACK[20]+20]]))
x
for i in range(0,1):
    code=DecodeTag((x_abs[OneSegStart[ACK[i] + 20]:OneSegEnd[ACK[i] + 20]]),(x[OneSegStart[ACK[i] + 20]:OneSegEnd[ACK[i] + 20]]))
    print(np.transpose(code))


plt.show()


end = time.clock()
print(end-start)
