import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time
import math
#import cupy as np
from scipy import signal
import pickle

from numba import jit
import os
from numba import vectorize

sampleRate = 2000000
@jit
def Add(a, b):
  return a + 1j*b

def InitEPC():
    MCode = np.fromfile(r'D:\Documents\TagArgus\exp\tash\multiple\sender\send7631.bin', dtype='byte')
    MCode = MCode*2-1
    epcList = list()
    codeList = list()
    epcList.append(7)
    epcList.append(6)
    epcList.append(3)
    epcList.append(1)
    codeList.append(MCode[0:int(4.41*2000)])
    codeList.append(MCode[int(4.41*2000*4):int(4.41 * 2000*5)])
    codeList.append(MCode[int(4.41 * 2000 * 8):int(4.41 * 2000 * 9)])
    codeList.append(MCode[int(4.41 * 2000 * 12):int(4.41 * 2000 * 13)])
    return epcList,codeList



start = time.clock()
#x=B[2::4]+1j*B[3::4]


RN16_length_min = int(7000 * sampleRate / 6000000)
epcList,codeList = InitEPC()

def GetEPC(code,epcList,codeList):
    max = 0
    epc = 0
    intev = int(sampleRate / 2000000)
    code = code[::intev]
    code = code[:int(4.41*2000)]
    for i in range(len(epcList)):
        temp = np.sum(codeList[i] * code)
        if temp > max:
            max = temp
            epc = epcList[i]
    return epc



def IsEdge(x,ind,errT,threhold):
    for i in range(0,errT):
        if x[ind+i] > threhold:
            return i
    for i in range(-errT,0):
        if x[ind+i] > threhold:
            return i
    return -100

def IsStart(x,ind,errT,threhold):
    if(ind > len(x) - 2000 * sampleRate / 6000000):
        return []
    result = []
    offset = IsEdge(x, ind, errT, threhold)
    if (offset != -100):
        result.append(ind + offset)
    ind = ind + int(150 * sampleRate / 6000000) +offset
    offset = IsEdge(x,ind,errT,threhold)
    if(offset != -100):
        result.append(ind + offset)
        ind = ind + offset + int(75 * sampleRate / 6000000)
        offset = IsEdge(x, ind, errT, threhold)
        if (offset != -100):
            result.append(ind + offset)
            ind = ind + offset + int(75* sampleRate / 6000000)
            offset = IsEdge(x, ind, errT, threhold)
            if (offset != -100):
                result.append(ind + offset)
                ind = ind + offset + int(150* sampleRate / 6000000)
                offset = IsEdge(x, ind, errT, threhold)
                if (offset != -100):
                    result.append(ind + offset)
                    ind = ind + offset + int(75* sampleRate / 6000000)
                    offset = IsEdge(x, ind, errT, threhold)
                    if (offset != -100):
                        result.append(ind + offset)
                        ind = ind + offset + int(225* sampleRate / 6000000)
                        offset = IsEdge(x, ind, errT, threhold)
                        if (offset != -100):
                            result.append(ind + offset)
                            ind = ind + offset + int(150* sampleRate / 6000000)
                            offset = IsEdge(x, ind, errT, threhold)
                            if (offset != -100):
                                result.append(ind + offset)
                                return result
    return []

def FindRN16Edges(diff_single_abs):
    dd = np.abs(diff_single_abs)
    tr = np.median(dd) * 10
    #print(tr)
    ind = np.where(dd > tr)[0]
    #print(np.diff(ind))
    result = []
    for i in range(0,len(ind)):
        temp = IsStart(dd,ind[i],2,tr)
        if(len(temp) > 0):
            #print('havestart')
            result = temp
            break
    if(len(result) == 0):
        return []
    ind = result[len(result)-1]
    while(ind < len(dd) - 100):
        ind = ind + int(75* sampleRate / 6000000)
        offset = IsEdge(dd, ind, 2, tr)
        if(offset != -100):
            ind = ind + offset
            result.append(ind)

    return result
def CalPhaseFromFile(filename,outputfile):
    B = np.fromfile(filename, dtype='float64')
    antennaArray = Add(B[0::4], B[1::4])
    antennaSingle = Add(B[2::4], B[3::4])
    #antennaArray = antennaArray [::3]
    #antennaSingle = antennaSingle[::3]
    diff_array = np.abs(np.diff(np.abs(antennaArray)))
    #print(np.max(diff_array))
    edge_index = np.where(diff_array > np.max(diff_array) * 0.8)[0]
    #print(diff_array[edge_index])
    array_offset = Counter(np.mod(edge_index, int(180*sampleRate / 6000000))).most_common(1)[0][0]
    array_offset = 10
    single_abs = np.abs(antennaSingle)
    index_low = np.where(single_abs <= np.max(single_abs) / 2)[0]
    index_high = np.where(single_abs > np.max(single_abs) / 2)[0]
    code = np.zeros(single_abs.shape)
    code[index_high] = 1
    code[index_low] = -1
    onelength = np.diff(index_low) - 1

    indexNzero = np.array(np.where((onelength > 0)))[0]
    NoneLength = onelength[indexNzero]
    OneSegStart = index_low[indexNzero]
    OneSegEnd = index_low[indexNzero + 1]
    #print(Counter(NoneLength))
    phaseList = list()
    for i in range(0, 64):
        phaseList.append(list())
    allPhaseList = list()
    allPhaseIndexList = list()
    allPhaseAnt = list()
    allPhaseStrength = list()
    allEPCList = list()
    RN16_index = np.where(NoneLength > RN16_length_min)[0]
    for index in RN16_index:
        start_index = int(OneSegStart[index]+1000 * sampleRate / 6000000)
        end_index = OneSegEnd[index]
        end_index = int(OneSegStart[index]+5000* sampleRate / 6000000)
        # plt.figure()
        # plt.plot(np.abs(antennaSingle[start_index:end_index]))
        # plt.show()
        s = 0
        epc = 0
        if(OneSegEnd[index] > 4.41 * sampleRate / 1000 ):
            s = OneSegEnd[index] - 4.41 * sampleRate / 1000
            epc = GetEPC(code[int(s):OneSegEnd[index]+100], epcList, codeList)
            print(epc)
        else:
            continue
        diff_single_abs = np.diff(single_abs[start_index:end_index])
        result = FindRN16Edges(diff_single_abs)
        if(len(result) == 0):
            continue
        # zeross = np.zeros((len(single_abs[start_index:end_index]),1))+0.035
        # zeross[result] = 0.036
        # plt.plot(zeross)
        # plt.show()
        start_seg = int((start_index - array_offset) / 180) + 1
        end_seg = int((end_index - array_offset) / 180)
        totalFUhao = 1
        #if(diff_single_abs[result[0]]< 0):
        if(single_abs[result[0]+start_index+int(10 * sampleRate / 6000000)] < single_abs[result[0]+start_index-int(10 * sampleRate / 6000000)]):
            totalFUhao = -1
        print(totalFUhao)
        for i in range(len(result)):
            index = start_index+result[i]+1
            # plt.plot(single_abs[index-100:index+100])
            # plt.show()
            if((index-array_offset) % int(180 * sampleRate / 6000000) < 20 * sampleRate / 6000000 or (index-array_offset) %int(180 * sampleRate / 6000000) > 160 * sampleRate / 6000000):
                continue
            j = (index-array_offset) // int(180 * sampleRate / 6000000)
            localFuhao = 1
            #if (diff_single_abs[result[i]] < 0):
            if (single_abs[result[i] + start_index + int(5 * sampleRate / 6000000)] < single_abs[result[i] + start_index - int(5 * sampleRate / 6000000)]):
                localFuhao = -1
            tagphase = np.angle((antennaArray[index+1]-antennaArray[index-1])*localFuhao*totalFUhao)
            phaseList[int(j) % 64].append(tagphase)
            allPhaseList.append(tagphase)
            allPhaseIndexList.append(index)
            allPhaseAnt.append(j % 64)
            allPhaseStrength.append(np.abs(antennaArray[index+1]-antennaArray[index-1])/np.mean(np.abs(np.diff(antennaArray[j*int(180 * sampleRate / 6000000)+array_offset+int(20 * sampleRate / 6000000):j*int(180 * sampleRate / 6000000)+array_offset+int(160 * sampleRate / 6000000)]))))
            allEPCList.append(epc)


    a = np.zeros([64, 1])
    for i in range(0, 64):
        a[i] = np.median(phaseList[i])
        #print(str(i)+' '+str(np.std((phaseList[i]))) + ' ' + str(np.median(phaseList[i])))
    #np.savetxt(outputfile, a)
    return allPhaseList,allPhaseIndexList,allPhaseAnt,allEPCList

def AnaOneFolder(folder):
    indd = 1

    start = time.clock()
    output = list()
    for filename in os.listdir(folder):
        if os.path.splitext(filename)[1] == '.bin':
            print(filename)
            pL, pIL, pAL,eL = CalPhaseFromFile(folder + r'\\' + filename, 'mm' + str(indd) + '.txt')
            t = float(filename.split('_')[0])
            print(t)
            for i in range(len(pL)):
                output.append(str(t + pIL[i] / sampleRate) + ' ' + str(pL[i]) + ' ' + str(pAL[i])+' '+str(eL[i]))
            indd = indd + 1
    end = time.clock()
    print(end - start)
    with open(folder + r'\\' + 'RFData.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in output)

def AnaMultipleFolders(fatherFolder):
    for foldername in os.listdir(fatherFolder):
        if os.path.isdir(fatherFolder+r'\\'+foldername):
            if os.path.exists(fatherFolder+r'\\'+foldername+r'\\'+'usrp_data'):
                AnaOneFolder(fatherFolder+r'\\'+foldername+r'\\'+'usrp_data')


#AnaMultipleFolders(r'E:\Data17')
AnaOneFolder(r'C:\usrp_data')
#CalPhaseFromFile(r'E:\Data17\2019-01-24-19-22-16-3529984\usrp_data\20190124192213.992_2ch.bin','aaa.txt')

