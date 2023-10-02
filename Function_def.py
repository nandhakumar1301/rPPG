#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import pywt
import ecgtemplates

try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
import scipy.signal as signal


class Detectors:
    

    def __init__(self, sampling_frequency = False):
        


        self.fs = sampling_frequency

       
        
        
        self.detector_list = [
            ["Two average",self.two_average_detector],
            ["Matched filter",self.matched_filter_detector],
            ["Wavelet transform",self.swt_detector],
            ["Christov",self.christov_detector],
            ["Hamilton",self.hamilton_detector],
            ["Pan Tompkins",self.pan_tompkins_detector],
            
        ]


# In[ ]:


def christov_detector(self, unfiltered_ecg):
       
        total_taps = 0

        b = np.ones(int(0.02*self.fs))
        b = b/int(0.02*self.fs)
        total_taps += len(b)
        a = [1]

        MA1 = signal.lfilter(b, a, unfiltered_ecg)

        b = np.ones(int(0.028*self.fs))
        b = b/int(0.028*self.fs)
        total_taps += len(b)
        a = [1]

        MA2 = signal.lfilter(b, a, MA1)

        Y = []
        for i in range(1, len(MA2)-1):
            
            diff = abs(MA2[i+1]-MA2[i-1])

            Y.append(diff)

        b = np.ones(int(0.040*self.fs))
        b = b/int(0.040*self.fs)
        total_taps += len(b)
        a = [1]

        MA3 = signal.lfilter(b, a, Y)

        MA3[0:total_taps] = 0

        ms50 = int(0.05*self.fs)
        ms200 = int(0.2*self.fs)
        ms1200 = int(1.2*self.fs)
        ms350 = int(0.35*self.fs)

        M = 0
        newM5 = 0
        M_list = []
        MM = []
        M_slope = np.linspace(1.0, 0.6, ms1200-ms200)
        F = 0
        F_list = []
        R = 0
        RR = []
        Rm = 0
        R_list = []

        MFR = 0
        MFR_list = []

        QRS = []

        for i in range(len(MA3)):

            # M
            if i < 5*self.fs:
                M = 0.6*np.max(MA3[:i+1])
                MM.append(M)
                if len(MM)>5:
                    MM.pop(0)

            elif QRS and i < QRS[-1]+ms200:
                newM5 = 0.6*np.max(MA3[QRS[-1]:i])
                if newM5>1.5*MM[-1]:
                    newM5 = 1.1*MM[-1]

            elif QRS and i == QRS[-1]+ms200:
                if newM5==0:
                    newM5 = MM[-1]
                MM.append(newM5)
                if len(MM)>5:
                    MM.pop(0)    
                M = np.mean(MM)    
            
            elif QRS and i > QRS[-1]+ms200 and i < QRS[-1]+ms1200:

                M = np.mean(MM)*M_slope[i-(QRS[-1]+ms200)]

            elif QRS and i > QRS[-1]+ms1200:
                M = 0.6*np.mean(MM)

            # F
            if i > ms350:
                F_section = MA3[i-ms350:i]
                max_latest = np.max(F_section[-ms50:])
                max_earliest = np.max(F_section[:ms50])
                F = F + ((max_latest-max_earliest)/150.0)

            # R
            if QRS and i < QRS[-1]+int((2.0/3.0*Rm)):

                R = 0

            elif QRS and i > QRS[-1]+int((2.0/3.0*Rm)) and i < QRS[-1]+Rm:

                dec = (M-np.mean(MM))/1.4
                R = 0 + dec


            MFR = M+F+R
            M_list.append(M)
            F_list.append(F)
            R_list.append(R)
            MFR_list.append(MFR)

            if not QRS and MA3[i]>MFR:
                QRS.append(i)
            
            elif QRS and i > QRS[-1]+ms200 and MA3[i]>MFR:
                QRS.append(i)
                if len(QRS)>2:
                    RR.append(QRS[-1]-QRS[-2])
                    if len(RR)>5:
                        RR.pop(0)
                    Rm = int(np.mean(RR))

        QRS.pop(0)
        
        return QRS

    
    


# In[ ]:


def hamilton_detector(self, unfiltered_ecg):
        
        f1 = 8/self.fs
        f2 = 16/self.fs

        b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')

        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        diff = abs(np.diff(filtered_ecg))

        b = np.ones(int(0.08*self.fs))
        b = b/int(0.08*self.fs)
        a = [1]

        ma = signal.lfilter(b, a, diff)

        ma[0:len(b)*2] = 0

        n_pks = []
        n_pks_ave = 0.0
        s_pks = []
        s_pks_ave = 0.0
        QRS = [0]
        RR = []
        RR_ave = 0.0

        th = 0.0

        i=0
        idx = []
        peaks = []  

        for i in range(len(ma)):

            if i>0 and i<len(ma)-1:
                if ma[i-1]<ma[i] and ma[i+1]<ma[i]:
                    peak = i
                    peaks.append(i)

                    if ma[peak] > th and (peak-QRS[-1])>0.3*self.fs:        
                        QRS.append(peak)
                        idx.append(i)
                        s_pks.append(ma[peak])
                        if len(n_pks)>8:
                            s_pks.pop(0)
                        s_pks_ave = np.mean(s_pks)

                        if RR_ave != 0.0:
                            if QRS[-1]-QRS[-2] > 1.5*RR_ave:
                                missed_peaks = peaks[idx[-2]+1:idx[-1]]
                                for missed_peak in missed_peaks:
                                    if missed_peak-peaks[idx[-2]]>int(0.360*self.fs) and ma[missed_peak]>0.5*th:
                                        QRS.append(missed_peak)
                                        QRS.sort()
                                        break

                        if len(QRS)>2:
                            RR.append(QRS[-1]-QRS[-2])
                            if len(RR)>8:
                                RR.pop(0)
                            RR_ave = int(np.mean(RR))

                    else:
                        n_pks.append(ma[peak])
                        if len(n_pks)>8:
                            n_pks.pop(0)
                        n_pks_ave = np.mean(n_pks)

                    th = n_pks_ave + 0.45*(s_pks_ave-n_pks_ave)

                    i+=1

        QRS.pop(0)

        return QRS


# In[ ]:


def swt_detector(self, unfiltered_ecg, MWA_name='cumulative'):
        
        
        maxQRSduration = 0.150 #sec
        swt_level=3
        padding = -1
        for i in range(1000):
            if (len(unfiltered_ecg)+i)%2**swt_level == 0:
                padding = i
                break

        if padding > 0:
            unfiltered_ecg = np.pad(unfiltered_ecg, (0, padding), 'edge')
        elif padding == -1:
            print("Padding greater than 1000 required\n")    

        swt_ecg = pywt.swt(unfiltered_ecg, 'db3', level=swt_level)
        swt_ecg = np.array(swt_ecg)
        swt_ecg = swt_ecg[0, 1, :]

        squared = swt_ecg*swt_ecg

        N = int(maxQRSduration*self.fs)
        mwa = MWA_from_name(MWA_name)(squared, N)
        mwa[:int(maxQRSduration*self.fs*2)] = 0

        filt_peaks = panPeakDetect(mwa, self.fs)
        
        return filt_peaks


# In[ ]:


def MWA_from_name(function_name):
    if function_name == "cumulative":
        return MWA_cumulative
    elif function_name == "convolve":
        return MWA_convolve
    elif function_name == "original":
        return MWA_original
    else: 
        raise RuntimeError('invalid moving average function!')

#Fast implementation of moving window average with numpy's cumsum function 
def MWA_cumulative(input_array, window_size):
    
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    
    for i in range(1,window_size):
        ret[i-1] = ret[i-1] / i
    ret[window_size - 1:]  = ret[window_size - 1:] / window_size
    
    return ret

#Original Function 
def MWA_original(input_array, window_size):

    mwa = np.zeros(len(input_array))
    mwa[0] = input_array[0]
    
    for i in range(2,len(input_array)+1):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i-window_size:i]        
        
        mwa[i-1] = np.mean(section)

    return mwa

#Fast moving window average implemented with 1D convolution 
def MWA_convolve(input_array, window_size):
    
    ret = np.pad(input_array, (window_size-1,0), 'constant', constant_values=(0,0))
    ret = np.convolve(ret,np.ones(window_size),'valid')
    
    for i in range(1,window_size):
        ret[i-1] = ret[i-1] / i
    ret[window_size-1:] = ret[window_size-1:] / window_size
    
    return ret

def two_average_detector(self, unfiltered_ecg, MWA_name='cumulative'):
       
        
        f1 = 8/self.fs
        f2 = 20/self.fs

        b, a = signal.butter(2, [f1*2, f2*2], btype='bandpass')

        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        window1 = int(0.12*self.fs)
        mwa_qrs = MWA_from_name(MWA_name)(abs(filtered_ecg), window1)

        window2 = int(0.6*self.fs)
        mwa_beat = MWA_from_name(MWA_name)(abs(filtered_ecg), window2)

        blocks = np.zeros(len(unfiltered_ecg))
        block_height = np.max(filtered_ecg)

        for i in range(len(mwa_qrs)):
            if mwa_qrs[i] > mwa_beat[i]:
                blocks[i] = block_height
            else:
                blocks[i] = 0

        QRS = []

        for i in range(1, len(blocks)):
            if blocks[i-1] == 0 and blocks[i] == block_height:
                start = i
            
            elif blocks[i-1] == block_height and blocks[i] == 0:
                end = i-1

                if end-start>int(0.08*self.fs):
                    detection = np.argmax(filtered_ecg[start:end+1])+start
                    if QRS:
                        if detection-QRS[-1]>int(0.3*self.fs):
                            QRS.append(detection)
                    else:
                        QRS.append(detection)

        return QRS    


# In[ ]:


def matched_filter_detector(self, unfiltered_ecg, template_file = False):
        
        current_dir = pathlib.Path(__file__).resolve()

        if template_file:
            template = np.loadtxt(template_file)
        else:
            if self.fs == 250:
                template = ecgtemplates.qrs_250Hz
            elif self.fs == 360:
                template = ecgtemplates.qrs_360Hz
            else:
                raise ValueError("!! No stock template for fs = {} !!".format(self.fs))

        f0 = 0.1/self.fs
        f1 = 48/self.fs

        b, a = signal.butter(4, [f0*2, f1*2], btype='bandpass')

        prefiltered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        matched_coeffs = template[::-1]  #time reversing template

        detection = signal.lfilter(matched_coeffs, 1, prefiltered_ecg)  # matched filter FIR filtering
        squared = detection*detection  # squaring matched filter output
        squared[:len(template)] = 0

        squared_peaks = panPeakDetect(squared, self.fs)
  
        return squared_peaks   

