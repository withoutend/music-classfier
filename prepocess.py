#import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.io import wavfile # get the api
#import math
import cmath
import numpy as np
#import seaborn

#filename excep .wav
#frame_length, overlap is passed by milli second
#bit_depth is set to be 16 bits
def fft_array(filename,frame_length=40,overlap=10,bit_depth=16):
    mag=[]
    phase=[]
    fs, data = wavfile.read(filename+'.wav') 
    square_sum=[left for left in data.T[0]]+[right for right in data.T[1]]
    rms=np.array([item/2 for item in square_sum])
    num_sample=int(fs*frame_length/1e3)
    x=np.linspace(0,1,num=num_sample)

    #for test
    #x=np.linspace(0,1,num=44100)
    #y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)
    #print(num_sample)
    begin_frame=0
    while((begin_frame+fs*frame_length/1e3)<rms.size):
        y=rms[begin_frame:int(begin_frame+fs*frame_length/1e3)]

        yy=fft(y)
        yreal = yy.real
        yimag = yy.imag

        yf=abs(yy)
        yf1=yf/len(x)
        yf2 = yf1[range(int(num_sample/2))] 

        xf = np.arange(len(y))
        xf1 = xf
        xf2 = xf[range(int(num_sample/2))]

        yp=np.array([cmath.phase(item) for item in yy])
        yp1=yp/num_sample
        yp2=yp[range(int(num_sample/2))]

        mag.append(yf2)
        phase.append(yp2)
        begin_frame+=int(fs*overlap/1e3)
    return [np.array(mag),np.array(phase)]

'''
how to use it
mag,phase=fft_array('untitled')
'''