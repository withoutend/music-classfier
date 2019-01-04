#import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.io import wavfile # get the api
#import math
import cmath
import numpy as np
import pydub
from pydub import AudioSegment
from os import path
import wave
#import seaborn

def fft_array(filename,channel=2,frame_length=40,overlap=10,bit_depth=16):
    mag=[]
    phase=[]
    full_mag=[]
    full_phase=[]
    fs, data = wavfile.read(filename+'.wav') 
    square_sum=[item for item in data.T[0]]
    for i in range(1,channel):
        square_sum+=[item for item in data.T[i]]
    rms=np.array([item/channel for item in square_sum])
    num_sample=int(fs*frame_length/1e3)
    window = np.hamming(num_sample)
    begin_frame=0
    while((begin_frame+fs*frame_length/1e3)<rms.size):
        y=rms[begin_frame:int(begin_frame+fs*frame_length/1e3)]
        
        yy=fft(np.multiply(window, y))

        yf=abs(yy)
        yf1=yf
        yf2 = yf1[0:int(num_sample/2)] 

        yp=np.array([cmath.phase(item) for item in yy])
        yp1=yp
        yp2=yp[0:int(num_sample/2)]
        full_mag.append(yf1)
        full_phase.append(yp1)
        mag.append(yf2)
        phase.append(yp2)
        begin_frame+=int(fs*(frame_length-overlap)/1e3)
    return np.array(mag),np.array(phase),np.array(full_mag),np.array(full_phase)

def ifft_array(mag,phase,frame_length=40,overlap=10,bit_depth=16,fs=44100):
    data=[]
    num_sample=mag[0].size#*2
    inverse_window=1./np.hamming(num_sample)
    frame_mag=[]
    frame_phase=[]
    frame_complex=[]
    for frame in range(mag.shape[0]):
        for item in mag[frame]:
            frame_mag.append(item)#*num_sample
        #for item in mag[frame][::-1]:
            #frame_mag.append(item)#*num_sample
       # print(frame_mag[0:10])
        for item in phase[frame]:
            frame_phase.append(item)
        #for item in phase[frame][::-1]:
            #frame_phase.append(-item)
        
        
        for i in range(len(frame_mag)):
            frame_complex.append(cmath.rect(frame_mag[i],frame_phase[i]))
        
        if(frame!=0):
            data=data[0:(len(data)-int(num_sample*overlap/frame_length))]
        data.extend(np.multiply(inverse_window,ifft(frame_complex).real))
        frame_mag=[]
        frame_phase=[]
        frame_complex=[]
        
    return np.array([item*2 for item in data],dtype=np.int16)

'''
mag,phase,x,y=fft_array('untitled')#[mag,phase]=
array=ifft_array(x,y)
wavfile.write('tem.wav',rate=44100,data=array[0:int(len(array)/2)])#data=array[0:int(len(array)/2)]
'''