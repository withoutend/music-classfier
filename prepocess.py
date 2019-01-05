import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.io import wavfile # get the api
import math
import cmath
import numpy as np
import pydub
from pydub import AudioSegment
from os import path
import wave
#import seaborn

def fft_array(filename,channels=2,frame_length=40,overlap=10,bit_depth=16,white_noise_std_in_dB=2**10):
    mag=[]
    phase=[]
    fs, data = wavfile.read(filename+'.wav')
    
    #compute average of all channel
    square_sum=[item for item in data.T[0]]
    for i in range(1,channels):
        square_sum+=[item for item in data.T[i]]
    rms=np.array([(item/channels) for item in square_sum])
    
    #add white noise
    rms+=np.random.normal(0, (2**bit_depth)*(10**(white_noise_std_in_dB/10)), size=rms.shape[0])
    
    
    num_sample=int(fs*frame_length/1e3)
    window = np.hamming(num_sample)
    begin_frame=0
    
    #fft frame by frame
    while((begin_frame+fs*frame_length/1e3)<rms.size):
        y=rms[begin_frame:int(begin_frame+fs*frame_length/1e3)]
        
        #apply hamming window
        yy=fft(np.multiply(window, y))

        yf=abs(yy)
        yf2 = yf[0:int(num_sample/2+1)] 

        yp=np.array([cmath.phase(item) for item in yy])
        yp2=yp[0:int(num_sample/2+1)]
        
        mag.append(yf2)
        phase.append(yp2)
        begin_frame+=int(fs*(frame_length-overlap)/1e3)

    return np.array(mag),np.array(phase)

def ifft_array(mag,phase,channels=2,frame_length=40,overlap=10,bit_depth=16,fs=44100):
    data=[]
    num_sample=(mag[0].size-1)*2
    inverse_window=1./np.hamming(num_sample)
    frame_mag=[]
    frame_phase=[]
    frame_complex=[]
    for frame in range(mag.shape[0]):
        
        #reconstruct complex array
        for item in mag[frame]:
            frame_mag.append(item)
        for item in mag[frame][::-1][1:-1]:
            frame_mag.append(item)
    
        for item in phase[frame]:
            frame_phase.append(item)
        for item in phase[frame][::-1][1:-1]:
            frame_phase.append(-item)  
        
        for i in range(len(frame_mag)):
            frame_complex.append(cmath.rect(frame_mag[i],frame_phase[i]))
        
        #apply ifft 
        if(frame!=0):
            data=data[0:(len(data)-int(num_sample*overlap/frame_length))]
        
        #apply inverse hamming window
        data.extend(np.multiply(inverse_window,ifft(frame_complex).real))
        
        #clear memory
        frame_mag=[]
        frame_phase=[]
        frame_complex=[]
    
    #get real part, convert to 16 bit-depth
    return np.array([item*channels for item in data],dtype=np.int16)

 def low_pass_filter(mag,phase,cut_off=500,decay_in_dB=-10,frame_length=40,overlap=10,bit_depth=16,fs=44100):
    num_sample=(mag[0].size-1)*2
    
    #compute frequency resolution
    f_resolution=fs/num_sample
    
    #scalar array for lpf
    lpf_mag=[]
    lpf_phase=[]
    
    #transfer dB into ratio
    decay=10**(decay_in_dB/10)
    for i in range(mag[0].size):
        f=i*f_resolution
        if(f>=cut_off):
            lpf_mag.append(decay**math.log(f/cut_off,2))  
            #lpf_phase.append(-math.atan(f/cut_off))
        else:
            lpf_mag.append(1)
            #lpf_phase.append(0)
        #do noting on phase
        lpf_phase.append(0)
        
    return np.multiply(mag,np.array(lpf_mag)),np.array(phase+np.array(lpf_phase))

'''
mag,phase=fft_array('untitled',white_noise_std_in_dB=-100)
mag,phase=low_pass_filter(mag,phase,cut_off=20000,decay_in_dB=-10)
array=ifft_array(mag,phase)
wavfile.write('tem.wav',rate=44100,data=array[0:int(len(array)/2)])
'''