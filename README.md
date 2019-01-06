prepocess.py

#### fft_array ####
takes filename(can only take wave file, remeber discard .wav), frame_length(=40ms), overlap(=10ms), bit_depth(=16 bits) as input parameters. 
return 2d ndarray, first dimension is nth frame, second is frequency(resolution is sample_rate/number_of_samples_in_a_frame, by default is 25Hz). 


1/4 update
#### fft_array ####
use hamming window over all frames to avoid power leakage. 
add number of channels as a input parameter. 
also return full image of magnitude and phase at index 2,3.

#### ifft_array ####
take full image magnitude and phase as input parameters.
return 16 bit mono wav array.

1/5 update

#### fft_array ####
add std of white noise as a input parameter.
correct the mag and phase and angle pi and 0.
delete the return of full mag and full phase .

#### ifft_array ####
change input parameters, take half image magnitude and phase as input parameter.

#### low_pass_filter ####
take magnitude, phase, cuttoff, and decay per otave as input parameters.
return mag and phase.

1/6 update

#### fft_array ####
add interval as a input parameter to slice the wav file. 
