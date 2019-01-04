prepocess.py

#### fft_array ####
takes filename(can only take wave file, remeber discard .wav), frame_length(=40ms), overlap(=10ms), bit_depth(=16 bits) as input parameters
return 2d ndarray, first dimension is nth frame, second is frequency(resolution is sample_rate/number_of_samples_in_a_frame, by default is 25Hz)


1/4 update
#### fft_array ####
use hamming window in frame to avoid power leakage
add number of channels as new ainput parameter
also return full image of magnitude and phase at index 2,3.

#### ifft_array ####
take full image magnitude and phase as input parameter
return 16 bit mono wav array