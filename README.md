prepocess.py

#### fft_array ####
takes filename(can only take wave file, remeber discard .wav), frame_length(=40ms), overlap(=10ms), bit_depth(=16 bits) as input parameters
return 2d ndarray, first dimension is nth frame, second is frequency(resolution is sample_rate/number_of_samples_in_a_frame, by default is 25Hz)