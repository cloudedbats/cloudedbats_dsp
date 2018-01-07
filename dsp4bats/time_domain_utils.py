#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017-2018 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import numpy as np
import scipy.signal
import librosa
#import wave_file_utils

class SignalUtil():
    """ """
    def __init__(self, 
                 sampling_freq=384000,
                 ):
        """ """
        self.sampling_freq = sampling_freq
        self.array_in_sec = None

    def get_array_in_sec(self, signal):
        """ Normally used for x array in sec. """
        return np.arange(0, len(signal)) / self.sampling_freq

    def noise_level(self, signal):
        """ """
        return np.sqrt(np.mean(np.square(signal)))

    def noise_level_in_db(self, signal):
        """ """
        return 20 * np.log10(self.noise_level(signal) / 1.0)
        
    def butterworth_filter(self, signal, 
                           low_freq_hz=None, # For highpass and bandpass filters
                           high_freq_hz=None, # For lowpass and bandpass filters
                           filter_order=9,  
                           bandstop=False): # Use both low_ and high_freq_hz for bandstop. 
        """ Filter. Butterworth. """
        nyquist = 0.5 * self.sampling_freq
        #
        if (low_freq_hz is not None) and (high_freq_hz is None) and (bandstop is False):
            low = low_freq_hz / nyquist
            b, a = scipy.signal.butter(filter_order, [low], btype='highpass')
        elif (low_freq_hz is None) and (high_freq_hz is None) and (bandstop is False):
            high = high_freq_hz / nyquist
            b, a = scipy.signal.butter(filter_order, [high], btype='lowpass')
        elif (low_freq_hz is None) and (high_freq_hz is None) and (bandstop is False):
            low = low_freq_hz / nyquist
            high = high_freq_hz / nyquist
            b, a = scipy.signal.butter(filter_order, [low, high], btype='bandpass')
        elif (low_freq_hz is not None) and (high_freq_hz is not None) and (bandstop is True):
            low = low_freq_hz / nyquist
            high = high_freq_hz / nyquist
            b, a = scipy.signal.butter(filter_order, [low, high], btype='bandstop')
        else:
            return signal
        # Apply folter on signal.
        # filtered_signal = scipy.signal.lfilter(b, a, signal)
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        #
        return filtered_signal
    
    def find_localmax(self, signal,
                      noise_threshold=0.0, # Range: [0.0, 1.0]. 
                      jump=None, 
                      frame_length=1024):
        """ """
        # Adjust for comparable results for low sampling rates.
        if self.sampling_freq < 300000:
            frame_length = int(frame_length / 2) 
        if jump is None:
            jump=int(self.sampling_freq/1000) # Default = 1 ms.
        y = signal.copy()
        if noise_threshold > 0.0:
            y[(np.abs(y) < noise_threshold)] = 0.0
        rmse = librosa.feature.rmse(y=y, hop_length=jump, frame_length=frame_length, center=True)
        locmax = librosa.util.localmax(rmse.T)
        maxindexlist = [index for index, a in enumerate(locmax) if a==True]
        # Original index list is related to jump length. Convert.
        index_list = librosa.frames_to_samples(maxindexlist, hop_length=jump)
        #
        return index_list

    def chirp_generator(self, 
                        start_freq_hz = 100000, 
                        end_freq_hz = 20000, 
                        duration_s = 0.008, 
                        chirp_interval_s = 0.1, 
                        max_amplitude = 0.3, 
                        noise_level = 0.002, 
                        number_of_chirps = 10, 
                        ):
        """ """
        # Create chirp. The shape is in between FM and QCF calls.
        time = np.linspace(0, duration_s, int(self.sampling_freq * duration_s))
        chirp = scipy.signal.waveforms.chirp(time, 
                                             f0=start_freq_hz, 
                                             f1=end_freq_hz, 
                                             t1=duration_s, 
                                             method='quadratic', 
                                             vertex_zero=False)
        # Apply window function and amplitude.
        chirp = chirp * scipy.signal.hanning(len(time)) * max_amplitude
        # Create silent part.
        silent_duration = chirp_interval_s - duration_s
        silent_half = np.zeros(int(self.sampling_freq * silent_duration / 2))
        # Build sequence.
        signal = []
        for index in range(number_of_chirps):
            signal = np.concatenate((signal, silent_half, chirp, silent_half))
        # Add noise.
        signal = signal + np.random.randn(len(signal)) * noise_level
        # 
        return signal


# === TEST ===    
if __name__ == "__main__":
    """ """
#     print('Test started.')
#      
#     # Sugnal util.
#     signal_util = SignalUtil(sampling_freq=384000)
#     # Create chirp.
#     signal = signal_util.chirp_generator() # Defaults only.
#     # Write to file in Time Expanded mode.
#     wave_writer = wave_file_utils.WaveFileWriter('test.wav',
#                                  sampling_freq=signal_util.sampling_freq,
#                                  time_expanded=True)
#     wave_writer.write_buffer(signal)
#     print('Out buffer length in sec: ', len(signal)/wave_writer.sampling_freq)
#     wave_writer.write_buffer(signal)
#     print('Out buffer length in sec: ', len(signal)/wave_writer.sampling_freq)
#     wave_writer.close()
#  
#     # Read file.
#     wave_reader = wave_file_utils.WaveFileReader('test.wav')
#     signal = wave_reader.read_buffer()
#     print('In buffer length in sec: ', len(signal)/wave_reader.sampling_freq)
#     signal = wave_reader.read_buffer()
#     print('In buffer length in sec: ', len(signal)/wave_reader.sampling_freq)
#     signal = wave_reader.read_buffer()
#     print('In buffer length in sec: ', len(signal)/wave_reader.sampling_freq)
#     wave_reader.close()
#      
#     print('Test ended.')

