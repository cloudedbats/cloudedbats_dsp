#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017-2018 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import numpy as np
import scipy.signal
import wave
#import scipy.io.wavfile
import librosa

class WaveFileReader():
    """ """
    def __init__(self, file_path=None):
        """ """
        self.clear()
        if file_path is not None:
            self.open(file_path)
        
    def clear(self):
        """ """
        self.wave_file = None
        self.channels = None
        self.samp_width = None
        self.frame_rate = None
        self.sampling_freq = None

    def open(self, file_path=None,
            convert_te=True):
        """ """
        if file_path is not None:
            self.file_path = file_path
        #
        if self.wave_file is not None:
            self.close()
        #
        self.wave_file = wave.open(self.file_path, 'rb')
        self.samp_width = self.wave_file.getsampwidth()
        self.frame_rate = self.wave_file.getframerate()
        #
        self.sampling_freq = self.frame_rate
        if convert_te:
            if self.sampling_freq < 192000:
                self.sampling_freq *= 10 # Must be Time Expanded.

    def read_buffer(self, buffer_size=None):
        """ """
        if self.wave_file is None:
            self.open()
        #    
        if buffer_size is None:
            buffer_size = self.sampling_freq # Read 1 sec as default.
        #
        frame_buffer = self.wave_file.readframes(buffer_size)
        # Convert to signal in the interval [-1.0, 1.0].
        signal = librosa.util.buf_to_float(frame_buffer, n_bytes=self.samp_width)
        #
        return signal       

    def close(self):
        """ """
        self.wave_file.close()
        self.wave_file = None

class WaveFileWriter():
    """ """
    def __init__(self, file_path=None,
                 channels = 1,
                 samp_width = 2,
                 sampling_freq = 384000,
                 frame_rate = 38400,
                 time_expanded = False,   
                ):
        """ """
        self.clear()
        self.file_path = file_path
        self.channels = channels
        self.samp_width = samp_width
        self.sampling_freq = sampling_freq
        self.frame_rate = frame_rate
        self.time_expanded = time_expanded
        
    def clear(self):
        """ """
        self.wave_file = None
        self.channels = None
        self.samp_width = None
        self.sampling_freq = None
        self.frame_rate = None
        self.time_expanded = False

    def open(self, file_path=None):
        """ """
        if file_path is not None:
            self.file_path = file_path
        #
        if self.wave_file is not None:
            self.close()
        #
        if self.time_expanded:
            self.frame_rate = int(self.sampling_freq / 10)
        #
        self.wave_file = wave.open(self.file_path, 'wb')
        self.wave_file.setnchannels(self.channels)
        self.wave_file.setsampwidth(self.samp_width)
        self.wave_file.setframerate(self.frame_rate)

    def write_buffer(self, signal):
        """ """
        if self.wave_file is None:
            self.open()
        # Convert from signal in the interval [-1.0, 1.0] to int16.
        signal_int16 = np.int16(signal * 32767)
        #
        self.wave_file.writeframes(signal_int16)

    def close(self):
        """ """
        self.wave_file.close()
        self.wave_file = None

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

    def find_localmax(self, signal,
                      noise_threshold=0.0, 
                      jump=None, 
                      frame_length=1024):
        """ """
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
        # Create chirp.
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
    print('Test started.')
    
    # Chirp generator.
    signal_util = SignalUtil(sampling_freq=384000)
    signal = signal_util.chirp_generator() # Defaults only.
    # Write to file in Time Expanded mode.
    wave_writer = WaveFileWriter('test.wav',
                                 sampling_freq=signal_util.sampling_freq,
                                 time_expanded=True)
    wave_writer.write_buffer(signal)
    print('Out buffer length in sec: ', len(signal)/wave_writer.sampling_freq)
    wave_writer.write_buffer(signal)
    print('Out buffer length in sec: ', len(signal)/wave_writer.sampling_freq)
    wave_writer.close()

    # Read file.
    wave_reader = WaveFileReader('test.wav')
    signal = wave_reader.read_buffer()
    print('In buffer length in sec: ', len(signal)/wave_reader.sampling_freq)
    signal = wave_reader.read_buffer()
    print('In buffer length in sec: ', len(signal)/wave_reader.sampling_freq)
    signal = wave_reader.read_buffer()
    print('In buffer length in sec: ', len(signal)/wave_reader.sampling_freq)
    wave_reader.close()
    
    print('Test ended.')

