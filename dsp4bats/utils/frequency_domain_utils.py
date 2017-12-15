#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import numpy as np
import scipy.signal
import soundfile
import librosa

class DbfsSpectrumUtil():
    """ """
    def __init__(self, 
                 window_size=512,
                 window_function='hann',
                 sampling_frequency=384000,
                 ):
        """ """
        self.window_size = window_size
        self.sampling_frequency = sampling_frequency
        self.bins_in_hz = None
        
        self.window = None
        if window_function.lower() in ['hanning', 'hann']:
            self.window = np.hanning(self.window_size)
        elif window_function.lower() in ['hamming', 'ham']:
            self.window = np.hamming(self.window_size)
        elif window_function.lower() in ['blackman', 'black']:
            self.window = np.blackman(self.window_size)
        elif window_function.lower() in ['blackmanharris', 'blackman-harris']:
            self.window = scipy.signal.blackmanharris(self.window_size)
        else:
            raise UserWarning("Invalid window function name.")

        # Max db value in window. DBFS = db full scale. Half spectrum used.
        self.dbfs_max = np.sum(self.window) / 2 

    def get_freq_bins_in_hz(self):
        """ Converts frequency bins to array in Hz. Calculated on demand. """
        # From "0" to "< FS/2".
        if self.bins_in_hz is None:      
#            self.bins_in_hz = np.fft.rfftfreq(self.window_size)[1:] * self.sampling_frequency
            self.bins_in_hz = np.fft.rfftfreq(self.window_size)[:-1] * self.sampling_frequency
#             bins = np.fft.rfftfreq(self.window_size)[:-1]
#             self.bins_in_hz = (bins + (bins[1] / 2)) * self.sampling_frequency # TODO: Adjust up in frequency???
        #
        return self.bins_in_hz

    def calc_dbfs_matrix(self, signal, matrix_size=128, jump=384):
        """ Convert frame to dBFS spectrum. """
        matrix = np.full([matrix_size, int(self.window_size / 2)], -100.0) # Default = -100 dBFS.
        #hop_length = int(self.rate/1000) # 1 ms.
        signal_len = len(signal)      
        row_number = 0
        start_index = 0
        while (row_number < matrix_size) and ((start_index + jump) < signal_len):
            spectrum = self.calc_dbfs_spectrum(signal[start_index:start_index+self.window_size])
            if spectrum is not False:
                matrix[row_number] = spectrum
            row_number += 1
            start_index += jump
        #   
        return matrix

    def calc_dbfs_spectrum(self, signal):
        """ Convert frame to dBFS spectrum. """
        signal_len = len(signal)
        if signal_len == self.window_size:
            frame = signal * self.window
        elif signal_len > self.window_size:
            frame = signal[:self.window_size] * self.window
        else:
            return False
        # Calc dBFS spectrum.
        spectrum = np.fft.rfft(frame)[:-1]
        dbfs_spectrum = 20 * np.log10(np.abs(spectrum) / self.dbfs_max)
        #
        return dbfs_spectrum


# === MAIN ===    
if __name__ == "__main__":
    """ """
    print('Test started.')
    print('Test ended.')
