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
                 kaiser_beta=14,
                 sampling_freq=384000,
                 ):
        """ """
        self.window_size = window_size
        self.sampling_freq = sampling_freq
        self.bins_in_hz = None
        self.dbfs_matrix = None
        
        self.window = None
        if window_function.lower() in ['hanning', 'hann']:
            self.window = np.hanning(self.window_size)
        elif window_function.lower() in ['hamming', 'ham']:
            self.window = np.hamming(self.window_size)
        elif window_function.lower() in ['blackman', 'black']:
            self.window = np.blackman(self.window_size)
        elif window_function.lower() in ['blackmanharris', 'blackman-harris']:
            self.window = scipy.signal.blackmanharris(self.window_size)
        elif window_function.lower() in ['kaiser']:
            self.window = scipy.signal.kaiser(self.window_size, kaiser_beta)
        else:
            raise UserWarning("Invalid window function name.")

        # Max db value in window. DBFS = db full scale. Half spectrum used.
        self.dbfs_max = np.sum(self.window) / 2 

    def get_freq_bins_in_hz(self):
        """ Converts frequency bins to array in Hz. Calculated on demand. """
        # From "0" to "< FS/2".
        if self.bins_in_hz is None:      
#            self.bins_in_hz = np.fft.rfftfreq(self.window_size)[1:] * self.sampling_freq
            self.bins_in_hz = np.fft.rfftfreq(self.window_size)[:-1] * self.sampling_freq
#             bins = np.fft.rfftfreq(self.window_size)[:-1]
#             self.bins_in_hz = (bins + (bins[1] / 2)) * self.sampling_freq # TODO: Adjust up in frequency???
        #
        return self.bins_in_hz

    def calc_dbfs_matrix(self, signal, matrix_size=128, jump=384):
        """ Convert frame to dBFS spectrum. """
        # Reuse the same matrix for fast processing.
        if self.dbfs_matrix is None:
            self.dbfs_matrix = np.full([matrix_size, int(self.window_size / 2)], -120.0) # Default = -120 dBFS.
        else:
            self.dbfs_matrix.fill(-120) # Default = -120 dBFS.
        #hop_length = int(self.rate/1000) # 1 ms.
        signal_len = len(signal)      
        row_number = 0
        start_index = 0
        while (row_number < matrix_size) and ((start_index + jump) < signal_len):
            spectrum = self.calc_dbfs_spectrum(signal[start_index:start_index+self.window_size])
            if spectrum is not False:
                self.dbfs_matrix[row_number] = spectrum
            row_number += 1
            start_index += jump
        #   
        return self.dbfs_matrix

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

    def interpolation_spectral_peak(self, spectrum_db):
        """ Quadratic interpolation of spectral peaks. 
            https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        """
        peak_bin = spectrum_db.argmax()
        if (peak_bin == 0) or (peak_bin >= len(spectrum_db) - 1):
            y0 = 0
            y1 = spectrum_db[peak_bin]
            y2 = 0
            x_adjust = 0.0
        else:
            y0, y1, y2 = spectrum_db[peak_bin-1:peak_bin+2]
            x_adjust = (y0 - y2) / 2 / (y0 - y1*2 + y2)
        # 
        peak_frequency = (peak_bin + x_adjust) * self.sampling_freq / self.window_size
        # Peak magnitude.
        peak_magnitude = y1 - (y0 - y2) * x_adjust / 4
        #
        return peak_frequency, peak_magnitude

# === MAIN ===    
if __name__ == "__main__":
    """ """
    print('Test started.')
    dsu = DbfsSpectrumUtil(window_size=16)
    freq, mag = dsu.calc_max_freq_from_db_spectrum(np.array([40,50,11,0,-10,-10,-10,5,10,5,-10,-10,-10,-10,30,20,]))
    print('Freq: ', freq, '   Magnitude: ', mag)
    print('Test ended.')
