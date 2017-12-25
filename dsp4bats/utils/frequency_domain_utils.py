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
#         self.dbfs_matrix = None
        
        self.window = None
        if window_function.lower() in ['hanning', 'hann']:
            self.window = np.hanning(self.window_size)
        elif window_function.lower() in ['blackman', 'black']:
            self.window = np.blackman(self.window_size)
        elif window_function.lower() in ['blackmanharris', 'blackman-harris', 'blackh']:
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
            self.bins_in_hz = np.fft.rfftfreq(self.window_size)[:-1] * self.sampling_freq
        #
        return self.bins_in_hz

    def calc_dbfs_matrix(self, signal, 
                         matrix_size=128, 
                         jump=None):
        """ Convert frame to dBFS spectrum. """
        if jump is None:
            jump=self.sampling_freq/1000 # Default = 1 ms.
            
#         # Reuse the same matrix for fast processing.
#         if self.dbfs_matrix is None:
#             self.dbfs_matrix = np.full([matrix_size, int(self.window_size / 2)], -120.0) # Default = -120 dBFS.
#         else:
#             self.dbfs_matrix.fill(-120) # Default = -120 dBFS.

        dbfs_matrix = np.full([matrix_size, int(self.window_size / 2)], -120.0) # Default = -120 dBFS.

        signal_len = len(signal)      
        row_number = 0
        start_index = 0
        while (row_number < matrix_size) and ((start_index + jump) < signal_len):
            spectrum = self.calc_dbfs_spectrum(signal[start_index:start_index+self.window_size])
            if spectrum is not False:
#                 self.dbfs_matrix[row_number] = spectrum
                dbfs_matrix[row_number] = spectrum
            row_number += 1
            start_index += jump
        #   
#         return self.dbfs_matrix
        return dbfs_matrix

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

    def interpolation_of_spectral_peak(self, spectrum_db):
        """ Quadratic interpolation of spectral peaks. Read more at:
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
        # Peak amplitude.
        peak_amplitude = y1 - (y0 - y2) * x_adjust / 4
        #
        return peak_frequency, peak_amplitude

    def chirp_metrics_header(self):
        """ """
        return ['peak_freq_hz', 'peak_dbfs', 
                'start_freq_hz', 'end_freq_hz', 
                'max_freq_hz', 'min_freq_hz', 
                'duration_ms', 
                'peak_signal_index', 'start_signal_index', 'end_signal_index']
        
    def chirp_metrics(self, signal, peak_position, 
                      jump_factor=4000, # Jump factor: 4000 = 0.25 ms.
                      high_pass_filter_freq_hz=15000,
                      threshold_dbfs = -50.0, 
                      threshold_dbfs_below_peak = 15.0, 
                      max_frames_to_check=100, 
                      max_silent_slots=8, 
                      debug=False):
        """ Extracts chirp metrics based on peak freq/"""
        signal_length = len(signal)
        # Expected results.
        peak_freq_hz = None
        peak_dbfs = None
        start_freq_hz = None
        end_freq_hz = None
        max_freq_hz = None
        min_freq_hz = None
        duration_ms = None
        # Indexes for results.
        peak_index = None
        start_index = None
        end_index = None
        max_freq_index = None
        min_freq_index = None
        # Resolution in time.
        jump = int(self.sampling_freq / jump_factor)
        # Used to decide when to stop checking.
        negative_index_counter = 0
        positive_index_counter = 0
        # Loop over frames. Switch between positive and negative side.
        for ix in range(1, max_frames_to_check):
            # Jump 0,1,-1,2,-2,3,-3...
            index = int(ix / 2)
            if (ix % 2) != 0: # Modulo operator.
                index *= -1
            if index < 0:
                if negative_index_counter > max_silent_slots:
                    if positive_index_counter > max_silent_slots:
                        # if debug: print('DEBUG: Break-pos: ', index)
                        break # Done.
                    else:
                        continue # Don't check after silent part.
            else:
                if positive_index_counter > max_silent_slots:
                    if negative_index_counter > max_silent_slots:
                        # if debug: print('DEBUG: Break-neg: ', index)
                        break # Done.
                    else:
                        continue # Don't check after silent part.
            # Check if still inside signal.
            start = peak_position + jump * index
            if start < 0:
                negative_index_counter = max_silent_slots + 10 # Finished.
                continue
            if start+self.window_size >= signal_length:
                positive_index_counter = max_silent_slots + 10 # Finished.
                continue
            # Calculate spectrum in dBFS.            
            spectrum = self.calc_dbfs_spectrum(signal[start:start+self.window_size])
            if spectrum is False:
                continue
            # Calculate frequency and dBFS by interpolation over spectral bins. 
            bin_freq_hz, bin_dbfs = self.interpolation_of_spectral_peak(spectrum)
            # Check peak and adjust if the original peak_position was wrong..
            if (peak_dbfs is None) or (peak_dbfs < bin_dbfs):
                peak_dbfs = bin_dbfs
                peak_freq_hz = bin_freq_hz
                peak_index = index
            # Check levels.
            if (bin_dbfs > peak_dbfs - threshold_dbfs_below_peak) and \
               (bin_dbfs > threshold_dbfs):
                # Metric start_freq.
                if (start_index is None) or (start_index > index ):
                    start_freq_hz = bin_freq_hz 
                    start_index = index
                # Metric end.
                if (end_index is None) or (end_index < index): 
                    end_freq_hz = bin_freq_hz
                    end_index = index
                # Metric max_freq.
                if (max_freq_index is None) or (max_freq_hz < bin_freq_hz):
                    max_freq_hz = bin_freq_hz
                    max_freq_index = index
                # Metric min_freq.
                if (min_freq_index is None) or (min_freq_hz > bin_freq_hz):
                    min_freq_hz = bin_freq_hz
                    min_freq_index = index
                # Used to decide when to stop checking.    
                if index < 0:
                    negative_index_counter = 0
                else:
                    positive_index_counter = 0
            else:
                # Used to decide when to stop checking.
                if index < 0:
                    negative_index_counter += 1
                else:
                    positive_index_counter += 1
        
        # Loop finished.
        if start_index is not None:
            # Apply high pass filter.
            if peak_freq_hz < high_pass_filter_freq_hz: 
                return False # No peak above thos frequency was found.
            #
            peak_signal_index = peak_position + jump * peak_index
            start_signal_index = peak_position + jump * start_index
            end_signal_index = peak_position + jump * end_index
            duration_ms = (end_index - start_index + 1) * jump / self.sampling_freq * 1000
            # Print for debug.
            if debug:
                print('Peak index: ', peak_signal_index, 
                      '  peak freq: ', np.round(peak_freq_hz/1000, 3), 
                      '  peak dbfs: ', np.round(peak_dbfs, 1), \
                      '  bw: ', np.round(np.absolute(start_freq_hz - end_freq_hz)/1000, 3), 
                      '  start freq: ', np.round(start_freq_hz/1000, 3), 
                      '  end freq: ', np.round(end_freq_hz/1000, 3), 
                      '  min freq: ', np.round(min_freq_hz/1000, 3), 
                      '  max freq: ', np.round(max_freq_hz/1000, 3), 
                      '  duration (ms): ', duration_ms  )
            #
            return (peak_freq_hz, peak_dbfs, 
                    start_freq_hz, end_freq_hz, 
                    max_freq_hz, min_freq_hz, 
                    duration_ms, 
                    peak_signal_index, start_signal_index, end_signal_index)
        else:
            return False

    def chirp_shape_row_header(self):
        """ """
        return ['time_s', 'frequency_hz', 'amplitude_dbfs', 'signal_index']
        
    def chirp_shape(self, signal, peak_position, 
                    start_index=None, 
                    stop_index=None, 
                    jump_factor=8000, # Jump factor: 8000 = 0.125 ms.
                    max_size=256):
        """ To be used for plotting similar to ZC (Zero Crossing). """
        # Create a matrix with one row for each 0.125 ms. Size 256*(window_size/2). 
        jump = int(self.sampling_freq / jump_factor) 
        if start_index is None:
            start_index = int(peak_position - (max_size * jump / 2))
        if stop_index is not None:
            max_size = int((stop_index - start_index) / jump)
        # Make it wider.
        start_index -= jump * 5
        max_size += 5
        # Calculate matrix.                
        matrix = self.calc_dbfs_matrix(signal[start_index:], matrix_size=max_size, jump=jump)
        # Get max dBFS value. (Note: Not needed now, maybe later...)
        # row, col = np.unravel_index(matrix.argmax(), matrix.shape)
        # calc_peak_freq_hz, calc_peak_dbfs = self.interpolation_of_spectral_peak(matrix[row])
        #
        result_table = []
        for spectrum_index, spectrum in enumerate(matrix):
            # Interpolate.
            freq_hz, amp_db = self.interpolation_of_spectral_peak(spectrum)
            #
            signal_index = start_index + spectrum_index * jump
            time_s = np.round(signal_index / self.sampling_freq, 5)
            frequency_hz = np.round(freq_hz, 0)
            amplitude_dbfs = np.round(amp_db, 1)
            # Add to result.
            result_table.append([time_s, frequency_hz, amplitude_dbfs, signal_index])
        #
        return result_table


# === MAIN ===    
if __name__ == "__main__":
    """ """
    print('Test started.')
    dsu = DbfsSpectrumUtil(window_size=16)
    freq, amp_db = dsu.interpolation_of_spectral_peak(np.array([0,0,0,0,0,0,0,3,10,3,0,0,0,0,0,0,]))
    print('Freq: ', freq, '   amp(db): ', amp_db)
    freq, amp_db = dsu.interpolation_of_spectral_peak(np.array([0,0,0,0,0,0,0,3,10,7,0,0,0,0,0,0,]))
    print('Freq: ', freq, '   amp(db): ', amp_db)
    print('Test ended.')
