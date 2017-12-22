#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import numpy as np
import librosa
###import soundfile
import wave
import pathlib
import utils.time_domain_utils
import utils.frequency_domain_utils

class BatAnalysisTool():
    """ """
    def __init__(self, wave_file_path):
        """ """
        self.wave_file_path = wave_file_path

    def analyse_file(self):
        """ """
        # Prepare file.
        wave_file = wave.open(self.wave_file_path, 'rb')
        # channels = wave_file.getnchannels()
        samp_width = wave_file.getsampwidth()
        frame_rate = wave_file.getframerate()
        
        # Adjust for TE.
        sampling_freq = frame_rate
        if sampling_freq < 192000:
            sampling_freq *= 10
        
        # Prepare utilities.
        signal_util = utils.time_domain_utils.SignalUtil(sampling_freq)
        spectrum_util = utils.frequency_domain_utils.DbfsSpectrumUtil(window_size=128,
                                                                      window_function='kaiser',
                                                                      kaiser_beta=14,
                                                                      sampling_freq=sampling_freq)
        # Open file to write results.
        out_file_name = pathlib.Path(self.wave_file_path).stem + '_ANALYSIS_RESULTS.txt'
        with pathlib.Path(out_file_name).open('w') as out_file:
            out_header = ['time_s', 'frequency_khz', 'dbfs', 'pulse_peak_khz', 'pulse_peak_dbfs']
            out_file.write('\t'.join(map(str, out_header)) + '\n')

            found_peak_counter = 0
            buffer_number = 0

            # Read frames for 1 sec.
            buffer_size = sampling_freq
            frame_buffer_1s = wave_file.readframes(buffer_size)
            # Read until end of file.
            while len(frame_buffer_1s) > 0:
                # Transform from int to float in the range [-1, 1].
                signal_1s = librosa.util.buf_to_float(frame_buffer_1s, n_bytes=samp_width)
                  
                # === Time domain. ===
                # Get noise level for 1 sec.
                noise_level = signal_util.noise_level(signal_1s)
                noise_level_db = signal_util.noise_level_in_db(signal_1s)
                print('Noise level: ', np.round(noise_level, 2), '   Noise (db): ', np.round(noise_level_db, 2))

                # Find peaks in time domain.
                peaks = signal_util.calc_localmax(signal=signal_1s,
                                                  noise_threshold=noise_level*4.0, # Threshold.
                                                  hop_length=int(sampling_freq/1000), # Jump 1 ms.
                                                  frame_length=1024) # Window size.
                
                # === Extract metrics and print result. ===
                
                peak_number = 0
                for peak_position in peaks:
                     
                    # Results.
                    peak_freq = None
                    peak_dbfs = None
                    start_freq = None
                    end_freq = None
                    max_freq = None
                    min_freq = None
                    duration = None
                    
                    # Indexes for results.
                    peak_index = None
                    start_index = None
                    end_index = None
                    max_freq_index = None
                    min_freq_index = None
                    
                    wsize = spectrum_util.window_size
                    jump = int(sampling_freq/1000/4)

                    max_frames_to_check = 100
                    negative_index_counter = 0
                    positive_index_counter = 0
                    for ix in range(1, max_frames_to_check):
                        # Jump 0,1,-1,2,-2,3,-3...
                        index = int(ix / 2)
                        if (ix % 2) != 0: # Modulo operator.
                            index *= -1
                            if negative_index_counter > 5:
                                continue # Don't check after silent part.
                        else:
                            if positive_index_counter > 5:
                                continue # Don't check after silent part.
                        
                        start = peak_position + jump * index
                        
                        spectrum = spectrum_util.calc_dbfs_spectrum(signal_1s[start:start+wsize])
                        if spectrum is False:
                            continue
                         
                        bin_freq_hz, bin_dbfs = spectrum_util.interpolation_spectral_peak(spectrum)
                        
                        # Peak.
                        if (peak_dbfs is None) or (peak_dbfs < bin_dbfs):
                            peak_dbfs = bin_dbfs
                            peak_freq = bin_freq_hz
                            peak_index = index

                        if (bin_dbfs > peak_dbfs - 15.0) and \
                           (bin_dbfs > -55.0):
                            
                            if (start_index is None) or (start_index > index ):
                                start_freq = bin_freq_hz 
                                start_index = index
                            if (end_index is None) or (end_index < index): 
                                end_freq = bin_freq_hz
                                end_index = index
                            
                            # Max.
                            if (max_freq_index is None) or (max_freq < bin_freq_hz):
                                max_freq_dbfs = bin_dbfs
                                max_freq = bin_freq_hz
                                max_freq_index = index
                            # Min.
                            if (min_freq_index is None) or (min_freq > bin_freq_hz):
                                min_freq_dbfs = bin_dbfs
                                min_freq = bin_freq_hz
                                min_freq_index = index
                                
                            if index < 0:
                                negative_index_counter = 0
                            else:
                                positive_index_counter = 0
                        else:
                            if index < 0:
                                negative_index_counter += 1
                            else:
                                positive_index_counter += 1
                                
                                
                    if start_index is not None:

                        # High pass filter.
                        if peak_freq < 15000: 
                            continue
                        
                        wav_file_index = buffer_number * buffer_size + peak_position + peak_index
                        time_s = wav_file_index / sampling_freq
                        duration = (end_index - start_index + 1) * jump / sampling_freq
                        
                        print('time (sec): ', np.round(time_s, 5),
#                               ' Index in file: ', wav_file_index, 
                              '  peak freq: ', np.round(peak_freq/1000, 3), 
                              '  bw: ', np.round(np.absolute(start_freq - end_freq)/1000, 3), 
                              '  peak dbfs: ', np.round(peak_dbfs, 1), \
                              '  start freq: ', np.round(start_freq/1000, 3), 
                              '  end freq: ', np.round(end_freq/1000, 3), 
                              '  min freq: ', np.round(min_freq/1000, 3), 
                              '  max freq: ', np.round(max_freq/1000, 3), 
                              '  duration (ms): ', duration * 1000  )
                        #
                        found_peak_counter += 1    
                    #
                    
                    
                    # === Create text file for plotting in ZC style. ===
                    
                    # Create a matrix with one row for each 0.125 ms. Size 256*(window_size/2). 
                    size = 256 # From -16 ms to + 16 ms * 8 per ms.
                    jump = int(sampling_freq/1000/8) # Jump 0.25 ms.
                    start_index = int(peak_position - (size * jump / 2))                
                    matrix = spectrum_util.calc_dbfs_matrix(signal_1s[start_index:], matrix_size=size, jump=jump)
                     
                    # Get max dBFS value.
                    row, col = np.unravel_index(matrix.argmax(), matrix.shape)
                    calc_peak_freq_hz, calc_peak_dbfs = spectrum_util.interpolation_spectral_peak(matrix[row])
                     
                    if (calc_peak_freq_hz > 15000) and (calc_peak_dbfs > -50):
                    
                        if calc_peak_dbfs > noise_level_db + 3.0:
#                             print('Peak: ' + str(freq_hz[col]/1000) + ' kHz' + '   dBFS: ' + str(max_dbfs) +  '   Pos: ' + str(row) + ', ' + str(col))
#                             found_peak_counter += 1
                            
                            # Prepare for file.
                            plot_threshold = np.maximum(calc_peak_dbfs - 25.0, -50.0)
                            for spectrum_index, spectrum in enumerate(matrix):
                                freq_hz, dbfs = spectrum_util.interpolation_spectral_peak(spectrum)
                                
                                if dbfs > plot_threshold:
                                    out_row = []
                                    # 'time_s'
                                    file_index = ((buffer_number * buffer_size) + start_index + (spectrum_index * jump))
                                    out_row.append(str(np.round(file_index / sampling_freq, 5)))
                                    # 'peak_khz'
                                    out_row.append(np.round(freq_hz/1000, 3))
                                    # 'dbfs'
                                    out_row.append(np.round(dbfs, 2))
                                    # 'pulse_peak_khz'
                                    out_row.append(np.round(calc_peak_freq_hz/1000, 3))
                                    # 'pulse_peak_dbfs'
                                    out_row.append(np.round(calc_peak_dbfs, 2))
                                    # Print to file.
                                    out_file.write('\t'.join(map(str, out_row)) + '\n')
                    
                    peak_number += 1
                #
                buffer_number += 1
                # Read next buffer.
                frame_buffer_1s = wave_file.readframes(buffer_size)
            
            print('Detected peak counter: ' + str(found_peak_counter))



# === MAIN ===    
if __name__ == "__main__":
    """ """
    print('Test started.\n')
    
    bat = BatAnalysisTool(wave_file_path='sin50khz_TE384.wav')
    bat.analyse_file()
    print('Mdau_TE384\n')
    bat = BatAnalysisTool(wave_file_path='../notebooks/data_in/Mdau_TE384.wav')
    bat.analyse_file()
    print('\nPpip_TE384\n')
    bat = BatAnalysisTool(wave_file_path='../notebooks/data_in/Ppip_TE384.wav')
    bat.analyse_file()
    print('\nMyotis-Plecotus-Eptesicus_TE384\n')
    bat = BatAnalysisTool(wave_file_path='../notebooks/data_in/Myotis-Plecotus-Eptesicus_TE384.wav')
    bat.analyse_file()

    print('\nTest ended.')
