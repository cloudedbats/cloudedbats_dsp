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
        spectrum_hann256 = utils.frequency_domain_utils.DbfsSpectrumUtil(window_size=256,
                                                                         window_function='hann',
                                                                         sampling_freq=sampling_freq)
        spectrum_hann512 = utils.frequency_domain_utils.DbfsSpectrumUtil(window_size=512,
                                                                         window_function='hann',
                                                                         sampling_freq=sampling_freq)
        spectrum_hann1024 = utils.frequency_domain_utils.DbfsSpectrumUtil(window_size=1024,
                                                                          window_function='hann',
                                                                          sampling_freq=sampling_freq)
        spectrum_blackh512 = utils.frequency_domain_utils.DbfsSpectrumUtil(window_size=512,
                                                                           window_function='blackman-harris',
                                                                           sampling_freq=sampling_freq)
        spectrum_blackh1024 = utils.frequency_domain_utils.DbfsSpectrumUtil(window_size=1024,
                                                                           window_function='blackman-harris',
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
                #
                peak_number = 0
                for peak_position in peaks:
                    wav_file_index = buffer_number * buffer_size + peak_position
#                     print('Time (sec): ' + str(np.round(wav_file_index / sampling_freq, 2)) + '   Index in file: ' + str(file_index) )
                     
                    # === Frequency domain, 1 ms. ===
                    #spectrum_util = spectrum_hann256
                    #spectrum_util = spectrum_hann512
                    #spectrum_util = spectrum_blackh512
                    spectrum_util = spectrum_blackh1024
                    size = spectrum_util.window_size
                    jump = int(sampling_freq/1000/4)
                     
                     
                    spectrum = spectrum_util.calc_dbfs_spectrum(signal_1s[peak_position-int(size/2):peak_position+int(size/2)])
                    if spectrum is False:
                        continue
                     
                    index_max = spectrum.argmax()
                    peak_pos_dbfs = spectrum[index_max]
 
                    first_ix = None
                    last_ix = None
                    max_dbfs = None
                    max_khz = None
                     
                    for index in range(-30, 30):
                        start = peak_position + jump * index
                        spectrum = spectrum_util.calc_dbfs_spectrum(signal_1s[start:start+size])
                        if spectrum is False:
                            print('ERROR. Start: ' + str(start) + '  tot: ' + str(len(signal_1s)) )
                            continue
                         
                        index_max = spectrum.argmax()
                        dbfs = spectrum[index_max]
                        freq_hz, mag, x_adjust = spectrum_util.interpolation_spectral_peak(spectrum)
                         
                        if (dbfs > peak_pos_dbfs -10.0) and (dbfs > -55.0):
                            if first_ix is None: 
                                first_ix = index
                            last_ix = index
                            if max_dbfs is None:
                                max_dbfs = dbfs
                                max_khz = freq_hz
                            if max_dbfs < dbfs:
                                max_dbfs = dbfs
                                max_khz = freq_hz
                                                             
                    if last_ix is not None:
                        print('Freq: ', np.round(max_khz/1000, 3), '   dBFS: ', np.round(max_dbfs, 1), \
                              '   bw (ms): ', (last_ix - first_ix + 1) * jump / sampling_freq * 1000  )
                        found_peak_counter += 1    
                    #
 
                    
                    
                    # === Frequency domain, slow. ===
                    
                    # Create a matrix with one row for each 0.125 ms. Size 256*(window_size/2). 
                    size = 256 # From -16 ms to + 16 ms * 8 per ms.
                    jump = int(sampling_freq/1000/8) # Jump 0.25 ms.
                    start_index = int(peak_position - (size * jump / 2))                
                    matrix = spectrum_util.calc_dbfs_matrix(signal_1s[start_index:], matrix_size=size, jump=jump)
                     
                    # Get max dBFS value.
                    row, col = np.unravel_index(matrix.argmax(), matrix.shape)
                    peak_freq_hz, peak_dbfs, x_adjust = spectrum_util.interpolation_spectral_peak(matrix[row])
                     
                    if (peak_freq_hz > 15000) and (peak_dbfs > -50):
                     
                        if peak_dbfs > noise_level_db + 3.0:
#                             print('Peak: ' + str(freq_hz[col]/1000) + ' kHz' + '   dBFS: ' + str(max_dbfs) +  '   Pos: ' + str(row) + ', ' + str(col))
                            found_peak_counter += 1
                             
                         
                            # Prepare for file.
                            plot_threshold = np.maximum(peak_dbfs - 25.0, -50.0)
                            for spectrum_index, spectrum in enumerate(matrix):
                                freq_hz, dbfs, x_adjust = spectrum_util.interpolation_spectral_peak(spectrum)
                                
                                
                                
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
                                    out_row.append(np.round(peak_freq_hz/1000, 3))
                                    # 'pulse_peak_dbfs'
                                    out_row.append(np.round(peak_dbfs, 2))
                                    # Print to file.
                                    out_file.write('\t'.join(map(str, out_row)) + '\n')
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    peak_number += 1
                #
                buffer_number += 1
                # Read next buffer.
                frame_buffer_1s = wave_file.readframes(buffer_size)
            
            print('Detected peak counter: ' + str(found_peak_counter))
                    
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
    #                     if spectrum is False:
    #                         continue
    #                     index_max = spectrum.argmax()
    #                     if freq_hz_win[index_max] < 15000:
    #                         print('too low')
    #                         continue
    #                     if spectrum[index_max] < -50:
    #                         print('too weak')
    #                         continue
    #                     
    #                     print('OK. Freq: ' + str(freq_hz_win[index_max]) + '   dBFS: ' + str(spectrum[index_max]) + '   amp: ' + str(signal_1s[file_index + index_max]))    
                        
                        
    #                     # Get max dBFS value.
    #                     row, col = np.unravel_index(matrix.argmax(), matrix.shape)
    #                     freq_hz = spectrum_util.get_freq_bins_in_hz()
    #                     max_dbfs = np.round(matrix[row, col], 2)
    #                     
    #                     if (freq_hz[col] > 15000) and (max_dbfs > -50):
    #                     
    #                         if max_dbfs > noise_level_db + 5.0:
    # #                             print('Peak: ' + str(freq_hz[col]/1000) + ' kHz' + '   dBFS: ' + str(max_dbfs) +  '   Pos: ' + str(row) + ', ' + str(col))
    #                             found_peak_counter += 1
    #                             
    #                         
    #                             # Prepare for file.
    #                             plot_threshold = np.maximum(max_dbfs - 15.0, -50.0)
    #                             for spectrum_index, spectrum in enumerate(matrix):
    #                                 index_max = spectrum.argmax()
    #                                 if spectrum[index_max] > plot_threshold:
    #                                     out_row = []
    #                                     # 'time_s'
    #                                     file_index = ((buffer_number * blocksize) + start_index + (spectrum_index * jump))
    #                                     out_row.append(str(np.round(file_index / sampling_freq, 5)))
    #                                     # 'peak_khz'
    #                                     out_row.append(np.round(freq_hz[index_max]/1000, 5))
    #                                     # 'dbfs'
    #                                     out_row.append(np.round(spectrum[index_max], 2))
    #                                     # 'signal_peak_khz'
    #                                     out_row.append(np.round(max_dbfs, 2))
    #                                     # Print to file.
    #                                     out_file.write('\t'.join(map(str, out_row)) + '\n')
                            
      
                        
                        
                        
                        
                        
                        
                        
                        
    #                     # === Frequency domain, slow. ===
    #                     
    #                     
    #     
    #                     
    #                     # Create a matrix with one row for each 0.125 ms. Size 256*(window_size/2). 
    #                     size = 256 # From -16 ms to + 16 ms * 8 per ms.
    #                     jump = int(sampling_freq/1000/8) # Jump 0.25 ms.
    #                     start_index = int(peak_position - (size * jump / 2))                
    #                     matrix = spectrum_util.calc_dbfs_matrix(signal_1s[start_index:], matrix_size=size, jump=jump)
    #                     
    #                     # Get max dBFS value.
    #                     row, col = np.unravel_index(matrix.argmax(), matrix.shape)
    #                     freq_hz = spectrum_util.get_freq_bins_in_hz()
    #                     max_dbfs = np.round(matrix[row, col], 2)
    #                     
    #                     if (freq_hz[col] > 15000) and (max_dbfs > -50):
    #                     
    #                         if max_dbfs > noise_level_db + 5.0:
    # #                             print('Peak: ' + str(freq_hz[col]/1000) + ' kHz' + '   dBFS: ' + str(max_dbfs) +  '   Pos: ' + str(row) + ', ' + str(col))
    #                             found_peak_counter += 1
    #                             
    #                         
    #                             # Prepare for file.
    #                             plot_threshold = np.maximum(max_dbfs - 15.0, -50.0)
    #                             for spectrum_index, spectrum in enumerate(matrix):
    #                                 index_max = spectrum.argmax()
    #                                 if spectrum[index_max] > plot_threshold:
    #                                     out_row = []
    #                                     # 'time_s'
    #                                     file_index = ((buffer_number * blocksize) + start_index + (spectrum_index * jump))
    #                                     out_row.append(str(np.round(file_index / sampling_freq, 5)))
    #                                     # 'peak_khz'
    #                                     out_row.append(np.round(freq_hz[index_max]/1000, 5))
    #                                     # 'dbfs'
    #                                     out_row.append(np.round(spectrum[index_max], 2))
    #                                     # 'signal_peak_khz'
    #                                     out_row.append(np.round(max_dbfs, 2))
    #                                     # Print to file.
    #                                     out_file.write('\t'.join(map(str, out_row)) + '\n')
    #                         
    #                     peak_number += 1
                

# === MAIN ===    
if __name__ == "__main__":
    """ """
    print('Test started.\n')
    
#     bat = BatAnalysisTool(wave_file_path='sin30khz_TE384.wav')
#     bat.analyse_file()
    
    
    print('Mdau_TE384\n')
    bat = BatAnalysisTool(wave_file_path='../notebooks/data_in/Mdau_TE384.wav')
    bat.analyse_file()
    print('\nPpip_TE384\n')
    bat = BatAnalysisTool(wave_file_path='../notebooks/data_in/Ppip_TE384.wav')
    bat.analyse_file()
#     print('\nMyotis-Plecotus-Eptesicus_TE384\n')
#     bat = BatAnalysisTool(wave_file_path='../notebooks/data_in/Myotis-Plecotus-Eptesicus_TE384.wav')
#     bat.analyse_file()
    print('\nTest ended.')
