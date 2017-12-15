#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import numpy as np
import librosa
import soundfile
import utils.time_domain_utils
import utils.frequency_domain_utils

class BatAnalysisTool():
    """ """
    def __init__(self, file_path):
        """ """
        self.file_path = file_path

    def analyse_file(self):
        """ """
        # Get file info.
        sampling_frequency = soundfile.info(self.file_path).samplerate
        # Adjust for TE.
        if sampling_frequency < 192000:
            sampling_frequency *= 10

        # Prepare utilities.
        signal_util = utils.time_domain_utils.SignalUtil(sampling_frequency)
        spectrum_util = utils.frequency_domain_utils.DbfsSpectrumUtil(window_size=1024,
                                                                      window_function='blackman-harris',
                                                                      sampling_frequency=sampling_frequency)
        # Open file and read blocks. Read 1 sec blocks (size = sampling rate) 
        found_peak_counter = 0
        block_number = 0
        block_generator = soundfile.blocks(self.file_path, blocksize=sampling_frequency)
        for signal_block in block_generator:
             
            
            # Get noise level for 1 sec.
            noise_level = signal_util.noise_level(signal_block)
            noise_level_db = signal_util.noise_level_in_db(signal_block)
            print('Noise level: ' + str(np.round(noise_level, 2)) + '   Noise (db): ' + str(np.round(noise_level_db, 2)))
            
            # Find peaks in time domain.
            peaks = signal_util.calc_localmax(signal=signal_block,
                                              noise_threshold=noise_level*4.0, # Threshold.
                                              hop_length=int(sampling_frequency/1000), # Jump 1 ms.
                                              frame_length=1024) # Window size.
            #
            peak_number = 0
            for peak_position in peaks:
                
                file_index = block_number * sampling_frequency + peak_position
                print('Time (sec): ' + str(np.round(file_index / sampling_frequency, 2)) + '   Index in file: ' + str(file_index) )
                
                # Create a matrix with one row for each 0.25 ms. Size 128*128. 
                jump = int(sampling_frequency/1000/4) # Jump 0.25 ms.
                size = 128 # From -16 ms to + 16 ms.
                start_index = int(peak_position - (size * jump / 2))                
                matrix = spectrum_util.calc_dbfs_matrix(signal_block[start_index:], matrix_size=size, jump=jump)
                
                # Get max dBFS value.
                row, col = np.unravel_index(matrix.argmax(), matrix.shape)
                freq_hz = spectrum_util.get_freq_bins_in_hz()
                max_dbfs = np.round(matrix[row, col], 2)
                
                if max_dbfs > noise_level_db + 5.0:
                    print('Peak: ' + str(freq_hz[col]/1000) + ' kHz' + '   dBFS: ' + str(max_dbfs) +  '   Pos: ' + str(row) + ', ' + str(col))
                    found_peak_counter += 1
                    
                peak_number += 1
            
            block_number += 1
                    
        print('Detected peak counter: ' + str(found_peak_counter))
                

# === MAIN ===    
if __name__ == "__main__":
    """ """
    print('Test started.\n')
    print('Mdau_TE384\n')
    bat = BatAnalysisTool(file_path='../notebooks/data_in/Mdau_TE384.wav')
    bat.analyse_file()
    print('\nPpip_TE384\n')
    bat = BatAnalysisTool(file_path='../notebooks/data_in/Ppip_TE384.wav')
    bat.analyse_file()
    print('\nMyotis-Plecotus-Eptesicus_TE384\n')
    bat = BatAnalysisTool(file_path='../notebooks/data_in/Myotis-Plecotus-Eptesicus_TE384.wav')
    bat.analyse_file()
    print('\nTest ended.')
