#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017-2018 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import pathlib
import datetime
import numpy as np
import pandas as pd
#
folium_installed = True
try:
    import folium # For maps.
except:
    folium_installed = False
#
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import sys
# sys.path.append('..')
# import dsp4bats 

# Parts of dsp4bats. 
import wave_file_utils
import time_domain_utils
import frequency_domain_utils 

class BatfilesScanner():
    """ """
    def __init__(self,
                 batfiles_dir='batfiles',
                 scanning_results_dir='batfiles_results',
                 sampling_freq=384000, 
                 debug=False,
                 ):
        """ """
        self.batfiles_dir = batfiles_dir
        self.scanning_results_dir = scanning_results_dir
        self.sampling_freq = sampling_freq        
        self.debug = debug
        #
        self.file_utils = wave_file_utils.WurbFileUtils()
        self.files_df = None 

    def create_list_of_files(self):
        """ """
        # Exists directory for results? Create if not.
        if not pathlib.Path(self.scanning_results_dir).exists():
            pathlib.Path(self.scanning_results_dir).mkdir(parents=True)
        # Read files to dataframe.
        self.file_utils.find_sound_files(dir_path=self.batfiles_dir, 
                                         recursive=False, 
                                         wurb_files_only=False)
        self.files_df = self.file_utils.get_dataframe()
        if self.debug:
            print('Number of wave files found: ', len(self.files_df))
    
    def scan_files(self, 
                # Time domain parameters.
                time_filter_low_limit_hz=15000,
                time_filter_high_limit_hz=None,
                localmax_noise_threshold_factor=1.2, 
                localmax_jump_factor=2000, 
                localmax_frame_length=512, 
                # Frequency domain parameters.
                freq_window_size=128, 
                freq_filter_low_hz=15000, 
                freq_threshold_below_peak_db=40.0, 
                freq_threshold_dbfs =-50.0, 
                freq_jump_factor=4000, 
                freq_max_frames_to_check=200, 
                freq_max_silent_slots=8, 
                ):
        """ """
        # Exists directory for results? Create if not.
        if not pathlib.Path(self.scanning_results_dir).exists():
            pathlib.Path(self.scanning_results_dir).mkdir(parents=True)
        # Read files to dataframe.
        self.file_utils.find_sound_files(dir_path=self.batfiles_dir, 
                                         recursive=False, 
                                         wurb_files_only=False)
        self.files_df = self.file_utils.get_dataframe()
        if self.debug:
            print('Number of wave files found: ', len(self.files_df))
        
        for file_path in self.files_df.abs_file_path:
            #
            if self.debug:
                print('\n', 'Scanning file: ', file_path)
            # Read signal from file. Length 1 sec.
            wave_reader = wave_file_utils.WaveFileReader(file_path)
            # samp_width = wave_reader.samp_width
            sampling_freq = wave_reader.sampling_freq
            if sampling_freq != self.sampling_freq:
                if self.debug:
                    print('\n', 'Error: Wrong sampling frequency in file: ', sampling_freq,
                          '   Expected: ', self.sampling_freq, '\n')
                    continue
            # Create dsp4bats utils.
            signal_util = time_domain_utils.SignalUtil(sampling_freq)
            spectrum_util = frequency_domain_utils.DbfsSpectrumUtil(window_size=freq_window_size,
                                                      window_function='kaiser',
                                                      kaiser_beta=14,
                                                      sampling_freq=sampling_freq)
            # Prepare output file for metrics. Create on demand.
            metrics_file_name = pathlib.Path(file_path).stem + '_Metrics.txt'
            out_header = spectrum_util.chirp_metrics_header()
            out_file = None
            # Read file.
            checked_peaks_counter = 0
            found_peak_counter = 0
            acc_checked_peaks_counter = 0
            acc_found_peak_counter = 0
            buffer_number = 0
            # Read buffer, 1 sec.
            signal_1sec = wave_reader.read_buffer()
            
            # Iterate over buffers.
            while len(signal_1sec) > 0:
                # Get noise level for 1 sec buffer.
                raw_noise_level = signal_util.noise_level(signal_1sec)
                raw_noise_level_db = signal_util.noise_level_in_db(signal_1sec)
                #
                signal_1sec = signal_util.butterworth_filter(signal_1sec, 
                                                             low_freq_hz=time_filter_low_limit_hz,
                                                             high_freq_hz=time_filter_high_limit_hz)
                # Get noise level for 1 sec buffer after filtering.
                noise_level = signal_util.noise_level(signal_1sec)
                noise_level_db = signal_util.noise_level_in_db(signal_1sec)
                if self.debug:
                    print('Noise level (before filter):', np.round(noise_level, 5), 
                          '(', np.round(raw_noise_level, 5), ')', 
                          ' Noise (db):', np.round(noise_level_db, 2), 
                          '(', np.round(raw_noise_level_db, 5), ')'
                          )
                # Find peaks in time domain.
                peaks = signal_util.find_localmax(signal=signal_1sec,
                                                  noise_threshold=noise_level * localmax_noise_threshold_factor, 
                                                  jump=int(sampling_freq/localmax_jump_factor), 
                                                  frame_length=localmax_frame_length) # Window size.
    
                checked_peaks_counter = len(peaks)
                acc_checked_peaks_counter += len(peaks)
                found_peak_counter = 0
                
                for peak_position in peaks:
        
                    # Extract metrics.
                    result = spectrum_util.chirp_metrics(
                                                signal=signal_1sec, 
                                                peak_position=peak_position, 
                                                jump_factor=freq_jump_factor, 
                                                high_pass_filter_freq_hz=freq_filter_low_hz, 
                                                threshold_dbfs = freq_threshold_dbfs, 
                                                threshold_dbfs_below_peak = freq_threshold_below_peak_db, 
                                                max_frames_to_check=freq_max_frames_to_check, 
                                                max_silent_slots=freq_max_silent_slots, 
                                                debug=False)
    
                    if result is False:
                        continue # 
                    else:
                        result_dict = dict(zip(out_header, result))
                        ## out_row = [result_dict.get(x, '') for x in out_header]
                        # Add buffer steps to peak_signal_index, start_signal_index and end_signal_index.
                        out_row = []
                        for key in out_header:
                            if '_signal_index' in key:
                                # Adjust index if more than one buffer was read.
                                index = int(result_dict.get(key, 0))
                                index += buffer_number * signal_util.sampling_freq
                                out_row.append(index)
                            else:
                                out_row.append(result_dict.get(key, ''))
                        # Write to file.
                        if out_file is None:
                            out_file = pathlib.Path(self.scanning_results_dir, metrics_file_name).open('w')
                            out_file.write('\t'.join(map(str, out_header)) + '\n')# Read until end of file.
                        #
                        out_file.write('\t'.join(map(str, out_row)) + '\n')
                        #
                        found_peak_counter += 1
                        acc_found_peak_counter += 1

                if self.debug:
                    print('Buffer: Detected peak counter: ', str(found_peak_counter),
                          '  of ', checked_peaks_counter, ' checked peaks.') 
                #
                buffer_number += 1
                # Read next buffer.
                signal_1sec = wave_reader.read_buffer()
                
            # Done.
            if self.debug:
                print('Summary: Detected peak counter: ', str(acc_found_peak_counter),
                      '  of ', acc_checked_peaks_counter, ' checked peaks.') 
            wave_reader.close()
            if out_file is None:
                print('\n', 'Warning: No detected peaks found. No metrics produced.', '\n') 
            else: 
                out_file.close()
    
    def plot_results(self, 
                     figsize_width=16, 
                     figsize_height=10, 
                     dpi=150,
                     plot_min_time_s=0.0, # None: Automatic. 
                     plot_max_time_s=None, # None: Automatic. 
                     plot_max_freq_khz=200, # None: Automatic. 
                     plot_max_interval_s=0.2, # None: Automatic. 
                     plot_max_duration_ms=20, # None: Automatic. 
                    ):
        """ """
        for file_path in self.files_df.abs_file_path:
            # Update for each figure.
            fig_min_time_s = plot_min_time_s
            fig_max_time_s = plot_max_time_s
            fig_max_freq_khz = plot_max_freq_khz
            fig_max_interval_s = plot_max_interval_s
            fig_max_duration_ms = plot_max_duration_ms

            # Prepare file pathes.
            metrics_file_path = pathlib.Path(file_path).stem + '_Metrics.txt'
            metrics_file_path = pathlib.Path(self.scanning_results_dir, metrics_file_path)
            plot_file_path = pathlib.Path(file_path).stem + '_Plot.png'
            plot_file_path = pathlib.Path(self.scanning_results_dir, plot_file_path)
            # Check if metrics exists.
            if not pathlib.Path(metrics_file_path).exists():
                continue
            #
            if self.debug:
                print('Plot to file: ', plot_file_path)
            # Read dataframe.
            metrics_df = pd.read_csv(metrics_file_path, sep="\t")
            # Plot time instead of index. Add columns to dataframe.
            metrics_df['time_peak_s'] = metrics_df.peak_signal_index / self.sampling_freq
            metrics_df['time_start_s'] = metrics_df.start_signal_index / self.sampling_freq
            metrics_df['time_end_s'] = metrics_df.end_signal_index / self.sampling_freq
            # Calculate intervals between chirps.
            metrics_df['interval_s'] = metrics_df.time_peak_s.diff()
            if fig_max_interval_s is None:
                metrics_df.loc[(metrics_df.interval_s > 0.2)] = np.NaN # Too long interval.
            else:
                metrics_df.loc[(metrics_df.interval_s > fig_max_interval_s)] = np.NaN # Too long interval.
            # Calculate if automatic.
            if fig_min_time_s is None:
                fig_min_time_s = metrics_df.time_peak_s.min() - 0.1
            if fig_max_time_s is None:
                fig_max_time_s = metrics_df.time_peak_s.max() + 0.1
            if fig_max_freq_khz is None:
                fig_max_freq_khz = self.sampling_freq / 1000 / 2 # Nyquist.
            if fig_max_interval_s is None:
                fig_max_interval_s = metrics_df.interval_s.max() + 0.1
            if fig_max_duration_ms is None:
                fig_max_duration_ms = metrics_df.duration_ms.max() + 0.1
            
            if len(metrics_df.time_peak_s) == 0:
                return # Empty.
    
            fig, (ax1, ax2) = plt.subplots(2,1,
                                    figsize=(figsize_width, figsize_height), 
                                    dpi=dpi,
                                    #sharex=True
                                    )
            # ax1 - Peak freq, etc.
            cs1 = ax1.scatter(
                        x=metrics_df.time_peak_s,
                        y=metrics_df.peak_freq_khz,
                        s=150,
                        edgecolors='black', 
                        linewidth=0.5, 
                        c=metrics_df.peak_dbfs,
                        cmap=plt.get_cmap('YlOrRd'),  #'YlOrRd', 'Reds'
                        alpha=0.7)
            ax1.vlines(x=metrics_df.time_peak_s, 
                       ymin=metrics_df.start_freq_khz, 
                       ymax=metrics_df.end_freq_khz,
    #                    ymin=metrics_df.max_freq_khz, 
    #                    ymax=metrics_df.min_freq_khz,
                       linewidth=0.5, 
                       alpha=0.8
                      )
            ax1.hlines(y=metrics_df.peak_freq_khz, 
                       xmin=metrics_df.time_start_s, 
                       xmax=metrics_df.time_end_s,
                       linewidth=0.5, 
                       alpha=0.8
                      )
            cbar = fig.colorbar(cs1, ax=ax1, label='dBFS')
            ax1.set_xlim((fig_min_time_s, fig_max_time_s))
            ax1.set_ylim((0, fig_max_freq_khz))
            ax1.minorticks_on()
            ax1.grid(which='major', linestyle='-', linewidth='0.5', alpha=0.6)
            ax1.grid(which='minor', linestyle='-', linewidth='0.5', alpha=0.3)
            ax1.tick_params(which='both', top='off', left='off', right='off', bottom='off')
            
            # ax2 - Interval.
            ax2.scatter(
                         x=metrics_df.time_peak_s,
                         y=metrics_df.interval_s,
                         s=20,
                         color='blue',
                         label='Interval (s)',
                         alpha=0.5
                        )    
            ax2.hlines(y=metrics_df.interval_s, 
                       xmin=metrics_df.time_peak_s - metrics_df.interval_s, 
                       xmax=metrics_df.time_peak_s,
                       linewidth=0.5, 
                       color='blue',
                       label=None,
                       alpha=0.8
                      )
            ax2.set_xlim((fig_min_time_s, fig_max_time_s))
            ax2.set_ylim((0, fig_max_interval_s))
            ax2.minorticks_on()
            ax2.grid(which='major', linestyle='-', linewidth='0.5', alpha=0.6)
            ax2.grid(which='minor', linestyle='-', linewidth='0.5', alpha=0.3)
            ax2.tick_params(which='both', top='off', left='off', right='off', bottom='off') 
            
            # ax3 - Duration.
            ax3 = ax2.twinx()
            ax3.scatter(
                         x=metrics_df.time_peak_s,
                         y=metrics_df.duration_ms,
                         s=20,
                         marker='s',
                         color='red',
                         label='Duration (ms)',
                         alpha=0.5
                        )
            ax3.set_xlim((fig_min_time_s, fig_max_time_s))
            ax3.set_ylim((0, fig_max_duration_ms))
            
            # Legends
            ax2.legend(loc='upper left')
            ax3.legend(loc='upper right')
            
            # Adjust size on second diagram.
            pos = ax1.get_position()
            pos2 = ax2.get_position()
            ax2.set_position([pos.x0,pos2.y0,pos.width,pos2.height])
            ax3.set_position([pos.x0,pos2.y0,pos.width,pos2.height])
            
            # Titles and labels.
            file_name = pathlib.Path(metrics_file_path).name
            fig.suptitle('Metrics from: ' + file_name, 
                         fontsize=12, fontweight='bold')
            ax1.set_title('Peak and start/end frequencies, amplitude and start/stop time')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Frequency (kHz)')
            ax2.set_xlabel('Time (s)')
            ax2.set_title('Interval (s) and duration (ms)')
            ax2.set_ylabel('Interval (s)')
            ax3.set_ylabel('Duration (ms)')
            
            # Save and plot.
            fig.savefig(str(plot_file_path))
            #
            plt.close()       
    
    def plot_positions_on_map(self, map_file_path=None):
        """ Plots positions on an interactive OpenStreetMap by using the folium library. 
            Note: A map can only be created if lat/long is in file name. """
        if folium_installed == False:
            if self.debug:
                print('\n', 'Warning: Position map not created. Folium is not installed.', '\n')
            return
        # Create name if not specified.
        if map_file_path is None:
            map_file_path=str(pathlib.Path(self.scanning_results_dir, 
                                           'positions_map.html'))        
        # Remove rows with no position.
        files_with_pos_df = pd.DataFrame(self.files_df)
        files_with_pos_df.latlong_str.replace('', np.nan, inplace=True)
        files_with_pos_df.dropna(subset=['latlong_str'], inplace=True) 
        if len(files_with_pos_df) > 0:
            # Group by positions and count files at each position.
            distinct_df = pd.DataFrame(
                    {'file_count' : files_with_pos_df.groupby( ['latlong_str', 
                                                                'latitude_dd', 
                                                                'longitude_dd']).size()
                    }).reset_index()
            # Add a column for description to be shown when hovering over point in map.
            distinct_df['description'] = 'Pos: ' + distinct_df['latlong_str'] + \
                                         ' Count: ' + distinct_df['file_count'].astype(str)
            # Use the mean value as center for the map.
            center_lat = distinct_df.latitude_dd.mean()
            center_long = distinct_df.longitude_dd.mean()
            # Create map object.
            map_osm = folium.Map(location=[center_lat, center_long], zoom_start=8)
            # Loop over positions an create markers.
            for long, lat, desc in zip(distinct_df.longitude_dd.values,
                                     distinct_df.latitude_dd.values,
                                     distinct_df.description.values):
                # The description column is used for popup messages.
                marker = folium.Marker([lat, long], popup=desc).add_to(map_osm)            
            # Write to html file.
            map_osm.save(map_file_path)
            if self.debug:
                print('Position map saved here: ', map_file_path)
        else:
            if self.debug:
                print('\n', 'Warning: Position map not created. Lat/long positions are missing.', '\n')
    


# === MAIN ===    
if __name__ == "__main__":
    """ """
    print('Batfile scanner started. ',  datetime.datetime.now())
    
    scanner = BatfilesScanner(
                batfiles_dir='batfiles',
                scanning_results_dir='batfiles_results',
#                 sampling_freq= 500000, 
                debug=True) # True: Print progress information.
        
    # Get files.
    scanner.create_list_of_files()
    
    # Scan all files and extract metrics.
    print('\n', 'Scanning files. ',  datetime.datetime.now(), '\n')
    scanner.scan_files(
                # Time domain parameters.
                time_filter_low_limit_hz=30000, 
                time_filter_high_limit_hz=None, 
                localmax_noise_threshold_factor=3.0, 
                localmax_jump_factor=1000, 
                localmax_frame_length=512, 
                # Frequency domain parameters.
                freq_window_size=128, 
                freq_filter_low_hz=30000, 
                freq_threshold_below_peak_db=20.0, 
                freq_threshold_dbfs =-50.0, 
                freq_jump_factor=2000, 
                freq_max_frames_to_check=100, 
                freq_max_silent_slots=8, 
                )
    
    # Plot the content of the "*_Metrics.txt" files as Matplotlib plots.
    print('\n', 'Creates plots. ',  datetime.datetime.now(), '\n')
    scanner.plot_results(
                figsize_width=16, 
                figsize_height=10, 
                dpi=80,
                plot_min_time_s=None, # None: Automatic. 
                plot_max_time_s=None, # 1.0, # None: Automatic. 
                plot_max_freq_khz=120, # None: Automatic.  
                plot_max_interval_s=0.2, # None: Automatic.  
                plot_max_duration_ms=20, # None: Automatic.  
                )
    
    # If the file names contains latitude/longitude information an 
    # interactive map (html) will be generated.
    print('\n', 'Creates map. ',  datetime.datetime.now(), '\n')
    scanner.plot_positions_on_map()

    print('\n', 'Batfile scanner ended. ',  datetime.datetime.now(), '\n')

