# #!/usr/bin/python3
# # -*- coding:utf-8 -*-
# 
# import sys
# #sys.path.append('.')
# 
# import matplotlib
# matplotlib.use('Agg')
# 
# import datetime
# import dsp4bats
# 
# print('Batfile scanner started. ',  datetime.datetime.now())
# 
# scanner = dsp4bats.BatfilesScanner(
#             batfiles_dir='data/batfiles',
#             scanning_results_dir='data/batfiles_results',
#             sampling_freq= 500000, #384000, # True sampling frequency (before TE). 
#             debug=True) # True: Print progress information.
#     
# # Get files.
# scanner.create_list_of_files()
# 
# # Scan all files and extract metrics.
# print('\n', 'Scanning files. ',  datetime.datetime.now(), '\n')
# scanner.scan_files(
#             # Time domain parameters.
#             time_filter_low_limit_hz=30000, # Lower limit for highpass or bandpass filter.
#             time_filter_high_limit_hz=None,  # Upper limit for lowpass or bandpass filter.
#             localmax_noise_threshold_factor=2.0, # Multiplies the detected noise level by this factor. 
#             localmax_jump_factor=1000, # 1000 gives 1 ms jumps, 2000 gives 0.5 ms jumps.
#             localmax_frame_length=1024, # Frame size to smooth the signal.
#             # Frequency domain parameters.
#             freq_window_size=128, # 
#             freq_filter_low_hz=30000, # Don't use peaks below this limit. 
#             freq_threshold_below_peak_db=20.0, # Threshold calculated in relation to chirp peak level.
#             freq_threshold_dbfs =-50.0, # Absolute threshold in dbms. 
#             freq_jump_factor=2000, # 1000 gives 1 ms jumps, 2000 gives 0.5 ms jumps.  
#             freq_max_frames_to_check=100, # Max number of jump steps to calculate metrics.  
#             freq_max_silent_slots=8, # Number of jump steps to detect start/end of chirp.
#             )
# 
# # Plot the content of the "*_Metrics.txt" files as Matplotlib plots.
# print('\n', 'Creates plots. ',  datetime.datetime.now(), '\n')
# scanner.plot_results(
#             # Figure settings.
#             figsize_width=16, 
#             figsize_height=10, 
#             dpi=80,
#             # Plot settings.
#             plot_min_time_s=0, # None: Automatic. 
#             plot_max_time_s=1.0, # None: Automatic. 
#             plot_max_freq_khz=200, # None: Automatic.  
#             plot_max_interval_s=0.2, # None: Automatic.  
#             plot_max_duration_ms=20, # None: Automatic.  
#             )
# 
# # If the file names contains latitude/longitude information an 
# # interactive map (html) will be generated.
# print('\n', 'Creates map. ',  datetime.datetime.now(), '\n')
# scanner.plot_positions_on_map()
# 
# print('\n', 'Batfile scanner ended. ',  datetime.datetime.now(), '\n')
#     
#     
