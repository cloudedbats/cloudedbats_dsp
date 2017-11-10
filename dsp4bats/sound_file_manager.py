#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import pathlib
import re
import dateutil.parser
import pandas as pd

class SoundFileManager(object):
    """ Class for sound file management. """
    
    def __init__(self): 
        """ """
        self._todo = None
        
        self._soundfiles_df = None        
    
    def find_sound_files(self, dir_path='.', recursive=False):
        """ A dataframe is used to store found files. """

        columns = ['a', 'b', 'c', 'd', 'e', 'f', ]
        data = []

        self._soundfiles_df = pd.DataFrame(data, columns=columns)
        self._soundfiles_df.reset_index()
        
        path_list = []
        if recursive:
            path_list = sorted(pathlib.Path(dir_path).glob('**/*.wav'))
        else:
            path_list = sorted(pathlib.Path(dir_path).glob('*.wav'))
            
        for filepath in path_list:
            meta_dict = self.extract_metadata(filepath)
            
            print('')
            print(pathlib.Path(filepath).name)
            #print(meta_dict)
            for key in sorted(meta_dict.keys()):
                print('- ' + key + ': ' + str(meta_dict[key]))
            
            

        
    def extract_metadata(self, filepath):
        """ Used to extract file name parts from sound files created by CloudedBats-WURB.
            Format: <recorder-id>_<time>_<position>_<rec-type>_<comments>.wav
            Example: wurb1_20170611T005215+0200_N57.6548E12.6711_TE384_Mdau-in-tandem.wav
        """
        meta_dict = {}
        path = pathlib.Path(filepath)
        
        # File and dir info.
        meta_dict['file_path'] = str(filepath)
        meta_dict['abs_file_path'] = str(path.absolute().resolve())
        meta_dict['dir_path'] = str(path.parent)
        meta_dict['file_name'] = path.name
        meta_dict['file_stem'] = path.stem
        
        # Extract parts based on format.
        parts = path.stem.split('_')
        
        # Check if the file is a WURB generated/formatted file.
        if path.suffix not in ['.wav']:
            return None
        if len(parts) >= 4:
            rec_type = parts[3]
            if (len(rec_type) >= 4) and \
               (parts[3][0:2] in ['TE', 'FS']):
                pass
            else:
                return meta_dict
        else:
            return meta_dict
        
        # Detector id.
        if len(parts) > 0:
            meta_dict['detector_id'] = parts[0]
            
        # Datetime in ISO format.
        if len(parts) > 1:
            try:
                meta_dict['datetime_iso'] = parts[1]
                meta_dict['datetime'] = dateutil.parser.parse(parts[1])
            except:
                meta_dict['datetime_iso'] = ''
                meta_dict['datetime'] = ''
                
        # Latitude/longitude.
        if len(parts) > 2:
            latlong_str = parts[2].upper()
            meta_dict['latlong_str'] = latlong_str
            # Extract lat-DD and long-DD.
            try:
                ns_start = re.search(r'[NS]', latlong_str).span(0)[0]
                ew_start = re.search(r'[EW]', latlong_str).span(0)[0]
                latitude_dd = float(latlong_str[ns_start+1:ew_start])
                longitude_dd = float(latlong_str[ew_start+1:])
                if latlong_str[ns_start] == 'S':
                    latitude_dd *= -1.0
                if latlong_str[ew_start] == 'W':
                    longitude_dd *= -1.0
                meta_dict['latitude_dd'] = latitude_dd
                meta_dict['longitude_dd'] = longitude_dd
            except:
                pass
            
        # Framerates.
        if len(parts) > 3:
            meta_dict['rec_type'] = parts[3]
            try:
                frame_rate = float(parts[3][2:])
                meta_dict['frame_rate_hz'] = str(round(frame_rate * 1000.0))
                if parts[3][0:2] == 'TE':
                    meta_dict['is_te'] = True # TE, Time Expanded.
                    meta_dict['file_frame_rate_hz'] = str(round(frame_rate * 100.0))
                else:
                    meta_dict['is_te'] = False # FS, Full Scan.
                    meta_dict['file_frame_rate_hz'] = str(round(frame_rate * 1000.0))
            except:
                pass
        
        # Comments. All parts above index 4.
        if len(parts) > 4:
            meta_dict['comments'] = '_'.join(parts[4:])
        
        return meta_dict
        
        