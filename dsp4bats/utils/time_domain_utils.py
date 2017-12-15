#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
# Copyright (c) 2017 Arnold Andreasson 
# License: MIT License (see LICENSE.txt or http://opensource.org/licenses/mit).

import numpy as np
import scipy.signal
import librosa

class SignalUtil():
    """ """
    def __init__(self, 
                 sampling_frequency=384000,
                 ):
        """ """
        self.sampling_frequency = sampling_frequency
        self.array_in_sec = None

    def get_array_in_sec(self, signal):
        """ Used as x array in sec. """
        return np.arange(0, len(signal)) / self.sampling_frequency

    def noise_level(self, signal):
        """ """
        return np.sqrt(np.mean(np.square(signal)))

    def noise_level_in_db(self, signal):
        """ """
        return 20 * np.log10(self.noise_level(signal) / 1.0)

    def calc_localmax(self, signal,
                           noise_threshold=0.0, 
                           hop_length=384, 
                           frame_length=1024):
        """ """
        y = signal.copy()
        if noise_threshold > 0.0:
            y[(np.abs(y) < noise_threshold)] = 0.0
        rmse = librosa.feature.rmse(y=y, hop_length=hop_length, frame_length=frame_length, center=True)
        locmax = librosa.util.localmax(rmse.T)
        maxindexlist = [index for index, a in enumerate(locmax) if a==True]

        # Original index list is related to hop_length. Convert.
        index_list = librosa.frames_to_samples(maxindexlist, hop_length=hop_length)
        
        return index_list
    