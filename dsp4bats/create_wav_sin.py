#!/usr/bin/python3
# -*- coding:utf-8 -*-

import wave
import array
import numpy as np

#p = pyaudio.PyAudio()

volume = 0.5     # range [0.0, 1.0]
fs = 38400       # sampling rate, Hz, must be integer
duration = 1.0   # in seconds, may be float
f = 5000        # sine frequency, Hz, may be float

out = wave.open('sin50khz_TE384.wav', mode='wb')
out.setparams((1, 2, fs, 0, 'NONE', 'not compressed'))

# generate samples, note conversion to float32 array
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

data = array.array('h')
for i in range(len(samples)):
    data.append(int(32767*volume*samples[i]))

out.writeframes(data.tostring())

out.close()
