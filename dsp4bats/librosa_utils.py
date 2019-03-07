#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Project: http://cloudedbats.org
#
# Source code from Librosa: https://librosa.github.io
# Reason: "It is too complicated to use pyinstaller for Windows and MacOS
#          with the whole librosa lib and all dependencies." 

 
# Copyright (c) 2013--2017, librosa development team.
# 
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


import numpy as np
from numpy.lib.stride_tricks import as_strided


def librosa_frame(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames.

    This implementation uses low-level stride manipulation to avoid
    redundant copies of the time series data.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Time series to frame. Must be one-dimensional and contiguous
        in memory.

    frame_length : int > 0 [scalar]
        Length of the frame in samples

    hop_length : int > 0 [scalar]
        Number of samples to hop between frames

    Returns
    -------
    y_frames : np.ndarray [shape=(frame_length, N_FRAMES)]
        An array of frames sampled from `y`:
        `y_frames[i, j] == y[j * hop_length + i]`

    Raises
    ------
    ParameterError
        If `y` is not contiguous in memory, not an `np.ndarray`, or
        not one-dimensional.  See `np.ascontiguous()` for details.

        If `hop_length < 1`, frames cannot advance.

        If `len(y) < frame_length`.

    Examples
    --------
    Extract 2048-sample frames from `y` with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.util.frame(y, frame_length=2048, hop_length=64)
    array([[ -9.216e-06,   7.710e-06, ...,  -2.117e-06,  -4.362e-07],
           [  2.518e-06,  -6.294e-06, ...,  -1.775e-05,  -6.365e-06],
           ...,
           [ -7.429e-04,   5.173e-03, ...,   1.105e-05,  -5.074e-06],
           [  2.169e-03,   4.867e-03, ...,   3.666e-06,  -5.571e-06]], dtype=float32)

    '''

    if not isinstance(y, np.ndarray):
        raise Exception('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))

    if y.ndim != 1:
        raise Exception('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))

    if len(y) < frame_length:
        raise Exception('Buffer is too short (n={:d})'
                             ' for frame_length={:d}'.format(len(y), frame_length))

    if hop_length < 1:
        raise Exception('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise Exception('Input buffer must be contiguous.')

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


def librosa_rms(y=None, S=None, frame_length=2048, hop_length=512,
        center=True, pad_mode='reflect'):
    '''Compute root-mean-square (RMS) value for each frame, either from the
    audio samples `y` or from a spectrogram `S`.

    Computing the RMS value from audio samples is faster as it doesn't require
    a STFT calculation. However, using a spectrogram will give a more accurate
    representation of energy over time because its frames can be windowed,
    thus prefer using `S` if it's already available.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        (optional) audio time series. Required if `S` is not input.

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude. Required if `y` is not input.

    frame_length : int > 0 [scalar]
        length of analysis frame (in samples) for energy calculation

    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.core.stft` for details.

    center : bool
        If `True` and operating on time-domain input (`y`), pad the signal
        by `frame_length//2` on either side.

        If operating on spectrogram input, this has no effect.

    pad_mode : str
        Padding mode for centered analysis.  See `np.pad` for valid
        values.

    Returns
    -------
    rms : np.ndarray [shape=(1, t)]
        RMS value for each frame


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.rms(y=y)
    array([[ 0.   ,  0.056, ...,  0.   ,  0.   ]], dtype=float32)

    Or from spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> rms = librosa.feature.rms(S=S)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> plt.semilogy(rms.T, label='RMS Energy')
    >>> plt.xticks([])
    >>> plt.xlim([0, rms.shape[-1]])
    >>> plt.legend(loc='best')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()

    Use a STFT window of constant ones and no frame centering to get consistent
    results with the RMS computed from the audio samples `y`

    >>> S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
    >>> librosa.feature.rms(S=S)

    '''
    if y is not None and S is not None:
        raise ValueError('Either `y` or `S` should be input.')
    if y is not None:
### Always mono...
###        y = to_mono(y)
        if center:
            y = np.pad(y, int(frame_length // 2), mode=pad_mode)

###        x = util.frame(y,
        x = librosa_frame(y,
                       frame_length=frame_length,
                       hop_length=hop_length)
### Only used for time domain.
###    elif S is not None:
###        x, _ = _spectrogram(y=y, S=S,
###                            n_fft=frame_length,
###                            hop_length=hop_length)
    else:
        raise ValueError('Either `y` or `S` must be input.')
    return np.sqrt(np.mean(np.abs(x)**2, axis=0, keepdims=True))


def librosa_localmax(x, axis=0):
    """Find local maxima in an array `x`.

    An element `x[i]` is considered a local maximum if the following
    conditions are met:

    - `x[i] > x[i-1]`
    - `x[i] >= x[i+1]`

    Note that the first condition is strict, and that the first element
    `x[0]` will never be considered as a local maximum.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> librosa.util.localmax(x)
    array([False, False, False,  True, False,  True, False,  True], dtype=bool)

    >>> # Two-dimensional example
    >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
    >>> librosa.util.localmax(x, axis=0)
    array([[False, False, False],
           [ True, False, False],
           [False,  True,  True]], dtype=bool)
    >>> librosa.util.localmax(x, axis=1)
    array([[False, False,  True],
           [False, False,  True],
           [False, False,  True]], dtype=bool)

    Parameters
    ----------
    x     : np.ndarray [shape=(d1,d2,...)]
      input vector or array

    axis : int
      axis along which to compute local maximality

    Returns
    -------
    m     : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local maximality along `axis`

    """

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    x_pad = np.pad(x, paddings, mode='edge')

    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)

    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x > x_pad[tuple(inds1)]) & (x >= x_pad[tuple(inds2)])


def librosa_frames_to_samples(frames, hop_length=512, n_fft=None):
    """Converts frame indices to audio sample indices.

    Parameters
    ----------
    frames     : number or np.ndarray [shape=(n,)]
        frame index or vector of frame indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `n_fft / 2`
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : number or np.ndarray
        time (in samples) of each given frame number:
        `times[i] = frames[i] * hop_length`

    See Also
    --------
    frames_to_time : convert frame indices to time values
    samples_to_frames : convert sample indices to frame indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> tempo, beats = librosa.beat.beat_track(y, sr=sr)
    >>> beat_samples = librosa.frames_to_samples(beats)
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)

