import threading, time
from typing import Tuple

import cv2
import numpy as np
import scipy.signal as sp
from scipy.fftpack import dct

from vuzic.util import NP_FMT, log_power_spectrum
from vuzic.audio.Processor import Processor
from vuzic.bus import bus


class ModulationFrequency(Processor):
    def __init__(self, n_frames: int, n_buckets: int, n_channels: int, show_anim: bool=True):
        super().__init__()

        self.n_frames = n_frames
        self.n_buckets = n_buckets
        self.n_channels = n_channels
        self.show_anim = show_anim

        self.shift = 1

        self.window = np.tile(sp.blackmanharris(self.n_frames), (self.n_buckets, 1)).T
        self.window = np.tile(self.window, (2, 1, 1))
        self.in_frames = np.zeros((n_channels, self.n_frames, self.n_buckets))
        self.out_frames = np.zeros((n_channels, self.n_frames // 2 - self.shift + 1, self.n_buckets))
        self.index = 0

        self.stream_thread = None
        self.done = False

    def process(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:
        for frame in frames:
            self.in_frames[:, self.index] = frame
            self.index = (self.index + 1) % self.n_frames
            # if self.index == 0:
            #     print("index reset!", time.time())
        
        frames = np.concatenate((self.in_frames[:, self.index:], self.in_frames[:, :self.index]), axis=1)
        frames = self.window * frames

        # for chan in range(self.n_channels):
        chan = 0
        self.out_frames[chan] = np.log(1 + np.square(np.abs(
            np.fft.rfft(frames[chan], axis=0)[self.shift:]
        )) / (self.n_frames // 2))
        #self.out_frames[:] = log_power_spectrum(frames)

    def cv_animation(self):
        key = None
        alpha = 1.0
        while not self.done and key != ord('q'):
            data = self.out_frames[0, :64]

            mx = np.max(data)
            alpha = .95 * alpha + .05 * mx
            
            # mean = np.mean(data)
            # std = np.std(data)
            # data[data < (mean + std)] = 0

            data = np.repeat(np.flip(data.T, axis=0), 32, axis=0)
            data = np.repeat(data, 16, axis=1)
            scaled = cv2.convertScaleAbs(data, alpha=255/alpha)
            image = cv2.blur(scaled, (8, 8))
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

            cv2.imshow('Modulation Frequency', image)

            key = cv2.waitKey(30)

        self.end_stream()

    def register_input(self, **kwargs):
        pass

    def begin(self):
        def _set_done(*args):
            self.done = True

        bus.subscribe('end_stream', _set_done)

        if self.child_processor:
            self.child_processor.begin()
        
        def _begin():
            try:
                self.cv_animation()
            except:
                self.end_stream()
                raise

        if self.show_anim:
            t = threading.Thread(target=_begin)
            t.start()
