import threading
from typing import Tuple

import cv2
import numpy as np
import scipy.signal as sp
from scipy.fftpack import dct

from vuzic.util import NP_FMT, log_power_spectrum, log_power
from vuzic.audio.Processor import Processor
from vuzic.bus import bus


class Cepstrum(Processor):
    def __init__(self, n_frames: int, n_samples: int, fs: int, n_channels: int, show_anim=True):
        super().__init__()

        self.n_frames = n_frames
        self.n_samples = n_samples
        self.fs = fs
        self.n_channels = n_channels
        self.show_anim = True

        self.window = sp.hamming(self.n_samples)
        self.frames = np.zeros((n_channels, self.n_frames, self.n_samples // 2))
        self.index = 0

        self.stream_thread = None
        self.done = False

    def process(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:
        for frame in frames:
            wframe = self.window * frame
            spec = log_power_spectrum(wframe)
            cept = dct(spec, axis=1)
            cept = log_power(np.abs(cept))
            # if self.index == 0:
            #     print(cept)
            self.frames[:, self.index] = cept

            self.index = (self.index + 1) % self.n_frames

            if self.child_processor:
                self.child_processor.process(cept)
    
    def cv_animation(self):
        key = None
        alpha = 1.0
        while not self.done and key != ord('q'):
            alpha = .95 * alpha + .05 * np.max(self.frames)
            
            data = np.flip(
                    np.transpose(
                        np.r_[self.frames[0, self.index:, :64], self.frames[0, :self.index, :64]]
                    ),
                0)
            data = np.repeat(data, 8, axis=0)

            scaled = cv2.convertScaleAbs(data, alpha=255/alpha)
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
            image = cv2.blur(image, (5,3))
            cv2.imshow('Cepstrum', image)

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
