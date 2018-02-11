import threading, time
from typing import Tuple

import cv2
import numpy as np
import scipy.signal as sp
from scipy.fftpack import dct

from vuzic.util import NP_FMT, log_power_spectrum, PhaseLockedLoop
from vuzic.audio.Processor import Processor
from vuzic.bus import bus


class ModulationFrequency(Processor):
    def __init__(self, n_frames: int, n_buckets: int, n_channels: int, fs: int, show_anim: bool=True):
        super().__init__()

        self.n_frames = n_frames
        self.n_buckets = n_buckets
        self.n_channels = n_channels
        self.show_anim = show_anim
        self.fs = 4 * fs / self.n_frames

        self.window = np.tile(sp.blackmanharris(self.n_frames), (self.n_buckets, 1)).T
        #self.window = np.tile(self.window, (2, 1, 1))
        self.in_frames = np.zeros((self.n_frames, self.n_buckets))
        self.out_frames = np.zeros((self.n_frames // 2, self.n_buckets))
        self.index = 0

        filter =  np.r_[
            0.2, 0.3, 0.5, 0.8, np.ones(self.n_frames//4 - 4 - 16, dtype=NP_FMT), np.arange(1.0, 0, -1.0/(self.n_frames//4 + 16))
        ]
        self.out_filter = np.tile(filter, (self.n_buckets, 1)).T

        self.peak_filter = np.array([0.2, 0.7, 1.0, 0.7, 0.2]) #, (self.n_buckets, 1))
        self.peaks = np.zeros(self.n_buckets)
        self.phases = np.zeros(self.n_buckets)
        self.peak_osc = np.zeros(self.n_buckets)
        self.osc_index = 0

        self.phase_ease = .2
        self.peak_ease = .2

        #self.plls = [PhaseLockedLoop()

        self.stream_thread = None
        self.done = False

    def process(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:
        for frame in frames:
            frame = np.average(frame, axis=0)
            self.in_frames[self.index] = frame
            self.index = (self.index + 1) % self.n_frames
            # if self.index == 0:
            #     print("index reset!", time.time())
        
        frames = np.concatenate((self.in_frames[self.index:], self.in_frames[:self.index]), axis=0)
        frames = self.window * frames

        fft = np.fft.rfft(frames, axis=0)[1:self.n_frames//2+1]
        self.out_frames = np.log(1 + np.square(np.abs(fft)) / (self.n_frames // 2))
   
        self.out_frames *= self.out_filter

        peaks = np.zeros((self.n_buckets, self.n_frames//2))
        for i in range(self.n_buckets):
            peaks[i] = np.convolve(self.out_frames[:, i], self.peak_filter, mode='same')
        self.peaks = np.floor(self.peak_ease * np.argmax(peaks, axis=1) + (1-self.peak_ease) * self.peaks).astype(np.int16)
        peak_f = 2 * np.pi * self.peaks / self.fs
        #print(self.peak_f)
        
        for i in range(self.n_buckets):
            self.phases[i] = self.phase_ease * np.angle(fft[self.peaks[i], i]) + (1-self.phase_ease) * self.phases[i]
        #print(self.phases)
        # if self.index == 0:
        #     print("RESET")
        #print(self.peaks[8])
        self.osc_index += 1
        self.peak_osc = np.cos( peak_f * self.osc_index + self.phases )
        #print(self.peak_osc[8])

    def cv_animation(self):
        key = None
        alpha = 1.0
        while not self.done and key != ord('q'):
            #
            # mod freq display
            #

            data = self.out_frames[:128]

            mx = np.max(data)
            alpha = .95 * alpha + .05 * mx
            
            for i in range(self.n_buckets):
                argmax = np.min((self.peaks[i], 127))
                data[argmax, i] = mx
                # if i == 8 and self.peak_osc[8] > .5:
                #     data[:, i] = mx
                data[:, self.peak_osc > .5] = mx

            data = np.repeat(np.flip(data.T, axis=0), 96, axis=0)
            data = np.repeat(data, 24, axis=1)
            scaled = cv2.convertScaleAbs(data, alpha=255/alpha)
            image = cv2.blur(scaled, (8, 8))
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

            cv2.imshow('Modulation Frequency', image)

            #
            # peak tracker display
            #

            data = np.tile(self.peak_osc, (640, 1))
            data = np.repeat(data, 32, axis=1)
            image = cv2.convertScaleAbs(data+1, alpha=128).T

            cv2.imshow('Peak Oscillators', image)

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
            except Exception as e:
                self.end_stream()
                raise

        if self.show_anim:
            t = threading.Thread(target=_begin)
            t.start()
