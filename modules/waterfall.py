import numpy as np
import scipy.signal as sp
import scipy.fftpack as fp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

import threading
from typing import Tuple

from ..audio.Processor import Processor
from ..bus import bus
from vuzic.util import Bucketer, log_power_spectrum, NP_FMT


class Waterfall(Processor):

    def __init__(self, n_frames: int, n_buckets: int = 16, show_anim=True):
        super().__init__()

        # Variables that need to be registered
        self.n_frames = 0
        self.fs = 0
        self.n_channels = 0
        self.n_frames = 0

        self.n_frames = n_frames
        self.n_buckets = n_buckets
        self.bucket_repeat = 512 // n_buckets
        self.show_anim = show_anim

        self.ffts = np.zeros(0)
        self.buckets = np.zeros(0)
        self.fft_index = 0

        self.stream_thread = None
        self.done = False

    def register_input(self, n_samples, fs, n_channels):
        self.n_samples = n_samples
        self.fs = fs
        self.n_channels = n_channels

        self.ffts = np.zeros((n_channels, self.n_frames, n_samples // 2), dtype=NP_FMT)
        self.buckets = np.zeros((n_channels, self.n_frames, self.n_buckets), dtype=NP_FMT)

        self.bucketer = Bucketer(n_samples // 2, self.n_buckets, n_channels, 40, fs // 2)
        self.window = sp.hanning(self.n_samples)

    def process(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:
        frame0, frame1 = frames

        wframe0 = self.window * frame0
        wframe1 = self.window * frame1

        self.ffts[:, self.fft_index] = fft0 = log_power_spectrum(wframe0)
        self.buckets[:, self.fft_index] = buckets0 = self.bucketer.bucket(fft0)
       
        self.ffts[:, self.fft_index+1] = fft1 = log_power_spectrum(wframe1)
        self.buckets[:, self.fft_index+1] = buckets1 = self.bucketer.bucket(fft1)
       
        self.fft_index = (self.fft_index + 2) % self.n_frames

        if self.child_processor:
            self.child_processor.process((buckets0, buckets1))

    def mp_animation(self):
        fig, ax = plt.subplots()
        im = plt.imshow(self.ffts, animated=True, cmap='magma', interpolation='gaussian')

        def init():
            # ax.set_xscale(0, self.fs // 2)
            # ax.set_yscale(0, self.n_frames * self.n_samples // 2 // self.fs)
            return im,

        def update(_):
            im.set_data(
                np.flip(
                    np.transpose(
                        np.r_[self.ffts[self.fft_index:], self.ffts[:self.fft_index]]
                    ),
                0)
            )
            return im,

        _ = FuncAnimation(fig, update, init_func=init, blit=True, interval=0)

        try:
            plt.show()
        except AttributeError:
            # hmm, this gets thrown when I click the X button..
            pass

        self.done = True

    def cv_animation(self):
        key = None
        print("press 'q' to quit")
        alpha1 = 1.0
        alpha2 = 1.0
        while not self.done and key != ord('q'):
            alpha1 = .95 * alpha1 + .05 * np.max(self.ffts)
            alpha2 = .95 * alpha2 + .05 * np.max(self.buckets)


            data = np.flip(
                    np.transpose(
                        np.r_[self.ffts[0, self.fft_index:], self.ffts[0, :self.fft_index]]
                    ),
                0)
  
            scaled = cv2.convertScaleAbs(data, alpha=255/alpha1)
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_OCEAN)
            image = cv2.blur(image, (5,3))
            cv2.imshow('Spectrum', image)


            repeat = np.repeat(self.buckets, self.bucket_repeat, axis=2)
            shape = np.r_[repeat[0, self.fft_index:], repeat[0, :self.fft_index]].T
            if self.n_channels == 2:
                shape = np.c_[
                    shape,
                    np.r_[repeat[1, self.fft_index::-1], repeat[1, :self.fft_index:-1]].T,
  
                ]
            data = np.flip(shape, 0)

            scaled = cv2.convertScaleAbs(data, alpha=255/alpha2)
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_PARULA)
            image = cv2.blur(image, (5,5))
            cv2.imshow('Buckets', image)

            key = cv2.waitKey(30)

        print("exiting..")
        self.end_stream()

    def begin(self):

        def _set_done(*args):
            self.done = True

        bus.subscribe("end_stream", _set_done)
        
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


if __name__ == '__main__':
    W = Waterfall(400)
