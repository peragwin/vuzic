import numpy as np
import scipy.signal as sp
import scipy.fftpack as fp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

import threading
from typing import Tuple

from processor import Processor
from util import Bucketer, log_power_spectrum, NP_FMT


class Waterfall(Processor):

    def __init__(self, n_frames: int, n_samples: int, fs: int, n_buckets: int = 16, *args, **kwargs):
        super().__init__(n_samples=n_samples, fs=fs, *args, **kwargs)

        self.n_frames = n_frames
        self.n_buckets = n_buckets
        self.bucket_repeat = 512 // n_buckets
        self.ffts = np.zeros((n_frames, self.n_samples // 2), dtype=NP_FMT)
        self.buckets = np.zeros((n_frames, 512), dtype=NP_FMT)
        self.fft_index = 0

        self.stream_thread = None

        self.bucketer = Bucketer(n_samples // 2, n_buckets, 40, fs // 2)

        self.done = False

    def process_frames(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:
        frame0, frame1 = frames

        window = sp.hanning(self.n_samples)
        wframe0 = window * frame0
        wframe1 = window * frame1

        self.ffts[self.fft_index] = fft = log_power_spectrum(wframe0)
        self.buckets[self.fft_index] = np.repeat(
            self.bucketer.bucket(fft), self.bucket_repeat)
       
        self.ffts[self.fft_index+1] = fft = log_power_spectrum(wframe1)
        self.buckets[self.fft_index+1] = np.repeat(
            self.bucketer.bucket(fft), self.bucket_repeat)
       
        self.fft_index = (self.fft_index + 2) % self.n_frames

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
        alpha = 1.0
        beta = 1.0
        while not self.done and key != ord('q'):
            alpha = .95 * alpha + .05 * np.max(self.ffts)
            data = np.flip(
                    np.transpose(
                        np.r_[self.ffts[self.fft_index:], self.ffts[:self.fft_index]]
                    ),
                0)
            # if mx == 0:
            scaled = cv2.convertScaleAbs(data, alpha=255/alpha)
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_OCEAN)
            image = cv2.blur(image, (5,3))
            cv2.imshow('Spectrum', image)

            #print(image[0][0])

            beta = .95 * beta + .05 * np.max(self.buckets)
            data = np.flip(
                    np.transpose(
                        np.r_[self.buckets[self.fft_index:], self.buckets[:self.fft_index]]
                    ),
                0)
            # if mx == 0:
            scaled = cv2.convertScaleAbs(data, alpha=255/beta)
            image = cv2.applyColorMap(scaled, cv2.COLORMAP_OCEAN)
            image = cv2.blur(image, (5,15))
            cv2.imshow('Buckets', image)


            key = cv2.waitKey(30)

        print("exiting..")
        self.end_stream()

    def begin(self):
        t = threading.Thread(target=self.cv_animation)
        t.start()

