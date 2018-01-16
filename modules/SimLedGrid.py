import numpy as np
import scipy.signal as sp
import cv2

import threading
from typing import Tuple

from ..audio.Processor import Processor
from vuzic.util import TriangleBucketer, power_spectrum, log_power, NP_FMT
from ..bus import bus

from .DemoDisplay import DemoDisplay

class SimLedGrid(Processor):

    def __init__(self,
                 mode: str,
                 # stereo or mono
                 n_channels: int,
                 # number of rows of LEDs
                 n_buckets: int,
                 # number of LEDs per row
                 n_frames: int,
                 # number of samples in the fft's that are input to process
                 n_samples: int,
                 fs: int,
                 *args, **kwargs):

        super().__init__()

        self.n_channels = n_channels
        self.n_buckets = n_buckets
        self.n_frames = n_frames

        #self.bucketer = Bucketer(n_samples, n_buckets, n_channels, 40, fs // 2)
        self.bucketer = TriangleBucketer(n_samples, n_buckets, n_channels, 40, fs // 2)

        # n_buckets x n_frames grid of 24bit RGB color values
        self.led_grid = np.zeros((n_buckets, n_frames), dtype=np.int32)

        self.done = False

        self.mode_funcs = {
            'raw': self.raw_process_mode,
            'by_row': self.by_row_process_mode,
        }
        self.mode_animations = {
            'raw': self.raw_mode_animation,
            'by_row': self.by_row_mode_animation,
        }
        assert mode in self.mode_funcs
        self.mode = mode

        if mode == 'by_row':
            self.by_row_init()

    def register_input(self, **kwargs) -> None:
        pass

    def process(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:

        for f in frames:
            self.process_frame(f)

    def process_frame(self, frame: np.ndarray) -> None:
        self.mode_funcs[self.mode](frame)

    def raw_process_mode(self, frame: np.ndarray) -> None:
        fft = power_spectrum(frame)
        buckets = self.bucketer.bucket(fft)
        center = np.reshape(np.r_[buckets, buckets], (2, len(buckets))).T
        #print(center)

        hlf = self.n_frames // 2
        #print(self.led_grid[:, :hlf-1].shape)
        #print(center.shape)
        self.led_grid = np.concatenate(
            (self.led_grid[:, 1:hlf], center, self.led_grid[:, hlf:2*hlf-1]),
            axis=1,
        )

    def by_row_init(self):
        self.sub_processor = DemoDisplay(n_buckets=self.n_buckets, n_frames=self.n_frames)

    def by_row_process_mode(self, frame: np.ndarray) -> None:
        fft = power_spectrum(frame)
        buckets = self.bucketer.bucket(fft)[:,::-1]
        buckets = log_power(buckets)
        self.led_grid = self.sub_processor.process(buckets)

    def raw_mode_animation(self):
        print("press 'q' to quit")
        key = None
        alpha = 1.
        while not self.done and key != ord('q'):
            alpha = .975 * alpha + .025 * np.max(self.led_grid)

            scaled = cv2.convertScaleAbs(self.led_grid, alpha=255/alpha)
            colored = cv2.applyColorMap(scaled, cv2.COLORMAP_RAINBOW)

            # repeat 32 x 8 to make display 
            grid = np.repeat(colored, 32, axis=0)
            grid = np.repeat(grid, 8, axis=1)

            grid = cv2.blur(grid, (9,9))
            cv2.imshow('LED Grid', grid)

            key = cv2.waitKey(30)

        self.end_stream()

    def by_row_mode_animation(self):
        key = None
        while not self.done and key != ord('q'):
            #print(self.led_grid.shape)
            grid = np.moveaxis(self.led_grid, 0, 1)
            # repeat 32 x 8 to make display 
            grid = np.repeat(grid, 32, axis=0)
            grid = np.repeat(grid, 8, axis=1)

            grid = cv2.blur(grid, (9,9)).astype(np.uint8)
            cv2.imshow('LED Grid', grid)

            key = cv2.waitKey(30)           

        self.end_stream()

    def begin(self):
        def _set_done(*args):
            self.done = True
        bus.subscribe("end_stream", _set_done)

        _func = self.mode_animations[self.mode]
        t = threading.Thread(target=_func)
        t.start()


if __name__ == '__main__':
    # Test sim grid
    G = SimLedGrid('by_row', 16, 60, n_samples=1024)
