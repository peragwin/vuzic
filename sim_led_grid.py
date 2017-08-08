import numpy as np
import cv2

import threading
from collections import namedtuple
from typing import Tuple

from processor import Processor
from util import Bucketer, log_power_spectrum, NP_FMT


class SimLedGrid(Processor):

    def __init__(self,
                 mode: str,
                 n_channels: int,
                 n_per_channel: int,
                 n_samples: int,
                 fs: int,
                 *args, **kwargs):

        super().__init__(n_samples=n_samples, fs=fs, *args, **kwargs)

        self.n_channels = n_channels
        self.n_per_channel = n_per_channel

        self.bucketer = Bucketer(n_samples // 2, n_channels, 40, fs // 2)

        # n_channels x n_per_channel grid of 24bit RGB color values
        self.led_grid = np.zeros((n_channels, n_per_channel), dtype=np.int32)

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

    def process_frames(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:
        for f in frames:
            self.process_frame(f)

    def process_frame(self, frame: np.ndarray) -> None:
        self.mode_funcs[self.mode](frame)

    def raw_process_mode(self, frame: np.ndarray) -> None:
        fft = log_power_spectrum(frame)
        buckets = self.bucketer.bucket(fft)[::-1]
        center = np.reshape(np.r_[buckets, buckets], (2, len(buckets))).T
        #print(center)

        hlf = self.n_per_channel // 2
        #print(self.led_grid[:, :hlf-1].shape)
        #print(center.shape)
        self.led_grid = np.concatenate(
            (self.led_grid[:, 1:hlf], center, self.led_grid[:, hlf:2*hlf-1]),
            axis=1,
        )

    def by_row_init(self):
        self.sub_processor = DemoDisplay(num_channels=self.n_channels, num_per_channel=self.n_per_channel)

    def by_row_process_mode(self, frame: np.ndarray) -> None:
        fft = log_power_spectrum(frame)
        buckets = self.bucketer.bucket(fft)[::-1]
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
        func = self.mode_animations[self.mode]
        t = threading.Thread(target=func)
        t.start()


class DemoDisplay:
    """ for lack of a better name... 
        TODO: explain what this does
    """

    Config = namedtuple("config", [
        'global_brightness',
        'direction',
        'amp_gain',
        'diff_gain',
        'amp_offset',
        'brightness',
        #'time_period',
        'space_period',
        'channel_sync',
    ])

    class Drivers:
        def __init__(self, amp: np.ndarray, ph: np.ndarray):
            self.amplitude = amp
            self.phase = ph
            # TODO: have both phase immediate and phase accumulated fields

    FilterValues = namedtuple("filterValues", [
        'gain',
        'diff',
    ])

    def __init__(self,
                 num_channels:    int = 16,
                 num_per_channel: int = 60,
                 direction:    str   = 'center',
                 amp_gain:     int   = 50,
                 diff_gain:    float = 3.5e-4, # I think this is approx the value that was being used
                 amp_offset:   int   = 400,
                 brightness:   int   = 800,
                 #time_period:  int   = 300,
                 space_period: int   = 100,
                 channel_sync: float = 4.5e-4, #1.8e-3, # this one too
                ):

        self.num_channels = num_channels
        self.num_per_channel = num_per_channel

        self.config = self.Config(
            127,
            direction,
            amp_gain,
            diff_gain,
            amp_offset,
            brightness,
            #time_period,
            space_period,
            channel_sync,
        )

        self.drivers = self.Drivers(
            np.zeros(num_channels, dtype=np.float64),
            np.zeros(num_channels, dtype=np.float64),
        )

        self.iter_count = 0

        # values are by order then channel because it makes it look cleaner in applyFilter
        self.filter_values = self.FilterValues(
            np.zeros((2, num_channels), dtype=np.float64),
            np.zeros((2, num_channels), dtype=np.float64),
        )

        # Using a second order IIR with negative feedback for each channel,
        # implemented in two stages for simplicity
        self.filter_params = self.FilterValues(
            np.zeros((num_channels, 2, 2), dtype=np.float64),
            np.zeros((num_channels, 2, 2), dtype=np.float64),
        )

        for i in range(num_channels):
            self.filter_params.gain[i] = np.array([
                [ 1.   , 0.5  ], # first elem is the gain on that channel
                [- .005,  .995], # this is the lowpass for the feedback mechanism which controls sensitivity
            ])
            self.filter_params.diff[i] = np.array([
                [ 1.  , 0.5 ],
                [- .04,  .96],
            ])

    def applyFilters(self, frame: np.ndarray) -> None:
        diff_input = np.zeros((2, self.num_channels), dtype=np.float64)
        self.applyFilter(frame, self.filter_values.gain, self.filter_params.gain, diff_input)
        self.applyFilter(diff_input[0], self.filter_values.diff, self.filter_params.diff)

    def applyFilter(self,
                    inpt: np.ndarray,
                    output: np.ndarray,
                    params: np.ndarray,
                    diff_input: np.ndarray = None):

        # apply in two stages, feeding the output of the first filter to the second
        for order in range(2):
            #print(params[:,order])
            a, b = params[:,order].T
            ae = a * inpt + b * output[order]
            
            if diff_input is not None:
                diff_input[order] = ae - output[order] # diff since previous output

            inpt = ae
            output[order] = ae

        # finally, apply the output of the second filter as feedback 
        output[0] += output[1]

    def applyChannelEffects(self):
        dg = self.config.diff_gain
        ag = self.config.amp_gain
        ao = self.config.amp_offset

        #print(self.filter_values.gain[0,:], self.filter_values.diff[0,:])

        self.drivers.phase -= dg * np.fabs(self.filter_values.diff[0,:])
        self.drivers.phase += .001 # add a constant opposing force makes an interesting heatmap of activity
        self.drivers.phase %= 2 * np.pi # numpy ftw
        self.drivers.amplitude = ao + ag * self.filter_values.gain[0,:]

    def applyChannelSync(self):
        avg_phase = np.average(self.drivers.phase)
        diff = avg_phase - self.drivers.phase
        diff *= diff * np.sign(diff)
        self.drivers.phase += self.config.channel_sync * diff

    def getPixelBlock(self, col_num: int) -> np.ndarray:
        """ returns a column of pixel colors in RGB format """
        br = self.config.brightness
        gbr = self.config.global_brightness
        amp = self.drivers.amplitude
        phase = self.drivers.phase
        ws = 2 * np.pi / self.config.space_period # spacial frequency
        ph_s = ws*col_num

        colors = np.array([
            np.sin(phase + ph_s),
            np.sin(phase + ph_s + 2*np.pi/3),
            np.sin(phase + ph_s - 2*np.pi/3),
        ])
        #print(np.sum(np.fabs(colors), axis=0)[0])
        # normailize brightness across hues
        colors /= np.sum(np.fabs(colors), axis=0)

        colors = gbr / br * (br + amp * colors)

        # colors = br + amp * np.array([
        #     np.sin(phase + ph_s),
        #     np.sin(phase + ph_s + 2*np.pi/3),
        #     np.sin(phase + ph_s - 2*np.pi/3),
        # ])
        # #print(np.sum(np.fabs(colors), axis=0)[0])
        # # normailize brightness across hues
        # colors *= gbr / np.sum(colors, axis=0)

        #print(colors.T[0])
        colors = np.fmax(np.fmin(colors, 255), 0)

        colors = np.floor(colors)

        return colors.T # transpose is an array of pixels with [r,g,b] values

    def process(self, frame: np.ndarray) -> np.ndarray:
        """ process a single frame consisting of a power spectrum already
            split into self.num_channels buckets
            :return: a np.array[num_channels][num_per_channel] for display """

        self.applyFilters(frame)
        self.applyChannelEffects()
        self.applyChannelSync()

        grid = np.zeros((self.num_per_channel, self.num_channels, 3), dtype=np.float64)
        half_len = self.num_per_channel // 2
        for i in range(half_len):
            col = self.getPixelBlock(i)
            grid[half_len + i] = col
            grid[half_len - 1 - i] = col

        return grid


if __name__ == '__main__':
    # Test sim grid
    G = SimLedGrid('by_row', 16, 60, n_samples=1024, fs=44100)
    D = DemoDisplay()