import numpy as np
from collections import namedtuple

from .DemoConfig import Config

class DemoDisplay:
    """ for lack of a better name... 
        TODO: explain what this does
    """

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
                 n_buckets:    int = 16,
                 n_frames: int = 60,
                 direction:    str   = 'center',
                 amp_gain:     int   = 20,
                 diff_gain:    float = 8e-4, # I think this is approx the value that was being used
                 amp_offset:   int   = 400,
                 brightness:   int   = 800,
                 space_period: int   = 150,
                 channel_sync: float = 1.8e-3, #1.8e-3, # this one too
                ):

        self.n_buckets = n_buckets
        self.n_frames = n_frames

        self.config = Config(
            127,
            direction,
            amp_gain,
            diff_gain,
            amp_offset,
            brightness,
            space_period,
            channel_sync,
        )

        self.drivers = self.Drivers(
            np.zeros(n_buckets, dtype=np.float64),
            np.zeros(n_buckets, dtype=np.float64),
        )

        self.iter_count = 0

        # values are by order then channel because it makes it look cleaner in applyFilter
        self.filter_values = self.FilterValues(
            np.zeros((2, n_buckets), dtype=np.float64),
            np.zeros((2, n_buckets), dtype=np.float64),
        )

        # Using a second order IIR with negative feedback for each channel,
        # implemented in two stages for simplicity
        self.filter_params = self.FilterValues(
            np.zeros((n_buckets, 2, 2), dtype=np.float64),
            np.zeros((n_buckets, 2, 2), dtype=np.float64),
        )

        for i in range(n_buckets):
            self.filter_params.gain[i] = np.array([
                [ .95   , 0.1  ], # first elem is the gain on that channel
                [ .005,  -.995], # this is the lowpass for the feedback mechanism which controls sensitivity
            ])
            self.filter_params.diff[i] = np.array([
                [ .95  , 0.1 ],
                [ .04,  -.96],
            ])

    def applyFilters(self, frame: np.ndarray) -> None:
        diff_input = np.zeros((2, self.n_buckets), dtype=np.float64)
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

    # XXX figure out why some channels seem to get "stuck" inverted 

    def applyChannelEffects(self):
        dg = self.config.diff_gain
        ag = self.config.amp_gain
        ao = self.config.amp_offset

        #print(self.filter_values.gain[0,:], self.filter_values.diff[0,:])

        gain = self.filter_values.gain[0,:]
        diff = self.filter_values.diff[0,:]
        #scale = (1+ np.log(gain)) / (1+np.log(np.sum(gain*gain)))

        self.drivers.phase -= dg * np.log(1+np.fabs(diff))
        self.drivers.phase += .001 # add a constant opposing force makes an interesting heatmap of activity
        #self.drivers.phase = (self.drivers.phase % (24 * np.pi)) - 12*np.pi # numpy ftw
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
            split into self.n_buckets buckets
            :return: a np.array[n_buckets][n_frames] for display """

        # average left and right
        if len(frame) == 2:
            frame = np.mean(frame, axis=0)

        self.applyFilters(frame)
        self.applyChannelEffects()
        self.applyChannelSync()

        grid = np.zeros((self.n_frames, self.n_buckets, 3), dtype=np.float64)
        half_len = self.n_frames // 2
        for i in range(half_len):
            col = self.getPixelBlock(i)
            grid[half_len + i] = col
            grid[half_len - 1 - i] = col

        return grid


if __name__ == '__main__':
    D = DemoDisplay()
