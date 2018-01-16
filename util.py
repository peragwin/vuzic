import numpy as np
import pyaudio

NP_FMT = np.float64
PA_FMT = pyaudio.paInt16
PA_DEVICE_KEYS = [
    'hostApi',
    'index',
    'maxInputChannels',
    'maxOutputChannels',
    'name',
]

def power_spectrum(frame: np.ndarray) -> np.ndarray:
    ln = (len(frame[0]) - 1)  // 2
    return np.square(
        np.abs(np.fft.rfft(frame, axis=1)[:, 1:])
    ) / ln

def log_power(frame: np.ndarray) -> np.ndarray:
    out = frame[:]
    out[out <= 1e-20] = 1e-20
    return 10 * np.log10(out)

def log_power_spectrum(frame: np.ndarray) -> np.ndarray:
    spec = power_spectrum(frame)
    spec[spec <= 1e-20] = 1e-20
    return 10 * np.log10(spec)


class LogScale:
    """ convert from linear to log scale """
    @staticmethod
    def To(val):
        return np.log(val)

    @staticmethod
    def From(val):
        return np.exp(val)


class MelScale:
    """ convert from frequency to mel scale """
    @staticmethod
    def To(val):
        return 1127 * np.log(1 + val / 700.)

    @staticmethod
    def From(val):
        return 700 * (np.exp(val / 1127.0) - 1)


class Bucketer:
    """ bucketer slices a spectrum array into :n: buckets based on indices
        derived from :scale:
    """
    scales = {
        'log': LogScale(),
        'mel': MelScale(),
    }

    def __init__(self,
                 frame_size: int,
                 n_buckets: int,
                 n_channels: int,
                 f_min: int,
                 f_max: int,
                 scale: str = 'mel'):

        self.n_buckets = n_buckets
        self.n_channels = n_channels
        self.f_min = f_min
        self.f_max = f_max
        self.frame_size = frame_size

        assert scale in self.scales, \
            "scale must be one of {}".format(self.scales.keys())
        self.scale = self.scales[scale]

        buckets = self.get_freq_buckets(self.scale, f_min, f_max, n_buckets)
        self.indices = np.int32(
            np.ceil(frame_size * buckets / f_max)
        )

    @staticmethod
    def get_freq_buckets(s, f_min: int, f_max: int, N: int) -> np.ndarray:
        return s.From(np.linspace(s.To(f_min), s.To(f_max), N))

    def bucket(self, frame: np.ndarray) -> np.ndarray:
        bucket = np.zeros((self.n_channels, self.n_buckets), dtype=NP_FMT)
        a = 0
        for i in range(self.n_buckets):
            b = self.indices[i]
            bucket[:, i] = np.average(frame[:, a:b], axis=1)
            a = b
        return bucket


class TriangleBucketer(Bucketer):

    # a value of 1 will have 0% overlap
    # a value of 2 will put 75% of the area inside the bucket
    # since the triangle will be of value 0.5 at each edge
    triangle_scale = 2

    def triangle(self, scale: int, center: float) -> np.ndarray:
        """ returns a triangle function around `center` with zeros at +/- `scale` """

        x = np.r_[:self.frame_size]
        out = np.zeros(self.frame_size, dtype=NP_FMT)
        left = center - scale
        right = center + scale
        
        out[x <= left] = 0
        out[x >= right] = 0
        
        first_half = np.logical_and(left < x, x <= center)
        out[first_half] = (x[first_half] - left) / (center - left)
        
        second_half = np.logical_and(center <= x, x < right)
        out[second_half] = (right - x[second_half]) / (right - center)
        
        return out / np.sum(out)

    def filterbank(self) -> np.ndarray:
        """ construct a matrix of coefficients that will be dotted with a frame """
        filters = np.zeros((self.n_buckets, self.frame_size), dtype=NP_FMT)
        a = 0
        for i in range(self.n_buckets):
            b = self.indices[i]
            center = (a + b) / 2
            scale = self.triangle_scale * (b - a) / 2
            filters[i] = self.triangle(scale, center)
            a = b
        return filters

    def bucket(self, frame: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_channels, self.n_buckets), dtype=NP_FMT)
        filters = self.filterbank().T
        for i in range(self.n_channels):
            out[i] = np.dot(frame[i], filters)
        return out
