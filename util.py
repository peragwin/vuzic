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
    ln = len(frame) // 2
    return ln * np.square(
        np.fft.rfft(frame)[:ln]
    )


def log_power_spectrum(frame: np.ndarray) -> np.ndarray:
    return 10 * np.log10(1 + power_spectrum(frame))


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
                 N: int,
                 f_min: int,
                 f_max: int,
                 scale: str = 'mel'):

        self.N = N
        self.f_min = f_min
        self.f_max = f_max

        assert scale in self.scales, \
            "scale must be one of {}".format(self.scales.keys())
        self.scale = self.scales[scale]

        buckets = self.get_freq_buckets(self.scale, f_min, f_max, N)
        self.indices = np.int32(
            np.ceil(frame_size * buckets / f_max)
        )
        print("indices", self.indices)

    @staticmethod
    def get_freq_buckets(s, f_min: int, f_max: int, N: int) -> np.ndarray:
        return s.From(np.linspace(s.To(f_min), s.To(f_max), N))

    def bucket(self, frame: np.ndarray) -> np.ndarray:
        bucket = np.zeros(self.N, dtype=np.float64)
        a = 0
        for i in range(self.N):
            b = self.indices[i]
            bucket[i] = np.average(frame[a:b])
            a = b
        return bucket
