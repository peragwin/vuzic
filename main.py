import sys

from audio_context import AudioCtx
from waterfall import Waterfall

from util import Bucketer, log_power_spectrum, NP_FMT

def main(device: int = -1):
    print("Displaying spectral waterfall for input stream...")

    N_SAMPLES = 1024
    FS = 44100
    #DEVICE_INDEX = 0 # -1 is Auto

    W = Waterfall(400, n_samples=N_SAMPLES, fs=FS)
    A = AudioCtx(W, n_samples=N_SAMPLES, fs=FS, device_index=device)

    A.run()

if __name__ == '__main__':
    device = -1
    if len(sys.argv) > 1:
        device = int(sys.argv[1])
    main(device)
