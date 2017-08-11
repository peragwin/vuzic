import sys

from audio_context import AudioCtx
from waterfall import Waterfall
from sim_led_grid import SimLedGrid
from processor import Multiprocessor

import graphene

from server import Server

from util import Bucketer, log_power_spectrum, NP_FMT

def main(device: int = -1, mode: str = 'by_row'):
    print("Displaying spectral waterfall for input stream...")

    N_SAMPLES = 1024
    FS = 44100
    #DEVICE_INDEX = 0 # -1 is Auto

    W = Waterfall(400, n_samples=N_SAMPLES, fs=FS)
    G = SimLedGrid(mode, 16, 60, N_SAMPLES, FS)
    P = Multiprocessor([W, G])

    A = AudioCtx(P, n_samples=N_SAMPLES, fs=FS, device_index=device)

    dc = G.sub_processor.config
    schema = graphene.Schema(query=dc.Query, mutation=dc.Mutations)
    S = Server(schema=schema)

    A.run()

if __name__ == '__main__':
    device = -1
    mode = 'by_row'
    if len(sys.argv) > 1:
        device = int(sys.argv[1])
    if len(sys.argv) > 2:
        mode = str(sys.argv[2])
    main(device)
