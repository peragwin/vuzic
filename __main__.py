import sys
import graphene

from .audio.DeviceContext import DeviceContext
from .audio.Processor import RawInputProcessor, MultiProcessor
from .modules.waterfall import Waterfall
from .modules.SimLedGrid import SimLedGrid
from .server import Server
from .util import Bucketer, log_power_spectrum, NP_FMT

def main(device: int = -1, mode: str = 'by_row'):
    print("Displaying spectral waterfall for input stream...")

    N_SAMPLES = 1024
    FS = 44100
    #DEVICE_INDEX = 0 # -1 is Auto

    P = RawInputProcessor(n_samples=N_SAMPLES, n_channels=2, fs=FS)

    W = Waterfall(400, show_anim=False)
    G = SimLedGrid(mode, n_buckets=16, n_frames=60, n_samples=N_SAMPLES, n_channels=2, fs=FS)
    
    P.attach_child(MultiProcessor([W, G]))

    A = DeviceContext(P, n_samples=N_SAMPLES, n_channels=2, fs=FS, device_index=device)

    # todo: apply median filter to denoise
    # dc = G.sub_processor.config
    # schema = graphene.Schema(query=dc.Query, mutation=dc.Mutations)
    # S = Server(schema=schema)
    # S.start()
    # P.exit_funcs.append(S.stop)

    A.run()

if __name__ == '__main__':
    device = -1
    mode = 'by_row'
    if len(sys.argv) > 1:
        device = int(sys.argv[1])
    if len(sys.argv) > 2:
        mode = str(sys.argv[2])
    main(device)
