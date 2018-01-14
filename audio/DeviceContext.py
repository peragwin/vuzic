import asyncio
import pyaudio
import threading
import time
import numpy as np
from pprint import pprint
from typing import Coroutine, Any, Callable, Tuple

from .Processor import Processor

from ..util import NP_FMT, PA_FMT, PA_DEVICE_KEYS
from ..bus import bus


class DeviceContext:

    def __init__(self,
                 processor: Processor,
                 n_samples: int = 1024,
                 n_channels: int = 1,
                 fs: int = 44100,
                 device_index: int = -1,
                 print_devices: bool = False):

        self.n_samples = n_samples
        assert int(2 ** np.log2(n_samples)) == n_samples  # power of 2 for fft
        self.n_channels = n_channels
        self.fs = fs

        self.pa = pa = pyaudio.PyAudio()

        if device_index < 0 or print_devices:
            for n in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(n)
                relevant_info = {k: info[k] for k in PA_DEVICE_KEYS}
                pprint(relevant_info)
                if device_index < 0 and "default" in info['name']:
                    device_index = info['index']

        self.device_index = device_index

        self.stream_thread = None
        self.stop_streaming = False
        
        def end_stream():
            self.stop_streaming = True
        bus.subscribe("end_stream", end_stream)

        processor.register_input(n_samples=n_samples,
                                 fs=fs,
                                 n_channels=n_channels)
        self.processor = processor

    def new_input_streamer(self) -> Callable[[], Coroutine[Any, Any, np.ndarray]]:
        stream = self.pa.open(format=PA_FMT,
                              channels=self.n_channels,
                              rate=self.fs,
                              input=True,
                              input_device_index=self.device_index,
                              frames_per_buffer=self.n_samples)

        async def _streamer() -> np.ndarray:
            return np.fromstring(
                stream.read(self.n_samples, exception_on_overflow=False),
                dtype=np.int16)

        return _streamer

    def new_test_streamer(self) -> Callable[[], Coroutine[Any, Any, np.ndarray]]:
        async def _streamer() -> np.ndarray:
            r = np.arange(0, self.n_samples, dtype=NP_FMT)
            r *= 2000 * 2 * np.pi / self.fs
            return np.cos(r)

        return _streamer

    async def stream(self):
        stream = self.new_input_streamer()
        # stream = self.new_test_streamer()
        print("..begin streaming")

        while True:
            data = await stream()
            if self.stop_streaming:
                break
            self.process_samples(data)

    def process_samples(self, samples: np.ndarray) -> None:
        self.processor.process(samples)

    def begin_stream(self):
        # Create an event loop to host the audio streamer
        loop = asyncio.get_event_loop()

        # Do streaming in a separate thread
        def _stream():
            loop.run_until_complete(self.stream())

        self.stream_thread = threading.Thread(target=_stream, name="streamer")
        self.stream_thread.start()

    def end_stream(self):
        bus.publish("end_stream")

    def begin_processing(self):
        self.processor.begin()

    def run(self):
        self.begin_stream()
        time.sleep(0.2) # delay to get some data so that imshow automatically normalizes okay
        self.begin_processing()
