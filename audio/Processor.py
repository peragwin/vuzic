import numpy as np
from typing import Tuple, List, Callable, Dict
from ..util import NP_FMT
from ..bus import bus
from abc import abstractmethod, ABCMeta


class Processor(metaclass=ABCMeta):
    @abstractmethod
    def process(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def register_input(self, **kwargs) -> None:
        self.input_registered = True
        if self.child_processor:
            self.child_processor.register_input(**kwargs)

    def __init__(self, **kwargs) -> None:
        self.input_registered = False
        self.args = kwargs
        self.child_processor = None

    def attach_child(self, child_processor) -> None:
        assert isinstance(child_processor, Processor)
        self.child_processor = child_processor
        child_processor.register_input(**self.args)

    def begin(self):
        assert self.input_registered, "must register input parameters"
        if self.child_processor:
            self.child_processor.begin()

    def end_stream(self):
        bus.publish("end_stream")


class RawInputProcessor(Processor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_samples = 0
        self.n_channels = 0

        self.buffer = np.zeros(0)
        self.buffer_toggle = 0
        self.idx_first_quarter = 0
        self.idx_last_quarter = 0

    def register_input(self, **kwargs) -> None:

        self.n_samples = n_samples = kwargs['n_samples']
        self.n_channels = n_channels = kwargs['n_channels']

        # going to 50% overlap between frames, so n_samples * 2
        self.buffer = np.zeros((n_channels, n_samples * 2), dtype=NP_FMT)
        self.idx_first_quarter = n_samples // 2
        self.idx_last_quarter = n_samples + self.idx_first_quarter

        super().register_input(**kwargs)

    def get_overlapping_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :returns: two 50% overlapping frames extracted from the buffer to be passed into ffts or whatnot
            The first frame is composed of the last half of previous samples plus the first half of newest
            The second is composed of only the newest samples
        """
        # these are the indices which mark the edges of the frames in the order that they are used
        a, b, c, d, e, f, g, h = 0, 0, 0, 0, 0, 0, 0, 0
        if self.buffer_toggle == 0:
            a = self.idx_last_quarter
            b = self.n_samples * 2
            c = 0
            d = self.idx_first_quarter
            e = c
            f = d
            g = d
            h = self.n_samples
        else:
            a = self.idx_first_quarter
            b = self.n_samples
            c = b
            d = self.idx_last_quarter
            e = self.n_samples
            f = self.idx_last_quarter
            g = f
            h = self.n_samples * 2
        first = np.c_[self.buffer[:, a:b], self.buffer[:, c:d]]
        second = np.c_[self.buffer[:, e:f], self.buffer[:, g:h]]
        return first, second

    def process_raw(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ process_samples adds samples to a circular buffer """
        for i in range(self.n_channels):
            if self.buffer_toggle == 0:
                self.buffer[i, self.n_samples:] = data[i::self.n_channels]
            else:
                self.buffer[i, :self.n_samples] = data[i::self.n_channels]

        frame0, frame1 = self.get_overlapping_frames()

        self.buffer_toggle ^= 1

        return frame0, frame1

    def process(self, data: np.ndarray) -> None:
        """ Function to call process_frames with frames created by process_raw
            allows other processors to share a common raw buffer
        """
        

        out = self.process_raw(data)

        super().process(data)
        if self.child_processor:
            self.child_processor.process(out)




# class ChainProcessor(Processor):
#     def __init__(self, processors: List[Processor]):
#         self.processors = processors

#     def begin(self):
#         for p in self.processors:
#             p.begin()


class Multiprocessor(Processor):
    def __init__(self, processors: List[Processor]):
        self.processors = processors

        self.exit_funcs = []

    def register_input(*args, **kwargs):
        for p in self.processors:
            p.register_input(*args, **kwargs)

    def process(self, data) -> None:
        for p in self.processors:
            p.process(data)

    def begin(self):
        for p in self.processors:
            p.begin()
