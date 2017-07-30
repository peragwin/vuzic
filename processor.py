import numpy as np
from typing import Tuple, List
from util import NP_FMT

class Processor:

    def __init__(self, n_samples: int, fs: int):
        self.n_samples = n_samples
        self.fs = fs

         # going to 50% overlap between frames, so n_samples * 2
        self.buffer = np.zeros(self.n_samples * 2, dtype=NP_FMT)
        self.buffer_toggle = 0
        self.idx_first_quarter = self.n_samples // 2
        self.idx_last_quarter = self.n_samples + self.idx_first_quarter

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
        first = np.r_[self.buffer[a:b], self.buffer[c:d]]
        second = np.r_[self.buffer[e:f], self.buffer[g:h]]
        return first, second

    def process_raw(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ process_samples adds samples to a circular buffer """
        if self.buffer_toggle == 0:
            self.buffer[self.n_samples:] = data
        else:
            self.buffer[:self.n_samples] = data

        frame0, frame1 = self.get_overlapping_frames()

        self.buffer_toggle ^= 1

        return frame0, frame1

    def process_frames(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:
    	""" Function to be defined by individual processor modules """
    	pass

    def process(self, data: np.ndarray) -> None:
    	""" Function to call process_frames with frames created by process_raw
			allows other processors to share a common raw buffer
    	"""
    	self.process_frames(
    		self.process_raw(data))

    def begin(self):
        pass


class Multiprocessor(Processor):
	def __init__(processors: List[Processor]):
		self.processors = processors

	def process_frames(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:
		for p in self.processors:
			p.process_frames(frames)

	def begin(self):
		for p in self.processors:
			p.begin()

