import numpy as np
import cv2

import threading

from processor import Processor

class SimLedGrid(Processor):

	def __init__(self, n_channels: int, n_per_channel: int, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.n_channels = n_channels
		self.n_per_channel = n_per_channel

		# n_channels x n_per_channel grid of 24bit RGB color values
		self.led_grid = np.zeros((n_channels, n_per_channel), dtype=np.int32)

	def process_frames(self, frames: Tuple[np.ndarray, np.ndarray]) -> None:

	def cv_animation(self):
		

	def begin(self):
		t = threading.Thread(target=self.cv_animation)
		t.start()

if __name__ == '__main__':
	# Test sim grid
	G = SimLedGrid(16, 60, n_samples=1024, fs=44100)