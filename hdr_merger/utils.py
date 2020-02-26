import numpy as np

class hdr_merger():
	def __init__(self, height, width, dtype, norm=False, max=255, norm_dtype='uint8'):
		self.frame_scales = [np.zeros((height*width,1), dtype=dtype), np.zeros((height*width,1), dtype=dtype)]
		self.flags_scales = [np.zeros((height*width,1), dtype='uint8'), np.zeros((height*width,1), dtype='uint8')]
		self.frame_merged = np.zeros((height*width,1), dtype=dtype)
		self.norm = norm
		self.max = max
		self.norm_dtype = norm_dtype
		self.updated = False

	def merge(self, frame, flags):
		if not flags[0,0] in [4,12]:
			if self.norm:
				self.frame_scales[1][:,:] = hdr_merger.norm(frame, max=self.max, dtype=self.norm_dtype)
			else:
				self.frame_scales[1][:,:] = frame
			self.flags_scales[1][:,:] = flags
			self.frame_merged[:,:] = self.frame_scales[0]
			mask = (np.bitwise_or(self.flags_scales[0] == 4, self.flags_scales[0] == 8)).copy()
			self.frame_merged[mask] = self.frame_scales[1][mask]
			self.updated = True
		else:
			if self.norm:
				self.frame_scales[0][:,:] = hdr_merger.norm(frame, max=self.max, dtype=self.norm_dtype)
			else:
				self.frame_scales[0][:,:] = frame
			self.flags_scales[0][:,:] = flags
			self.updated = False
		return self.frame_merged

	def is_updated(self):
		return self.updated

	@staticmethod
	def	norm(frame, max=255, dtype='uint8'):
		frame = frame - frame.min()
		if frame.max() != 0:
			frame = max*(frame/frame.max())
		return frame.astype(dtype)