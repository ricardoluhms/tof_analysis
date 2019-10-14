import cv2, numpy as np

class filter():
	def __init__(self, height, width):
		self.height = height
		self.width = width

	@staticmethod
	def	norm(frame, max=255, dtype='uint8'):
		frame = frame - frame.min()
		if frame.max() != 0:
			frame = (max*frame/frame.max())
		return frame.astype(dtype)

	def apply(self, frame):
		frame = frame.reshape((self.height, self.width))
		frame = self.filter(frame)
		frame = frame.reshape((-1,1))
		return frame

	def filter(self, frame):
		pass


class filter_int(filter):
	def __init__(self, height, width):
		filter.__init__(self, height, width)

	def apply(self, frame):
		frame = frame.reshape((self.height, self.width))
		frame_min = frame.min()
		frame = frame-frame_min
		frame_max = frame.max()
		frame = self.filter(frame)
		frame = frame_max*frame.astype('float32')/255+frame_min
		frame = frame.reshape((-1,1))
		return frame


class median_filter(filter_int):
	def __init__(self, height, width, kernel_size=3):
		filter_int.__init__(self, height, width)
		self.kernel_size = kernel_size

	def filter(self, frame):
		frame = cv2.medianBlur(filter.norm(frame, dtype='uint8'), self.kernel_size)
		return frame


class mean_filter(filter_int):
	def __init__(self, height, width, kernel_size=3):
		filter_int.__init__(self, height, width)
		self.kernel_size = kernel_size

	def filter(self, frame):
		frame = cv2.blur(filter.norm(frame, dtype='uint8'), self.kernel_size)
		return frame


class gaussian_filter(filter):
	def __init__(self, height, width, kernel_size=3, std=1):
		filter.__init__(self, height, width)
		self.kernel_size = kernel_size
		self.std = std

	def filter(self, frame):
		frame = cv2.GaussianBlur(frame,(self.kernel_size, self.kernel_size),self.std)
		return frame


class bilateral_filter(filter):
	def __init__(self, height, width, kernel_size=3, std=1):
		filter.__init__(self, height, width)
		self.kernel_size = kernel_size
		self.std = std

	def filter(self, frame):
		frame = cv2.bilateralFilter(frame,self.kernel_size,self.std,self.std)
		return frame


class edge_filter(filter):
	def __init__(self, height, width, kernel_size=3):
		filter.__init__(self, height, width)
		self.lap_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
		self.kernel_size = kernel_size

	def filter(self, frame):
		edge = cv2.filter2D(frame, -1, self.lap_kernel)
		edge = filter.norm(edge)
		edge = cv2.threshold(edge,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.kernel_size,self.kernel_size)))
		# edge = cv2.erode(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.kernel_size-2,self.kernel_size-2)))
		# cv2.imshow('edge', edge)
		edge = edge/edge.max()
		#cv2.imshow('ed', edge)
		frame = frame*np.abs(edge-1)
		#frame = frame*edge
		return frame