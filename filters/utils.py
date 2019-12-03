import cv2, numpy as np

class filter():
	"""
	Abstract class for all filters. After creating an object ANY_filter() call method apply()
	To create a new filter just overwrite the filter method as the reshape is done before and after.
	"""
	def __init__(self, height, width):
		self.height = height
		self.width = width

	@staticmethod
	def	norm(frame, max=255, dtype='uint8'):
		frame = frame - frame.min()
		if frame.max() != 0:
			frame = max*(frame/frame.max())
		return frame.astype(dtype)

	def apply(self, frame):
		frame = frame.reshape((self.height, self.width))
		frame = self.filter(frame)
		frame = frame.reshape((-1,1))
		return frame

	def filter(self, frame):
		pass


class filter_int(filter):
	"""
	Abstract class for filters, but with normalized input for method filter().
	After creating an object ANY_filter() call method apply()
	"""
	def __init__(self, height, width):
		filter.__init__(self, height, width)

	def apply(self, frame):
		frame = frame.reshape((self.height, self.width))
		dtype = frame.dtype
		frame_min = frame.min()
		frame = frame-frame_min
		frame_max = frame.max()
		frame = (255*(frame/frame_max)).astype('uint8')
		frame = self.filter(frame)
		frame = (frame_max*(frame.astype('float64')/255)+frame_min).astype(dtype)
		frame = frame.reshape((-1,1))
		return frame


class median_filter(filter_int):
	"""
	Filter class using the median inside a window with size=kernel_size
	"""
	def __init__(self, height, width, kernel_size=3):
		filter_int.__init__(self, height, width)
		self.kernel_size = kernel_size

	def filter(self, frame):
		frame = cv2.medianBlur(frame, self.kernel_size)
		return frame


class mean_filter(filter_int):
	"""
	Filter class using the mean inside a window with size=kernel_size
	"""
	def __init__(self, height, width, kernel_size=3):
		filter_int.__init__(self, height, width)
		self.kernel_size = kernel_size

	def filter(self, frame):
		frame = cv2.blur(frame, (self.kernel_size,self.kernel_size))
		return frame


class gaussian_filter(filter):
	def __init__(self, height, width, kernel_size=3, std=1):
		filter.__init__(self, height, width)
		self.kernel_size = kernel_size
		self.std = std

	def filter(self, frame):
		frame = cv2.GaussianBlur(frame,(self.kernel_size, self.kernel_size),self.std)
		return frame


class bilateral_filter(filter_int):
	def __init__(self, height, width, kernel_size=3, std_color=75, std_space=75):
		filter_int.__init__(self, height, width)
		self.kernel_size = kernel_size
		self.std_color = std_color
		self.std_space = std_space

	def filter(self, frame):
		frame = cv2.bilateralFilter(frame,self.kernel_size,self.std_color,self.std_space)
		return frame


class weighted_least_square_filter(filter_int):
	'''
	Tried to implement however too much memory is used and i did not fully understand the papers
	notation:
	https://arxiv.org/pdf/1705.01674.pdf
	http://publish.illinois.edu/visual-modeling-and-analytics/files/2014/10/FGS-TIP.pdf
	'''
	def __init__(self, height, width, kernel_size=3, std_color=75, std_space=75):
		filter_int.__init__(self, height, width)
		self.kernel_size = kernel_size
		self.std_color = std_color
		self.std_space = std_space
		
		self.x, self.y = np.meshgrid(np.arange(0,(self.width), dtype='uint16'), np.arange(0,self.height, dtype='uint16'))


	def filter(self, frame):
		w = np.exp(-np.square(np.square(self.x_i.ravel()-self.x_j.ravel())-np.square(self.y_i.ravel()-self.y_j.ravel()))/(2*np.square(self.std_space)))*np.exp(-np.square(frame[self.y_i.ravel(), self.x_i.ravel()]-frame[self.y_j.ravel(), self.x_j.ravel()])/(2*np.square(self.std_color)))
		return frame


class edge_filter(filter):
	"""
	Filter class that attibute the value 0 to countour/edges.
	Used to delete flying pixels.
	"""
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
		frame = frame*np.abs(edge-1)
		return frame


class guided_filter(filter):
	def __init__(self, height, width, kernel_size=3, eps=0.2):
		filter.__init__(self, height, width)
		self.kernel_size = kernel_size
		self.eps = eps

	def apply(self, guide, frame):
		frame = frame.reshape((self.height, self.width))
		guide = guide.reshape((self.height, self.width))
		dtype = frame.dtype
		frame_min = frame.min()
		frame = frame-frame_min
		frame_max = frame.max()
		frame = (255*(frame/frame_max)).astype('uint8')
		frame = self.filter(filter.norm(guide), frame)
		frame = (frame_max*(frame.astype('float64')/255)+frame_min).astype(dtype)
		frame = frame.reshape((-1,1))
		return frame

	def filter(self, guide, frame):
		frame = cv2.ximgproc.guidedFilter(guide, frame, self.kernel_size, self.eps).reshape((-1,1))
		return frame


class temporal_filter():
	"""
	Generic class for temporal filters. It constructs a history of frames
	with length history_len and always keeps the latest frame in the last
	position. Use method filter to create a new filter
	"""
	def __init__(self, history_len):
		self.history_len = history_len
		self.history = None

	def apply(self, frame):
		if type(self.history) == type(None):
			self.history = frame.reshape((-1,1))
		elif self.history.shape[1] < self.history_len:
			self.history = np.hstack([self.history, frame.reshape((-1,1))])
		else:
			self.history[:,:-1] = self.history[:,1:]
			self.history[:,-1:] = frame.reshape((-1,1))
		frame = self.filter(frame).reshape((-1,1))
		return frame

	def filter(self, frame):
		pass

class temporal_mean_filter(temporal_filter):
	"""
	Use the history created to compute the mean and return
	"""
	def __init__(self, history_len):
		temporal_filter.__init__(self,history_len)

	def filter(self,frame):
		frame = self.history.mean(axis=1)
		return frame

class temporal_median_filter(temporal_filter):
	"""
	Use the history created to compute the median and return
	"""
	def __init__(self, history_len):
		temporal_filter.__init__(self,history_len)

	def filter(self,frame):
		frame = np.median(self.history, axis=1)
		return frame

class temporal_bilateral_filter(temporal_filter):
	"""
	Use the history created as a image to compute the bilateral filter in the time domain
	"""
	def __init__(self, history_len,std_color=15,std_space=15):
		temporal_filter.__init__(self,history_len)
		self.std_color = std_color
		self.std_space = std_space

	def filter(self,frame):
		frames = cv2.bilateralFilter(self.history,self.history_len,self.std_color,self.std_space)
		frame = frames[:,0]
		return frame

def heat_map(img, norm=True):
	"""
	DO NOT USE. WILL BE DELETED IN THE FUTURE
	Heat map depracated function.
	"""
	if norm:
		img = img - np.min(img)
		img = np.multiply((6 * 256 - 1), np.divide(img, np.max(img)))
	blue = img.copy()
	green = img.copy()
	red = img.copy()


	#blue = blue
	green[green <= 1 * 256 - 1] = 0  
	#red = red

	blue[np.bitwise_and(blue > 1 * 256 - 1, blue <= 2 * 256 - 1)] = 255
	green[np.bitwise_and(green > 1 * 256 - 1, green <= 2 * 256 - 1)] = 0
	red[np.bitwise_and(red > 1 * 256 - 1, red <= 2 * 256 - 1)] = 2 * 256 - 1 - red[np.bitwise_and(red > 1 * 256 - 1, red <= 2 * 256 - 1)]

	blue[np.bitwise_and(blue > 2 * 256 - 1, blue <= 3 * 256 - 1)] = 255
	green[np.bitwise_and(green > 2 * 256 - 1, green <= 3 * 256 - 1)] = green[np.bitwise_and(green > 2 * 256 - 1, green <= 3 * 256 - 1)] - 2 * 256
	red[np.bitwise_and(red > 2 * 256 - 1, red <= 3 * 256 - 1)] = 0

	blue[np.bitwise_and(blue > 3 * 256 - 1, blue <= 4 * 256 - 1)] = 4 * 256 - 1 - blue[np.bitwise_and(blue > 3 * 256 - 1, blue <= 4 * 256 - 1)]
	green[np.bitwise_and(green > 3 * 256 - 1, green <= 4 * 256 - 1)] = 255
	red[np.bitwise_and(red > 3 * 256 - 1, red <= 4 * 256 - 1)] = 0

	blue[np.bitwise_and(blue > 4 * 256 - 1, blue <= 5* 256 - 1)] = 0 
	green[np.bitwise_and(green > 4 * 256 - 1, green <= 5 * 256 - 1)] = 255
	red[np.bitwise_and(red > 4 * 256 - 1, red <= 5 * 256 - 1)] = red[np.bitwise_and(red > 4 * 256 - 1, red <= 5 * 256 - 1)] - 4 * 256

	blue[np.bitwise_and(blue > 5 * 256 - 1, blue <= 6 * 256 - 1)] = 0 
	green[np.bitwise_and(green > 5 * 256 - 1, green <= 6 * 256 - 1)] = 6 * 256 - 1 - green[np.bitwise_and(green > 5 * 256 - 1, green <= 6 * 256 - 1)]
	red[np.bitwise_and(red > 5 * 256 - 1, red <= 6 * 256 - 1)] = 255

	h, w = img.shape
	img = np.zeros((h, w, 3), dtype='uint8')
	img[:, :, 0] = blue
	img[:, :, 1] = green
	img[:, :, 2] = red
	return img