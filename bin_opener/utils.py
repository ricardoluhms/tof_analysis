import numpy as np


class bin_opener():
	"""
	Abstract class for all opners. After creating an object ANY_opener() call method open() and them use read()
	To create a new opener just change dtype and, IF NEEDED, overwrite reshape or read_transformation.
	"""
	def __init__(self, file, height, width, dtype):
		self.file = file
		self.height = height
		self.width = width
		self.dtype = dtype
		self.i = 0
		self.buffer = None

	def open(self,):
		with open(self.file, 'rb') as f:
			self.buffer = np.frombuffer(f.read(), dtype=self.dtype).copy()
		self.reshape()
		return self

	def reshape(self,):
		self.buffer = self.buffer.reshape((-1,1))

	def read(self):
		if self.i >= self.buffer.shape[0]//(self.width*self.height):
			return False, None
		data = self.buffer[self.width*self.height*self.i:self.width*self.height*(self.i+1),:]
		data = self.read_transformation(data)
		self.i += 1
		return True, data

	def read_transformation(self,data):
		data = data.reshape((self.width,self.height)).transpose().reshape((-1,1))
		return data


class PointCloud_opener(bin_opener):
	def __init__(self,file,height,width):
		bin_opener.__init__(self,file,height,width,'float32')

	def reshape(self,):
		self.buffer = self.buffer.reshape((-1,4))
		self.buffer = self.buffer[np.bitwise_and(np.bitwise_and(self.buffer[:,0] != 6, self.buffer[:,1] != 6),np.bitwise_and(self.buffer[:,2] != 6, self.buffer[:,3] != 6)), :]

	def read_transformation(self,data):
		return data

class Amplitude_opener(bin_opener):
	def __init__(self,file,height,width):
		bin_opener.__init__(self,file,height,width,'uint16')

class Ambient_opener(bin_opener):
	def __init__(self,file,height,width):
		bin_opener.__init__(self,file,height,width,'uint8')

class Depth_opener(bin_opener):
	def __init__(self,file,height,width):
		bin_opener.__init__(self,file,height,width,'float32')

class Phase_opener(bin_opener):
	def __init__(self,file,height,width):
		bin_opener.__init__(self,file,height,width,'uint16')

#'Ambient',
#'0Amplitude','AmplitudeAvg','AmplitudeStd',
#'Depth','DepthAvg','DepthStd','Distance',
#'Phase','PhaseAvg','PhaseStd',
#'PointCloud'
# np.uint8,
# np.uint16,np.uint16,np.uint16,
# np.float32,np.float32,np.float32,np.float32,
# np.uint16,np.uint16,np.uint16,
# np.float32