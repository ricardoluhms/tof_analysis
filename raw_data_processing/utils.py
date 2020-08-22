import numpy as np
'''
Review if it should be put into ONE class instead!
'''
class processor():
	def __init__(self, height, width, f1, f2, focal_length, pixel_size, dealiased_mask):
		self.height = height
		self.width = width
		self.dealiased_mask = dealiased_mask

		self.c = 299792458
		self.freq = None
		self.R = None
		self.ma = None
		self.mb = None
		self.set_parameters(f1, f2)

		self.x_alpha = None
		self.y_alpha = None
		self.focal_length=focal_length;self.pixel_size=pixel_size;
		self.get_sensor_parameters()

	def get_sensor_parameters(self):
		sensor_x, sensor_y = np.meshgrid(np.arange(0,self.width), np.arange(0,self.height))
		sensor_x = (sensor_x * self.pixel_size) + self.pixel_size / 2 - (self.pixel_size * self.width) / 2
		sensor_y = (sensor_y * self.pixel_size) + self.pixel_size / 2 - (self.pixel_size * self.height) / 2
		self.x_alpha = np.arctan(sensor_x/self.focal_length).reshape((-1,1))
		self.y_alpha = np.arctan(sensor_y/self.focal_length).reshape((-1,1))
		self.beta = np.arctan(np.sqrt(np.square(sensor_x)+np.square(sensor_y))/self.focal_length).reshape((-1,1))

	def set_parameters(self, f1, f2):
		self.freq = np.gcd(int(f1),int(f2))
		self.R = self.c/(2*self.freq)
		self.ma = f1/self.freq
		self.mb = f2/self.freq

	def process(self):
		pass

class phase2depth(processor):
	def __init__(self, height, width,f1=4e7,f2=6e7,focal_length=3.33e-3,pixel_size=15e-6,dealiased_mask=2, filter=None):
		processor.__init__(self, height, width,f1,f2,focal_length,pixel_size,dealiased_mask)
		self.filter = filter

	def phase2distance(self, phase):
		phase[phase>2**12-1] = 2**12-1
		phase = phase.reshape((-1,1))
		distance = (phase*self.R*2**(5-self.dealiased_mask))/(self.ma*self.mb*2**12)
		return distance

	def process(self, phase):
		distance = self.phase2distance(phase)
		depth = distance*np.cos(self.beta)
		if type(self.filter) != type(None):
			depth = self.filter.apply(depth)
		return depth.astype('float32')

class depth2point_cloud(processor):
	def __init__(self, height, width,f1=4e7,f2=6e7,focal_length=3.33e-3,pixel_size=15e-6,dealiased_mask=2, filter=None):
		processor.__init__(self, height, width,f1,f2,focal_length,pixel_size,dealiased_mask)
		self.filter = filter

	def depth2point_cloud(self, z):
		z = z.reshape((-1,1))
		x = z*np.tan(self.x_alpha)
		y = z*np.tan(self.y_alpha)
		point_cloud = np.hstack([x,y,z])
		return point_cloud.astype('float32')

	def process(self, z):
		if type(self.filter) != type(None):
			z = self.filter.apply(z)
		point_cloud = self.depth2point_cloud(z)
		return point_cloud.astype('float32')


class phase2point_cloud(phase2depth,depth2point_cloud):
	def __init__(self, height, width,f1=4e7,f2=6e7,focal_length=3.33e-3,pixel_size=15e-6,dealiased_mask=2, filter=None):
		phase2depth.__init__(self, height, width,f1,f2,focal_length,pixel_size,dealiased_mask)
		self.filter = filter

	def process(self, phase):
		distance = self.phase2distance(phase)
		z = distance*np.cos(self.beta)
		if type(self.filter) != type(None):
			z = self.filter.apply(z)
		point_cloud = self.depth2point_cloud(z)
		return point_cloud.astype('float32')