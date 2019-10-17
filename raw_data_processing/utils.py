import numpy as np
class processor():
	def __init__(self, height, width, f1, f2, focal_length, pixel_size):
		self.height = height
		self.width = width

		self.c = 299792458
		self.freq = None
		self.R = None
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

	def get_distances(self, phase):
		phase[phase>2**12-1] = 2**12-1
		phase = phase.reshape((-1,1))
		distance = phase*self.R/(2**12)
		return distance

	def process(self):
		pass


class phase2depth(processor):
	def __init__(self, height, width,f1=4e7,f2=6e7,focal_length=3.33e-3,pixel_size=15e-6):
		processor.__init__(self, height, width,f1=f1,f2=f2,focal_length=focal_length,pixel_size=pixel_size)

	def process(self, phase):
		distance = self.get_distances(phase)
		depth = distance*np.cos(self.beta)
		return depth


class phase2point_cloud(processor):
	def __init__(self, height, width,f1=4e7,f2=6e7,focal_length=3.33e-3,pixel_size=15e-6):
		processor.__init__(self, height, width,f1=f1,f2=f2,focal_length=focal_length,pixel_size=pixel_size)

	def process(self, phase):
		distance = self.get_distances(phase)
		z = distance*np.cos(self.beta)
		x = z*np.tan(self.x_alpha)
		y = z*np.tan(self.y_alpha)
		point_cloud = np.hstack([x,y,z])
		return point_cloud