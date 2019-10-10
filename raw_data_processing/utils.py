import numpy as np
class processor():
	def __init__(self):
		pass

	def process(self):
		pass


class phase2depth(processor):
	def __init__(self, height, width, f1=4e7, f2=6e7):
		processor.__init__(self)
		self.height = height
		self.width = width

		self.c = 299792458
		self.freq = None
		self.R = None
		self.set_parameters(f1, f2)

		self.x_alpha = None
		self.y_alpha = None
		self.get_sensor_parameters()

	def get_sensor_parameters(self):
		focal_length=3.33e-3;pixel_size=15e-6;
		sensor_x, sensor_y = np.meshgrid(np.arange(0,self.width), np.arange(0,self.height))
		sensor_x = (sensor_x * pixel_size) + pixel_size / 2 - (pixel_size * self.width) / 2
		sensor_y = (sensor_y * pixel_size) + pixel_size / 2 - (pixel_size * self.height) / 2
		sensor_x = np.abs(sensor_x)
		sensor_y = np.abs(sensor_y)
		self.x_alpha = np.arctan2(sensor_x, focal_length).reshape((-1,1))
		self.y_alpha = np.arctan2(sensor_y, focal_length).reshape((-1,1))

	def set_parameters(self, f1, f2):
		self.freq = np.gcd(int(f1),int(f2))
		self.R = self.c/(2*self.freq)

	def process(self, phase):
		phase = phase.reshape((-1,1))
		distance = phase*self.R / (2*np.pi) / 1000
		depth = (np.cos(self.y_alpha)*distance)*np.cos(self.x_alpha)
		return depth


# class depth(processor):
# 	def __init__(self, f1=4e7, f2=6e7):
# 		processor.__init__(self)
# 		self.freq = np.gcd(f1,f2)
# 		self.c = 299792458
# 		self.R = self.c/(2*self.freq)

# 	def process(self,):	