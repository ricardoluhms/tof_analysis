from real_time_acquisition.basler.utils import *
from raw_data_processing.utils import *
from filters.utils import *
from viewer.utils import *
from read_write.utils import *
import cv2, numpy as np

class lens_callibrate():
	def __init__(self,height,width,mapx,mapy):
		self.height = height
		self.width = width
		self.mapx = mapx
		self.mapy = mapy

	def apply(self,frame):
		return cv2.remap(frame.reshape((self.height,self.width)),self.mapx,self.mapy,cv2.INTER_LINEAR).reshape((-1,1))

import time
class FPS():
	def __init__(self):
		self.t0=time.time()

	def measure(self):
		self.t1 = time.time()
		print(1/(self.t1-self.t0))
		self.t0 = self.t1


if __name__=='__main__':
	pcv.height=480;pcv.width=640
	cap = point_cloud()
	
	pc = np.zeros((480*640,3), dtype='float32')
	fps = FPS()
	exp_dir = 'test/'
	check_dir(exp_dir)
	x_writer = writer(exp_dir+'x.out',height=480,width=640,dtype='float32')
	y_writer = writer(exp_dir+'y.out',height=480,width=640,dtype='float32')
	z_writer = writer(exp_dir+'z.out',height=480,width=640,dtype='float32')
	amplitude_writer = writer(exp_dir+'amplitude.out',height=480,width=640,dtype='uint16')
	count=0
	while 1:
		ret, [x,y,z,amplitude] = cap.read()
		if not ret:
			cap.release()
			cap = capture()
			continue
		if count==90:
			break

		x_writer.write(x); y_writer.write(y);z_writer.write(z); amplitude_writer.write(amplitude)

		pcv.imshow('depth', z, heat_map=True)
		pcv.imshow('amplitude', amplitude, heat_map=True)
		key_pressed = cv2.waitKey(1) & 0xff
		if key_pressed in [27, ord('q')]:
			break
		if key_pressed in [ord('s')]:
			pc = np.hstack([x,y,z])
			pcv.pcshow(pc)
		count +=1

	cap.release()