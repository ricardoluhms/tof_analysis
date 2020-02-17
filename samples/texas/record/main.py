from real_time_acquisition.texas.utils import *
from raw_data_processing.utils import *
from filters.utils import *
from viewer.utils import *
from hdr_merger.utils import *
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
	camera_matrix = np.load('camera_matrix.npz')
	mapx = camera_matrix['mapx']; mapy = camera_matrix['mapy']

	capture = raw_processed
	# capture = depth
	cap = capture()
	processor = phase2point_cloud(240,320)#,filter=median_filter(240,320,kernel_size=5))

	remap = lens_callibrate(240,320,mapx,mapy)
	pc = np.zeros((240*320,3), dtype='float32')
	fps = FPS()
	phase_hdr = hdr_merger(240, 320, 'uint16', norm=False, max=255, norm_dtype='uint8')
	amplitude_hdr = hdr_merger(240, 320, 'uint16', norm=True, max=2**12-1, norm_dtype='uint16')
	phase_processor = phase2point_cloud(240,320,f1=16e6,f2=24e6,focal_length=3.33e-3,pixel_size=15e-6,dealiased_mask=3, filter=median_filter(240,320,kernel_size=7))

	exp_dir = '10022020/exp4/illum/'
	check_dir(exp_dir)
	phase_writer = writer(exp_dir+'phase.out',dtype='uint16')
	amplitude_writer = writer(exp_dir+'amplitude.out',dtype='uint16')
	ambient_writer = writer(exp_dir+'ambient.out',dtype='uint8')
	flags_writer = writer(exp_dir+'flags.out',dtype='uint8')

	count=0
	while 1:
		ret, [phase, amplitude, ambient, flags] = cap.read()
		if not ret:
			cap.release()
			cap = capture()
			continue
		if count==90:
			break

		# '''
		phase_writer.write(phase); amplitude_writer.write(amplitude);
		ambient_writer.write(ambient); flags_writer.write(flags)
		# '''

		pcv.imshow('phase', phase, heat_map=True)
		pcv.imshow('', amplitude, heat_map=True)
		key_pressed = cv2.waitKey(1) & 0xff
		if key_pressed in [27, ord('q')]:
			break
		if key_pressed in [ord('s')]:
			pc = phase_processor.process(phase)
			pcv.pcshow(pc)
		count +=1

	cap.release()
