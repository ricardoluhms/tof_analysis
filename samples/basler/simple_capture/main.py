from real_time_acquisition.basler.utils import *
from raw_data_processing.utils import *
from filters.utils import *
from viewer.utils import *
from interface.utils import *
import cv2, numpy as np

class gui(interface):
	def __init__(self):
		interface.__init__(self)
		self.capture = point_cloud
		self.cap = self.capture()
		self.x, self.y, self.z, self.amplitude, self.confidence = None, None, None, None, None
		self.pc = np.zeros((480*640,3), dtype='float32')
		self.scale=2
		self.font_size=1
		self.select = 0

	def get_frame(self):
		ret, [self.x, self.y, self.z, self.amplitude] = self.cap.read()
		while not ret:
			ret, [self.x, self.y, self.z, self.amplitude] = self.cap.read()
			self.cap.release()
			self.cap = self.capture()

		if self.select==0:
			img2show = pcv.heat_map(self.amplitude.reshape((480,640)))
			img2show = cv2.putText(img2show, 'amp', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)
			img2show = cv2.putText(img2show, 'amp', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
		elif self.select==1:
			img2show = pcv.heat_map(self.z.reshape((480,640)))
			img2show = cv2.putText(img2show, 'dpth', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)
			img2show = cv2.putText(img2show, 'dpth', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
		return img2show

	def main(self):
		x = self.x.reshape((480,640))
		y = self.y.reshape((480,640))
		z = self.z.reshape((480,640))
		amplitude = self.amplitude.reshape((480,640))

		'''
		If there's any bndbox in the screen it will show the mean+-std of x, y, z and amplitude
		'''
		self.img2show = cv2.resize(self.img2show, (self.img2show.shape[1]*self.scale, self.img2show.shape[0]*self.scale))
		for bndbox in self.bndboxes:
			mean_amp = amplitude[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_amp = amplitude[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'amp:%.1f+-%.1f'%(mean_amp,std_amp), (self.scale*bndbox[2], self.scale*bndbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'amp:%.1f+-%.1f'%(mean_amp,std_amp), (self.scale*bndbox[2], self.scale*bndbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

			mean_x = x[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_x = x[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'x:%.1f+-%.1f'%(mean_x,std_x), (self.scale*bndbox[2], self.scale*bndbox[1]+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'x:%.1f+-%.1f'%(mean_x,std_x), (self.scale*bndbox[2], self.scale*bndbox[1]+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

			mean_y = y[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_y = y[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'y:%.1f+-%.1f'%(mean_y,std_y), (self.scale*bndbox[2], self.scale*bndbox[1]+45), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'y:%.1f+-%.1f'%(mean_y,std_y), (self.scale*bndbox[2], self.scale*bndbox[1]+45), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

			mean_z = z[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_z = z[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'z:%.1f+-%.1f'%(mean_z,std_z), (self.scale*bndbox[2], self.scale*bndbox[1]+70), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'z:%.1f+-%.1f'%(mean_z,std_z), (self.scale*bndbox[2], self.scale*bndbox[1]+70), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

		self.img2show = cv2.resize(self.img2show, (self.img2show.shape[1]//self.scale, self.img2show.shape[0]//self.scale))

		cv2.imshow('',self.img2show)

		'''
		Section to handle keys
		'''
		key_pressed = cv2.waitKey(1000//60) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey() & 0xff
		if key_pressed in [27, ord('q')]:
			self.keep_going = False
		if key_pressed == ord('s'):
			self.pc = np.hstack([self.x,self.y,self.z])
			pcv.pcshow(self.pc)
		if key_pressed == ord(','):
			self.select = 1 if self.select==0 else self.select-1
		if key_pressed == ord('.'):
			self.select = 0 if self.select==1 else self.select+1


if __name__=='__main__':
	a = gui()
	a.loop()
	a.cap.release()