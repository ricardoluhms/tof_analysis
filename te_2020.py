from read_write.utils import *
from raw_data_processing.utils import *
from filters.utils import *
from viewer.utils import *
from interface.utils import *
import cv2, numpy as np

class gui(interface):
	def __init__(self):
		interface.__init__(self)
		self.x, self.y, self.z, self.amplitude = None, None, None, None
		self.pc = np.zeros((240*320,3), dtype='float32')
		self.scale=2
		self.font_size=1
		self.phase_processor = phase2depth(240, 320,f1=16e6,f2=24e6,focal_length=3.33e-3,pixel_size=15e-6,dealiased_mask=2, filter=None)
		self.select = 0

		self.scale_min = 0
		self.scale_max = 0
		self.phase_max = self.phase_processor.R * self.phase_processor.ma * self.phase_processor.mb * 2**12/(self.phase_processor.R*2**(5-self.phase_processor.dealiased_mask))

		dist = '10000'
		self.exp_dir = '/home/vinicius/tof/texas/record/26022020/%s/'%dist
		self.phase_reader = reader(self.exp_dir+'phase.out')
		self.amplitude_reader = reader(self.exp_dir+'amplitude.out')
		self.ambient_reader = reader(self.exp_dir+'ambient.out')
		self.flags_reader = reader(self.exp_dir+'flags.out')

		self.camera_matrix = np.load('camera_matrix.npz')
		self.mapx = self.camera_matrix['mapx']; self.mapy = self.camera_matrix['mapy']

	def get_frame(self):
	
		ret, self.phase = self.phase_reader.read()
		_, self.amplitude = self.amplitude_reader.read()
		_, self.ambient = self.ambient_reader.read()
		_, self.flags = self.flags_reader.read()
		if not ret:
			self.phase_reader.reset()
			self.amplitude_reader.reset()
			self.ambient_reader.reset()
			self.flags_reader.reset()

		self.phase[self.phase>self.phase_max] = self.phase_max
		self.depth = self.phase_processor.process(self.phase)

		'''
		Select which image to show in the click and drag window
		'''
		if self.select==0:
			self.scale_min = self.amplitude.min()
			self.scale_max = self.amplitude.max()
			img2show = pcv.heat_map(self.amplitude.reshape((240,320)))
			# img2show = pcv.heat_map((6*256-1)*(self.amplitude.reshape((240,320))/4095), norm=False)
			img2show = cv2.putText(img2show, 'amp', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)
			img2show = cv2.putText(img2show, 'amp', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
		elif self.select==1:
			self.scale_min = self.phase.min()
			self.scale_max = self.phase.max()
			# img2show = pcv.heat_map(self.phase.reshape((240,320)))
			img2show = pcv.heat_map((6*256-1)*(self.phase.reshape((240,320))/4095), norm=False)
			img2show = cv2.putText(img2show, 'phs', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)
			img2show = cv2.putText(img2show, 'phs', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
		elif self.select==2:
			self.scale_min = self.depth.min()
			self.scale_max = self.depth.max()
			# img2show = pcv.heat_map(self.depth.reshape((240,320)))
			img2show = pcv.heat_map((6*256-1)*(self.depth.reshape((240,320))/18), norm=False)
			img2show = cv2.putText(img2show, 'dpth', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)
			img2show = cv2.putText(img2show, 'dpth', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
		elif self.select==3:
			self.scale_min = self.ambient.min()
			self.scale_max = self.ambient.max()
			# img2show = pcv.heat_map(self.ambient.reshape((240,320)))
			img2show = pcv.heat_map((6*256-1)*(self.ambient.reshape((240,320))/4095), norm=False)
			img2show = cv2.putText(img2show, 'amb', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)
			img2show = cv2.putText(img2show, 'amb', (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
		return img2show

	def main(self):
		amplitude = self.amplitude.reshape((240,320))
		phase = self.phase.reshape((240,320))
		ambient = self.ambient.reshape((240,320))
		depth = self.depth.reshape((240,320))
		phase_z = (self.phase*np.cos(self.phase_processor.beta)).reshape((240,320))

		'''
		If there's any bndbox in the screen it will show the mean+-std of amplitude, phase, ambient and depth
		'''
		self.img2show = cv2.resize(self.img2show, (self.img2show.shape[1]*self.scale, self.img2show.shape[0]*self.scale))
		for bndbox in self.bndboxes:
			mean_amp = amplitude[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_amp = amplitude[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'amp:%.1f+-%.1f'%(mean_amp,std_amp), (self.scale*bndbox[2], self.scale*bndbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'amp:%.1f+-%.1f'%(mean_amp,std_amp), (self.scale*bndbox[2], self.scale*bndbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

			mean_phs = phase[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_phs = phase[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'phs:%.1f+-%.1f'%(mean_phs,std_phs), (self.scale*bndbox[2], self.scale*bndbox[1]+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'phs:%.1f+-%.1f'%(mean_phs,std_phs), (self.scale*bndbox[2], self.scale*bndbox[1]+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

			mean_phsz = phase_z[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_phsz = phase_z[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'phsz:%.1f+-%.1f'%(mean_phsz,std_phsz), (self.scale*bndbox[2], self.scale*bndbox[1]+45), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'phsz:%.1f+-%.1f'%(mean_phsz,std_phsz), (self.scale*bndbox[2], self.scale*bndbox[1]+45), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

			mean_amb = ambient[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_amb = ambient[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'amb:%.1f+-%.1f'%(mean_amb,std_amb), (self.scale*bndbox[2], self.scale*bndbox[1]+70), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'amb:%.1f+-%.1f'%(mean_amb,std_amb), (self.scale*bndbox[2], self.scale*bndbox[1]+70), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

			mean_dep = depth[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].mean()
			std_dep = depth[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]].std()
			cv2.putText(self.img2show, 'dph:%.2f+-%.2f'%(mean_dep,std_dep), (self.scale*bndbox[2], self.scale*bndbox[1]+95), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (0,0,0), 3)
			cv2.putText(self.img2show, 'dph:%.2f+-%.2f'%(mean_dep,std_dep), (self.scale*bndbox[2], self.scale*bndbox[1]+95), cv2.FONT_HERSHEY_COMPLEX_SMALL, self.font_size, (255,255,255), 2)

		self.img2show = cv2.resize(self.img2show, (self.img2show.shape[1]//self.scale, self.img2show.shape[0]//self.scale))

		# cv2.imshow('',self.img2show)
		pcv.put_scale('', self.img2show, self.scale_min, self.scale_max)

		'''
		Section to handle keys
		'''
		key_pressed = cv2.waitKey(1000//60) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey() & 0xff
		if key_pressed in [27, ord('q')]:
			self.keep_going = False
		if key_pressed == ord('s'):
			self.pc = self.depth_processor.process(self.depth)
			pcv.pcshow(self.pc)
		if key_pressed == ord(','):
			self.select = 3 if self.select==0 else self.select-1
		if key_pressed == ord('.'):
			self.select = 0 if self.select==3 else self.select+1


if __name__=='__main__':
	a = gui()
	a.loop()