from filters.utils import *
from bin_opener.utils import *
from viewer.utils import *
from raw_data_processing.utils import *
import cv2, numpy as np


class callibrate():
	def __init__(self, height, width):
		self.k = np.array([
			[0,0,0],
			[0,0,0],
			])
		self.p = np.array([
			[0,0],
			])

		self.height = height
		self.width = width

		x, y = np.meshgrid(np.arange(0,self.width), np.arange(0,self.height))
		self.x = x.astype('int'); self.y = y.astype('int')
		x = (x - self.width / 2) 
		y = (y - self.height / 2)
		self.r = np.sqrt(np.square(x)+np.square(y))

		# radial correction
		self.x_remap = x*(1+self.k[0,0]*np.power(self.r, 2)+self.k[0,1]*np.power(self.r, 4)+self.k[0,2]*np.power(self.r, 6))/(1+self.k[1,0]*np.power(self.r, 2)+self.k[1,1]*np.power(self.r, 4)+self.k[1,2]*np.power(self.r, 6))
		self.y_remap = y*(1+self.k[0,0]*np.power(self.r, 2)+self.k[0,1]*np.power(self.r, 4)+self.k[0,2]*np.power(self.r, 6))/(1+self.k[1,0]*np.power(self.r, 2)+self.k[1,1]*np.power(self.r, 4)+self.k[1,2]*np.power(self.r, 6))		

		# tangencial correction
		self.x_remap += 2*self.p[0,0]*x*y + self.p[0,1]*(np.square(self.r)+2*np.square(x))
		self.y_remap += self.p[0,0]*(np.square(self.r)+2*np.square(y)) + 2*self.p[0,1]*x*y

		# make sense, but should be revised
		self.x_remap = (self.x_remap + self.width / 2)
		self.y_remap = (self.y_remap + self.height / 2)


		self.x_remap[self.x_remap<0] = 0
		self.y_remap[self.y_remap<0] = 0

		self.x_remap[self.x_remap>=self.width] = self.width-1
		self.y_remap[self.y_remap>=self.height] = self.height-1

		self.x_remap = self.x_remap.astype('int')
		self.y_remap = self.y_remap.astype('int')

	def apply(self, frame):
		frame = frame.reshape((self.height,self.width))
		output = np.zeros((self.height, self.width), dtype=frame.dtype)
		output[self.y_remap.ravel(), self.x_remap.ravel()] = frame[self.y.ravel(), self.x.ravel()]
		# output[self.y.ravel(), self.x.ravel()] = frame[self.y_remap.ravel(), self.x_remap.ravel()]
		output = output.reshape((-1,1))
		return output


experiments = np.array([
	[12,11,10, 9, 8,7.5, 7, 6, 5, 4, 3, 2, 1,0.5],#distances
	[ 1, 4, 7,10,13, 16,19,22,25,28,31,34,37, 40],#s/f
	[ 2, 5, 8,11,14, 17,20,23,26,29,32,35,38, 41],#f 1
	[ 3, 6, 9,12,15, 18,21,24,27,30,33,36,39, 42],#f 2
	])


if __name__=='__main__':
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp001/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp005/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp013/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp021/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp026/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Exp_0_Pan_90_Tilt_0_ilum_y_reflex_n_dist_1512/'
	path = '/media/vinicius/048A82A468318E17/datasets/tof/EXPERIMENTOS 2 TEXAS/Exp32/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimento_RUA_PUC/Filtro2/Filtro2_video1/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimento_RUA_PUC/Sem Filtro/Bateria_veiculos1_sem filtro/'


	pc_opener = PointCloud_opener(path+'PointCloud.bin',240,320).open()
	amp_opener = Amplitude_opener(path+'Amplitude.bin',240,320).open()
	phs_opener = Phase_opener(path+'Phase.bin',240,320).open()
	amb_opener = Ambient_opener(path+'Ambient.bin',240,320).open()

	z_filters = [
		# temporal_bilateral_filter(3),
		# median_filter(240,320,kernel_size=15),
		# bilateral_filter(240,320,kernel_size=15,std_color=15,std_space=15),
	]
	
	# phase_processor = phase2depth(240,320)
	phase_processor = phase2point_cloud(240,320)
	depth_processor = depth2point_cloud(240,320)

	cv2.namedWindow('pc_data', cv2.WINDOW_NORMAL)
	cv2.namedWindow('pc_calc', cv2.WINDOW_NORMAL)
	cv2.namedWindow('img', cv2.WINDOW_NORMAL)
	while 1:
		ret, pc_data = pc_opener.read()
		amp_data = amp_opener.read()[1]
		phs_data = phs_opener.read()[1]
		amb_data = amb_opener.read()[1]
		if ret != True:
			break


		amp_data[amp_data>amp_data.mean()+3*amp_data.std()] = amp_data.mean()+3*amp_data.std()
		pc_data[:,3][pc_data[:,3]>pc_data[:,3].mean()+3*pc_data[:,3].std()] = pc_data[:,3].mean()+3*pc_data[:,3].std()
		pc_calc = phase_processor.process(phs_data)

		# with open('pc_data%s.bin'%path[-3:-1], 'wb') as f:
		# 	f.write(pc_data.tobytes())
		# exit()

		# gt = np.where(experiments[1:,:] == int(path[-3:-1]))[1]
		# gt = experiments[0, gt]
		# pc_data[:,2] += phase_processor.R if gt >= 7.5 else 0
		# pc_calc[:,2] += phase_processor.R if gt >= 7.5 else 0

		pc_orig = pc_calc.copy()
		for f in z_filters:
			pc_calc[:, 2:3] = f.apply(pc_calc[:, 2:3])
		pc_calc[:, 0:3] = depth_processor.process(pc_calc[:,2:3])


		# mask = cv2.threshold(filter.norm(amp_data.reshape((240,320))),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		mask = (filter.norm(pc_data[:,3]) == 255).reshape((240,320)).astype('uint8')*255
		delta = (pc_calc[:,2:3] - pc_data[:,2:3])[mask.ravel()]
		print('delta:',delta.min(),delta.max(),delta.mean(),delta.std())
		print('measure0:', pc_data[mask.ravel()!=0,2].mean(), pc_data[mask.ravel()!=0,2].std())
		print('measure1:', pc_calc[mask.ravel()!=0,2].mean(), pc_calc[mask.ravel()!=0,2].std())


		# pc_calc[mask.ravel()==0,2] = 0
		# pc_data[mask.ravel()==0,2] = 0
		cv2.imshow('mask', mask)
		pc_data[:,0] -= 15
		pcs = np.vstack([pc_calc, pc_data[:,:3]])

		pcv.imshow('pc_data', pc_data)
		pcv.imshow('pc_calc', pc_calc)
		key_pressed = cv2.waitKey(33) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [ord('s')]:
			pcv.pcshow(pcs)
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [27, ord('q')]:
			break