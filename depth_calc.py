from filters.utils import *
from bin_opener.utils import *
from viewer.utils import *
from raw_data_processing.utils import *
import cv2, numpy as np

if __name__=='__main__':
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp001/'
	path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp005/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp013/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp021/'
	pc_opener = PointCloud_opener(path+'PointCloud.bin',240,320);pc_opener.open()
	amp_opener = Amplitude_opener(path+'Amplitude.bin',240,320);amp_opener.open()
	phs_opener = Phase_opener(path+'Phase.bin',240,320);phs_opener.open()
	phase_processor = phase2depth(240,320)
	cv2.namedWindow('delta', cv2.WINDOW_NORMAL)
	edfr = edge_filter(240,320)
	while 1:
		ret, pc_data = pc_opener.read()
		amp_data = amp_opener.read()[1]
		phs_data = phs_opener.read()[1]
		if ret != True:
			break

		amp_data[amp_data>amp_data.mean()+3*amp_data.std()]=amp_data.mean()+3*amp_data.std()
		depth_real = pc_data[:,2]
		depth_calc = phase_processor.process(phs_data)
		delta = np.abs(depth_real.ravel() - depth_calc.ravel())#.mean()
		print('depth_real:', depth_real.min(), depth_real.max(), depth_real.mean(), depth_real.std())
		print('depth_calc:', depth_calc.min(), depth_calc.max(), depth_calc.mean(), depth_calc.std())
		print('delta:', delta.min(), delta.max(), delta.mean(), delta.std())

		pcv.imshow('depth_real', depth_real)
		pcv.imshow('depth_calc', depth_calc)
		cv2.imshow('delta', heat_map(delta.reshape((240,320))))
		pcv.imshow('', pc_data)
		key_pressed = cv2.waitKey(33) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [ord('s')]:
			pcv.pcshow(pc_data)
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [27, ord('q')]:
			break