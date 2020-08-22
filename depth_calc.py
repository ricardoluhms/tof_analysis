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
	point_cloud_processor = phase2point_cloud(240,320)
	cv2.namedWindow('delta', cv2.WINDOW_NORMAL)
	edfr = edge_filter(240,320)
	while 1:
		ret, pc_data = pc_opener.read()
		amp_data = amp_opener.read()[1]
		phs_data = phs_opener.read()[1]
		if ret != True:
			break

		amp_data[amp_data>amp_data.mean()+3*amp_data.std()]=amp_data.mean()+3*amp_data.std()
		pc_calc = point_cloud_processor.process(phs_data)
		pc_calc[:,0:1] = edfr.apply(pc_calc[:,0:1])
		pc_calc[:,1:2] = edfr.apply(pc_calc[:,1:2])
		pc_calc[:,2:3] = edfr.apply(pc_calc[:,2:3])
		delta = np.abs(pc_data[:,:3] - pc_calc)

		pcv.imshow('delta', pc_calc)
		key_pressed = cv2.waitKey(33) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [ord('s')]:
			pcv.pcshow(pc_calc)
			# pcv.pcshow(pc_data)
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [27, ord('q')]:
			break