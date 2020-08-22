from utils import *
import cv2, numpy as np


if __name__=='__main__':
	# opener = PointCloud_opener('/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp007/PointCloud.bin',240,320)
	# opener = Amplitude_opener('/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp007/Amplitude.bin',240,320)
	# opener = Depth_opener('/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp007/Depth.bin',240,320)
	opener = Phase_opener('/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp007/Phase.bin',240,320)
	opener.open()
	while 1:
		ret, data = opener.read()
		if ret != True:
			break
		# depth = data[:,2].reshape((240,320))
		depth = data.reshape((240,320))
		depth = depth/depth.max()

		cv2.imshow('', depth)
		key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [27, ord('q')]:
			break