from filters.utils import *
from bin_opener.utils import *
from viewer.utils import *
import cv2, numpy as np


def hist_per_error(data, error, mask, pace=1):
	e = []
	for bin in range(data[mask==255].min(), data[mask==255].max()+1, pace):
		e_temp = error[np.bitwise_and(mask==255,np.bitwise_and(data>=bin, data<bin+1))].mean()
		e.append(0 if np.isnan(e_temp) else e_temp)
	e = np.array(e)
	from matplotlib import pyplot
	bins = np.arange(data[mask==255].min(), data[mask==255].max()+1, pace)
	pyplot.subplot(1,2,1)
	pyplot.hist(bins,bins, weights=e)
	pyplot.subplot(1,2,2)
	counts, bins = np.histogram(data[mask==255].ravel(), bins=np.arange((data[mask==255].min()-1)/pace, (data[mask==255].max()+1)/pace, dtype='int')*pace)
	pyplot.hist(bins[:-1], bins, weights=counts)
	pyplot.show()


if __name__=='__main__':
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp001/'
	path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp005/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp013/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp021/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp026/'
	# path='/media/vinicius/048A82A468318E17/datasets/tof/Exp_1_Pan_90_Tilt_0_ilum_y_reflex_n_dist_1512/'
	pc_opener = PointCloud_opener(path+'PointCloud.bin',240,320).open()
	amp_opener = Amplitude_opener(path+'Amplitude.bin',240,320).open()
	phs_opener = Phase_opener(path+'Phase.bin',240,320).open()


	edfr = edge_filter(240,320)
	while 1:
		ret, pc_data = pc_opener.read()
		amp_data = amp_opener.read()[1]
		phs_data = phs_opener.read()[1]
		if ret != True:
			break

		mask = cv2.threshold(filter.norm(amp_data.reshape((240,320))),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		depth = pc_data[:,2].reshape((240,320))
		# pc_data[mask.ravel()==0,:] = 0

		amp_data[amp_data>amp_data.mean()+3*amp_data.std()]=amp_data.mean()+3*amp_data.std()

		# ===============================================================
		# 			getting bndboxes and filtering by size
		# ===============================================================
		_, connec, stats, centroids = cv2.connectedComponentsWithStats(mask.reshape((240,320)))
		labels = np.unique(connec)
		bndboxes = np.zeros((labels.shape[0]-1, 4), dtype='uint16')
		for c in range(1,labels.shape[0]):
			coord_y, coord_x = np.where(connec==c)
			bndboxes[c-1,:] = np.array([coord_x.min(), coord_y.min(), coord_x.max(), coord_y.max()], dtype='uint16')
		bndboxes = bndboxes[(bndboxes[:,2]-bndboxes[:,0])*(bndboxes[:,3]-bndboxes[:,1])>100, :]

		# ===============================================================
		# 			measurement and error
		# ===============================================================
		error = (depth-1.013)
		# error = (depth-3.009)
		# error = (depth-4.834)
		mean = depth[mask!=0].mean(); std = depth[mask!=0].std()
		measure = depth[np.bitwise_and(mask!=0,np.bitwise_and(depth>=mean-2*std, depth<=mean+2*std))]
		print('measure:%f+-%f [m]'%(measure.mean(),measure.std()))
		print('  error:%f+-%f [m]'%(error[mask!=0].mean(),error[mask!=0].std()))

		# ===============================================================
		# 			drawing squares and computing error per square
		# ===============================================================
		z_bndboxes = filter.norm(cv2.cvtColor(pc_data[:,3].reshape((240,320)), cv2.COLOR_GRAY2BGR))
		pixels = 20
		for bndbox in bndboxes:
			z_bndboxes = cv2.rectangle(z_bndboxes, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), (255,0,0), 1)
			for y in range((bndbox[3]-bndbox[1])//pixels):
				y_0 = (y)*pixels + bndbox[1]
				y_1 = (y+1)*pixels + bndbox[1]
				pixels_error = []
				for x in range((bndbox[2]-bndbox[0])//pixels):
					x_0 = (x)*pixels + bndbox[0]
					x_1 = (x+1)*pixels + bndbox[0]
					pixels_mask = np.zeros(z_bndboxes.shape[:2], dtype='uint8')
					pixels_mask[y_0:y_1, x_0:x_1] = 1
					pixels_error.append(error[np.bitwise_and(mask!=0, pixels_mask!=0)].mean())
					# z_bndboxes = cv2.putText(z_bndboxes, '%.4f'%pixels_error, (x_0, y_0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0))
					z_bndboxes = cv2.rectangle(z_bndboxes, (x_0, y_0), (x_1, y_1), (255,0,0), 1)
				print(pixels_error)
				print('-'*90)
			print('*'*90)
		cv2.imshow('z_bndboxes', z_bndboxes)

		# hist_per_error(phs_data.reshape((240,320)), error, mask)

		# ===============================================================
		# 			showing results
		# ===============================================================
		pcv.imshow('', pc_data)
		pcv.imshow('a', amp_data)
		key_pressed = cv2.waitKey(33) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [ord('s')]:
			pcv.pcshow(pc_data)
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [27, ord('q')]:
			break