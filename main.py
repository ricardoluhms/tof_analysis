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
	pyplot.hist(bins[:-1],bins, weights=counts)
	pyplot.show()


if __name__=='__main__':
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp005/'
	# path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp013/'
	path = '/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp021/'
	pc_opener = PointCloud_opener(path+'PointCloud.bin',240,320);pc_opener.open()
	amp_opener = Amplitude_opener(path+'Amplitude.bin',240,320);amp_opener.open()
	phs_opener = Phase_opener(path+'Phase.bin',240,320);phs_opener.open()
	median_filter = mean_filter(240, 320, kernel_size=5)
	edge_filter = edge_filter(240, 320, kernel_size=5)
	while 1:
		ret, pc_data = pc_opener.read()
		amp_data = amp_opener.read()[1]
		phs_data = phs_opener.read()[1]
		if ret != True:
			break

		# pc_data[:,0] = z_filter.apply(pc_data[:,0])
		# pc_data[:,1] = z_filter.apply(pc_data[:,1])
		# pc_data[:,2] = median_filter.apply(pc_data[:,2])
		# pc_data[:,2] = edge_filter.apply(pc_data[:,2])


		mask = cv2.threshold(filter.norm(pc_data[:,-1].reshape((240,320))),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		depth = pc_data[:,2].reshape((240,320))
		pc_data[mask.ravel()==0,:] = 0

		a = amp_data[mask.ravel()==255]
		print(a.mean(), a.min(), a.max(), np.median(a), a.std()), exit()


		# error = (depth-1.013)
		# error = (depth-3.009)
		error = (depth-4.834)
		# error[np.bitwise_and(error >= -0.030, error <= 0.030)] = 0
		# print((error[mask==255]!=0).sum()/(mask==255).sum()*100)
		mean = depth[mask==255].mean(); std = depth[mask==255].std()
		measure = depth[np.bitwise_and(mask==255,np.bitwise_and(depth>=mean-2*std, depth<=mean+2*std))]
		print('measure:%f+-%f [m]'%(measure.mean(),measure.std()))
		print('  error:%f+-%f [m]'%(error[mask==255].mean(),error[mask==255].std()))

		
		# hist_per_error(phs_data.reshape((240,320)), error, mask)


		pcv.imshow('', pc_data)
		key_pressed = cv2.waitKey(33) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [ord('s')]:
			pcv.pcshow(pc_data)
			key_pressed = cv2.waitKey(0) & 0xff
		if key_pressed in [27, ord('q')]:
			break