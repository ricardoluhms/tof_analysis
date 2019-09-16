import cv2
import numpy as np
import open3d as o3d
from utils import *
from matplotlib import pyplot


if __name__=='__main__':
	with open('/media/vinicius/048A82A468318E17/datasets/tof/Experimentos_atenuadores/Exp007/PointCloud.bin', 'rb') as f:
		buffer = np.frombuffer(f.read(), dtype='float32').copy()
	buffer = buffer.reshape((-1,4))
	buffer = buffer[np.bitwise_and(np.bitwise_and(buffer[:,0] != 6, buffer[:,1] != 6),np.bitwise_and(buffer[:,2] != 6, buffer[:,3] != 6)), :]

	z_filter = median_filter(240, 320, kernel_size=3)

	pcd = o3d.geometry.PointCloud()
	cv2.namedWindow('', cv2.WINDOW_NORMAL)
	for i in range(len(buffer)//(320*240)):
		data = buffer[320*240*i:320*240*(i+1),:]
		x = data[:,0:1]
		y = data[:,1:2]
		z = data[:,2:3]
		a = data[:,3:4]
		# a[a >= a.mean()+10*a.std()] = 0

		# z = z_filter.apply(z)
		
		xyz = np.hstack([x,y,z])
		# xyz = xyz[np.bitwise_and(xyz[:,0]>=-1, xyz[:,0]<=1),:]
		# xyz = xyz[np.bitwise_and(xyz[:,1]>=-1, xyz[:,1]<=1),:]
		# xyz = xyz[np.bitwise_and(xyz[:,2]>= 1, xyz[:,2]<=2),:]
		# xyz = xyz[xyz[:,2]>0]
		

		# xyz[:,1] = -xyz[:,1]
		# xyz[:,2] = -xyz[:,2]
		# pcd.points = o3d.utility.Vector3dVector(xyz)

		depth = z.reshape((240,320)).copy()
		amplitude = a.reshape((240,320)).copy()
		x = filter.norm(x.reshape((240,320)),dtype='uint8')
		y = filter.norm(y.reshape((240,320)),dtype='uint8')
		z = filter.norm(z.reshape((240,320)),dtype='uint8')
		a = filter.norm(a.reshape((240,320)),dtype='uint8')
		mask = cv2.threshold(a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		# mask = np.zeros((240,320),dtype='uint8'); mask[49:83, 126:215] = 255
		
		xyz = xyz[mask.ravel()==255,:]
		xyz[:,1] = -xyz[:,1]
		xyz[:,2] = -xyz[:,2]
		pcd.points = o3d.utility.Vector3dVector(xyz)
		cv2.imshow('mask', mask)

		error = depth-1.013
		measure = depth[np.bitwise_and(mask==255,np.bitwise_and(depth>=depth[mask==255].mean()-2*depth[mask==255].std(), depth<=depth[mask==255].mean()+2*depth[mask==255].std()))].mean()
		
		counts, bins = np.histogram(error.ravel(), bins=np.arange((error.min()-1)*2, (error.max()+1)*2, dtype='int')/2)
		pyplot.hist(bins[:-1], bins, weights=counts)
		pyplot.show()
		# pyplot.title('VIIONxDOPPLER Deltas')
		# pyplot.xlabel('Deltas')
		# pyplot.ylabel('Counts')
		# pyplot.savefig('delta_histogram_no_abs.png');pyplot.clf()
		print(measure,error[mask==255].mean(), error[mask==255].std())

		
		cv2.imshow('', np.hstack([x,y,z,a]))
		key_pressed = cv2.waitKey(33) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey() & 0xff
		if key_pressed in [ord('s')]:
			o3d.visualization.draw_geometries([pcd])
		if key_pressed in [27, ord('q')]:
			break