import cv2
import numpy as np
import open3d as o3d


def	norm(frame, max=255, dtype='uint8'):
	frame = frame - frame.min()
	if frame.max() != 0:
		frame = (max*frame/frame.max())
	return frame.astype(dtype)

def median_filter(frame, kernel_size):
	frame = frame.reshape((240, 320))
	frame_min = frame.min()
	frame = frame-frame_min
	frame_max = frame.max()
	frame = cv2.medianBlur(norm(frame, dtype='uint8'), kernel_size)
	frame = frame_max*frame.astype('float32')/255+frame_min
	frame = frame.reshape((-1,1))
	return frame


def mean_filter(frame, kernel_size):
	frame = frame.reshape((240, 320))
	frame_min = frame.min()
	frame = frame-frame_min
	frame_max = frame.max()
	frame = cv2.blur(norm(frame, dtype='uint8'), (kernel_size,kernel_size))
	frame = frame_max*frame.astype('float32')/255+frame_min
	frame = frame.reshape((-1,1))
	return frame


def gaussian_filter(frame, kernel_size, std=0):
	frame = frame.reshape((240, 320))
	frame = cv2.GaussianBlur(frame,(kernel_size, kernel_size),std)
	frame = frame.reshape((-1,1))
	return frame


def bilateral_filter(frame, kernel_size, std=75):
	frame = frame.reshape((240, 320))
	frame = cv2.bilateralFilter(frame,kernel_size,std,std)
	frame = frame.reshape((-1,1))
	return frame


def upsample(frame, factor=2):
	frame = frame.reshape((240, 320))
	frame = cv2.resize(frame, None, fx=factor, fy=factor)
	frame = frame.reshape((-1,1))
	return frame


def edge(frame, kernel_size):
	lap_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
	frame = frame.reshape((240, 320))
	edge = cv2.filter2D(frame, -1, lap_kernel)
	edge = norm(edge)
	edge = cv2.threshold(edge,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size)))
	# edge = cv2.erode(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size-2,kernel_size-2)))
	# cv2.imshow('edge', edge)
	edge = edge.reshape((-1,1))
	return edge/edge.max()


def meanshift_3d(pcd):
	threshold_euclidian_distance = 0.15
	threshold_new_euclidian_distance = 0.05
	feature_dimension = 3
	feature_space = pcd
	mean = np.zeros((feature_dimension),dtype='float32')
	mean = feature_space[np.random.randint(feature_space.shape[0]), :]
	new_mean = np.zeros((feature_dimension),dtype='float32')
	while 1:	
		euclidian_distance = np.sqrt(np.sum(np.square(np.subtract(mean, feature_space)), axis=1))
		clusters = feature_space[euclidian_distance < threshold_euclidian_distance, :]
		non_clusters = feature_space[euclidian_distance >= threshold_euclidian_distance, :]
		new_mean = np.mean(clusters, axis=0)
		new_euclidian_distance = np.sqrt(np.sum(np.square(np.subtract(new_mean, mean))))
		if new_euclidian_distance < threshold_new_euclidian_distance:
			print(non_clusters.shape)
			if clusters.shape[0] > 10:
				pass
				# pcd[
				# 	np.bitwise_and(
				# 		np.bitwise_and(
				# 			np.bitwise_and(pcd[:,0]>=clusters[:,0].min(), pcd[:,0]<=clusters[:,0].max()),
				# 			np.bitwise_and(pcd[:,1]>=clusters[:,1].min(), pcd[:,1]<=clusters[:,1].max())
				# 				),
				# 		np.bitwise_and(pcd[:,2]>=clusters[:,2].min(), pcd[:,2]<=clusters[:,2].max()))
				# 	, 2] = new_mean[2]
			else:
				pcd[
					np.bitwise_and(
						np.bitwise_and(
							np.bitwise_and(pcd[:,0]>=clusters[:,0].min(), pcd[:,0]<=clusters[:,0].max()),
							np.bitwise_and(pcd[:,1]>=clusters[:,1].min(), pcd[:,1]<=clusters[:,1].max())
								),
						np.bitwise_and(pcd[:,2]>=clusters[:,2].min(), pcd[:,2]<=clusters[:,2].max()))
					, 2] = 0
			feature_space = non_clusters
			if feature_space.shape[0] == 0:
				break
			mean = feature_space[0,:]
		else:
			mean = new_mean
	return pcd



class analog_filter():
	def __init__(self, history_size=15, fps=30):
		self.history_size = history_size
		self.fps = fps
		self.time_constant = 1e-1
		self.frame_number = 0
		self.history = np.zeros((240, 320, self.history_size), dtype='float32')
		self.frame_temp = np.zeros((240, 320), dtype='float32')
		self.g = np.flip(1/self.time_constant*(np.exp((-np.arange(0,self.history_size)/self.fps)/(self.time_constant))), 0)
		# print(self.g)


	def apply(self, frame):
		if self.frame_number < self.history_size:
			self.history[:,:,self.frame_number] = frame
			self.frame_number += 1
			self.frame_temp = (self.history*np.flip(self.g,0)/self.fps).sum(2)
		else:
			self.history[:,:,:-1] = self.history[:,:,1:]
			self.history[:,:,-1] = frame
			self.frame_temp = (self.history*self.g/self.fps).sum(2)
		return self.frame_temp/1.1759088243857097


class temporal_filter():
	def __init__(self, history_size):
		self.frame_number = 0
		self.history_size = history_size
		self.history = np.zeros((240, 320, self.history_size), dtype='float32')
		self.frame_temp = np.zeros((240, 320), dtype='float32')


	def apply(self, frame):
		if self.frame_number < self.history_size:
			self.history[:,:,self.frame_number] = frame
			self.frame_number += 1
		else:
			# mean = self.history.mean(axis=2)
			# std = self.history.std(axis=2)
			# indexes = np.bitwise_and(
			# 		frame>(mean+2*std),
			# 		frame<(mean-2*std))
			# frame[indexes] = mean[indexes]#np.median(self.history, axis=2)[indexes]
			self.history[:,:,:-1] = self.history[:,:,1:]
			self.history[:,:,-1] = frame
		self.frame_temp[:,:] = np.median(self.history[:,:,:self.frame_number+1], axis=2)
		return self.frame_temp


	def bgsb(self, frame):
		frame = cv2.absdiff(frame, self.frame_temp)
		frame = norm(frame, dtype='uint8')
		frame = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		return (frame/frame.max()).astype('uint8')


if __name__ == '__main__':
	# sampling = 1/30;time = 10;T = 1e-1
	# t = np.arange(0,time,sampling);x = np.ones((int(time/sampling),));g = 1/T*np.flip(np.exp(-t/T),0);y = 0
	# for i in range(1,int(time/sampling)):
	# 	y = (x[:i]*g[-i:]*sampling).sum()
	# print(y)
	# exit()
	with open('/media/vinicius/048A82A468318E17/datasets/tof/Exp_1_Pan_90_Tilt_0_ilum_y_reflex_n_dist_1512/PointCloud.bin', 'rb') as f:
	# with open('/media/vinicius/048A82A468318E17/datasets/tof/Exp_2_Pan_90_Tilt_0_ilum_y_reflex_y_dist_1512/PointCloud.bin', 'rb') as f:
	# with open('/media/vinicius/048A82A468318E17/datasets/tof/Exp_7_Pan_5_Tilt_0_ilum_y_reflex_n_dist_1161/PointCloud.bin', 'rb') as f:
	# with open('/media/vinicius/048A82A468318E17/datasets/tof/Exp_9_Pan_5_Tilt_0_ilum_y_reflex_y_dist_1161/PointCloud.bin', 'rb') as f:
		buffer = np.frombuffer(f.read(), dtype='float32').copy()
	buffer = buffer.reshape((-1,4))
	buffer = buffer[np.bitwise_and(np.bitwise_and(buffer[:,0] != 6, buffer[:,1] != 6),np.bitwise_and(buffer[:,2] != 6, buffer[:,3] != 6)), :]

	mask = np.zeros((240, 320), dtype='float32')
	size = 10
	mask[size:-size,size:-size] = 1
	# mask[120-size:120+size,160-size:160+size] = 1
	# mask[120-size:120+size,1] = 1
	# mask[:,160-size:160+size] = 1

	n = 5
	# xt = temporal_filter(n)
	# yt = temporal_filter(n)
	# zt = temporal_filter(n)
	# at = temporal_filter(n)
	xt = analog_filter()
	yt = analog_filter()
	zt = analog_filter()
	at = analog_filter()


	pcd = o3d.geometry.PointCloud()
	# vis = o3d.visualization.Visualizer()
	# vis.create_window()
	

	cv2.namedWindow('', cv2.WINDOW_NORMAL)
	for i in range(len(buffer)//(320*240)):
		# print(i)
		data = buffer[320*240*i:320*240*(i+1),:]
		x = data[:,0:1]
		y = data[:,1:2]
		z = data[:,2:3]
		a = data[:,3:4]
		a[a >= a.mean()+3*a.std()] = 0


		kernel_size = 3; std = 5
		# x = gaussian_filter(x, kernel_size, std=std)
		# y = gaussian_filter(y, kernel_size, std=std)
		# z = gaussian_filter(z, kernel_size, std=std)
		# x = bilateral_filter(x, kernel_size, std=std)
		# y = bilateral_filter(y, kernel_size, std=std)
		# z = bilateral_filter(z, kernel_size, std=std)
		# x = median_filter(x, 3)
		# y = median_filter(y, 3)
		# z = median_filter(z, 3)
		# x_fltrd = xt.apply(x.reshape((240, 320)))
		# y_fltrd = yt.apply(y.reshape((240, 320)))
		# z_fltrd = zt.apply(z.reshape((240, 320)))
		# x = x_fltrd.reshape((-1,1))
		# y = y_fltrd.reshape((-1,1))
		# z = z_fltrd.reshape((-1,1))


		# fshift = np.fft.fftshift(np.fft.fft2(x.reshape((240,320))))
		# x = np.fft.ifft2(np.fft.ifftshift(mask*fshift))
		# x = x.reshape((-1,1))
		# fshift = np.fft.fftshift(np.fft.fft2(y.reshape((240,320))))
		# y = np.fft.ifft2(np.fft.ifftshift(mask*fshift))
		# y = y.reshape((-1,1))
		# fshift = np.fft.fftshift(np.fft.fft2(z.reshape((240,320))))
		# z = np.fft.ifft2(np.fft.ifftshift(mask*fshift))
		# z = z.reshape((-1,1))


		# kernel_size = 3
		# x_edge_mask = edge(x, kernel_size)
		# z = z*np.abs(x_edge_mask-1)
		# y_edge_mask = edge(y, kernel_size)
		# z = z*np.abs(y_edge_mask-1)
		# z_edge_mask = edge(z, kernel_size)
		# z = z*np.abs(z_edge_mask-1)

		
		xyz = np.hstack([x,y,z])

		
		# y_edge_mask = edge(y, 5)
		# z = z*np.abs(y_edge_mask-1)
		# z_edge_mask = edge(z, 5)
		# z = z*np.abs(z_edge_mask-1)
		# xyz_outlier = xyz[np.hstack([x,y,z])[:,2]==0,:]
		# xyz = xyz[np.hstack([x,y,z])[:,2]>0,:]

		# xyz = meanshift_3d(xyz.copy())
		# xyz_outlier = xyz[xyz_new[:,2]==0,:]
		# xyz = xyz_new

		# outlier = o3d.geometry.PointCloud()
		# xyz_outlier[:,1] = -xyz_outlier[:,1]
		# xyz_outlier[:,2] = -xyz_outlier[:,2]
		# outlier.points = o3d.utility.Vector3dVector(xyz_outlier)
		# outlier.paint_uniform_color([1, 0, 0])


		# xyz = xyz[np.bitwise_and(xyz[:,0]>=-1, xyz[:,0]<=1),:]
		# xyz = xyz[np.bitwise_and(xyz[:,1]>=-1, xyz[:,1]<=1),:]
		# xyz = xyz[np.bitwise_and(xyz[:,2]>= 1, xyz[:,2]<=2.5),:]
		xyz = xyz[xyz[:,2]>0]
		

		xyz[:,1] = -xyz[:,1]
		xyz[:,2] = -xyz[:,2]
		pcd.points = o3d.utility.Vector3dVector(xyz)
		# pcd.paint_uniform_color([0.8, 0.8, 0.8])


		# vis.add_geometry(pcd)
		# vis.update_geometry()
		# vis.poll_events()
		# vis.update_renderer()

		
		x = norm(x.reshape((240,320)),dtype='uint8')
		y = norm(y.reshape((240,320)),dtype='uint8')
		z = norm(z.reshape((240,320)),dtype='uint8')
		a = norm(a.reshape((240,320)),dtype='uint8')
		cv2.imshow('', np.hstack([x,y,z,a]))
		key_pressed = cv2.waitKey(33) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey() & 0xff
		if key_pressed in [ord('s')]:
			# cl, ind = o3d.geometry.statistical_outlier_removal(pcd,nb_neighbors=40,std_ratio=2.0)
			# inlier_cloud = o3d.geometry.select_down_sample(pcd, ind)
			# outlier_cloud = o3d.geometry.select_down_sample(pcd, ind, invert=True)
			# outlier_cloud.paint_uniform_color([1, 0, 0])
			# inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
			# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])			
			o3d.visualization.draw_geometries([pcd])
			# o3d.visualization.draw_geometries([pcd, outlier])
			# vis = o3d.visualization.Visualizer()
			# vis.create_window()
		if key_pressed in [27, ord('q')]:
			break
