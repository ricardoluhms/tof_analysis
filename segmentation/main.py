import cv2
import numpy as np
import open3d as o3d


def	norm(frame, max=255, dtype='uint8'):
	frame = frame - frame.min()
	if frame.max() != 0:
		frame = (max*frame/frame.max())
	return frame.astype(dtype)


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


class bndboxes_tracker():
	def __init__(self):
		self.tracking_th = 0.5
		self.bndboxes_old = None
		self.ids = None
		self.last_id = 0
		

	def track(self, bndboxes_new):
		if type(self.bndboxes_old) != type(None):
			i, ii = np.meshgrid(np.arange(0,self.bndboxes_old.shape[0]), np.arange(0,bndboxes_new.shape[0]))
			x_A = np.hstack([bndboxes_new[ii.ravel(), 0:0+1], self.bndboxes_old[i.ravel(), 0:0+1]]).max(1).reshape(i.shape)
			y_A = np.hstack([bndboxes_new[ii.ravel(), 1:1+1], self.bndboxes_old[i.ravel(), 1:1+1]]).max(1).reshape(i.shape)
			x_B = np.hstack([bndboxes_new[ii.ravel(), 2:2+1], self.bndboxes_old[i.ravel(), 2:2+1]]).min(1).reshape(i.shape)
			y_B = np.hstack([bndboxes_new[ii.ravel(), 3:3+1], self.bndboxes_old[i.ravel(), 3:3+1]]).min(1).reshape(i.shape)
			inter_w = x_B-x_A
			inter_h = y_B-y_A
			inter_area = inter_w*inter_h
			no_inter_area = np.bitwise_or(
				bndboxes_new[ii.ravel(), 2:2+1] <= self.bndboxes_old[i.ravel(), 0:0+1],
				bndboxes_new[ii.ravel(), 0:0+1] >= self.bndboxes_old[i.ravel(), 2:2+1])
			no_inter_area = np.bitwise_or(no_inter_area,np.bitwise_or(
				bndboxes_new[ii.ravel(), 3:3+1] <= self.bndboxes_old[i.ravel(), 1:1+1],
				bndboxes_new[ii.ravel(), 1:1+1] >= self.bndboxes_old[i.ravel(), 3:3+1]))
			no_inter_area = no_inter_area.reshape(i.shape)
			inter_area[no_inter_area] = 0
			area_A = (bndboxes_new[:,2]-bndboxes_new[:,0])*(bndboxes_new[:,3]-bndboxes_new[:,1])
			area_B = (self.bndboxes_old[:,2]-self.bndboxes_old[:,0])*(self.bndboxes_old[:,3]-self.bndboxes_old[:,1])
			union_area = (area_A[ii.ravel()]+area_B[i.ravel()]).reshape(i.shape)-inter_area
			iou = inter_area / union_area
			ids = np.zeros((iou.shape[0],), dtype='uint16')

			if iou.shape[0]!=0 and iou.shape[1]==0:
				ids = np.arange(self.last_id+1,self.last_id+1+bndboxes_new.shape[0], dtype='uint16').reshape((-1,1))
				self.last_id = ids.max()
			elif iou.shape[0]!=0 and iou.shape[1]!=0:
				for k in range(iou.shape[0]):
					if iou[k,:].max()>=self.tracking_th:
						idx = np.argmax(iou[k,:])
						ids[k] = self.ids[idx,0]
						iou[:,idx] = 0
					else:
						ids[k] = self.last_id+1#np.max([ids.max(),self.ids.max()])+1
						self.last_id+=1

			self.ids = ids.reshape((-1,1))
		else:
			self.ids = np.arange(0,bndboxes_new.shape[0], dtype='uint16').reshape((-1,1))
		self.bndboxes_old = bndboxes_new
		return self.ids



class background_subtraction():
	def __init__(self, history_size, sampling_freq):
		self.detection_th = 0.001
		self.sampling_freq = sampling_freq
		self.frames_number = 0
		self.frames_saved = 0
		self.history_size = history_size
		self.history = np.zeros((240, 320, self.history_size), dtype='float32')
		self.background = np.zeros((240, 320), dtype='float32')
		self.foreground = np.zeros((240, 320), dtype='float32')


	def apply(self, frame):
		frame = frame.reshape((240, 320))
		if self.frames_number % self.sampling_freq == 0:
			if self.frames_saved < self.history_size:
				self.history[:,:,self.frames_saved] = frame
				self.frames_saved += 1
			else:
				self.history[:,:,:-1] = self.history[:,:,1:]
				self.history[:,:,-1] = frame
		self.background[:,:] = np.median(self.history[:,:,:self.frames_saved+1], axis=2)
		self.foreground = cv2.absdiff(frame, self.background)
		self.foreground = norm(self.foreground, dtype='uint8')
		self.foreground = cv2.threshold(self.foreground,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		# cv2.imshow('fg', foreground)
		self.foreground = (self.foreground/self.foreground.max()).astype('uint8')
		self.frames_number += 1
		return self.foreground.reshape((-1,1))


	def set_foreground(self, mask):
		self.foreground = mask


	def get_foreground_bndboxes(self):
		# _, connec = cv2.connectedComponents(fg.reshape((240,320)))
		_, connec, stats, centroids = cv2.connectedComponentsWithStats(fg.reshape((240,320)))
		labels = np.unique(connec)
		bndboxes = np.zeros((labels.shape[0]-1, 4), dtype='uint16')
		for c in range(1,labels.shape[0]):
			coord_y, coord_x = np.where(connec==c)
			bndboxes[c-1,:] = np.array([coord_x.min(), coord_y.min(), coord_x.max(), coord_y.max()], dtype='uint16')
			# bndboxes[c-1,:] = np.array([coord_x.min(), coord_y.min(), coord_x.max(), coord_y.max(), centroids[c,0], centroids[c,1]], dtype='uint16')
		bndboxes = bndboxes[(bndboxes[:,2]-bndboxes[:,0])*(bndboxes[:,3]-bndboxes[:,1]) > 240*320*self.detection_th,:]
		return bndboxes


	def get_foreground_position(self,x,y,z):
		_, connec, stats, centroids = cv2.connectedComponentsWithStats(fg.reshape((240,320)))
		labels = np.unique(connec)
		position = np.zeros((labels.shape[0]-1, 3), dtype='float32')
		bndboxes = np.zeros((labels.shape[0]-1, 4), dtype='uint16')
		for c in range(1,labels.shape[0]):
			coord_y, coord_x = np.where(connec==c)
			bndboxes[c-1,:] = np.array([coord_x.min(), coord_y.min(), coord_x.max(), coord_y.max()], dtype='uint16')
			x_masked = x*(connec==c)
			y_masked = y*(connec==c)
			z_masked = z*(connec==c)
			x_mean = x_masked[x_masked!=0].mean()
			y_mean = y_masked[y_masked!=0].mean()
			z_mean = z_masked[z_masked!=0].mean()
			x_std = x_masked[x_masked!=0].std()
			y_std = y_masked[y_masked!=0].std()
			z_std = z_masked[z_masked!=0].std()
			position[c-1,0] = np.mean(x_masked[np.bitwise_and(x_masked>=x_mean-2*x_std, x_masked<=x_mean+2*x_std)])
			position[c-1,1] = np.mean(y_masked[np.bitwise_and(y_masked>=y_mean-2*y_std, y_masked<=y_mean+2*y_std)])
			position[c-1,2] = np.mean(z_masked[np.bitwise_and(z_masked>=z_mean-2*z_std, z_masked<=z_mean+2*z_std)])
			# position[c-1,0] = np.mean(x[connec==c])
			# position[c-1,1] = np.mean(y[connec==c])
			# position[c-1,2] = np.mean(z[connec==c])
			# position[c-1,0] = np.median(x[connec==c])
			# position[c-1,1] = np.median(y[connec==c])
			# position[c-1,2] = np.median(z[connec==c])
			# position[c-1,0] = x[centroids[c-1,1].astype('uint16'),centroids[c-1,0].astype('uint16')]
			# position[c-1,1] = y[centroids[c-1,1].astype('uint16'),centroids[c-1,0].astype('uint16')]
			# position[c-1,2] = z[centroids[c-1,1].astype('uint16'),centroids[c-1,0].astype('uint16')]
		position = position[(bndboxes[:,2]-bndboxes[:,0])*(bndboxes[:,3]-bndboxes[:,1]) > 240*320*self.detection_th,:]
		return position


if __name__ == '__main__':
	with open('/media/vinicius/048A82A468318E17/datasets/tof/Exp_1_Pan_90_Tilt_0_ilum_y_reflex_n_dist_1512/PointCloud.bin', 'rb') as f:
	# with open('/media/vinicius/048A82A468318E17/datasets/tof/Exp_2_Pan_90_Tilt_0_ilum_y_reflex_y_dist_1512/PointCloud.bin', 'rb') as f:
	# with open('/media/vinicius/048A82A468318E17/datasets/tof/Exp_7_Pan_5_Tilt_0_ilum_y_reflex_n_dist_1161/PointCloud.bin', 'rb') as f:
	# with open('/media/vinicius/048A82A468318E17/datasets/tof/Exp_9_Pan_5_Tilt_0_ilum_y_reflex_y_dist_1161/PointCloud.bin', 'rb') as f:
		buffer = np.frombuffer(f.read(), dtype='float32').copy()
	buffer = buffer.reshape((-1,4))
	buffer = buffer[np.bitwise_and(np.bitwise_and(buffer[:,0] != 6, buffer[:,1] != 6),np.bitwise_and(buffer[:,2] != 6, buffer[:,3] != 6)), :]


	pcd = o3d.geometry.PointCloud()
	pcd_bg = o3d.geometry.PointCloud()
	# vis = o3d.visualization.Visualizer()
	# vis.create_window()


	fgbg = background_subtraction(10, 6)
	fgbg2 = background_subtraction(10, 6)
	trkr = bndboxes_tracker()
	
	car = []
	position_old = np.zeros((0,3))
	ids_old = np.zeros((0,1))

	cv2.namedWindow('', cv2.WINDOW_NORMAL)
	cv2.namedWindow('z_bndboxes', cv2.WINDOW_NORMAL)
	for i in range(len(buffer)//(320*240)):
		data = buffer[320*240*i:320*240*(i+1),:]
		x = data[:,0:1]
		y = data[:,1:2]
		z = data[:,2:3]
		a = data[:,3:4]
		a[a >= a.mean()+4*a.std()] = 0

		
		# fg = fgbg.apply(z)
		fg = np.bitwise_or(fgbg.apply(z), fgbg2.apply(a))

		kernel_size = 3
		y_edge_mask = edge(y, kernel_size)
		z = z*np.abs(y_edge_mask-1)
		z_edge_mask = edge(z, kernel_size)
		z = z*np.abs(z_edge_mask-1)
		fg[z==0]=0
		# fg = cv2.dilate(fg.reshape((240,320)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).reshape((-1,1))
		fg = cv2.erode(fg.reshape((240,320)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))).reshape((-1,1))
		fg = cv2.dilate(fg.reshape((240,320)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).reshape((-1,1))
		
		
		fgbg.set_foreground(fg)
		bndboxes = fgbg.get_foreground_bndboxes()
		position = fgbg.get_foreground_position(x.reshape(-1,320), y.reshape(-1,320), z.reshape(-1,320))
		ids = trkr.track(bndboxes)
		if ids.shape[0]!=0:
			for idx in range(ids.max()+1):
				if (ids[:,0]==idx).sum()!=0:
					delta_s = (np.sqrt(np.square(position[ids[:,0]==idx,:] - position_old[ids_old[:,0]==idx,:]).sum()))*30
					print(idx, delta_s)
					if idx==29 and delta_s>0:
						if len(car)>5:
							if delta_s > np.mean(car) - 2*np.std(car) and delta_s < np.mean(car) + 2*np.std(car):
								car.append(delta_s)
						else:
							car.append(delta_s)
		print(np.mean(car), np.std(car), np.median(car))
		position_old = position
		ids_old = ids
		print('='*90)


		xyz_bg = np.hstack([x*(1-fg),y*(1-fg),z*(1-fg)])
		# xyz_bg = xyz_bg[np.bitwise_and(xyz_bg[:,0]>-1, xyz_bg[:,0]<1)]
		# xyz_bg = xyz_bg[np.bitwise_and(xyz_bg[:,1]>-1, xyz_bg[:,1]<1)]
		# xyz_bg = xyz_bg[np.bitwise_and(xyz_bg[:,2]>1, xyz_bg[:,2]<2.5)]
		xyz_bg = xyz_bg[xyz_bg[:,2]>0]
		xyz_bg[:,1] = -xyz_bg[:,1]
		xyz_bg[:,2] = -xyz_bg[:,2]
		pcd_bg.points = o3d.utility.Vector3dVector(xyz_bg)
		pcd_bg.paint_uniform_color([0.8, 0.8, 0.8])


		xyz = np.hstack([x*fg,y*fg,z*fg])
		# xyz = np.hstack([x,y,z])
		# xyz = xyz[np.bitwise_and(xyz[:,0]>-1, xyz[:,0]<1)]
		# xyz = xyz[np.bitwise_and(xyz[:,1]>-1, xyz[:,1]<1)]
		# xyz = xyz[np.bitwise_and(xyz[:,2]>1, xyz[:,2]<2.5)]
		xyz = xyz[xyz[:,2]>0]
		xyz[:,1] = -xyz[:,1]
		xyz[:,2] = -xyz[:,2]
		pcd.points = o3d.utility.Vector3dVector(xyz)
		pcd.paint_uniform_color([1, 0, 0])


		# vis.add_geometry(pcd)
		# vis.add_geometry(pcd_bg)
		# vis.update_geometry()
		# vis.poll_events()
		# vis.update_renderer()

		
		x = norm(x.reshape((240,320)),dtype='uint8')
		y = norm(y.reshape((240,320)),dtype='uint8')
		z = norm(z.reshape((240,320)),dtype='uint8')
		a = norm(a.reshape((240,320)),dtype='uint8')
		fg = norm(fg.reshape((240,320)),dtype='uint8')
		
		z_bndboxes = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
		for bndbox in np.hstack([bndboxes, ids]):
			z_bndboxes = cv2.rectangle(z_bndboxes, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), (255,0,0), 1)
			z_bndboxes = cv2.putText(z_bndboxes, '%d'%bndbox[4], (bndbox[0], bndbox[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0))
		cv2.imshow('z_bndboxes', z_bndboxes)

		cv2.imshow('', np.hstack([x,y,z,a,fg]))
		key_pressed = cv2.waitKey(33) & 0xff
		if key_pressed in [32, ord('p')]:
			key_pressed = cv2.waitKey() & 0xff
		if key_pressed in [ord('s')]:
			o3d.visualization.draw_geometries([pcd, pcd_bg])
		if key_pressed in [27, ord('q')]:
			break