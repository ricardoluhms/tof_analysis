import cv2
import numpy as np
import open3d as o3d


class pcv():#point cloud viewer
	pcd = o3d.geometry.PointCloud()

	@staticmethod
	def	norm(frame, max=255, dtype='uint8'):
		frame = frame - frame.min()
		if frame.max() != 0:
			frame = max*(frame/frame.max())
		return frame.astype(dtype)

	@staticmethod
	def imshow(window_name, point_cloud, height=240, width=320):
		frames = []
		if len(point_cloud.shape) == 1:
			point_cloud = point_cloud.reshape((-1,1))
		for channel in range(point_cloud.shape[1]):
			frame = point_cloud[:,channel].reshape((height, width))
			frame = pcv.norm(frame)
			cv2.imshow(window_name, frame)
			frames.append(frame)
		frames = np.hstack(frames)
		cv2.imshow(window_name, frames)

	@staticmethod
	def pcshow(point_cloud, hide_zero_plane=True):
		if point_cloud.shape[1] >= 3:
			point_cloud = point_cloud[:,:3]
			if hide_zero_plane:
				point_cloud = point_cloud[point_cloud[:,2]>0,:]
			point_cloud[:,1] = -point_cloud[:,1]
			point_cloud[:,2] = -point_cloud[:,2]
			pcv.pcd.points = o3d.utility.Vector3dVector(point_cloud)
			o3d.visualization.draw_geometries([pcv.pcd])
		else:
			print('Not a point cloud')