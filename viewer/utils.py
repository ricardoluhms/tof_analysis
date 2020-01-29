import cv2
import numpy as np
import open3d as o3d


class pcv():#point cloud viewer
	pcd = o3d.geometry.PointCloud()
	height = 240
	width = 320

	@staticmethod
	def set_shape(height, width):
		pcv.height = height
		pcv.width = width

	@staticmethod
	def	norm(frame, max=255, dtype='uint8'):
		frame = frame - frame.min()
		if frame.max() != 0:
			frame = max*(frame/frame.max())
		return frame.astype(dtype)

	@staticmethod
	def heat_map(img, norm=True):
		"""
		Create a heat map based on pixel intensity from purple to red function.
		"""
		if norm:
			img = img - np.min(img)
			img = np.multiply((6 * 256 - 1), np.divide(img, np.max(img)))
		blue = img.copy()
		green = img.copy()
		red = img.copy()

		#blue = blue
		green[green <= 1 * 256 - 1] = 0  
		#red = red

		blue[np.bitwise_and(blue > 1 * 256 - 1, blue <= 2 * 256 - 1)] = 255
		green[np.bitwise_and(green > 1 * 256 - 1, green <= 2 * 256 - 1)] = 0
		red[np.bitwise_and(red > 1 * 256 - 1, red <= 2 * 256 - 1)] = 2 * 256 - 1 - red[np.bitwise_and(red > 1 * 256 - 1, red <= 2 * 256 - 1)]

		blue[np.bitwise_and(blue > 2 * 256 - 1, blue <= 3 * 256 - 1)] = 255
		green[np.bitwise_and(green > 2 * 256 - 1, green <= 3 * 256 - 1)] = green[np.bitwise_and(green > 2 * 256 - 1, green <= 3 * 256 - 1)] - 2 * 256
		red[np.bitwise_and(red > 2 * 256 - 1, red <= 3 * 256 - 1)] = 0

		blue[np.bitwise_and(blue > 3 * 256 - 1, blue <= 4 * 256 - 1)] = 4 * 256 - 1 - blue[np.bitwise_and(blue > 3 * 256 - 1, blue <= 4 * 256 - 1)]
		green[np.bitwise_and(green > 3 * 256 - 1, green <= 4 * 256 - 1)] = 255
		red[np.bitwise_and(red > 3 * 256 - 1, red <= 4 * 256 - 1)] = 0

		blue[np.bitwise_and(blue > 4 * 256 - 1, blue <= 5* 256 - 1)] = 0 
		green[np.bitwise_and(green > 4 * 256 - 1, green <= 5 * 256 - 1)] = 255
		red[np.bitwise_and(red > 4 * 256 - 1, red <= 5 * 256 - 1)] = red[np.bitwise_and(red > 4 * 256 - 1, red <= 5 * 256 - 1)] - 4 * 256

		blue[np.bitwise_and(blue > 5 * 256 - 1, blue <= 6 * 256 - 1)] = 0 
		green[np.bitwise_and(green > 5 * 256 - 1, green <= 6 * 256 - 1)] = 6 * 256 - 1 - green[np.bitwise_and(green > 5 * 256 - 1, green <= 6 * 256 - 1)]
		red[np.bitwise_and(red > 5 * 256 - 1, red <= 6 * 256 - 1)] = 255

		h, w = img.shape[:2]
		img = np.zeros((h, w, 3), dtype='uint8')
		img[:, :, 0] = blue
		img[:, :, 1] = green
		img[:, :, 2] = red
		return img

	@staticmethod
	def imshow(window_name, point_cloud, heat_map=False, heat_map_norm=True):
		frames = []
		if len(point_cloud.shape) == 1:
			point_cloud = point_cloud.reshape((-1,1))
		for channel in range(point_cloud.shape[1]):
			frame = point_cloud[:,channel].reshape((pcv.height, pcv.width))
			if heat_map:
				frame = pcv.heat_map(frame, norm=heat_map_norm)
			else:
				frame = pcv.norm(frame)
			frames.append(frame)
		frames = np.hstack(frames)
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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

class cpcv():#continuous point cloud viewer
	pcd = o3d.geometry.PointCloud()
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	height = 240
	width = 320
	count = 0

	@staticmethod
	def pcshow(point_cloud, hide_zero_plane=True, count=1):
		if point_cloud.shape[1] >= 3:
			point_cloud = point_cloud[:,:3]
			if hide_zero_plane:
				point_cloud = point_cloud[point_cloud[:,2]>0,:]
			point_cloud[:,1] = -point_cloud[:,1]
			point_cloud[:,2] = -point_cloud[:,2]
			cpcv.pcd.points = o3d.utility.Vector3dVector(point_cloud)
			if cpcv.count < count:
				cpcv.vis.add_geometry(cpcv.pcd)
				cpcv.count += 1
			cpcv.vis.update_geometry()
			cpcv.vis.poll_events()
			cpcv.vis.update_renderer()
		else:
			print('Not a point cloud')