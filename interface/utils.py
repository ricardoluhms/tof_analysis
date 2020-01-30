import cv2, numpy as np, os

class Data():
	def __init__(self, points=None):
		if type(points) == type([]):
			self.data = np.array(points)
		elif type(points) == type(np.array([])):
			self.data = points
		elif type(points) == type(None):
			self.data = np.array([])

	def append(self, points):
		if self.data.size == 0:
			if type(points) == type(Points):
				self.data = points.data
			elif type(points) == type(np.array([])):
				self.data = points
			elif type(points) == type([]):
				self.data = np.array(points)
		else:	
			if type(points) == type(Points):
				self.data = np.vstack([self.data, points.data])
			elif type(points) == type(np.array([])):
				self.data = np.vstack([self.data, points])
			elif type(points) == type([]):
				self.data = np.vstack([self.data, np.array(points)])

	def delete(self, idx):
		self.data = np.delete(self.data, idx, axis=0)

	def find(self, point):
		return np.where((self.data==point).all(axis=1))[0]


	def __iter__(self):
		self.i = 0
		return self

	def __next__(self):
		if self.data.size==0:
			raise StopIteration
		else:
			if self.i < self.data.shape[0]:
				r = self.data[self.i, :]
				self.i += 1
				return r
			else:
				raise StopIteration

	def is_empty(self):
		return False if self.data.size != 0 else True

	def delete_save(self, filename):
		if os.path.isfile(filename):
			os.remove(filename)

	def save(self, filename):
		if self.data.size != 0:
			with open(filename, 'wb') as f:
				f.write(('%d,%d,%s\n'%(self.data.shape[0], self.data.shape[1], str(self.data.dtype))).encode())
				f.write(self.data.tobytes())
		else:
			self.delete_save(filename)

	def load(self, filename):
		if os.path.isfile(filename):
			with open(filename, 'rb') as f:
				height, width, dtype = f.readline().decode().split(',')
				height, width, dtype = int(height), int(width), dtype[:-1]
				self.data = np.frombuffer(f.read(),dtype=dtype).reshape((height, width)).copy()

class Points(Data):
	def __init__(self, points=None):
		Data.__init__(self, points)

	def centers(self):
		return self.data[:,0:1+1]

	def draw(self, img, color=(255,0,0)):
		if self.data.size != 0:
			for point in self.centers():
				img = cv2.circle(img, (int(point[0]), int(point[1])), 1, color, -1)
		return img

class Bndboxes(Data):
	def __init__(self, points=None):
		Data.__init__(self, points)

	def centers(self):
		return self.data[:,4:5+1]

	def bndboxes(self):
		return self.data[:,0:3+1]

	def bndboxes_shape(self):
		return self.data[:,6:7+1]

	def draw(self, img, color=(255,0,0), thickness=2):
		if self.data.size != 0:
			for bndbox in self.bndboxes():
				img = cv2.rectangle(img, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), color, thickness)
		return img

	def draw_centers(self, img, color=(255,0,0)):
		for point in self.centers():
			img = cv2.circle(img, (int(point[0]), int(point[1])), 1, color, -1)
		return img

class mouse:
	point = np.array([0,0], dtype='uint16')
	first_click_point = np.array([0,0], dtype='uint16')
	right_click_point = np.array([0,0], dtype='uint16')
	first_click = False
	second_click = False
	right_click = False

def callback(event,x,y,flags,param):
	global m
	if event == cv2.EVENT_LBUTTONDOWN:
		pass
	if event == cv2.EVENT_LBUTTONUP:
		m.point = np.array([x,y], dtype='uint16')
		if m.first_click:
			m.first_click = False
			m.second_click = True
		else:
			m.first_click = True
			m.first_click_point = np.array([x,y], dtype='uint16')
	if m.first_click:
		m.point = np.array([x,y], dtype='uint16')

	if event == cv2.EVENT_RBUTTONUP:
		if m.right_click:
			m.right_click = False
		else:
			m.right_click = True
			m.right_click_point = np.array([x,y], dtype='uint16')

m = mouse()
class interface():
	'''
	Class to add click and drag function to create bndboxes. The get_frame method and main method should be overwritten
	by a child class in order to pass the frame and process the acquired information, also the loop method should be 
	used to run the code. Example bellow...
	'''
	def __init__(self,):
		self.keep_going = True
		self.key_pressed = None

		self.get_picture_callback = None
		self.main_callback = None

		self.img2show = None
		
		self.bndboxes = Bndboxes()
		self.points = Points()
		self.bndbox2insert = np.array([0,0,0,0])

	def get_frame(self):
		return None

	def main(self):
		pass

	def loop(self):
		cv2.namedWindow('', cv2.WINDOW_GUI_NORMAL)
		cv2.setMouseCallback('',callback)
		while self.keep_going:
			img = self.get_frame()
			self.img2show = img.copy()
			self.img2show = self.bndboxes.draw(self.img2show, color=(0,0,0), thickness=2)
			self.img2show = self.bndboxes.draw(self.img2show, color=(255,255,255), thickness=1)

			if m.first_click:
				self.bndbox2insert = np.hstack([m.first_click_point, m.point])
				self.bndbox2insert = np.array([self.bndbox2insert[np.array([0,2])].min(), self.bndbox2insert[np.array([1,3])].min(),
										self.bndbox2insert[np.array([0,2])].max(), self.bndbox2insert[np.array([1,3])].max()], dtype='uint16')
				self.img2show = cv2.rectangle(self.img2show, (self.bndbox2insert[0], self.bndbox2insert[1]), (self.bndbox2insert[2], self.bndbox2insert[3]), (0,0,0), 2)
				self.img2show = cv2.rectangle(self.img2show, (self.bndbox2insert[0], self.bndbox2insert[1]), (self.bndbox2insert[2], self.bndbox2insert[3]), (255,0,0), 1)
			
			if m.second_click:
				m.second_click = False
				self.bndboxes.append(np.expand_dims(self.bndbox2insert, axis=0))

			if m.right_click:
				m.right_click = False
				d=Data()
				for bndbox in self.bndboxes:
					if m.right_click_point[0] > bndbox[0] and m.right_click_point[0] < bndbox[2] and m.right_click_point[1] > bndbox[1] and m.right_click_point[1] < bndbox[3]:
						idx = self.bndboxes.find(bndbox)
						distance = np.sqrt(np.square(m.right_click_point-(0.5*(bndbox[2:]+bndbox[:2]))).sum())
						d.append(np.expand_dims(np.hstack([idx, distance]), 0))
				if d.data.size != 0:
					self.bndboxes.delete(d.data[np.argmin(d.data[:,1]),0])
			self.main()


class test(interface):
	def __init__(self):
		interface.__init__(self)

	def main(self):
		scale=3
		font=2
		self.img2show = cv2.resize(self.img2show, (self.img2show.shape[1]*scale, self.img2show.shape[0]*scale))
		for bndbox in self.bndboxes:
			cv2.putText(self.img2show, '%d'%(1234567890), (scale*bndbox[0], scale*bndbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, font, (0,0,0), 2)
			cv2.putText(self.img2show, '%d'%(1234567890), (scale*bndbox[0], scale*bndbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, font, (255,0,0), 1)
		self.img2show = cv2.resize(self.img2show, (self.img2show.shape[1]//scale, self.img2show.shape[0]//scale))
		cv2.imshow('',self.img2show)
		key_pressed = cv2.waitKey(1000//30) & 0xff
		if key_pressed in [27, ord('q')]:
			self.keep_going = False

def get_frame():
	return 255*np.ones((240,320,3), dtype='uint8')

if __name__=='__main__':
	a = test()
	a.get_frame = get_frame
	a.loop()