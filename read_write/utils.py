import cv2,numpy as np

import os
def check_dir(dir):
	if dir[-1] == '/':
		dir = dir[:-1]
	dir_tree = dir.split('/')
	path = ''
	for d in dir_tree:
		if not os.path.isdir(path+d+'/'):
			os.mkdir(path+d+'/')
		path += d + '/'

class writer():
	def __init__(self,file_name,height=240,width=320,dtype='uint16'):
		self.f = open(file_name, 'wb')
		self.f.write(np.array(height, dtype='uint32').tobytes())
		self.f.write(b'\n')
		self.f.write(np.array(width, dtype='uint32').tobytes())
		self.f.write(b'\n')
		dtype += ' ' * (7-len(dtype))
		self.f.write(dtype.encode())
		self.f.write(b'\n')

	def write(self, frame):
		self.f.write(frame.tobytes())

	def release(self):
		self.f.flush()

class reader():
	def __init__(self,file_name):
		self.f = open(file_name, 'rb')
		self.f.seek(0, 2);file_size = self.f.tell();self.f.seek(0, 0)
		self.height, self.width, self.dtype, self.buffer = self.f.read(5*2+8).split(b'\n')
		self.height = np.frombuffer(self.height, dtype='uint32')[0]
		self.width = np.frombuffer(self.width, dtype='uint32')[0]
		self.dtype = self.dtype.decode().replace(' ', '')
		self.data_size = len(np.array(0,dtype=self.dtype).tobytes())
		self.frames_count = (file_size - self.f.tell())//(self.height*self.width*self.data_size)
		self.frame_counter = 0

	def read(self):
		data_buffer = self.f.read(self.height*self.width*self.data_size)
		frame = np.frombuffer(data_buffer, dtype=self.dtype).reshape((-1,1))
		self.frame_counter += 1
		ret = True if self.frame_counter < self.frames_count else False
		return ret, frame

	def release(self):
		self.f.flush()