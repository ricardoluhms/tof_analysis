import cv2, numpy as np


import subprocess as sp
import multiprocessing as mp
import time, os
class ti_tof():
	def __init__(self,mode,output_format):
		self.mode = mode
		self.output_format = output_format
		self.bufsize = None; self.get_bufsize()
		self.tsensor = None; self.tillum = None
		self.q_out = mp.Queue(10)
		self.p = mp.Process(target=ti_tof.capture, args=(self,self.q_out,))
		self.p.start()

	def get_bufsize(self):
		#calculating buffer size
		self.bufsize = 0
		for s in self.output_format:
			self.bufsize += 240*320*s[1]

	def capture(self,q_out,):
		#exe path
		folders = os.path.realpath(__file__).split('/')[:-1]
		path = ''
		for folder in folders:
			if folder == '':
				continue
			path = path + '/' + folder
		
		#calling c++ voxel sdk code
		p = sp.Popen([
			'%s/DepthCapture'%(path),
			'-v', '451',
			'-p', '9105',
			'-s', '12016791199903',
			'-f', 'none',
			'-n', '1',
			'-t', self.mode
		], stdout=sp.PIPE, bufsize = self.bufsize)
		# ], stdout=sp.PIPE, stderr=sp.PIPE, bufsize = self.bufsize)
		p.stdout.readline()
		p.stdout.readline()
		while 1:
			buf = p.stdout.read(self.bufsize+8)
			# print(len(buf))
			try:
				q_out.put(buf, timeout=0)
			except:
				_ = q_out.get()
				q_out.put(buf, timeout=0)

	def read(self,):
		try:
			buf = self.q_out.get(timeout=1)
		except:
			return False, [None]*len(self.output_format)
		begin = 0
		output = []
		info = np.frombuffer(buf[:8], dtype='uint32')
		if info.size == 0:
			return False, [None]*len(self.output_format)
		self.tsensor = info[0]
		self.tillum = info[1]
		buf = buf[8:]
		for s in self.output_format:
			end = 240*320*s[1]
			data = np.frombuffer(buf[begin:begin+end], dtype=s[0]).copy().reshape((-1,1))
			output.append(data)
			begin = begin+end
		return True, output

	def release(self,):
		self.p.terminate()

class depth(ti_tof):
	def __init__(self,):
		ti_tof.__init__(self, 'depth', [
			['float32', 4],
			['float32', 4],
		])

class raw_processed(ti_tof):
	def __init__(self,):
		ti_tof.__init__(self, 'raw_processed', [
			['uint16', 2],
			['uint16', 2],
			['uint8', 1],
			['uint8', 1],
		])

class point_cloud(ti_tof):
	def __init__(self,):
		ti_tof.__init__(self, 'pointcloud', [
			['float32', 4],
			['float32', 4],
			['float32', 4],
			['float32', 4],
		])

	def read(self,):
		try:
			buf = self.q_out.get(timeout=1)
		except:
			return False, [None, None, None]
		data = np.frombuffer(buf[:-self.output_format[0][1]*240*320], dtype=self.output_format[0][0]).copy().reshape((-1,3))
		if data.size == 0:
			return False, [None]*len(self.output_format)
		else:
			return True, [data[:,0:1], data[:,1:2], data[:,2:3]]