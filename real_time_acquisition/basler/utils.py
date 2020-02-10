import cv2, numpy as np


import subprocess as sp
import multiprocessing as mp
import time, os
class basler_tof():
	def __init__(self,output_format):
		self.output_format = output_format
		self.bufsize = None; self.get_bufsize()
		self.tsensor = None; self.tillum = None
		self.q_out = mp.Queue(10)
		self.p = mp.Process(target=basler_tof.capture, args=(self,self.q_out,))
		self.p.start()

	def get_bufsize(self):
		#calculating buffer size
		self.bufsize = 0
		for s in self.output_format:
			self.bufsize += 640*480*s[1]

	def capture(self,q_out,):
		#exe path
		folders = os.path.realpath(__file__).split('/')[:-1]
		path = ''
		for folder in folders:
			if folder == '':
				continue
			path = path + '/' + folder
		
		#calling c++ basler sdk code
		p = sp.Popen([
			'%s/DepthCapture'%(path),
		], stdout=sp.PIPE, bufsize = self.bufsize)
		# ], stdout=sp.PIPE, stderr=sp.PIPE, bufsize = self.bufsize)
		while 1:
			buf = p.stdout.read(self.bufsize)
			try:
				q_out.put(buf, timeout=0)
			except:
				_ = q_out.get()
				q_out.put(buf, timeout=0)

	def read(self,):
		try:
			buf = self.q_out.get(timeout=10)
		except:
			return False, [None]*len(self.output_format)
		begin = 0
		output = []
		for s in self.output_format:
			end = 640*480*s[1]
			data = np.frombuffer(buf[begin:begin+end], dtype=s[0]).copy().reshape((-1,1))
			output.append(data)
			begin = begin+end
		return True, output

	def release(self,):
		self.p.terminate()

class point_cloud(basler_tof):
	def __init__(self,):
		basler_tof.__init__(self, [
			['float32', 4],
			['float32', 4],
			['float32', 4],
			['uint16', 2],
			# ['uint16', 2],
		])

if __name__ == '__main__':
	cap = point_cloud()

	while 1:
		ret, [x, y, z, amplitude] = cap.read()
		if not ret:
			break

		cv2.imshow('x', x.reshape((480,640))/x.max())
		cv2.imshow('y', y.reshape((480,640))/y.max())
		cv2.imshow('z', z.reshape((480,640))/z.max())
		cv2.imshow('amplitude', amplitude.reshape((480,640)))
		key_pressed = cv2.waitKey(1)&0xff
		if key_pressed in [27, ord('q')]:
			break

	cap.release()