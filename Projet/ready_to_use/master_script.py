#-*- coding: utf8 -*
import test_interfaced as test
import visualInterface as script3
from skimage import io
import matplotlib.pyplot as plt
import math
from threading import Thread
import numpy as np
import time

def shortestOrientedAngle(fr, to):
	delta = to - fr
	return -((-delta)% 360) if (-delta)% 360 < delta%360 else delta%360

def norme(v):
	return math.sqrt(v[0]**2 + v[1]**2)

def printRobot(*s):
	print("[ROBOT]: ", end="")
	print(*s)

class WebcamVideoStream:
	def __init__(self, src):
		self.im = io.imread("1.jpg")
	def read(self):
		return self.im
class FakeRobot:
	def __init__(self):
		image = io.imread("1.jpg")
		pos, orient = script3.getRobotState(image)
		self.pos = pos
		self.orientation = orient
	def steer_in_place(self, angle):
		print("[FakeRobot] : received angle : {}°".format(angle))
		self.orientation+=math.radians(angle)*0.3
		self.orientation = self.orientation % (2.0 * math.pi)
		time.sleep(1)
	def move_forward(self, distance):
		print("[FakeRobot] : received distance : {}pix".format(distance))
		distance = distance * 0.13
		self.pos = (self.pos[0] + math.cos(self.orientation) * distance, self.pos[1] + math.sin(self.orientation) * distance)
		time.sleep(2)

robot = FakeRobot()

class Tracker(Thread):
	def __init__(self):
		Thread.__init__(self)
		self.wvs = WebcamVideoStream(src=0)
		self.last_pos = None
		self.last_orientation = None
		self.running = True
		self.refreshed = False

	def run_real(self):
		while self.running:
			image = self.wvs.read()
			self.last_pos, self.last_orientation = script3.getRobotState(image)
			self.refreshed = True
	def run(self):
		while self.running:
			time.sleep(1)
			self.last_pos, self.last_orientation = robot.pos, robot.orientation
			self.refreshed = True	

	def getPath(self):
		image = self.wvs.read()
		path = gibmeMyPath(image)
		return path		
		
	

class RobotWrapper:
	angle_step = 40.
	distance_step = 20.
	pix_to_cm = 1.0       #cm.pix^-1

	def __init__(self):
		self.coefficient_angle = 1.0
		self.coefficient_distance = 1.0
		self.tracker = Tracker()
		self.path = None

	# Distance in pix
	def moveForward(self, distance):
		printRobot("request to move forward : {}pix".format(distance)) 
		init_pos, _ = self.getState()
		distance = distance * RobotWrapper.pix_to_cm * self.coefficient_distance
		robot.move_forward(distance=distance)
		self.tracker.refreshed = False
		while self.tracker.refreshed == False:
			time.sleep(0.2)
		c_p,_ = self.getState()
		v = (init_pos[0] - c_p[0], init_pos[1] - c_p[1])
		d = norme(v)
		printRobot("get distance : {}pix".format(d))
		return d

	# orientation in radians
	def changeOrientation(self, orientation):
		orientation = math.degrees(orientation)%360
		printRobot("request for orientation : {:.1f}°".format(orientation))
		_, init_orientation = self.getState()
		angle = shortestOrientedAngle(math.degrees(init_orientation), orientation) * self.coefficient_angle
		robot.steer_in_place(angle=angle)
		self.tracker.refreshed = False
		while self.tracker.refreshed == False:
			time.sleep(0.2)
		_,c_o = self.getState()
		printRobot("get orientation : {:.1f}°".format(math.degrees(c_o)))
		return c_o
		
		
		
	def getState(self):
		pos = None
		orientation = None

		#  !! very small chance of concurrent problem
		while pos is None or orientation is None:
			pos, orientation = self.tracker.last_pos, self.tracker.last_orientation
			if pos is None or orientation is None:
				printRobot("Tracker not ready or not started")
				time.sleep(0.3)
			script3.EXECMONITORING()

		return pos, orientation



	def calibratedAngle(self):
		_, init_orientation = self.getState()
		p_o = init_orientation
		total_rotation = 0
		l_rotate = []
		while total_rotation < 360 and total_rotation > -360:
			robot.steer_in_place(angle=RobotWrapper.angle_step)
			_, c_o = self.getState()
			delta_o = ((math.degrees(c_o) - math.degrees(p_o))%360)
			total_rotation += delta_o
			l_rotate.append(delta_o)
			p_o = c_o
		
		self.coefficient_angle = RobotWrapper.angle_step / (sum(l_rotate) / float(len(l_rotate)))
	
	def calibratedDistance(self):
		dist = self.moveForward(RobotWrapper.distance_step)
		self.coefficient_distance = RobotWrapper.distance_step / dist

	# Dream for it to work, but it probably won't, corrector incoming..	
	def goto(self, pos):
		cp, co = self.getState()
		v = (pos[0] - cp[0], pos[1] - cp[1])
		needed_orientation = math.atan2(v[1], v[0])
		self.changeOrientation(needed_orientation)
		d = norme(v)
		self.moveForward(d)
	
	def followPath(self):
		for x,y in zip(self.path[0], self.path[1]):
			x = int(x)
			y = int(y)
			printRobot("Go to point ({}, {})..".format(x,y))
			self.goto((x,y))
		
	def runForYourLife(self):
		printRobot("Request for path..")
		self.path = self.tracker.getPath()
		printRobot("Starting tracker..")
		self.tracker.start()	
		printRobot("Etalonated angles..")
		self.calibratedAngle()
		printRobot("Step angle : {}, coefficient computed : {}".format(RobotWrapper.angle_step, self.coefficient_angle))
		printRobot("Etalonated distance..")
		self.calibratedDistance()
		printRobot("Step distance : {}, coefficient computed : {}".format(RobotWrapper.distance_step, self.coefficient_distance))
		printRobot("Begining to follow path..")
		self.followPath()
		printRobot("End of path !")
		self.tracker.running = False
	


# shapes : TOUTES les shapes
# digitShapes : que les shapes avec digits
def findHoles(digitShapes, shapes):
		return [(shape, script3.findSimilarShapes(shape, shapes)[1][0]) for shape in digitShapes]

def gibmeMyPath(im):
	shapes = script3.getShapes(im)
	
	digitShapes = [shape for shape in shapes if shape.digit != None]
	test.getDigitValue(digitShapes)

	digitShapes.sort(key=lambda x : x.digit.value)
	pairs = findHoles(digitShapes, shapes)
	path_x = []
	path_y = []
	for pair in pairs:
		path_x.append(pair[0].region.centroid[1])
		path_x.append(pair[1].region.centroid[1])
		path_y.append(im.shape[0] - pair[0].region.centroid[0])
		path_y.append(im.shape[0] - pair[1].region.centroid[0])


	res = script3.findSimilar(np.exp(np.arange(400) * 2*math.pi / 400), shapes)
	path_x.append(res[0][0].region.centroid[1])
	path_y.append(im.shape[0] - res[0][0].region.centroid[0])
	return path_x, path_y

def connectRobot():
	from iapr.robot import Robot
	from iapr.webcam import WebcamVideoStream
	instanceRobot = None
	

def main():
	prefix =  '1'
	im = io.imread("{}.jpg".format(prefix))
	curve_x, curve_y = gibmeMyPath(im)
	im = im[::-1, :, :]
	ax = script3.plotImage(im)
	script3.plotCurve(curve_x, curve_y, ax=ax)
	plt.show()

	r = None
	try:
		r = RobotWrapper()
		r.runForYourLife()
	except Exception as e:
		print(e)
		r.tracker.running = False

def main_old():
	prefix =  '1'
	im = io.imread("{}.jpg".format(prefix))
	shapes = script3.getShapes(im)
	
	digitShapes = [shape for shape in shapes if shape.digit != None]
	test.getDigitValue(digitShapes)

	digitShapes.sort(key=lambda x : x.digit.value)
	pairs = findHoles(digitShapes, shapes)
	curve_x = []
	curve_y = []
	for pair in pairs:
		curve_x.append(pair[0].region.centroid[1])
		curve_x.append(pair[1].region.centroid[1])
		curve_y.append(pair[0].region.centroid[0])
		curve_y.append(pair[1].region.centroid[0])

	#for el in pairs:
	#	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
	#	ax1.imshow(el[0].region.image)
	#	ax2.imshow(el[1].region.image)

	ax = script3.plotImage(im)
	script3.plotCurve(curve_x, curve_y, ax=ax)
	plt.show()
	
	


if __name__ == "__main__":
	main()
