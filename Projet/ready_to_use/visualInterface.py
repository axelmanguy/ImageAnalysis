#-*- coding:utf8 -*
# Usine à gaz v9.9999
import threading
import skimage
import os
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
from skimage.measure import find_contours
from skimage import measure
from scipy import ndimage
from skimage import filters
from scipy.optimize import curve_fit
from savitzky_golay import savitzky_golay

# Secret stuff
#from scipy.signal import argrelextrema


# Global resolution for shapes and digits
MIN_SIZE = 50
MIN_SIZE_DIGIT = 8
# Width of the noise for the white color
WIDTH_NOISE = 15
# For digits, for require height
MIN_HEIGHT_DIGIT = 20

#Global variable
robot_descriptor = None
l_accuracy_arrow = []

_MONITORING_CHANNELS = {}


def MONITORING(channel, subchannel, function, *param, clear=True):
	data = _MONITORING_CHANNELS.get(channel, [None, {}])
	data[1][subchannel] = (function, param, clear)
	_MONITORING_CHANNELS[channel] = data
	if threading.current_thread().__class__.__name__ == '_MainThread' and not __name__ == "__main__":
		EXECMONITORING()



def UNMONITORING(channel):
	data = _MONITORING_CHANNELS.get(channel, None )
	if data is not None:
		data[1] = None
	

def EXECMONITORING():
	plt.ion()
	for channel, data in _MONITORING_CHANNELS.items():
		if len(data) == 0:
			continue
		ax = data[0]
		if data[1] == None:
			plt.close(ax._mntfig)
			data[0] = None
			data[1] = {}
			continue
		for subchannel, (function, param, clear) in data[1].items():
			if clear and ax is not None:
				fig = ax._mntfig
				fig.clf()
				ax = fig.add_subplot(111)
				ax._mntfig = fig
			ax = function(*param, ax=ax)
		data[0] = ax
	plt.draw()
	plt.show()
	plt.pause(0.1)

class Digit:
	def __init__(self):
		self.region = None
		self.value = None 

class Shape:
	def __init__(self):
		self.region = None
		self.extracted_image = None
		self.contour = None
		self.fdescriptors = None
		self.digit = None


def plotHistogram(img, ax = None):
	hist = skimage.exposure.histogram(img)
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax._mntfig = fig
	ax.step(hist[1], hist[0])
	return ax

def plotImage(img, ax=None):
	if ax == None:
		fig, ax = plt.subplots(1, 1, figsize=(6, 6))
		ax._mntfig = fig
	ax.imshow(img)
	return ax


def getRegions(img, min_width, min_height):
	label_img = measure.label(img)
	regions_img = measure.regionprops(label_img)
	regions = []
	for region in regions_img:
		minr, minc, maxr, maxc = region.bbox
		if region.major_axis_length >  min_height and region.minor_axis_length >  min_width:
			regions.append(region)

	return regions


# Calcule le threshold en fittant une gaussienne
def getRGBThreshold(im):
	y = np.zeros((3, 256))
	for j in range(3):
		hist = skimage.exposure.histogram(im[:,:,j])
		y[j, hist[1]] = hist[0]
	
	ymin = np.min(y, axis=0)
	ymin = savitzky_golay(ymin, 11, 3)
	m = np.max(ymin)
	ym = np.argmax(ymin)
	coeff, _ = curve_fit(gaussian, range(256), ymin, [90.0, ym, m])

	## MONITORING ###
	rg = range(256)
	MONITORING(3, 1, plotCurve, rg, ymin)
	MONITORING(3, 2, plotCurve, rg, gaussian(rg, *coeff), clear=False)
	##            ###

	return coeff[1] - abs(coeff[0])*2.7


# Prend une image en base RGB en entrée
def getRegionsRGB(im, threshold, min_width, min_height):
	bool_im = ((im[:,:,2] < threshold)|(im[:,:,1] < threshold)|(im[:,:,0] < threshold))&(~scanning(im, WIDTH_NOISE, 0, 256))
	regions = getRegions(bool_im, min_width, min_height)
	return regions, bool_im

def notInBorder(im, regions):
	h, w = im.shape
	l_not_in_border = []
	for region in regions:
		(rmin, cmin, rmax, cmax) = region.bbox
		if not (rmin == 0 or rmax == h or cmin == 0 or cmax == w):
			l_not_in_border.append(region)
	return l_not_in_border

def bboxToIm(im, bbox):
	(rmin, cmin, rmax, cmax) = bbox
	return im[rmin:rmax, cmin:cmax]

def extractDigit(im_shape):
	regions = getRegions(~im_shape, MIN_SIZE_DIGIT, MIN_HEIGHT_DIGIT)
	selected_regions = regions
	not_in_border = notInBorder(~im_shape, selected_regions)

	if len(not_in_border) == 0:
		return None
	else:
		not_in_border.sort(key=lambda region : region.perimeter)
		return not_in_border[-1]

def getDescriptor(region):
	extended_image = np.zeros((region.image.shape[0] + 2, region.image.shape[1] + 2))
	extended_image[1:-1,1:-1] = region.filled_image

	contours = measure.find_contours(extended_image, 0.8)
	if len(contours) > 0:
		contours.sort(key=lambda x : x.shape[0], reverse=True)
		contour = contours[0]
		index = np.linspace(0, contour.shape[0]-1, 300, dtype=int)
		contour = contour[index]
		signal = contour[:, 0] + contour[:, 1] * 1.0j
		descriptor = np.fft.fft(signal)
		return descriptor, contour


def getShapes(im):
	threshold = getRGBThreshold(im)
	regions, bool_im_shapes = getRegionsRGB(im, threshold, MIN_SIZE, MIN_SIZE)
	l_shape = []
	for region in regions:
		shape = Shape()
		shape.region = region
		shape.extracted_image = bboxToIm(bool_im_shapes, region.bbox)
		region_digit = extractDigit(shape.extracted_image)
		if region_digit != None:
			shape.digit = Digit()
			shape.digit.region = region_digit

		desc, cont = getDescriptor(shape.region)
		shape.fdescriptors = desc
		shape.contour = cont
		l_shape.append(shape)

	return l_shape

def findSimilarShapes(reference_shape, l_shapes):
	r_descriptors = reference_shape.fdescriptors
	return findSimilar(r_descriptors, l_shapes)

def findSimilar(r_descriptors, l_shapes):
	r_desc_1 = np.max(np.abs(r_descriptors[1:]))

	#n_r_descn = np.abs(r_descriptors[-2:])
	#n_r_descn = n_r_descn / r_desc_1

	n_r_descp = np.abs(r_descriptors[1:10])
	n_r_descp = n_r_descp / r_desc_1

	res = []

	for shape in l_shapes:
		n_descriptorsp = np.abs(shape.fdescriptors[1:10])
		desc_1 = np.max(np.abs(shape.fdescriptors[1:]))
		n_descriptorsp = n_descriptorsp / desc_1
		#n_descriptorsn = np.abs(shape.fdescriptors[-2:])
		#n_descriptorsn = n_descriptorsn / desc_1
		mse = ((n_r_descp - n_descriptorsp)**2).sum() #+ ((n_r_descn - n_descriptorsn)**2).sum()
		mse = np.sqrt(mse)
		res.append([shape, mse])

	res.sort(key=lambda x : x[1])
	return res

def findSimilarWithScale(r_descriptors, l_shapes):
	res = []
	for shape in l_shapes:
		mse = ((np.abs(shape.fdescriptors[1:10]) - np.abs(r_descriptors[1:10]))**2).sum()
		mse += ((np.abs(shape.fdescriptors[-10:]) - np.abs(r_descriptors[-10:]))**2).sum()
		mse = np.sqrt(mse)
		res.append([shape, mse])
	res.sort(key=lambda x : x[1])
	return res

def initialiseRobotDescriptor(regions):
	global robot_descriptor
	regions.sort(key=lambda region : region.filled_area, reverse=True)
	if os.path.isfile("arrow.desc.npy"):
		robot_descriptor = np.load("arrow.desc.npy")
	else:
		for region in regions:
			MONITORING(9, 1, plotImage, region.image)
			if __name__ == "__main__":
				EXECMONITORING()
			r = str(input('Is this robot ? y/n\n'))
			if len(r) > 0 and r[0] == 'y':
				robot_descriptor, _ = getDescriptor(region)
				r = str(input('Save ? y/n\n'))
				if len(r) > 0 and r[0] == 'y':
					np.save("arrow.desc", robot_descriptor)
				break
		UNMONITORING(9)


def weightOrientation(region):
	y0, x0 = region.local_centroid
	orientation = region.orientation
	x1 = math.cos(orientation)
	y1 = math.sin(orientation)
	wx, wy = (x0 - region.image.shape[1]/2, region.image.shape[0]/2 - y0)
	sp = x1*wx + y1*wy
	if sp < 0:
		return (orientation + math.pi)% (2.0*math.pi)
	if sp >=0:
		return orientation

def getRobotState(im):
	threshold = getRGBThreshold(im)
	bool_im = inInter(im, 0, threshold*0.7)#&scanning(im, WIDTH_NOISE*2, 0, int(threshold*0.6))
	regions = getRegions(bool_im, 30, 50)
	if robot_descriptor is None:
		initialiseRobotDescriptor(regions)
	shapes = []
	for region in regions:
		shape = Shape()
		shape.fdescriptors, contour = getDescriptor(region)
		shape.region = region
		shape.contour = contour
		shapes.append(shape)
	candidate_regions = findSimilarWithScale(robot_descriptor,shapes)


	robot_region = candidate_regions[0][0].region

	orientation = weightOrientation(robot_region)
	(rmin, cmin, rmax, cmax) = robot_region.bbox
	pos = ((cmin + cmax)/2, im.shape[0] - (rmin + rmax)/2)
	pos = (int(pos[0]), int(pos[1]))


	## MONITORING ###
	accuracy = np.log2(candidate_regions[1][1]/candidate_regions[0][1])
	l_accuracy_arrow.append(accuracy)
	if len(l_accuracy_arrow) > 20:
		del l_accuracy_arrow[0]
	MONITORING(1, 1, plotCurve, (cmin, cmin, cmax, cmax, cmin), (rmin, rmax, rmax, rmin, rmin))
	MONITORING(1, 2, plotImage, bool_im, clear=False)
	MONITORING(2, 1, plotCurve, l_accuracy_arrow)
	##            ###
	return pos, orientation

def plotCurve(*c,ax = None):
	if ax == None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax._mntfig = fig
	ax.plot(*c)
	return ax	

def demo(i):
	plt.ion()
	im = io.imread('{}.jpg'.format(i))
	filterImageRGB(im)
	shapes = getShapes(im)
	ax = None
	
	
	
	ax = None	
	ax2 = None
	for shape in shapes:
		if shape.digit is not None:
			res = findSimilarShapes(shape, shapes)
			MONITORING(8, 1, plotImage, shape.extracted_image)
			MONITORING(7, 1, plotImage, res[1][0].extracted_image)
			MONITORING(6, 1, plotCurve, np.log2(res[2][1]/res[1][1]), 'ro', clear=False)
			plt.pause(2)
			EXECMONITORING()

	res = findSimilar(np.exp(np.arange(400) * 2*math.pi / 400), shapes)
	MONITORING(7, plotImage, res[0][0].extracted_image)
	EXECMONITORING()

	# Etat du robots
	#pos, orientation = getRobotState(im)
	#print("Robot currently at pos {} and has orientation {}".format(pos, orientation * 180 / math.pi))



def fancyAnimation():
	plt.ion()
	print("CTRL+C to stop animation")
	ax = None
	for i in range(1, 25):
		im = io.imread('Parcours_updated/{}.jpg'.format(i))
		filterImageRGB(im)	
		pos, orientation = getRobotState(im)
		print("Robot currently at pos {} and has orientation {:.0f}°".format(pos, orientation * 180 / math.pi))
		EXECMONITORING()
		

def scanning(im, inter, fr, to):
	res = (np.zeros(im.shape[0:2]) == 1) 
	for i in range(fr, to):
		res|=inInter(im, i, i+inter)

	return res
	
def gaussian(x, *param):
	s1, mu1, a1  = param
	return a1*np.exp(-(x-mu1)**2/(2.*(s1**2)))

	
def inInter(im, bi, bs):
	return (bi < im[:,:,0])&(im[:,:,0] < bs)&(bi < im[:,:,1])&(im[:,:,1] < bs)&(bi < im[:,:,2])&(im[:,:,2] < bs)


def filterImageRGB(im):
	for i in range(3):
		im[:,:, i] = filters.median(im[:,:,i], np.ones((2, 2)))


def main():
	#demo('1_bis_bis_very_mauvais')
	#i = '1'
	#im = io.imread('{}.jpg'.format(i))
	#pos, orientation = getRobotState(im)
	#print("Robot currently at pos {} and has orientation {:.0f}°".format(pos, orientation * 180 / math.pi))
	fancyAnimation()
	input("")



if __name__ == "__main__":
	main()
