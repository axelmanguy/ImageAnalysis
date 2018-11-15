#-*- coding:utf8 -*
import skimage
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage import io
from skimage import morphology
from skimage.measure import find_contours
from skimage import measure
from scipy import ndimage
from skimage import filters
from scipy.optimize import curve_fit

# Secret stuff
#from savitzky_golay import savitzky_golay
#from scipy.signal import argrelextrema

# Les thresholds NE sont plus hardcodés

# Global resolution for shapes and digits
MIN_SIZE = 20
MIN_SIZE_DIGIT = 5
# Width of the noise for the white color
WIDTH_NOISE = 15
# For digits, for require height
MIN_HEIGHT_DIGIT = 20

#Global variable
robot_descriptor = None
initiale_orientation = None


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
	ax.step(hist[1], hist[0])
	return ax

def plotImage(img, ax=None):
	if ax == None:
		fig, ax = plt.subplots(1, 1, figsize=(6, 6))
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
# Utilise la composante S de la base HSV. Prend une image en base
# HSV en entrée
def getRegionsHSV(im):
	s_im = im[:, :, 1]*255
	#filtered = filters.median(s_im, np.ones((2, 2)))
	bool_filtered = s_im > 100
	regions = getRegions(bool_filtered)
	return regions, bool_filtered



# Calcule le threshold en fittant une gaussienne
def getRGBThreshold(im):
	y = np.zeros((3, 256))
	for j in range(3):
		hist = skimage.exposure.histogram(im[:,:,j])
		y[j, hist[1]] = hist[0]
	ymin = np.min(y, axis=0)
	coeff, _ = curve_fit(gaussian, range(256), ymin, [1.0, 155, 1.0])
	return coeff[1] - abs(coeff[0])*2.7


# Prend une image en base RGB en entrée
def getRegionsRGB(im, threshold, min_width, min_height):
	#filtered = filters.median(im, np.ones((2, 2)))
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
		#assert len(not_in_border) == 1
		return not_in_border[-1]

def getDescriptor(region):
	extended_image = np.zeros((region.image.shape[0] + 2, region.image.shape[1] + 2))
	extended_image[1:-1,1:-1] = region.filled_image
	#extended_image = morphology.binary_opening(extended_image, np.ones((4, 4)))
	#extended_image = morphology.binary_opening(extended_image, np.ones((4, 4)))
	contours = measure.find_contours(extended_image, 0.8)
	if len(contours) > 0:
		contours.sort(key=lambda x : x.shape[0], reverse=True)
		contour = contours[0]
		index = np.linspace(0, contour.shape[0]-1, 400, dtype=int)
		contour = contour[index]
		descriptor = np.fft.fft(contour[:, 0] + contour[:, 1] * 1j)
		return descriptor, contour


def getShapes(im):
	threshold = getRGBThreshold(im)
	regions, bool_im_shapes = getRegionsRGB(im, threshold, MIN_SIZE, MIN_SIZE)
	#plotImage(bool_im_shapes)
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
	n_r_descn = np.abs(r_descriptors[-1:-20])
	r_desc_1 = np.abs(r_descriptors[0])
	n_r_descn = n_r_descn / r_desc_1

	n_r_descp = np.abs(r_descriptors[1:20])
	n_r_descp = n_r_descp / r_desc_1

	res = []

	for shape in l_shapes:
		n_descriptorsp = np.abs(shape.fdescriptors[1:20])
		desc_1 = np.abs(shape.fdescriptors[0])
		n_descriptorsp = n_descriptorsp / desc_1
		n_descriptorsn = np.abs(shape.fdescriptors[-1:-20])
		n_descriptorsn = n_descriptorsn / desc_1
		mse = ((n_r_descp - n_descriptorsp)**2).sum() + ((n_r_descn - n_descriptorsn)**2).sum()
		res.append([shape, mse])

	res.sort(key=lambda x : x[1])
	return res

def findSimilarWithScale(r_descriptors, l_shapes):

	res = []

	for shape in l_shapes:
		mse = ((np.abs(shape.fdescriptors[1:20]) - np.abs(r_descriptors[1:20]))**2).sum()
		mse += ((np.abs(shape.fdescriptors[-1:-20]) - np.abs(r_descriptors[-1:-20]))**2).sum()
		res.append([shape, mse])

	res.sort(key=lambda x : x[1])
	return res

def initialiseRobotDescriptor(regions):
	global robot_descriptor
	global initiale_orientation
	regions.sort(key=lambda region : region.filled_area, reverse=True)
	plt.ion()
	ax = None
	for region in regions:
		ax=plotImage(region.image, ax)
		r = str(raw_input('Is this robot ? y/n\n'))
		plt.draw()
		if r[0] == 'y':
			robot_descriptor, _ = getDescriptor(region)
			initiale_orientation = weightOrientation(region)
			return region


def weightOrientation(region):
	y0, x0 = region.local_centroid
	orientation = region.orientation
	x1 = math.cos(orientation)
	y1 = math.sin(orientation)
	wx, wy = (x0 - region.image.shape[1]/2, region.image.shape[0]/2 - y0)
	sp = x1*wx + y1*wy
	if sp < 0:
		return orientation + math.pi
	if sp >=0:
		return orientation

def getRobotState(im):
	threshold = getRGBThreshold(im)
	bool_im = inInter(im, 0, threshold)#&(scanning(im, 25, 0, 256))
	regions = getRegions(bool_im, 30, 50)
	if robot_descriptor is None:
		robot_region = initialiseRobotDescriptor(regions)
		orientation = initiale_orientation
	else:
		shapes = []
		for region in regions:
			shape = Shape()
			shape.fdescriptors, _ = getDescriptor(region)
			shape.region = region
			shapes.append(shape)
		candidate_regions = findSimilarWithScale(robot_descriptor,shapes)
		robot_region = candidate_regions[0][0].region
		orientation = weightOrientation(robot_region)
	(rmin, cmin, rmax, cmax) = robot_region.bbox
	pos = ((cmin + cmax)/2, im.shape[0] - (rmin + rmax)/2)
	pos = (int(pos[0]), int(pos[1]))
	return pos, orientation

def plotCurve(x, y, ax = None):
	if ax == None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	ax.plot(x, y)
	return ax	

def demo(i):
	plt.ion()
	im = io.imread('{}.jpg'.format(i))
	shapes = getShapes(im)
	
	
	ax = None	
	ax2 = None
	for shape in shapes:
		if shape.digit is not None:
			ax = plotImage(shape.extracted_image, ax)
			res = findSimilarShapes(shape, shapes)
			ax2 = plotImage(res[1][0].extracted_image, ax2)
			plt.pause(2)
			plt.draw()

	# Etat du robots
	pos, orientation = getRobotState(im)
	print("Robot currently at pos {} and has orientation {}".format(pos, orientation * 180 / math.pi))



def fancyAnimation():
	import time
	plt.ion()
	plt.show()
	print("CTRL+C to stop animation")
	ax = None
	for i in range(1, 25):
		im = io.imread('{}.jpg'.format(i))	
		pos, orientation = getRobotState(im)
		print("Robot currently at pos {} and has orientation {:.0f}°".format(pos, orientation * 180 / math.pi))
		ax = plotImage(im, ax)	
		plt.draw()
		plt.pause(1)
		

def scanning(im, inter, fr, to):
	res = (np.zeros(im.shape[0:2]) == 1) 
	for i in range(fr, to):
		res|=inInter(im, i, i+inter)

	return res
	
def gaussian(x, *param):
	s1, mu1, a1  = param
	return a1*np.exp(-(x-mu1)**2/(2.*s1**2))

	
def inInter(im, bi, bs):
	return (bi < im[:,:,0])&(im[:,:,0] < bs)&(bi < im[:,:,1])&(im[:,:,1] < bs)&(bi < im[:,:,2])&(im[:,:,2] < bs)




def main():
	demo('1_bis')
	fancyAnimation()



if __name__ == "__main__":
	main()




