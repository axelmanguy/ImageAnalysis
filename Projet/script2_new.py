#-*- coding:utf8 -*
import skimage
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import argrelextrema

from skimage import io
from skimage import morphology
from skimage.filters import threshold_minimum
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
from skimage.segmentation import active_contour

from skimage import transform
from skimage import feature
from skimage import measure
from scipy.ndimage import filters as sfilters
from scipy import ndimage
from skimage import filters

# Secret stuff
#from savitzky_golay import savitzky_golay


# Les thresholds sont hard-codés, mais ça fonctionne très bien comme ça
THREESHOLD_V = 70
THREESHOLD_S = 65
THRESHOLD_RGB = 100
MIN_SIZE = 20

# Implementation of flood fill : https://en.wikipedia.org/wiki/Flood_fill
def regionGrowing(z, bi, bs, seed):
	neighbours = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
	mask = np.zeros_like(z, dtype = bool)
	stack = [seed] # push start coordinate on stack

	while stack:
    		x, y = stack.pop()
    		mask[x, y] = True
    		for dx, dy in neighbours:
        		nx, ny = x + dx, y + dy
        		if (0 <= nx < z.shape[0]) and (0 <= ny < z.shape[1]) and (not mask[nx, ny]) and (z[nx, ny] <= bs) and (z[nx, ny] >= bi):
            			stack.append((nx, ny))

	
	return mask

# Implementation of flood fill : https://en.wikipedia.org/wiki/Flood_fill
def regionGrowingN(z, bi, bs, *seed):
	neighbours = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
	mask = np.zeros_like(z, dtype = bool)
	stack = [] # push start coordinate on stack
	stack.extend(seed)

	while stack:
    		x, y = stack.pop()
    		mask[x, y] = True
    		for dx, dy in neighbours:
        		nx, ny = x + dx, y + dy
        		if (0 <= nx < z.shape[0]) and (0 <= ny < z.shape[1]) and (not mask[nx, ny]) and (z[nx, ny] <= bs) and (z[nx, ny] >= bi):
            			stack.append((nx, ny))

	
	return mask
# Implementation of flood fill : https://en.wikipedia.org/wiki/Flood_fill
def regionGrowingBoolean(z, seed):
	neighbours = [(-1,0), (0,1), (1,0), (0,-1)]
	mask = np.zeros_like(z, dtype = bool)
	stack = [seed] # push start coordinate on stack

	while stack:
    		x, y = stack.pop()
    		mask[x, y] = True
    		for dx, dy in neighbours:
        		nx, ny = x + dx, y + dy
        		if (0 <= nx < z.shape[0]) and (0 <= ny < z.shape[1]) and (not mask[nx, ny]) and z[nx, ny]:
            			stack.append((nx, ny))

	
	return mask

# Dessine les Bbox sur une image
def plotBbox(img, ax):
	label_img = measure.label(img)
	regions = measure.regionprops(label_img)
	i = 0
	for props in regions:
		i+=1
		print("processing {}".format(i))
		y0, x0 = props.centroid
		orientation = props.orientation
		x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
		y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
		x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
		y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

		minr, minc, maxr, maxc = props.bbox
		bx = (minc, maxc, maxc, minc, minc)
		by = (minr, minr, maxr, maxr, minr)
		ax.plot(bx, by, '-b', linewidth=2.5)
	return regions

def plotBboxFromList(bboxs, ax, color='r'):
	for bbox in bboxs:
		minr, minc, maxr, maxc = bbox
		bx = (minc, maxc, maxc, minc, minc)
		by = (minr, minr, maxr, maxr, minr)
		ax.plot(bx, by, '-'+color, linewidth=2.5)		
	

# Retourne les Bbox, c'est à dire une liste de (ligne_min, colonne_min, ligne_max, colonne_max) 
def getBbox(img, use_min_size = True, min_size=MIN_SIZE):
	label_img = measure.label(img)
	regions = measure.regionprops(label_img)
	bboxs = []
	for props in regions:
		minr, minc, maxr, maxc = props.bbox
		if use_min_size and (maxr - minr) >  min_size and (maxc - minc) >  min_size:
			bboxs.append((minr, minc, maxr, maxc))
		elif not use_min_size:
			bboxs.append((minr, minc, maxr, maxc))
	return bboxs

def plotHistogram(img):
	hist = skimage.exposure.histogram(img)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.step(hist[1], hist[0])

def plotCurve(x, y):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x, y)	

def plotImage(img):
	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	ax.imshow(img, cmap='gray')
	return ax

# Retourne les Shapes c'est à dire une liste de (ligne_min, colonne_min, ligne_max, colonne_max), ainsi que l'image traitée en binaire.
# Utilise une image dans la base HSV
def getShapes(im, show_image = False):
	s_im = im[:, :, 1]
	filtered = filters.median(s_im, np.ones((2, 2)))
	bool_filtered = (filtered) > THREESHOLD_S
	bboxs = getBbox(bool_filtered)
	if show_image:
		ax = plotImage(bool_filtered)
		plotBboxFromList(bboxs, ax)

	return bboxs, bool_filtered

def getShapesRGB(im, show_image = False):
	#filtered = filters.median(im, np.ones((2, 2)))
	bool_im = (im[:,:,2] < THRESHOLD_RGB )|(im[:,:,1] < THRESHOLD_RGB )|(im[:,:,0] < THRESHOLD_RGB )
	bboxs = getBbox(bool_im)
	if show_image:
		ax = plotImage(bool_im)
		plotBboxFromList(bboxs, ax)

	return bboxs, bool_im
	
# Retourne la Bbox de la flêche du robot, ainsi que l'image traitée en binaire. 
# Utilise une image dans la base HSV. 
def getArrow(im, show_image = False):
	v_im = im[:, :, 2]*255
	#plotImage(v_im)
	bool_v_im = v_im < THREESHOLD_V

	#eroded_bool_v_im = morphology.binary_erosion(bool_v_im, np.ones((25, 25)))
	#plotImage(eroded_bool_v_im)
	#x, y = np.where(eroded_bool_v_im == True)
	#im_arrow = regionGrowingBoolean(bool_v_im, (x[0], y[0]))
	im_arrow = bool_v_im
	bboxs = getBbox(im_arrow, True, 70)
	if show_image:
		ax = plotImage(im_arrow)
		plotBboxFromList(bboxs, ax)
	return bboxs, im_arrow

# Retourne les Bbox qui ne sont pas sur le bord d'une image.
# Utile pour les digits.
def notInBorder(im, bboxs):
	h, w = im.shape
	l_not_in_border = []
	for (rmin, cmin, rmax, cmax) in bboxs:
		if not (rmin == 0 or rmax == h or cmin == 0 or cmax == w):
			l_not_in_border.append((rmin, cmin, rmax, cmax))
	return l_not_in_border
			


def extractDigit(im_shape):
	not_in_border = notInBorder(~im_shape, getBbox(~im_shape, True, 10))
	if len(not_in_border) == 0:
		return None
	else:
		assert len(not_in_border) == 1
		return not_in_border[0]

def bboxToIm(im, bbox):
	print(bbox)
	(rmin, cmin, rmax, cmax) = bbox
	return im[rmin:rmax, cmin:cmax]


# Recupere une liste composée de [image shape, image digit, Fourier descriptors]
def getShapes(im):
	bboxs, bool_im_shapes = getShapesRGB(im, True)
	l_shape = []
	for bbox in bboxs:
		#(rmin, cmin, rmax, cmax) = bbox
		im_shape = bboxToIm(bool_im_shapes , bbox)

		bbox_digit = extractDigit(im_shape)
		if bbox_digit != None:
			im_digit = bboxToIm(im_shape, bbox_digit)
		else:
			im_digit = None

		extended_image = np.zeros((im_shape.shape[0] + 50, im_shape.shape[1] + 50))
		extended_image[25:-25,25:-25] = im_shape
		contours = measure.find_contours(extended_image, 0.8)
		print(len(contours))
		if len(contours) > 0:
			contours.sort(key=lambda x : x.shape[0])
			shape_contour = contours[-1]
			descriptors = np.fft.fft(shape_contour[:, 0] + shape_contour[:, 1] * 1j)
			
		else:
			shape_contour = None
			descriptors = None
		l_shape.append([im_shape, im_digit, shape_contour, descriptors])

	return l_shape
		

# Retourne une liste composée de [shape, MSE] triée par ordre croissant de MSE
def findSimilarShapes(reference_shape, l_shapes):
	r_descriptors = reference_shape[3]
	r_desc_1 = np.absolute(r_descriptors[1])
	n_r_desc_2 = np.absolute(r_descriptors[2])/r_desc_1
	n_r_desc_3 = np.absolute(r_descriptors[3])/r_desc_1
	
	res = []

	for shape in l_shapes:
		descriptors = shape[3]
		desc_1 = np.absolute(descriptors[1])
		n_desc_2 = np.absolute(descriptors[2])/desc_1
		n_desc_3 = np.absolute(descriptors[3])/desc_1
		res.append([shape, (n_r_desc_2 - n_desc_2)**2 + (n_r_desc_3 - n_desc_3)**2])
	res.sort(key=lambda x : x[1])
	return res

def newmain():
	im = io.imread('./Parcours_updated/14.jpg')
	im = im[:,40:]
	im = im[:,:-40]
	# Get the shapes from the image
	shapes = getShapes(im)
	# Show digits that have been identify
	for shape in shapes:
		if shape[1] is not None:
			plotImage(shape[1])
	# Example of shape find by similarity
	ax = plotImage(shapes[3][0])
	res = findSimilarShapes(shapes[3], shapes)
	plotImage(res[1][0][0])
	

	
	plt.show()

def main():
	im = io.imread('.Parcours_updated/14.jpg')
	im = im[:,40:]
	im = im[:,:-40]
	hsv_im = skimage.color.convert_colorspace(im, 'RGB', 'HSV')


	bboxs_arrow, bool_im_arrow = getArrow(hsv_im)
	(rmin, cmin, rmax, cmax) = bboxs_arrow[0]
	bboxs, bool_im_shapes = getShapesRGB(im, True)
	pos = ((cmin + cmax)/2, (rmin + rmax)/2)
	print("Position du robot : ({}, {})".format(pos[0], pos[1]))
	ax = plotImage(bool_im_arrow)
	plotBboxFromList(bboxs_arrow, ax)

	(rmin, cmin, rmax, cmax) = bboxs[11]

	# Essayons d'extraire un digit
	im_shape = bool_im_shapes[rmin:rmax, cmin:cmax]
	# Un opening ça ne fait jamais de mal !
	im_shape = morphology.binary_opening(im_shape, np.ones((2, 2)))
	ax = plotImage(~im_shape)
	plotBbox(~im_shape, ax)
	# On ne garde que les bbox qui ne sont pas au bord
	not_in_border = notInBorder(~im_shape, getBbox(~im_shape, False))
	# On affiche (en rouge)
	plotBboxFromList(not_in_border, ax)

	extended_image = np.zeros((im_shape.shape[0] + 50, im_shape.shape[1] + 50))
	extended_image[25:-25,25:-25] = im_shape
	im_shape = extended_image
	ax = plotImage(im_shape)
	contours = measure.find_contours(im_shape, 0.8)
	for n, contour in enumerate(contours):
		print(contour.shape)
		ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
	descriptors = np.fft.fft(contour[:, 0] + contour[:, 1] * 1j)
	print(descriptors)
	plt.show()


if __name__ == "__main__":
	newmain()
