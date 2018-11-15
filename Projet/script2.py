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

from skimage import transform
from skimage import feature
from skimage import measure
from scipy.ndimage import filters as sfilters
from skimage import filters

from sklearn.neural_network import MLPClassifier
from skimage.transform import resize
import pickle

# Les thresholds sont hard-codés, mais ça fonctionne très bien comme ça
THREESHOLD_V = 80
THREESHOLD_S = 50
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
		if use_min_size and (maxr - minr) > MIN_SIZE and (maxc - minc) > MIN_SIZE:
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
	
# Retourne la Bbox de la flêche du robot, ainsi que l'image traitée en binaire. 
# Utilise une image dans la base HSV. 
def getArrow(im, show_image = False):
	v_im = im[:, :, 2]*255
	bool_v_im = v_im < THREESHOLD_V

	eroded_bool_v_im = morphology.binary_erosion(bool_v_im, np.ones((25, 25)))
	
	x, y = np.where(eroded_bool_v_im == True)
	im_arrow = regionGrowingBoolean(bool_v_im, (x[0], y[0]))

	bboxs = getBbox(im_arrow)
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

def digit_extract(bboxs,bool_im_shapes):
	model= pickle.load( open( "model2.p", "rb" ) )
	for bbox in bboxs:
		(rmin, cmin, rmax, cmax) = bbox
		im_shape = bool_im_shapes[rmin:rmax, cmin:cmax]
		im_shape = morphology.binary_opening(im_shape, np.ones((5, 5)))
		not_in_border = notInBorder(~im_shape, getBbox(~im_shape, False))
		print(not_in_border)
		for bounds in not_in_border:
			print(bounds)
			(rmin, cmin, rmax, cmax) = bounds
			shape= im_shape[rmin-2:rmax+2, cmin-2:cmax+2]
			resize_shape=resize(shape, (20,20))
			#then paste the image on a 28x28
			flatten=255*((resize_shape<1).flatten())
			pred=model.predict([flatten])
			plt.figure()
			plt.imshow(im_shape)
			plt.title(pred)


def main():
	im = io.imread('./Parcours_updated/4.jpg')
	im = im[:,40:]
	im = im[:,:-40]
	#plt.imshow(im)
	hsv_im = skimage.color.convert_colorspace(im, 'RGB', 'HSV')


	bboxs_arrow, bool_im_arrow = getArrow(hsv_im, False)
	(rmin, cmin, rmax, cmax) = bboxs_arrow[0]
	bboxs, bool_im_shapes = getShapes(hsv_im, False)
	print(bboxs)
	pos = ((cmin + cmax)/2, (rmin + rmax)/2)
	print("Position du robot : ({}, {})".format(pos[0], pos[1]))

	digit_extract(bboxs,bool_im_shapes)
	"""
	(rmin, cmin, rmax, cmax) = bboxs[15]
	for bbox in bboxs:
		(rmin, cmin, rmax, cmax) = bbox
		# Essayons d'extraire un digit
		im_shape = bool_im_shapes[rmin:rmax, cmin:cmax]
		# Un opening ça ne fait jamais de mal !
		im_shape = morphology.binary_opening(im_shape, np.ones((5, 5)))
		ax = plotImage(~im_shape)
		#plotBbox(~im_shape, ax)
		# On ne garde que les bbox qui ne sont pas au bord
		not_in_border = notInBorder(~im_shape, getBbox(~im_shape, False))
		# On affiche (en rouge)
		plotBboxFromList(not_in_border, ax)
	"""

	plt.show()


	


if __name__ == "__main__":
	main()
