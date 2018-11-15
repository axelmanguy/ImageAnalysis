#-*- coding: utf8 -*
import test_interfaced as test
import script3
from skimage import io
import matplotlib.pyplot as plt


# shapes : TOUTES les shapes
# digitShapes : que les shapes avec digits
def findHoles(digitShapes, shapes):
		return [(shape, script3.findSimilarShapes(shape, shapes)[1][0]) for shape in digitShapes]

def main():
	prefix =  './Parcours_updated/1'
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

	for i in range(len(curve_x)):
		print(curve_x[i],curve_y[i])
	#for el in pairs:
	#	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
	#	ax1.imshow(el[0].region.image)
	#	ax2.imshow(el[1].region.image)

	ax = script3.plotImage(im)
	script3.plotCurve(curve_x, curve_y, ax)
	plt.show()
	
	


if __name__ == "__main__":
	main()
