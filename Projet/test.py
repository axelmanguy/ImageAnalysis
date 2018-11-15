import numpy as np
import skimage.io
from script3 import *
from sklearn.neural_network import MLPClassifier
from skimage.transform import resize
from skimage.transform import rotate
from skimage.util import pad
#skimage.util.pad(array, pad_width, mode,
import pickle
digitlist=[1,2,3,4,5,7,6,8]
def preprocess_digit(imdigit):
	shape_im=imdigit
	shapedim=shape_im.shape
	#build background fro padding
	background=np.zeros((max(shapedim),max(shapedim)))
	bgdim=background.shape
	#paste image in background
	background[bgdim[0]//2-shapedim[0]//2:bgdim[0]//2-shapedim[0]//2+shapedim[0],
			   bgdim[1]//2-shapedim[1]//2:bgdim[1]//2-shapedim[1]//2+shapedim[1]]=shape_im
	#add a litle bit of padding to match with MNIST format
	resize_shape=resize(pad(background,8,'constant', constant_values=0), (28,28))
	#get rid of the blurring effect due to resize
	return resize_shape

if __name__=="__main__":
	im = io.imread('./Parcours_updated/1.jpg')
	im = im[:,40:]
	im = im[:,:-40]
	# Get the shapes from the image
	shapes = getShapes(im)
	# Show digits that have been identify
	model= pickle.load( open( "model4.p", "rb" ))
	for shape in shapes:
		if shape.digit is not None:
			#load image from region
			shape_im=shape.digit.region.image
			angle_to_rotate=(90-shape.digit.region.orientation*(180/np.pi))
			shape_rotated_1=rotate(shape_im,angle_to_rotate,resize=True)
			shape_rotated_2=rotate(shape_im,180+angle_to_rotate,resize=True)
			
			resize_shape_1=preprocess_digit(shape_rotated_1)
			resize_shape_2=preprocess_digit(shape_rotated_2)
			#flatten for classifier
			flatten_1=(255*((resize_shape_1).flatten()))
			flatten_2=(255*((resize_shape_2).flatten()))
			#make prediction
			proba=model.predict_proba([flatten_1,flatten_2])
			proba_max=np.max(proba,1)
			pred=model.predict([flatten_1,flatten_2])
			final_pred=pred[np.argmax(proba_max)]
			#if the final prediction is not in the digit list
			#the other is taken
			#if both prediction are wrong....well
			#if final_pred not in digitlist and pred[np.argmin(proba_max)] in digitlist:
			#	final_pred=pred[np.argmin(proba_max)]
			#elif final_pred not in digitlist and pred[np.argmin(proba_max)] not in digitlist:
			#	final_pred=9999

			#plotting
			plt.subplot(223)
			plt.imshow(shape_im,cmap='gray')
			plt.ylabel('original')
			plt.subplot(221)
			plt.imshow(resize_shape_1,cmap='gray')
			plt.title(str(pred[0])+" : "+str(proba_max[0]))
			plt.subplot(222)
			plt.imshow(resize_shape_2,cmap='gray')
			plt.title(str(pred[1])+" : "+str(proba_max[1]))
			plt.suptitle('Final Prediction: '+str(final_pred), fontsize=16, color='r')
			plt.show()
		
