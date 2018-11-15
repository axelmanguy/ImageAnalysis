import tarfile
import os

data_base_path = os.path.join(os.pardir, 'data')
data_folder = 'lab-02-data'
tar_path = os.path.join(data_base_path, data_folder + '.tar.gz')
with tarfile.open(tar_path, mode='r:gz') as tar:
    tar.extractall(path=data_base_path)

import skimage.io
from skimage import measure
from skimage import feature
from skimage import filters
from skimage import morphology
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from scipy.fftpack import fft, ifft
from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load images
data_base_path = os.path.join(os.pardir, 'data')
data_folder = 'lab-02-data'
#  Load zeros
zeros_path = os.path.join(data_base_path, data_folder, 'part1', '0')
zeros_names = [nm for nm in os.listdir(zeros_path) if '.png' in nm]  # make sure to only load .png
zeros_names.sort()  # sort file names
ic = skimage.io.imread_collection([os.path.join(zeros_path, nm) for nm in zeros_names])
zeros_im = skimage.io.concatenate_images(ic)
#  Load ones
ones_path = os.path.join(data_base_path, data_folder, 'part1', '1')
ones_names = [nm for nm in os.listdir(ones_path) if '.png' in nm]  # make sure to only load .png
ones_names.sort()  # sort file names
ic = skimage.io.imread_collection(([os.path.join(ones_path, nm) for nm in ones_names]))
ones_im = skimage.io.concatenate_images(ic)
#  Load twos
twos_path = os.path.join(data_base_path, data_folder, 'part2', '2')
twos_names = [nm for nm in os.listdir(twos_path) if '.png' in nm]  # make sure to only load .png
twos_names.sort()  # sort file names
ic = skimage.io.imread_collection([os.path.join(twos_path, nm) for nm in twos_names])
twos_im = skimage.io.concatenate_images(ic)
#  Load threes
threes_path = os.path.join(data_base_path, data_folder, 'part2', '3')
threes_names = [nm for nm in os.listdir(threes_path) if '.png' in nm]  # make sure to only load .png
threes_names.sort()  # sort file names
ic = skimage.io.imread_collection(([os.path.join(threes_path, nm) for nm in threes_names]))
threes_im = skimage.io.concatenate_images(ic)

feature_list_0,feature_list_0_3d=[],[]
feature_list_1,feature_list_1_3d=[],[]
feature_list_2,feature_list_2_3d=[],[]
feature_list_3,feature_list_3_3d=[],[]
# Plot images
fig, axes = plt.subplots(4, len(zeros_im), figsize=(12, 3))
for ax, im, nm in zip(axes[0], zeros_im, zeros_names):
    #regions detections
    label_img = measure.label(im>100) #converting to binary 
    #label_img= erosion(label_img,square(2))
    regions = measure.regionprops(label_img)
    feature_list_0_3d.append(0)
    for props in regions:
        #adding area and perimeter as feature
        feature_list_0_3d[-1]=max(feature_list_0_3d[-1],props.area)
    #contours definition using marching squares
    contours = measure.find_contours(im, 0.8)   
    #signal creation
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])   
    #fourier transform
    fourier=fft(signal)   
    #2 first coef extraction
    feature_list_0.append([np.abs(fourier[1]),np.abs(fourier[4])])   
    ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.axis('off')
for ax, im, nm in zip(axes[1], ones_im, ones_names):
    #regions detections
    label_img = measure.label(im>100) #converting to binary 
    #label_img= erosion(label_img,square(2))
    regions = measure.regionprops(label_img)
    feature_list_1_3d.append(0)
    for props in regions:
        #adding area and perimeter as feature
        feature_list_1_3d[-1]=max(feature_list_1_3d[-1],props.area)
    #contours definition using marching squares
    contours = measure.find_contours(im, 0.8)   
    #signal creation
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])   
    #fourier transform
    fourier=fft(signal)   
    #2 first coef extraction
    feature_list_1.append([np.abs(fourier[1]),np.abs(fourier[4])])   
    ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.axis('off')
for ax, im, nm in zip(axes[2], twos_im, twos_names):
    #regions detections
    label_img = measure.label(im>100) #converting to binary 
    #label_img= erosion(label_img,square(2))
    regions = measure.regionprops(label_img)
    feature_list_2_3d.append(0)
    for props in regions:
        #adding area and perimeter as feature
        feature_list_2_3d[-1]=max(feature_list_2_3d[-1],props.area)
    #contours definition using marching squares
    contours = measure.find_contours(im, 0.8)   
    #signal creation
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])   
    #fourier transform
    fourier=fft(signal)   
    #2 first coef extraction
    feature_list_2.append([np.abs(fourier[1]),np.abs(fourier[4])])   
    ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.axis('off')
for ax, im, nm in zip(axes[3], threes_im, threes_names):
    
    #regions detections
    label_img = measure.label(im>100) #converting to binary 
    #label_img= erosion(label_img,square(2))
    regions = measure.regionprops(label_img)
    feature_list_3_3d.append(0)
    for props in regions:
        #adding area and perimeter as feature
        feature_list_3_3d[-1]=max(feature_list_3_3d[-1],props.area)
        
    #contours definition using marching squares
    contours = measure.find_contours(im, 0.8)   
    #signal creation
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])   
    #fourier transform
    fourier=fft(signal)   
    #2 first coef extraction
    feature_list_3.append([np.abs(fourier[1]),np.abs(fourier[4])])   
    ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.axis('off')


feature_list_0,feature_list_1=np.array(feature_list_0),np.array(feature_list_1)
feature_list_2,feature_list_3=np.array(feature_list_2),np.array(feature_list_3)
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(feature_list_0.T[0], feature_list_0.T[1],feature_list_0_3d,label='0')
ax1.scatter(feature_list_1.T[0], feature_list_1.T[1],feature_list_1_3d,label='1')
ax1.scatter(feature_list_2.T[0], feature_list_2.T[1],feature_list_2_3d,label='2')
ax1.scatter(feature_list_3.T[0], feature_list_3.T[1],feature_list_3_3d,label='3')
ax1.legend()
plt.show()