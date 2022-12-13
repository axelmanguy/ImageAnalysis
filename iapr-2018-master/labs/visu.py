import tarfile
import os

data_base_path = os.path.join(os.pardir, 'data')
data_folder = 'lab-02-data'
tar_path = os.path.join(data_base_path, data_folder + '.tar.gz')
with tarfile.open(tar_path, mode='r:gz') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path=data_base_path)

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
fig, axes = plt.subplots(2, len(zeros_im), figsize=(12, 3))
fft_lst_1,fft_lst_0=[],[]
for ax, im, nm in zip(axes[0], zeros_im, zeros_names):
    #contours definition using marching squares
    contours = measure.find_contours(im, 1)
    #signal creation
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])#contours definition using marching squares
    fourier=fft(signal)
    fft_lst_0.append(fourier)
    ax.imshow(im,cmap='gray')
    ax.axis('off')
    ax.set_title(nm)
#loop on ones
for ax, im, nm in zip(axes[1], ones_im, ones_names):
    #contours definition using marching squares
    contours = measure.find_contours(im, 1)
    #signal creation
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])#contours definition using marching squares
    fourier=fft(signal)
    fft_lst_1.append(fourier)
    ax.imshow(im,cmap='gray')
    ax.axis('off')
    ax.set_title(nm)
    
fig2 = plt.figure()
for i,fourier in enumerate(fft_lst_0):
    plt.stem(abs(fft_lst_0[i]),label='0')
    plt.stem(abs(fft_lst_1[i]),label='1',markerfmt='og')
plt.xlim(0,10)
plt.show()