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

fft_lst_1,fft_lst_0=[],[]
fft_lst_2,fft_lst_3=[],[]
fig, axes = plt.subplots(4, len(zeros_im), figsize=(12, 3))
for ax, im, nm in zip(axes[0], zeros_im, zeros_names):
    #contours definition using marching squares
    contours = measure.find_contours(im, 0.8)
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])
    fourier=fft(signal)
    fft_lst_0.append(fourier)
    ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.axis('off')
for ax, im, nm in zip(axes[1], ones_im, ones_names):
    #contours definition using marching squares
    contours = measure.find_contours(im, 0.8)
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])
    fourier=fft(signal)
    fft_lst_1.append(fourier)
    ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.axis('off')
for ax, im, nm in zip(axes[2], twos_im, twos_names):
    #contours definition using marching squares
    contours = measure.find_contours(im, 0.8)
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])
    fourier=fft(signal)
    fft_lst_2.append(fourier)
    ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.axis('off')
for ax, im, nm in zip(axes[3], threes_im, threes_names):
    #contours definition using marching squares
    contours = measure.find_contours(im, 0.8)
    signal = contours[0][:, 0]+(1j*contours[0][:, 1])
    fourier=fft(signal)
    fft_lst_3.append(fourier)
    ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.axis('off')

fig2, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 5))
for i,fourier in enumerate(fft_lst_0):
    ax1.stem(abs(fft_lst_0[i][0:10]),label='0')
    ax1.stem(abs(fft_lst_1[i][0:10]),label='1',markerfmt='C1o')
    ax1.stem(abs(fft_lst_2[i][0:10]),label='2',markerfmt='C2o')
    ax1.stem(abs(fft_lst_3[i][0:10]),label='3',markerfmt='C3o')
    
    ax2.stem(abs(fft_lst_0[i][1:2]),label='0')
    ax2.stem(abs(fft_lst_1[i][1:2]),label='1',markerfmt='C1o')
    ax2.stem(abs(fft_lst_2[i][1:2]),label='2',markerfmt='C2o')
    ax2.stem(abs(fft_lst_3[i][1:2]),label='3',markerfmt='C3o')
ax1.set_title('10 first descriptors')
ax2.set_title('1st descriptor (module)')
plt.show()