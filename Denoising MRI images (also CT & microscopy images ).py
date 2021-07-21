# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:49:35 2021

@author: abc
"""

#Introduction of pydicom 

import matplotlib.pyplot as plt
import pydicom
import TiffImagePlugin

#read dataset
dataset = pydicom.dcmread("MRI_Images/CT_small.dcm")

#Extract pixel array from dataset
img = dataset.pixel_array

#see our image how it's look like
plt.imshow(img, cmap=plt.cm.bone)

#save our dataset
plt.imsave("MRI_Images/scm_to_tiff_converted.tif", img, cmap="gray")


######################################################################


#Gaussian 

from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

#Read our original image
noisy_img = img_as_float(io.imread("MRI_Images/MRI_noisy.tif"))

#Read our refernce image
ref_img = img_as_float(io.imread("MRI_Images/MRI_clean.tif"))

#Apply gaussian filter
gaussian_img = nd.gaussian_filter(noisy_img, sigma=5)

#Show the image
plt.imshow(gaussian_img, cmap="gray")

#Save the image
plt.imsave("MRI_Images/Gaussian_smoothed.tif", gaussian_img, cmap="gray")

#Print peark signal noise ratio
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
gaussian_cleaned_psnr = peak_signal_noise_ratio(ref_img, gaussian_img)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", gaussian_cleaned_psnr)


##################################################################################

#Bilateral 

from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

#read our original image
noisy_img = img_as_float(io.imread("MRI_Images/MRI_noisy.tif"))

#read reference image
ref_img = img_as_float(io.imread("MRI_Images/MRI_clean.tif"))

from skimage.restoration import (denoise_tv_bregman, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

#Apply bilateral
denoise_bilateral = denoise_bilateral(noisy_img, sigma_spatial=15, multichannel=False)

#print signal noise ratio
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
bilateral_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_bilateral)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", bilateral_cleaned_psnr)

#save our image
plt.imsave("MRI_Images/bilateral_smoothed.tif", denoise_bilateral, cmap="gray")

#show the image
plt.imshow(denoise_bilateral, cmap="gray")

####################################################################################

#TV Filter
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

#read original image
noisy_img = img_as_float(io.imread("MRI_Images/MRI_noisy.tif"))

#read refernence image
ref_img = img_as_float(io.imread("MRI_Images/MRI_clean.tif"))

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


#Define TV filter
denoise_TV = denoise_tv_chambolle(noisy_img, weight=0.3 , multichannel=False)

#print peak signal noise ratio
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
TV_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_TV)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", TV_cleaned_psnr)

#save our image
plt.imsave("MRI_Images/TV_smoothed.tif", denoise_TV, cmap="gray")

#show the image
plt.imshow(denoise_TV, cmap="gray")



########################################################################################

#Wavelet filter
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

#read original image
noisy_img = img_as_float(io.imread("MRI_Images/MRI_noisy.tif"))

#read reference image
ref_img = img_as_float("MRI_images/MRI_clean.tif")

from skimage.restoration import ( denoise_tv_chambolle,denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

#define wavelet filter
wavelet_smoothed = denoise_wavelet(noisy_img, multichannel=False, method="BayesShrink", mode="soft",
                                   rescale_sigma=True)

#Print peak signal noise ratio
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
Wavelet_cleaned_psnr = peak_signal_noise_ratio(ref_img, wavelet_smoothed)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", Wavelet_cleaned_psnr)

#Save the image
plt.imsave("MRI_Image/wavelet_smoothed.tif", wavelet_smoothed, cmap="gray")

#show the image
plt.imshow(wavelet_smoothed, cmap="gray")

 

##########################################################################################

#Anisotropic diffusion filter
import matplotlib.pyplot as plt
import cv2
from skimage import io
from medpy.filter.smoothing import anisotropic_diffusion
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio

#read our original image
noisy_img = img_as_float(io.imread("MRI_Images/MRI_noisy.tif", as_gray=True))

#read our reference image
ref_img = img_as_float(io.imread("MRI_Images/MRI_clean.tif"))

#define anisotropic filter
img_aniso_filtered = anisotropic_diffusion(noisy_img, niter=50, kappa=50, gamma=0.2, option=2)

#Print peak signal noise ratio 
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
anisotropic_cleaned_psnr = peak_signal_noise_ratio(ref_img, img_aniso_filtered)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", anisotropic_cleaned_psnr)

#save the image
plt.imsave("MRI_Images/anisotropic_denoised.tif", img_aniso_filtered, cmap="gray")

#show the image
plt.imshow(img_aniso_filtered, cmap="gray")





























































