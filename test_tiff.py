from math import remainder
import os
import cv2
from PIL import Image
from skimage import io
from sklearn.tree import plot_tree
from tifffile import imread
import torch
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange
import matplotlib.pyplot as plt

path_im = os.path.join(os.getcwd(), 'assets/mvadlmi/train/train/0a619ab32b0cd639d989cce1e1e17da0.tiff')
# path_im = os.path.join(os.getcwd(), 'assets/mvadlmi/train/train/0bb6facc3ebad1582fca77968ff757c6.tiff')

# im_scale = imread(path_im, key=0)
im_scale = imread(path_im, key=1)
# im_scale = imread(path_im, key=2)

im_tensor = torch.from_numpy(im_scale)
quantity_to_pad = im_tensor.shape[0] - im_tensor.shape[1]
padded_im_tensor = F.pad(im_tensor, pad=(0, 0, 0, quantity_to_pad), mode='constant', value=255).unsqueeze(0)

patch_size = 128

remaining_pixels = padded_im_tensor.shape[1] % patch_size
if remaining_pixels != 0:
    if (padded_im_tensor.shape[1] + remaining_pixels) % patch_size == 0 :
        # padd
        padded_im_tensor = F.pad(im_tensor, pad=(0, remaining_pixels, 0, remaining_pixels), mode='constant', value=255).unsqueeze(0)
    else:
        # crop
        padded_im_tensor = padded_im_tensor[:, 0:padded_im_tensor.shape[1]-remaining_pixels, 0:padded_im_tensor.shape[2]-remaining_pixels, :]

h=(padded_im_tensor.shape[1] // patch_size)
w=(padded_im_tensor.shape[2] // patch_size)
output_patches = rearrange(padded_im_tensor, 'b (h p1) (w p2) c -> b (h w) p1 p2 c', p1=patch_size, p2=patch_size, h=h, w=w)

percentage_blank = 0.2

mask = (1.0*(output_patches == 255)).sum(dim=(2, 3, 4))/(patch_size*patch_size*3) < percentage_blank # remove patch with only blanks pixels
non_white_patches = output_patches[mask]

print("end")