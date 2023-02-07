########################
# Image IO Tools
# Writer: kkang
# Final_update: 22.12.26
########################

import cv2
import torch
import numpy as np
from PIL import Image

#############################################
# preprocess(images, channel_order='RGB', to_tensor)
# input : numpy, np.uint8, 0~255, RGB, BHWC (HWC)
# output : numpy, np.float32, -1~1, RGB, BCHW
#############################################

#############################################
# postprocess(images)
# input : tensor, -1~1, RGB, BCHW
# output : np.uint8, 0~255, BGR, BHWC
#############################################

#############################################
# Lanczos_resizing(image_target, resizing_tuple=(256,256))
# input : -1~1, RGB, BCHW, Tensor
# output : -1~1, RGB, BCHW, Tensor
#############################################

def preprocess(images, channel_order='RGB', to_tensor=False):
    # input  : numpy, np.uint8, 0~255, RGB, (BHWC or HWC)
    # output : numpy, np.float32, -1~1, RGB, BCHW
	
	max_val = 1.0
	min_val = -1.0
	
	if len(images.shape) == 3:
		images = images[np.newaxis, :]
	
	B, H, W, C = images.shape

	if C == 3 and channel_order == 'BGR':
		images = images[:,:,:,::-1]

	images = images / 255.0 * (max_val - min_val) + min_val
	images = images.astype(np.float32).transpose(0, 3, 1, 2)

	if to_tensor:
		images = torch.tensor(images)

	return images

def postprocess(images):
	"""Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
	# input : tensor, -1~1, RGB, BCHW
	# output : np.uint8, 0~255, BGR, BHWC

	images = images.detach().cpu().numpy()
	images = (images + 1.) * 255. / 2.
	images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
	images = images.transpose(0, 2, 3, 1)[:,:,:,[2,1,0]]
	return images

def tensor2numpyimg(images):
	"""Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
	# input : tensor, -1~1, RGB, BCHW
	# output : np.uint8, 0~255, RGB, BHWC

	shape = images.shape
	images = images.detach().cpu().numpy()
	images = (images + 1.) * 255. / 2.
	images = np.clip(images + 0.5, 0, 255).astype(np.uint8)

	images = images.transpose(0, 2, 3, 1)
	if shape[1] == 1:
		return images.squeeze(-1)
	elif shape[1] == 3:
		return images

def numpyimg2tensor(images, channel_order='RGB'):
	"""Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
	# input : np.uint8, 0~255, RGB, (BHWC or HWC)
	# output: tensor, -1~1, RGB, BCHW 

	max_val = 1.0
	min_val = -1.0
	if len(images.shape) == 3:
		images = images[np.newaxis, :]
	
	B, H, W, C = images.shape
	if C == 3 and channel_order == 'BGR':
		images = images[:,:,:,::-1]

	images = images / 255.0 * (max_val - min_val) + min_val
	images = images.astype(np.float32).transpose(0, 3, 1, 2)
	return torch.tensor(images)

def Lanczos_resizing(image_target, resizing_tuple=(256,256)):
	# input : -1~1, RGB, BCHW, Tensor
	# output : -1~1, RGB, BCHW, Tensor
	image_target_resized = image_target.clone().cpu().numpy()
	image_target_resized = (image_target_resized + 1.) * 255. / 2.
	image_target_resized = np.clip(image_target_resized + 0.5, 0, 255).astype(np.uint8)

	image_target_resized = image_target_resized.transpose(0, 2, 3, 1)
	tmps = []
	for i in range(image_target_resized.shape[0]):
		tmp = image_target_resized[i]
		tmp = Image.fromarray(tmp) # PIL, 0~255, uint8, RGB, HWC
		tmp = np.array(tmp.resize(resizing_tuple, PIL.Image.LANCZOS))
		tmp = torch.from_numpy(preprocess(tmp[np.newaxis,:])).cuda()
		tmps.append(tmp)
	return torch.cat(tmps, dim=0)
