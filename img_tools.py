########################
# Image IO Tools
# Writer: kkang
# Final_update: 22.12.26
########################

import cv2
import torch
import numpy as np
from PIL import Image

############
# Preprocess
def numpyimg2tensor(images, channel_order='RGB'):
	"""Pre-processes images from `numpy.ndarray` to `torch.Tensor`."""
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

def pilimg2tensor(images):
	"""Pre-processes images from `PIL.Image` to `torch.Tensor`."""
	# input : uint8, 0~255, RGB, HWC
	# output: tensor, -1~1, RGB, BCHW
	max_val = 1.0
	min_val = -1.0

	print('!!!!!!!!!!!!!!!!!')
	print('not yet')
	raise

#############
# Postprocess
def tensor2numpyimg(input_tensor):
	"""Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
	# input : tensor, -1~1, RGB, BCHW
	# output : np.uint8, 0~255, RGB, BHWC
	input_tensor = input_tensor.detach().cpu().numpy()
	input_tensor = (input_tensor + 1.) * 255. / 2.
	output_image = np.clip(input_tensor + 0.5, 0, 255).astype(np.uint8)

	output_image = output_image.transpose(0, 2, 3, 1)
	if output_image.shape[-1] == 1:
		return output_image[:,:,:,0]
	elif output_image.shape[-1] == 1:
		return output_image
