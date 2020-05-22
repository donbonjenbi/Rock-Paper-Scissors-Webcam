''' utility script for running a preprocessing pipeline on all images within a directory'''

import numpy as np
import cv2
import os
import shutil
import random
import time
from os import listdir


DIRECTORY = 'datasets/unprocessed_external_RPS/rockpaperscissors'
# DIRECTORY = 'datasets/unprocessedRPS/test_dir'
COPY_TO_NEW_DIR = True
COUNTER = 0


def process_image(filepath):
	global COUNTER 
	COUNTER += 1
	print("#",COUNTER)
	img = cv2.imread(filepath)
	# print("loading image...", img.shape, filepath)

	## PROCESSING PIPELINE 
	#  -------------------
	#
	for _ in range(random.randint(0,2)): # randomly choose between 0°, 90°, and 180° rotation
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
	# img = crop_to_square(img)

	img = pad_to_square(img)
	bkg = crop_to_square(get_random_background())
	mask = get_chroma_mask(img, hsv_bounds = [[45,50,50],[150,255,255]]) # green
	img = scale_hue_in_masked(img, mask, hue_target = 15, blur = 15)
	img = replace_background(img, bkg, mask)
	# 
	#  -------------------

	# save and overwrite the original image
	# print("Saving image... ", img.shape, filepath)
	cv2.imwrite(filepath, img)


def show_result(img, wait_time = 500, print_text = False):
	scale = 1.0
	disp_img = cv2.resize(img, None, fx=scale, fy=scale)
	if print_text:
		print(print_text)
		print(img[:3,:3])
		print(img[120:123,120:123])
	cv2.imshow('image',disp_img)
	cv2.waitKey(wait_time) # time to wait to wait in ms.  0 will hold until keypress
	cv2.destroyAllWindows()


def crop_to_square(img):
	if img.shape[0] < img.shape[1]:
		margin = int((img.shape[1]-img.shape[0]) / 2)
		img = img[:,margin:-margin]
	elif img.shape[0] > img.shape[1]:
		margin = int((img.shape[0]-img.shape[1]) / 2)
		img = img[margin:-margin,:] 
	return img


def pad_to_square(img, fill_color = [50,200,60]): # green = [50,200,60]
	if img.shape[0] < img.shape[1]:
		margin = int((img.shape[1]-img.shape[0]) / 2)
		img_copy = cv2.copyMakeBorder(img,margin,margin,0,0,cv2.BORDER_CONSTANT, value = fill_color)
	elif img.shape[0] > img.shape[1]:
		margin = int((img.shape[0]-img.shape[1]) / 2)
		img_copy = cv2.copyMakeBorder(img,0,0,margin,margin,cv2.BORDER_CONSTANT, value = fill_color)
	return img_copy


def get_chroma_mask(img, hsv_bounds): 
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_bound = np.array(hsv_bounds[0])
	upper_bound = np.array(hsv_bounds[1])
	# create mask
	mask = cv2.inRange(hsv, lower_bound, upper_bound)
	mask = cv2.bitwise_not(mask)
	# smooth & soften the mask edges
	mask = cv2.erode(mask,None, iterations = 3)
	mask = cv2.dilate(mask,None, iterations = 1)
	mask = cv2.GaussianBlur(mask, (13, 13), 0)
	mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
	mask = cv2.GaussianBlur(mask, (5, 5), 0)
	return mask


def change_hue_in_range(img, hsv_range, hue_target, blur): 
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_bound, upper_bound = np.zeros(img.shape), np.zeros(img.shape)
	lower_bound[:,:] = hsv_range[0]
	upper_bound[:,:] = hsv_range[1]

	in_range = (hsv > lower_bound) & (hsv < upper_bound)
	# new_img = [[(hsv[i,j,:] + hsv_shift if np.all(in_range[i,j,:]) else hsv[i,j,:]) for j in range(in_range.shape[1])] for i in range(in_range.shape[0])]

	new_img = hsv
	for i in range(hsv.shape[0]):
		for j in range(hsv.shape[1]):
			if np.all(in_range[i,j,:]):
				new_img[i,j,0] = new_img[i,j,0] + (hue_target - new_img[i,j,0])*0.75
	
	new_img = np.array(new_img).astype(np.uint8)
	return cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)


def scale_hue_in_masked(img, mask, hue_target, blur): 
	hue_target = hue_target / 255
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) / 255

	mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)[1]
	mask = cv2.GaussianBlur(mask, (blur,blur),0)
	mask = 1 - (mask[:,:] / 255)

	hsv[:,:,0] = hsv[:,:,0] + (hue_target - hsv[:,:,0]) * mask

	hsv = np.array(hsv * 255).astype(np.uint8)
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def get_random_background():
	SOURCE = 'datasets/unprocessed_external_RPS/AirBNB_photos'
	BKG_TYPES = [
		# 'backyard',
		'bathroom',
		'bedroom',
		# 'decor',
		'dining-room',
		# 'entrance',
		# 'house-exterior',
		'kitchen',
		'living-room',
		]
	num_per_type = [len([f for f in os.listdir(os.path.join(SOURCE,BKG_TYPES[i]))]) for i in range(len(BKG_TYPES))]
	weighted_indexes = [j for dim in [[[i]] * num_per_type[i] for i in range(len(BKG_TYPES))] for j in dim]
	dir_index = random.choice(weighted_indexes)[0]
	dir_path = os.path.join(SOURCE,BKG_TYPES[dir_index])
	filenames = os.listdir(dir_path)
	filepath = os.path.join(dir_path,random.choice(filenames))
	rand_img = cv2.imread(filepath)
	# print("Loading bkg... ", rand_img.shape, filepath)
	return rand_img

def compile_last_folder(save_dir, num_imgs):
	save_dir = save_dir + "/none"
	try:
		os.mkdir(save_dir)
	except:
		shutil.rmtree(save_dir)
		os.mkdir(save_dir)

	for i in range(num_imgs):
		print("none #",i)
		filepath = save_dir + "/none_" + str(i).zfill(4) + ".png"
		img = crop_to_square(get_random_background())
		img = cv2.resize(img, (300,300))
		cv2.imwrite(filepath, img)



def replace_background(img, bkg, mask):
	fg_mask = mask
	bkg_mask = 255 - fg_mask
	fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
	bkg_mask = cv2.cvtColor(bkg_mask, cv2.COLOR_GRAY2BGR)
	bkg = cv2.resize(bkg, img.shape[:2])
	fg_part = (img / 255.0) * (fg_mask / 255.0)
	bkg_part = (bkg / 255.0) * (bkg_mask / 255.0)
	return np.uint8(cv2.addWeighted(fg_part, 255.0, bkg_part, 255.0, 0.0))


def process_dir(local_dir):
	'''recursively processes all files within the "local_dir" '''
	if os.path.isdir(local_dir):
		files = listdir(local_dir)
		for f in files:
			if f != '.DS_Store':
				local_file = os.path.join(local_dir,f)
				if os.path.isdir(local_file):
					process_dir(local_file)
				else:
					process_image(local_file)


if __name__ == '__main__':
	if COPY_TO_NEW_DIR:
		new_dir = DIRECTORY + "_copy"
		# try:
		# 	shutil.copytree(DIRECTORY,new_dir)
		# except: 
		# 	shutil.rmtree(new_dir)
		# 	"WARNING: copy folder already exists. removing it"
		# 	shutil.copytree(DIRECTORY,new_dir)
		# process_dir(new_dir)
		compile_last_folder(new_dir, 720)
	else:
		process_dir(DIRECTORY)
		compile_last_folder(DIRECTORY, 720)






