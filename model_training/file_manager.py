

import os
import numpy as np
import shutil
import random


SOURCE_DIR = {  # value is the "max" number of images to pull from each folder
	'datasets/my_RPS_images': 650, # 650
	'datasets/processed_external_RPS/other_sources': 65,
	'datasets/processed_external_RPS/rockpaperscissors_copy': 650,
	'datasets/tfds_RPS_images': 650,
	}

SAVE_DIR = 'datasets/full_RPS_dataset'
CLASSES = ['rock','paper','scissors','none']
SPLIT_NAMES = ['train','val','test']
SPLIT = [0.6, 0.3, 0.1] # [train, val, test]


def create_splits(source_dir, split):
	src_names, orig_totals  = load_directory(source_dir)
	max_images = SOURCE_DIR[source_dir]
	class_max = [total if total < max_images else max_images for total in orig_totals]

	train_names, val_names, test_names = {},{},{}
	for i in range(len(CLASSES)):
		random.shuffle(src_names[i])
		train_split = int(class_max[i]*SPLIT[0])
		val_split = train_split + int(class_max[i]*SPLIT[1])

		train_names[CLASSES[i]] = src_names[i][:train_split]
		val_names[CLASSES[i]] = src_names[i][train_split:val_split]
		test_names[CLASSES[i]] = src_names[i][val_split:class_max[i]]

	splits = {
		'train': train_names,
		'val': val_names,
		'test': test_names,
		}

	split_info = {
		'source_folder' : source_dir,
		'classes': CLASSES,
		'total_in_folder': orig_totals,
		'total_split': class_max,
		'train_split': [len(splits['train'][class_name]) for class_name in splits['train']],
		'val_split': [len(splits['val'][class_name]) for class_name in splits['val']],
		'test_split': [len(splits['test'][class_name]) for class_name in splits['test']]
	}
	
	for key in split_info.keys():
		print(key, split_info[key])

	return splits, split_info


def copy_to_dir(source_dir, save_dir, img_names):
	print("\nCopying files... \nfrom: ", source_dir, "\nto:", save_dir)
	count_a = 0
	for batch_name in img_names.keys():
		print("\t/",batch_name)
		for class_name in img_names[batch_name].keys():
			count_b = 0
			for filename in img_names[batch_name][class_name]:
				load_path = os.path.join(source_dir, class_name, filename)
				save_path = os.path.join(save_dir,batch_name,class_name)
				try: 
					os.makedirs(save_path)
				except:
					pass
				shutil.copy2(load_path, save_path)
				count_a += 1
				count_b += 1
			print("\t\t/",class_name,":",count_b,"files")
	print("total ", count_a, "files copied ")


def remove_from_dir(source_dir, target_dir):
	print("\nRemoving files...\n\tsource names:", source_dir, "\n\ttarget dir:", target_dir)
	count = 0
	for src_root, _, src_files in os.walk(source_dir):
		for src_file in src_files:
			if src_file != '.DS_Store':
				# check if it exists in the target_dir
				for tgt_root, tgt_dirs, tgt_files in os.walk(target_dir):
					for tgt_dir in tgt_dirs:
						tgt_path = os.path.join(tgt_root,tgt_dir,src_file)
						if os.path.exists(tgt_path):
							os.remove(tgt_path)
							count += 1
	print("Removed", count, "files...\n")



	# if os.path.isdir(local_dir):
	# 	files = listdir(local_dir)
	# 	for f in files:
	# 		if f != '.DS_Store':
	# 			local_file = os.path.join(local_dir,f)
	# 			if os.path.isdir(local_file):
	# 				process_dir(local_file)
	# 			else:
	# 				process_image(local_file)


def load_directory(directory): 
	filenames = [[name for name in os.listdir(os.path.join(directory, class_name)) if (os.path.isfile(os.path.join(directory,class_name,name)) and name != '.DS_Store')] for class_name in CLASSES]
	total_per_class = [len(class_type) for class_type in filenames]
	return filenames, total_per_class

def get_size(directory):
	size_MB = round(sum(os.path.getsize(os.path.join(dirpath,filename)) for dirpath, dirnames, filenames in os.walk(directory) for filename in filenames ) / (1024*1024),1)
	return size_MB

if __name__ == '__main__':
	try: 
		os.makedirs(SAVE_DIR)
	except:
		pass
	# print class totals in each folder
	for source in SOURCE_DIR.keys():
		_, totals = load_directory(source)
		print(source,":\t",totals, "\t(",get_size(source),"MB )")

	for source in SOURCE_DIR.keys():
		print("\n\n")
		remove_from_dir(source,SAVE_DIR)
		splits, _, = create_splits(source, SPLIT)
		copy_to_dir(source, SAVE_DIR, splits)

	print(f"\n\nFiles in {SAVE_DIR} ({get_size(SAVE_DIR)} MB)")
	for _dir in os.listdir(SAVE_DIR):
		_, total_in_dir = load_directory(os.path.join(SAVE_DIR,_dir))
		print("\t",_dir,":",total_in_dir)










