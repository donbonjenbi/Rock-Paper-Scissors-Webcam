IN_COLAB = False
# %tensorflow_version 2.x  # uncomment if in colab

import os
import sys
import time
import shutil
import requests
import numpy as np 
from os import listdir
import tensorflow as tf
from os.path import isfile
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("using python version: ",sys.version)
print("using Tensorflow version: ",tf.__version__)
print("Using GPU: ", tf.config.experimental.list_physical_devices('GPU'))
# logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)

# Loading the data:  
# 	if DATA_SOURCE is 'tfds' => downloads the official tfds dataset.  
# 	if DATA_SOURCE is 'local_folder' => uses dataset contained in LOCAL_DATASET_DIR
DATA_SOURCE = 'tfds'  
LOCAL_DATASET_DIR = 'datasets/donbonjenbi_RPS_dataset2'
CLASS_NAMES = ['rock','paper','scissors','none'] # only grabs from folders that match these names
REFRESH_DATASET = False # set this to true for the first run colab. it will load a local copy of the training data from G-drive to local folder before running training.  

# Training hyperparameters
LR_FIRST_ROUND = 0.0001 # typically 0.0001
EPOCHS_FIRST_ROUND = 1 # typically 10, reduce to 1 for initial test
LR_SECOND_ROUND = 0.000005  # typically 0.000001 to 0.000005
EPOCHS_SECOND_ROUND = 1 # typically 200, reduce to 1 for initial test
EARLY_STOPPING_PATIENCE = 6

# Saving the model
MODEL_NAME = 'RockPaperScissors_model'
MODEL_VERSION = 1  # the to use when saving the model


def load_and_preprocess_dataset(data_src = 'tfds'):
	print(f"\nLoading the data from {data_src}...")
	
	batch_size = 32
	input_size = (224,224)
	
	def preprocess_image(image,label = None):
		image = tf.image.resize(image, input_size) # note that this does NOT preserve aspect ratio.  Crop to a square first before resizing
		image /= 127.5
		image -= 1.  
		if DATA_SOURCE == 'tfds':
			return image, label
		elif DATA_SOURCE == 'local_folder':
			return image
	
	if data_src == 'tfds': 
		# load the dataset & info dict
		dataset_name = "rock_paper_scissors"
		dataset, info = tfds.load(dataset_name, as_supervised = True, with_info = True)
		dataset_size = info.splits["train"].num_examples

		# separate into train, val, & test splits
		test_set = tfds.load(dataset_name, split = tfds.Split.TEST, as_supervised = True)
		val_set = tfds.load(dataset_name, split ='train[:16%]', as_supervised = True)
		train_set = tfds.load(dataset_name, split ='train[16%:]', as_supervised = True)

		# shuffle and prefetch the data
		train_set = train_set.shuffle(500)
		train_set = train_set.map(preprocess_image).batch(batch_size).repeat().prefetch(1)
		val_set = val_set.map(preprocess_image).batch(batch_size).repeat().prefetch(1)
		test_set = test_set.map(preprocess_image).batch(batch_size).repeat().prefetch(1) 
		
		info_dict = {
			'batch_size': batch_size,
			'data_dir': 'tfds/'+dataset_name,
			'image_shape': list(info.features["image"].shape),
			'labels': info.features["label"].names,
			'train_size': int(round(info.splits["train"].num_examples * (1-0.16),0)),
			'val_size': int(round(info.splits["train"].num_examples * (0.16),0)),
			'test_size': info.splits['test'].num_examples,
			}

		print("class names: ", info_dict['labels'], "\nn_classes: ", info.features["label"].num_classes, "\nimg_shape: ", info_dict['image_shape'], "\nbatch size: ", batch_size, "\ntrain_size: ", info_dict['train_size'],"\nval_size: ", info_dict['val_size'],"\ntest_size: ", info_dict['test_size'])

	if data_src == 'local_folder': 
		data_dir = LOCAL_DATASET_DIR
		if IN_COLAB & REFRESH_DATASET:
			print("transferring training data from G-drive to local folder in colab...")
			shutil.copytree(os.path.join('drive/My Drive/Colab Notebooks/',LOCAL_DATASET_DIR),data_dir)

		datagen = ImageDataGenerator(
			dtype = 'int32', # has no effect on the output... probably a bug?  only outputs float32
			preprocessing_function = preprocess_image,
			rotation_range = 20, # in degrees of random rotation applied
			shear_range = 10, # degrees of random shear intensity
			zoom_range = 0.05, # ratio of max random zoom (or [lower,upper] to specify an offset zoom window)
			brightness_range = (0.5,1.5), # range of random brightness shifts. <1 darkens, >1 lightens, 1 applies no brightness shift
			channel_shift_range = 10, # range for random channel shift
			fill_mode = 'nearest', # one of {"constant", "nearest", "reflect" or "wrap"}
			horizontal_flip = True, # randomly flip along this axis
			vertical_flip = True # randomly flip along this axis
			)

		# load and iterate training dataset
		train_set = datagen.flow_from_directory(os.path.join(data_dir,'train'), class_mode='sparse', batch_size=batch_size, target_size = input_size, classes = CLASS_NAMES)
		val_set = datagen.flow_from_directory(os.path.join(data_dir,'val'), class_mode='sparse', batch_size = batch_size, target_size = input_size, classes = CLASS_NAMES)
		test_set = datagen.flow_from_directory(os.path.join(data_dir,'test'), class_mode='sparse', batch_size=batch_size, target_size = input_size, classes = CLASS_NAMES)
		## WARNING: 'target_size' down-scales without cropping (i.e. squeezes aspect ratio)
		
		info_dict = {
			'batch_size': batch_size,
			'data_dir': data_dir,
			'image_shape': [input_size[0],input_size[1],3],
			'labels': CLASS_NAMES,
			'train_size': train_set.samples,
			'val_size': val_set.samples,
			'test_size': test_set.samples,
			}

		print("class names: ", info_dict['labels'], "\nn_classes: ", len(info_dict["labels"]), "\nimg_shape: ", info_dict['image_shape'], "\nbatch size: ", batch_size, "\ntrain_size: ", info_dict['train_size'],"\nval_size: ", info_dict['val_size'],"\ntest_size: ", info_dict['test_size'])

	return train_set, val_set, test_set, info_dict




def create_model(num_classes = 3):
	# start with a pretrained model trained on ImageNet, replacing the top GlobalAvgPooling2D and Dense output layers
	base_model = keras.applications.mobilenet_v2.MobileNetV2(weights = "imagenet", include_top = False, input_shape = (224,224,3))
	avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
	output = keras.layers.Dense(num_classes, activation = "softmax")(avg)
	model = keras.Model(inputs = base_model.input, outputs = output)
	return model, base_model




def save_model(model, model_name, version):
	print("\nSaving the model locally...")
	directory = "drive/My Drive/Colab Notebooks/models" if IN_COLAB else "models"
	version = str(version).zfill(4)
	filepath = os.path.join(directory, model_name, version)
	print(f"saving model to '{filepath}'")
	tf.saved_model.save(model, filepath)




def train_model(model, base_model, train_set, val_set, test_set, info):
	print("\nTraining the model...")
	early_stopping_cb = keras.callbacks.EarlyStopping( #stops training if error hasnt improved in #epochs = patience
            patience = EARLY_STOPPING_PATIENCE, 
            restore_best_weights = True,
            )
	#freeze the transfer layers for first few epochs of training and start training.  
	for layer in base_model.layers:
	    layer.trainable = False
	optimizer = keras.optimizers.Nadam(learning_rate=LR_FIRST_ROUND, beta_1=0.9, beta_2=0.999)
	model.compile(
	    loss = "sparse_categorical_crossentropy", 
	    optimizer = optimizer, 
	    metrics = ["accuracy"]
	    )
	history = model.fit(
	    train_set, 
	    epochs = EPOCHS_FIRST_ROUND, # was 10, reducing to save runtime
	    steps_per_epoch = info['train_size']//info['batch_size'], 
	    validation_data = val_set,
	    validation_steps = info['val_size']//info['batch_size'], 
	    callbacks = [early_stopping_cb], 
	    verbose = 1,
	    ) 
	# then unfreeze and continue training
	for layer in base_model.layers:
	    layer.trainable = True
	optimizer = keras.optimizers.Nadam(learning_rate=LR_SECOND_ROUND, beta_1=0.9, beta_2=0.999)
	model.compile(
	    loss = "sparse_categorical_crossentropy", 
	    optimizer = optimizer, 
	    metrics = ["accuracy"]
	    )
	history = model.fit(
	    train_set, 
	    epochs = EPOCHS_SECOND_ROUND, 
	    steps_per_epoch = info['train_size']//info['batch_size'], 
	    validation_data = val_set,
	    validation_steps = info['val_size']//info['batch_size'], 
	    callbacks = [early_stopping_cb], 
	    verbose = 1,
	    )  
	history = model.evaluate(
	    test_set, 
	    steps = info['test_size']//info['batch_size'], 
	    )  




def convert_to_tflite(keras_model, save_model = False):
	directory = "drive/My Drive/Colab Notebooks/models" if IN_COLAB else "models"
	
	converter = tf.lite.TFLiteConverter.from_keras_model(keras_model) # use to load from a model object
	tflite_model = converter.convert()
	
	if save_model: 
		save_filename = os.path.join(directory,MODEL_NAME + '_TFLite',str(MODEL_VERSION).zfill(4),'converted_model.tflite')
		try:
			os.mkdir(os.path.dirname(save_filename))
		except:
			print(f"saving tflite version into existing folder: {os.path.dirname(save_filename)}")
		open(save_filename, "wb").write(tflite_model)
	return tflite_model




def load_images_from_folder(folder_path, verbose = True): 
	'''grabs all images within the target folder, 
		loading each into a dictionary containing the image and class label'''
	images = []
	def process_folder(local_folder):
		'''recursively processes all files within the 'local_folder' '''
		if os.path.isdir(local_folder):
			files = listdir(local_folder)
			for f in files:
				if f != '.DS_Store':
					local_file = os.path.join(local_folder,f)
					if os.path.isdir(local_file):
						process_folder(local_file)
					else:
						image_class = os.path.basename(os.path.dirname(local_file))
						class_index = None
						for i in range(len(CLASS_NAMES)):
							if image_class == CLASS_NAME[i]:
								class_index = i
						if verbose:
							print(f"Loading '{local_file}'... class = '{image_class}' [{class_index}]")
						new_image = {
								'image': plt.imread(local_file), 
								'label': class_index,
								'filepath': local_file,
								}
						images.append(new_image)
	process_folder(folder_path)
	print(f"loaded {len(images)} images...")
	return images



if __name__ == '__main__':
	# load the training dataset
	train_set, val_set, test_set, info = load_and_preprocess_dataset(data_src = DATA_SOURCE)
	# create and train the model
	model, base_model = create_model(num_classes = len(info["labels"]))
	# print(model.summary())
	train_model(model, base_model, train_set, val_set, test_set, info)
	# save model locally
	save_model(model, MODEL_NAME, MODEL_VERSION)
	# convert to tflite & save
	tflite_model = convert_to_tflite(model, save_model = True)
	# convert to tfjs and save


