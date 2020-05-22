import tensorflow as tf
import numpy as np
import os
import ExCh19
from os.path import join
from tensorflow.keras.preprocessing.image import ImageDataGenerator

load_dir = 'models/RockPaperScissors_model'
load_version = 20
load_filepath = os.path.join(load_dir, str(load_version).zfill(4))
save_dir = load_dir + "_TFLite"



def simple_convert():
	model_name = 'converted_model_simple.tflite'
	save_filename = join(save_dir,str(load_version).zfill(4),model_name)
	# run the conversion
	converter = tf.lite.TFLiteConverter.from_saved_model(load_filepath)
	tflite_model = converter.convert()
	# save the model
	save_model(tflite_model,save_filename)


def default_convert():
	'''
	The simplest way to create a small model is to quantize the weights 
	to 8 bits and quantize the inputs/activations "on-the-fly", during inference. 
	This has latency benefits, but prioritizes size reduction.
	During conversion, set the optimizations flag to optimize for size:
	'''
	model_name = 'converted_model_default.tflite'
	save_filename = join(save_dir,str(load_version).zfill(4),model_name)
	# run the conversion
	converter = tf.lite.TFLiteConverter.from_saved_model(load_filepath)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()
	# save the model
	save_model(tflite_model,save_filename)

def optimize_for_latency_convert():
	model_name = 'converted_model_latency.tflite'
	save_filename = join(save_dir,str(load_version).zfill(4),model_name)
	# run the conversion
	converter = tf.lite.TFLiteConverter.from_saved_model(load_filepath)
	converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	tflite_model = converter.convert()
	# save the model
	save_model(tflite_model,save_filename)

def optimize_for_size_convert():
	model_name = 'converted_model_size.tflite'
	save_filename = join(save_dir,str(load_version).zfill(4),model_name)
	# run the conversion
	converter = tf.lite.TFLiteConverter.from_saved_model(load_filepath)
	converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	tflite_model = converter.convert()
	# save the model
	save_model(tflite_model,save_filename)


def full_integer_quant_convert():
	'''
	We can get further latency improvements, reductions in peak memory usage, 
	and access to integer only hardware accelerators by making sure all model 
	math is quantized. To do this, we need to measure the dynamic range of 
	activations and inputs with a representative data set. 
	You can simply create an input data generator and provide it to our converter.
	'''
	model_name = 'converted_model_full_quant.tflite'
	save_filename = join(save_dir,str(load_version).zfill(4),model_name)
	# get a representative dataset
	ExCh19.LOCAL_DATASET_DIR = 'datasets/my_RPS_test_images'  # change to the location of the representative set
	_, val_set, _, test_set2, info = ExCh19.load_and_preprocess_dataset(data_src = 'local_folder')

	data = val_set.next()[0] 
	ds = tf.data.Dataset.from_tensor_slices((data)).batch(1).repeat()
	
	def representative_dataset_gen():
		for input_value in ds.take(32):
			yield [input_value]

	# run the conversion
	converter = tf.lite.TFLiteConverter.from_saved_model(load_filepath)
	# converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	converter.representative_dataset = representative_dataset_gen
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8
	tflite_model = converter.convert()
	# save the model
	save_model(tflite_model,save_filename)


def save_model(tflite_model, save_filename):
	try:
		os.mkdir(os.path.dirname(save_filename))
	except:
		print(f"saving tflite version into existing folder: {os.path.dirname(save_filename)}")
	open(save_filename, "wb").write(tflite_model)
	file_stats = os.stat(save_filename)
	print(f"saving tflite_model to:  {save_filename} ({round(file_stats.st_size / (1024 * 1024),1)} MB)\n")


if __name__ == '__main__':
	simple_convert()
	default_convert()
	optimize_for_latency_convert()
	optimize_for_size_convert()
	full_integer_quant_convert()













