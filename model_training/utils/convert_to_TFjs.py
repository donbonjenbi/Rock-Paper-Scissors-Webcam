import os
from os.path import join
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras


LOAD_DIR = 'models/RockPaperScissors_model'
LOAD_VERSION = 20
LOAD_FILEPATH = join(LOAD_DIR, str(LOAD_VERSION).zfill(4))
SAVE_DIR = LOAD_DIR + "TFjs"
SAVE_FILEPATH = join(SAVE_DIR, str(LOAD_VERSION).zfill(4))



def simple_convert():
	model = load_model()
	print("\nConverting model to tfjs...")
	try:
		os.mkdir(os.path.dirname(SAVE_FILEPATH))
	except:
		print(f"NOTE: save folder already exists.  Overwriting existing model in: '{os.path.dirname(SAVE_FILEPATH)}'")
	tfjs.converters.save_keras_model(model, SAVE_FILEPATH)
	print(f"Saved converted tfjs_model to:  {SAVE_FILEPATH} ({round(get_size(SAVE_FILEPATH) / (1024 * 1024),1)} MB)\n")



def load_model():
	print(f"Loading model {LOAD_FILEPATH}...\n")
	return keras.models.load_model(LOAD_FILEPATH) if LOAD_VERSION > 12 else print("error loading model (suggest using ExCh19 function instead)")



def get_size(target_dir):
	total_size = 0
	for dirpath, dirnames, filenames in os.walk(target_dir):
	    for f in filenames:
	        fp = os.path.join(dirpath, f)
	        # skip if it is symbolic link
	        if not os.path.islink(fp):
	            total_size += os.path.getsize(fp)
	return total_size



if __name__ == '__main__':
	simple_convert()














