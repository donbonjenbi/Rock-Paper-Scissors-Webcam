
# Rock Paper Scissors Webcam
This is a project to create a deep learning-based prediction model for real-time predicting "Rock" "Paper" or "Scissors" within a browser's webcam stream.  

To interact with it working live, visit: https://www.donbonjenbi.com

![Rock Paper Scissors Example](misc/example_screenshots.png)

It uses tensorflow.keras for model training, and tensorflowjs for model deployment.  

The model consists of: 
	- A MobileNetV2 backbone, starting with weights pre-trained on imagenet (https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
	- With the head replaced with our custom output layers 
	- With the weights re-trained on our custom dataset (/model_training/datasets/donbonjenbi_RPS_dataset2)
	- and finally converted from keras_model => tfjs for compression & final deployment


The deployment package uses:
	- flask as the webframework & URL routing
	- openCv.js for image preprocessing
	- `RPS_webcam_js/RPS_webcam/static/script.js` for processing the webcam stream & parsing the predictions
	- tensorflowjs for model deployment



## To train a model
- start your virtualenv
- navigate to the `/model_training` directory
- install required packages with pip: 
```
pip install -r requirements.txt
```
- Set the necessary variables for training: 
```
# Loading the data:  
# 	if DATA_SOURCE is 'tfds' => downloads the official tfds dataset.  
# 	if DATA_SOURCE is 'local_folder' => uses dataset contained in LOCAL_DATASET_DIR
DATA_SOURCE = 'tfds'  
LOCAL_DATASET_DIR = 'datasets/donbonjenbi_RPS_dataset2'
CLASS_NAMES = ['rock','paper','scissors','none'] # only grabs from folders that match these names
REFRESH_DATASET = False 		# set this to true for the first run colab. it will load a local copy of the training data from G-drive to local folder before running training.  

# Training hyperparameters
LR_FIRST_ROUND = 0.0001 		# typically 0.0001
EPOCHS_FIRST_ROUND = 1 			# typically 10, reduce to 1 for initial test
LR_SECOND_ROUND = 0.000005  	# typically 0.000001 to 0.000005
EPOCHS_SECOND_ROUND = 1 		# typically 200, reduce to 1 for initial test
EARLY_STOPPING_PATIENCE = 6

# Saving the model
MODEL_NAME = 'RockPaperScissors_model' 
MODEL_VERSION = 1  				# version_num to use while saving the new model
```
- run training:
```
python train_model.py
```


## To deploy the code & run on your local host
- Either use the default model, or replace the model files in `/RPS_webcam_js/RPS_webcam/static/models` with your preferred version.  
- start your virtualenv
- navigate to the `/RPS_webcam_js` directory:  
```
cd RPS_webcam_js
```
- install required packages with pip: 
```
pip install -r requirements.txt
```
- start the flask server: 
```
python3 main.py
```
- once the server is running, open port 8080 of the localhost in your web browser of choice to view it.  
```
127.0.0.1:8080
```
or
```
localhost:8080
```

## Enjoy! 
