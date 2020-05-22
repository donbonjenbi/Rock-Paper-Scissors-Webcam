
const FPS = 30;
const PRED_RATE = 5; // Hz
const _CLASSES = ['Rock','Paper','Scissors','None'];
var predicted_class = 'none';
var preds = [0.25,0.25,0.25,0.25];
var pred_index = 3;
var prev_update_t = Date.now();
var width = 640;
var height = 480;
var model;

const video = document.getElementById("videoElement");
const buffer = document.createElement('canvas');
const canvas = document.getElementById('canvas');
const canvas2 = document.getElementById('canvas2');

load_tf();

if (navigator.mediaDevices.getUserMedia) {       
	navigator.mediaDevices.getUserMedia({
		video: true,
		audio: false,
		facingMode: 'user' // or 'environment'
	}).then(function(stream) {
    	video.srcObject = stream;
	}).catch(function(err0r) {
		console.log("Something went wrong!");
	});
}


async function load_tf(){
	var get_URL = window.location;
	var model_URL = get_URL.protocol + '//' + get_URL.host + "/models/model.json";
	console.log("loading model at:  " + model_URL)
	try {
		model = await tf.loadLayersModel(model_URL);
		console.log("loaded tf");
	} catch (err){
		alert(err);
	}
}


function openCV_render() {
  let begin = Date.now();
  buffer.width = width;
  buffer.height = height;
  buffer.getContext('2d').drawImage(video, 0, 0);

  let src = cv.imread(buffer);

  src = image_processing_pipeline(src)

  cv.imshow('canvas', src);
  src.delete();
  setTimeout(openCV_render, 1000/FPS - (Date.now() - begin));
  // window.requestAnimationFrame(openCV_render);
}



function image_processing_pipeline(src) {
	let img = src

	cv.flip(img, img, 1);

	//draw the rectangle
	let p1 = new cv.Point(0,0);
	let p2 = new cv.Point(350,350);
	// let color = new cv.Scalar(0,0,255,255);
	// cv.rectangle(img,p1,p2,color,2,cv.LINE_AA,0);

	//start a pred update (in the background)
	update_preds(img, p1, p2);

	// update html text
	document.getElementById('pred_text').innerHTML = _CLASSES[pred_index];
	document.getElementById('pred_values').innerHTML = preds.map(function(each_el){return Number(each_el.toFixed(2));});;
	
	return img;
}



async function update_preds(img, p1, p2){
	let current_t = Date.now();

	if (current_t - prev_update_t > 1000/PRED_RATE){
		prev_update_t = current_t;

		// preprocessing pipeline
		let rect = new cv.Rect(p1.x, p1.y, p2.x, p2.y)
		let crop = img.roi(rect);
		cv.imshow('canvas2',crop)
		crop.delete();
		var pred_img = tf.browser.fromPixels(canvas2);
		pred_img = tf.image.resizeBilinear(pred_img,[224,224]).toFloat();
		pred_img = tf.sub(tf.div(pred_img, tf.scalar(127.5)),tf.scalar(1));
		pred_img = tf.reshape(pred_img,[1,224,224,3]);
		pred_img[0] = null;

		if (model) {
			// get the prediction
			preds = await model.predict(pred_img);
			// update the preds
			preds = Array.from(preds.dataSync())
			pred_index = preds.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
		}
	}
}


function openCvReady() {
  cv['onRuntimeInitialized']=()=>{ openCV_render();};
}

openCvReady()








