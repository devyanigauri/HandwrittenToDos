from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import tensorflow as tf
import keras
import numpy as np

def load_image(filename):
	# load the image
	img = load_img(filename, color_mode = 'grayscale', target_size=(128, 128))

	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 128, 128, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
from keras.initializers import glorot_uniform
def run_example():
    file = input("File name:")
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image",image)
    cv2.imwrite("test_image.jpg",image)
    img = load_image("test_image.jpg")
    fname = '../data/models/testModel'
    loaded_model = keras.models.load_model(fname)

    # with open('ocr_mdlstm_test16batchsize10epochs.json', 'r') as json_file:
    #     json_savedModel= json_file.read()
    # #load the model architecture
    # loaded_model = tf.keras.models.model_from_json(json_savedModel)

    loaded_model.summary()
    pred = np.argmax(loaded_model.predict(img), axis=-1)
    # print("Prediction:",digit[0])
    print("Model:")
    print(pred)
    while True:
        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
run_example()
