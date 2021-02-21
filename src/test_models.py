import os
import numpy as np
import matplotlib.pyplot as plt

import imghdr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2

batch_size = 16
drop_rate = 0.2

max_img_height = 100
max_img_width = 200
min_img_height = 30
min_img_width = 30

epochs = 20

# read entire file 
f = open('../data/iam/words.txt', mode='r')
raw = f.read()
f.close()

# remove descriptive lines
lines = raw.split('\n')
while lines[0][0] == '#':
    del lines[0]   

# convert file to data dictionary
data = [ (line.split(' ')[0:8]) + [' '.join(line.split(' ')[8:])] for line in lines ]
data = [{'id':x[0], 'quality':x[1], 'graylevel':x[2], 'x':int(x[3]), 'y':int(x[4]), 'w':int(x[5]), 'h':int(x[6]), 'tag':x[7], 'label':x[8] } for x in data]

# filter out data with segmentation quality 'err' or width too small 
data = list(filter(lambda x: x['quality']=='ok' and x['w']>=min_img_width and x['h']>=min_img_height and x['w']<=max_img_width and x['h']<=max_img_height, data))

images = np.array(['C:/Users/samgo/Desktop/cs512-f20-samuel-golden/project/data/iam/words/' + x['id'] + '.png' for x in data])
labels = np.array([x['label'] for x in data])

alphabet = set(c for label in [x['label'] for x in data] for c in label)

max_label_length = max(len(x['label']) for x in data)
print(max_label_length)

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(alphabet))
print("Characters present: ", alphabet)

char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(alphabet), num_oov_indices=0, mask_token=None
)

num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def split_data(images, labels, train_prop=0.7, valid_prop=0.2, test_prop=0.1):
    # ensure valid proportions
    assert round(train_prop + valid_prop + test_prop) == 1, 'proportions should sum to 1'
    
    count = len(images)
    # shuffle indices 
    idxs = np.arange(count)
    np.random.shuffle(idxs)
    
    # calculate indices
    train_idx = int(count * train_prop)
    valid_idx = int(count * valid_prop) + train_idx
    
    # split input data accordingly
    trainI, trainL = images[idxs[:train_idx]], labels[idxs[:train_idx]]
    validI, validL = images[idxs[train_idx:valid_idx]], labels[idxs[train_idx:valid_idx]]
    testI, testL = images[idxs[valid_idx:]], labels[idxs[valid_idx:]]
    
    return (trainI, trainL), (validI, validL), (testI, testL)
    
def resize_image_and_encode_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    #img = tf.image.resize(img, [max_img_height, max_img_width])
    img = 1 - img
    img = tf.image.resize_with_pad(img, max_img_height,max_img_width, method='bilinear')
    img = 1 - img
    img = tf.transpose(img, perm=[1, 0, 2])
    
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    
    while tf.shape(label)[0] < max_label_length:
        label = tf.concat([label, [len(alphabet)]],0)

    return {"image": img, "label": label}

(trainI, trainL), (validI, validL), (testI, testL) = split_data(images, labels)
print(trainI.shape)
print(trainL.shape)
print(validI.shape)
print(validL.shape)
print(testI.shape)
print(testL.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((trainI, trainL))

train_dataset = (
    train_dataset.map(
        resize_image_and_encode_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

valid_dataset = tf.data.Dataset.from_tensor_slices((validI, validL))
valid_dataset = (
    valid_dataset.map(
        resize_image_and_encode_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

test_dataset = tf.data.Dataset.from_tensor_slices((testI, testL))
test_dataset = (
    test_dataset.map(
        resize_image_and_encode_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def lambda_reverse_layer_A():
    return layers.Lambda(lambda t: keras.backend.reverse(t, 0))
    
def lambda_reverse_layer_B():
    return layers.Lambda(lambda t: keras.backend.reverse(t, 1))    

def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(max_img_width, max_img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((max_img_width // 4), (max_img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(alphabet) + 1, activation="softmax", name="dense_out")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


def LSTM4D(inp, mdlstm_units, dense_units, return_sequences=False, dense_act='tanh'):
	#w = layers.Bidirectional(layers.LSTM(mdlstm_units, return_sequences=return_sequences, dropout=drop_rate))(inp)
	w = layers.LSTM(mdlstm_units, return_sequences=return_sequences, dropout=drop_rate)(inp)
	w = layers.Dense(dense_units, activation=dense_act)(w)

	x = lambda_reverse_layer_A()(inp)
	#x = layers.Bidirectional(layers.LSTM(mdlstm_units, return_sequences=return_sequences, dropout=drop_rate))(x)
	x = layers.LSTM(mdlstm_units, return_sequences=return_sequences, dropout=drop_rate)(x)
	x = layers.Dense(dense_units, activation=dense_act)(x)
	x = lambda_reverse_layer_A()(x)

	y = lambda_reverse_layer_B()(inp)
	#y = layers.Bidirectional(layers.LSTM(mdlstm_units, return_sequences=return_sequences, dropout=drop_rate))(y)
	y = layers.LSTM(mdlstm_units, return_sequences=return_sequences, dropout=drop_rate)(y)
	y = layers.Dense(dense_units, activation=dense_act)(y)
	y = lambda_reverse_layer_B()(y)

	z = lambda_reverse_layer_A()(inp)
	z = lambda_reverse_layer_B()(z)
	#z = layers.Bidirectional(layers.LSTM(mdlstm_units, return_sequences=return_sequences, dropout=drop_rate))(z)
	z = layers.LSTM(mdlstm_units, return_sequences=return_sequences, dropout=drop_rate)(z)
	z = layers.Dense(dense_units, activation=dense_act)(z)
	z = lambda_reverse_layer_B()(z)
	z = lambda_reverse_layer_A()(z)

	added = layers.Add()([w,x,y,z])

	return added


def LSTMCell4D(inp, mdlstm_units, dense_units, return_sequences=False, dense_act='tanh'):
	w = layers.RNN(layers.LSTMCell(mdlstm_units), return_sequences=True)(inp)
	w = layers.Dense(dense_units, activation=dense_act)(w)

	x = lambda_reverse_layer_A()(inp)
	x = layers.RNN(layers.LSTMCell(mdlstm_units), return_sequences=True)(x)
	x = layers.Dense(dense_units, activation=dense_act)(x)
	x = lambda_reverse_layer_A()(x)

	y = lambda_reverse_layer_B()(inp)
	y = layers.RNN(layers.LSTMCell(mdlstm_units), return_sequences=True)(y)
	y = layers.Dense(dense_units, activation=dense_act)(y)
	y = lambda_reverse_layer_B()(y)

	z = lambda_reverse_layer_A()(inp)
	z = lambda_reverse_layer_B()(z)
	z = layers.RNN(layers.LSTMCell(mdlstm_units), return_sequences=True)(z)
	z = layers.Dense(dense_units, activation=dense_act)(z)
	z = lambda_reverse_layer_B()(z)
	z = lambda_reverse_layer_A()(z)

	added = layers.Add()([w,x,y,z])

	return added


def build_LSTM_model():
    dense_act = 'tanh'
    
    input_img = layers.Input(shape=(max_img_width, max_img_height, 1), name='image', dtype='float32')
    labels = layers.Input(name='label', shape=(None,), dtype='float32')

    inp = layers.Reshape(target_shape=(max_img_width, max_img_height))(input_img)

    mdlstm1 = LSTM4D(inp, 2, 4, return_sequences=True)
    mdlstm2 = LSTM4D(mdlstm1, 10, 16, return_sequences=True)
    mdlstm3 = LSTM4D(mdlstm2, 50, 40, return_sequences=True)
    
    out = layers.Dense(len(alphabet) + 1, activation='softmax', name='dense_out')(mdlstm3)
    classified = CTCLayer(name='ctc_loss')(labels, out)

    model = keras.models.Model(inputs=[input_img, labels], outputs=classified, name='ocr_mdlstm_test')

    model.compile(optimizer=keras.optimizers.Adam())

    return model

def build_LSTMCellwRNN_model(mdlstm_units=32, dense_units=200):
    dense_act = 'tanh'
    
    input_img = layers.Input(shape=(max_img_width, max_img_height, 1), name='image', dtype='float32')
    labels = layers.Input(name='label', shape=(None,), dtype='float32')

    input_reshaped = layers.Reshape(target_shape=(max_img_width, max_img_height))(input_img)

    x = layers.RNN(layers.LSTMCell(mdlstm_units), return_sequences=True)(input_reshaped)
    x = layers.Dense(100, activation=dense_act, name='x_out')(x)

    y = layers.Permute((2,1))(input_reshaped)
    y = layers.RNN(layers.LSTMCell(mdlstm_units), return_sequences=True)(y)
    y = layers.Dense(200, activation=dense_act, name='y_out')(y)
    y = layers.Permute((2,1))(y)
    print(x)
    print(y)
    
    added = layers.Add()([x,y])
    out = layers.Dense(len(alphabet) + 1, activation='softmax', name='dense_out')(added)
    classified = CTCLayer(name='ctc_loss')(labels, out)

    model = keras.models.Model(inputs=[input_img, labels], outputs=classified, name='LSTMlayerModel')

    model.compile(optimizer=keras.optimizers.Adam())

    return model


def save_model(id,model):
	modelJSON = model.to_json()
	with open('../data/models/'+str(id)+'.json', 'w') as f:
		f.write(modelJSON)
	model.save_weights('../data/models/'+str(id)+'.h5')
	print("Saved model "+str(id)+" to data directory.")


# Get the model
model = build_LSTM_model()
model.summary()


early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

save_model((str(model.name)+str(batch_size)+'batchsize'+str(epochs)+'epochs'), model)

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense_out").output
)
prediction_model.summary()

save_model((str(prediction_model.name)+str(batch_size)+'batchsize'+str(epochs)+'epochs'), prediction_model)

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_label_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Let's check results on some validation samples
for batch in test_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i].replace('[UNK]','')}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")

plt.show()