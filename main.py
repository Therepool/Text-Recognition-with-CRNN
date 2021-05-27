import numpy as np
import string
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from functions_for_model import prepare_data, create_model, ctc_lambda_func



char_list = string.ascii_letters+string.digits

path = 'data'


data_results = prepare_data(path)

# lists for training dataset
training_img = data_results['training_img']
training_txt = data_results['training_txt']
train_input_length = data_results['train_input_length']
train_label_length = data_results['train_label_length']
orig_txt = data_results['orig_txt']

# lists for validation dataset
valid_img = data_results['valid_img']
valid_txt = data_results['valid_txt']
valid_input_length = data_results['valid_input_length']
valid_label_length = data_results['valid_label_length']
valid_orig_txt = data_results['valid_orig_txt']

max_label_len = data_results['max_label_len']



train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = len(char_list))


model_results = create_model()

model = model_results['model']
inputs = model_results['inputs']
outputs = model_results['outputs']


model.summary()


labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

# model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

filepath = "trained_model2.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

batch_size = 32
epochs = 10
model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length],
          y=np.zeros(len(training_img)), batch_size=batch_size, epochs = epochs,
          validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length],
                             [np.zeros(len(valid_img))]), verbose = 1, callbacks = callbacks_list)