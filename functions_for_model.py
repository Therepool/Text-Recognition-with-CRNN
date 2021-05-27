import os
import fnmatch
import cv2
import numpy as np
import string


from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
import keras.backend as K

char_list = string.ascii_letters+string.digits


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return dig_lst


def prepare_data(path):
    # lists for training dataset
    training_img = []
    training_txt = []
    train_input_length = []
    train_label_length = []
    orig_txt = []

    # lists for validation dataset
    valid_img = []
    valid_txt = []
    valid_input_length = []
    valid_label_length = []
    valid_orig_txt = []

    max_label_len = 0

    i = 1
    flag = 0

    for root, dirnames, filenames in os.walk(path):

        for f_name in fnmatch.filter(filenames, '*.jpg'):
            # read input image and convert into gray scale image
            img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)

            # convert each image of shape (32, 128, 1)
            w, h = img.shape
            if h > 128 or w > 32:
                continue
            if w < 32:
                add_zeros = np.ones((32 - w, h)) * 255
                img = np.concatenate((img, add_zeros))

            if h < 128:
                add_zeros = np.ones((32, 128 - h)) * 255
                img = np.concatenate((img, add_zeros), axis=1)
            img = np.expand_dims(img, axis=2)

            # Normalize each image
            img = img / 255.

            # get the text from the image
            txt = f_name.split('_')[1]

            # compute maximum length of the text
            if len(txt) > max_label_len:
                max_label_len = len(txt)

            # split the data into validation and training dataset as 10% and 90% respectively
            if i % 10 == 0:
                valid_orig_txt.append(txt)
                valid_label_length.append(len(txt))
                valid_input_length.append(31)
                valid_img.append(img)
                valid_txt.append(encode_to_labels(txt))
            else:
                orig_txt.append(txt)
                train_label_length.append(len(txt))
                train_input_length.append(31)
                training_img.append(img)
                training_txt.append(encode_to_labels(txt))

            i += 1

    return {'training_img': training_img,
            'training_txt': training_txt,
            'train_input_length': train_input_length,
            'train_label_length': train_label_length,
            'orig_txt': train_label_length,
            'valid_img': valid_img,
            'valid_txt': valid_txt,
            'valid_input_length': valid_input_length,
            'valid_label_length': valid_label_length,
            'valid_orig_txt': valid_orig_txt,
            'max_label_len': max_label_len}



def create_model():

    # input with shape of height=32 and width=128

    inputs = Input(shape=(32, 128, 1))

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    # pooling layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
    # pooling layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs, outputs)

    return {'model': act_model, 'inputs': inputs, 'outputs': outputs}


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def predict_one_image(filename):

    global result
    char_list = string.ascii_letters + string.digits

    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

    w, h = img.shape

    dsize = (128, 32)

    if w < 32:
        add_zeros = np.ones((32 - w, h)) * 255
        img = np.concatenate((img, add_zeros))


    if h < 128:
        add_zeros = np.ones((32, 128 - h)) * 255
        img = np.concatenate((img, add_zeros), axis=1)

    if w > 32 or h > 128:

        img = cv2.resize(img, dsize)

    img = np.expand_dims(img, axis=2)

    # Normalize each image
    img = img / 255.

    img = img.reshape(-1, 32, 128, 1)

    model_results = create_model()

    model = model_results['model']

    # load the saved best model weights
    model.load_weights('trained_model.hdf5')

    # predict outputs on validation images
    prediction = model.predict(img)

    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                   greedy=True)[0][0])

    i = 0
    for x in out:

        result = ""

        for p in x:
            if int(p) != -1:
                result = result + char_list[int(p)]
        i += 1


    print(result)

    return result
