import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from keras.utils import plot_model
from tensorflow.keras import layers
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import tsaug as tsa
from scipy.io import loadmat,savemat
from tensorflow.keras.losses import categorical_crossentropy

def data_read(csv_path):
    df = pd.read_csv(csv_path)
    train_list = df.iloc[:,0].tolist()
    valid_list = df.iloc[:,1].tolist()
    test_list = df.iloc[:,2].tolist()
    train_list = [x.split('\\')[-1].split('_')[0] + '_' + x.split('\\')[-1].split('_')[1] + '.mat' for x in train_list]
    valid_list = [x.split('\\')[-1].split('_')[0] + '_' + x.split('\\')[-1].split('_')[1] + '.mat' for x in valid_list if x != "0"]
    test_list = [x.split('\\')[-1].split('_')[0] + '_' + x.split('\\')[-1].split('_')[1] + '.mat' for x in test_list if x != "0"]
    return train_list, valid_list, test_list

def normalize(input_list):
    x_nor = np.copy(input_list)
    for n,signal in enumerate(input_list):
        x_max = np.max(signal)
        x_min = np.min(signal)
        x_mean = np.mean(signal)
        x_nor[n] = (signal-x_min)/(x_max-x_min)
    return x_nor

def one_hot(lab, num_class):
    return np.eye(num_class)[lab]

def data_gen_fn(data_ls, data_pre_fix, batch_size=16, aug=True):
    data_path_ls = shuffle(data_ls)
    i = 0
    augmenter = (
        tsa.AddNoise(scale=0.05) @ 0.5 +
        tsa.TimeWarp(n_speed_change=5, max_speed_ratio=5) @ 0.5 +
        tsa.Drift(max_drift=0.4, n_drift_points=3) @ 0.5)

    while True:
        start = i * batch_size
        # print('len(data_path_ls):',len(data_path_ls))
        end = np.min([start + batch_size, len(data_path_ls)])

        BatchData_mri = list()
        BatchData_pnr = list()
        BatchData_vid = list()
        BatchData_emo = list()
        BatchData_p = list()
        for img_p in data_path_ls[start:end]:
            BatchData_p.append(img_p)
            mat_data = loadmat(data_pre_fix + img_p)
            mri = mat_data['mri_d']
            mri_n = normalize(mri)
            mri_n = np.transpose(mri_n, axes=[1,0])
            if aug:
                mri_n = augmenter.augment(mri_n)

            pnr = np.squeeze(mat_data['pnr_rate'])

            lab = mat_data['lab_d']
            vid = lab[0,0]
            emo = lab[0,1]

            BatchData_mri.append(mri_n.astype(np.float32))
            BatchData_pnr.append(pnr.astype(np.float32))
            BatchData_vid.append(one_hot(vid.astype(int)+1, 3))
            BatchData_emo.append(one_hot(emo.astype(int)+4, 9))

        yield [np.array(BatchData_mri), np.array(BatchData_pnr)],[np.array(BatchData_vid).astype(np.float32),np.array(BatchData_emo).astype(np.float32)]
        i = i+1
        if (i * batch_size) >= (len(data_ls)):
            data_path_ls = shuffle(data_ls)
            i = 0


def DL_CNN(input_shape=(25,246)):
    act = 'leaky_relu'
    input_layer_s = tf.keras.layers.Input(shape=input_shape, name="input_s")  # Variable-length sequence of ints
    input_layer_r = tf.keras.layers.Input(shape=(2), name="input_r")  # Variable-length sequence of ints

    x11c = tf.keras.layers.Conv1D(5000, kernel_size=10, strides=1, padding='same', activation=act)(input_layer_s)
    x12c = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x11c)
    x21c = tf.keras.layers.Conv1D(2000, kernel_size=3, strides=1, padding='same', activation=act)(x12c)
    x22c = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(x21c)
    x31c = tf.keras.layers.Conv1D(1000, kernel_size=1, strides=1, padding='same', activation=act)(x22c)
    x32c = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(x31c)
    LSTM_c1 = tf.keras.layers.GRU(32)(x22c) # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    x_fc = tf.keras.layers.Flatten()(LSTM_c1)
    x_d1c = tf.keras.layers.Dense(1000, activation=act)(x_fc)
    x_dd1c = tf.keras.layers.Dropout(0.5)(x_d1c)
    x_d2c = tf.keras.layers.Dense(500, activation=act)(x_dd1c)

    x11l = tf.keras.layers.Conv1D(5000, kernel_size=10, strides=1, padding='same', activation=act)(input_layer_s)
    x12l = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x11l)
    x21l = tf.keras.layers.Conv1D(2000, kernel_size=3, strides=1, padding='same', activation=act)(x12l)
    x22l = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(x21l)
    x31l = tf.keras.layers.Conv1D(1000, kernel_size=1, strides=1, padding='same', activation=act)(x22l)
    x32l = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(x31l)
    LSTM_l1 = tf.keras.layers.GRU(128)(x22l) # Reduce sequence of embedded words in the body into a single 32-dimensional vector
    x_fl = tf.keras.layers.Flatten()(LSTM_l1)
    x_d1l = tf.keras.layers.Dense(1000, activation=act)(x_fl)
    x_d1l = tf.keras.layers.Dropout(0.5)(x_d1l)
    x_d2l = tf.keras.layers.Dense(500, activation=act)(x_d1l)

    x_dl_1 = tf.keras.layers.Dense(128, activation=act)(input_layer_r)
    x_dp_1 = tf.keras.layers.Dropout(0.5)(x_dl_1)
    x_dl_2 = tf.keras.layers.Dense(64, activation=act)(x_dp_1)

    x = tf.keras.layers.concatenate([x_d2c, x_d2l, x_dl_2]) # Merge all available features into a single large vector via concatenation

    class_pred = tf.keras.layers.Dense(3, activation='softmax', name="class")(x) # Stick a logistic regression for priority prediction on top of the features
    level_pred = tf.keras.layers.Dense(9, activation='softmax', name="level")(x) # Stick a department classifier on top of the features

    model = tf.keras.Model(inputs=[input_layer_s,input_layer_r], outputs=[class_pred, level_pred])# Instantiate an end-to-end model predicting both priority and department
    return model



######## Training
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    epochs = 1000
    batch_size = 4

    csv_split = 'levelsplit.csv'
    data_root = './Train/'

    x_train, x_val, x_test = data_read(csv_split)
    train_data_gn = data_gen_fn(x_train, data_root, batch_size=batch_size, aug=True)
    val_data_gn = data_gen_fn(x_val, data_root, batch_size=1, aug=False)

    steps_train_epoch = int(np.ceil(len(x_train) / batch_size))
    steps_val_epoch = int(np.ceil(len(x_val) / 1))

    model = DL_CNN()
    model.summary()

    ##### model optimizer and loss function
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    mse_loss = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")

    opt = tf.keras.optimizers.SGD(learning_rate=1e-6)
    model.compile(optimizer=opt, loss={'class':cce_loss, 'level':mse_loss}, loss_weights={'class':0.3,'level':0.7}, metrics=['accuracy'])
    
    ##### callbacks model monitor
    model_path = './weights/model_{epoch:03d}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_level_accuracy', verbose=2, save_best_only=True, mode='max')
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, mode='min', baseline=None)
    csv_logger = tf.keras.callbacks.CSVLogger('./weights/train_log.csv')

    ##### model training
    history = model.fit(train_data_gn,
            epochs=epochs, 
            verbose='auto',
            steps_per_epoch=steps_train_epoch,
            validation_data=val_data_gn,
            validation_steps=steps_val_epoch,
            callbacks=[checkpoint, csv_logger, EarlyStopping])
