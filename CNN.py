import sys
import numpy as np
from keras.models import load_model
from keras.optimizer_v2.adam import Adam
import os
from scipy.io import loadmat
# from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import Sequential, Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, BatchNormalization, Subtract, ConvLSTM2D, Conv3D, Dropout
import pickle
import scipy.io
from keras.callbacks import ReduceLROnPlateau


def SRCNN_model(nSym, nSC):
    model = Sequential()
    model.add(
        Conv2D(64, kernel_size=(9, 9), activation='relu', kernel_initializer='he_normal', input_shape=(nSC, nSym, 1),
               padding='same'))
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(Conv2D(1, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def DNCNN_model(nSym, nSC):
    input = Input(shape=(nSC, nSym, 1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(18):
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    x = Subtract()([input, x])  # input - noise
    model = Model(inputs=input, outputs=x)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    print(model.summary())
    return model


# General Parameters
configuration_mode = len(sys.argv)
SNR_index = np.arange(0, 45, 5)
val_split = 0.25
nSym = 100
if configuration_mode == 11:
    # We are running the training phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    wi_configuration = sys.argv[3]
    modulation_order = sys.argv[4]
    scheme = sys.argv[5]
    training_snr = sys.argv[6]
    cnn_type = sys.argv[7]
    cnn_input = int(sys.argv[8])
    epoch = int(sys.argv[9])
    batch_size = int(sys.argv[10])

    mat = loadmat('./{}_{}_{}_{}_{}_CNN_training_dataset_{}.mat'.format(mobility, channel_model, wi_configuration,
                                                                        modulation_order, scheme, training_snr))
    Dataset = mat['CNN_Datasets']
    Dataset = Dataset[0, 0]
    X = Dataset['Train_X']
    Y = Dataset['Train_Y']
    print('Loaded Dataset Inputs: ', X.shape)
    print('Loaded Dataset Outputs: ', Y.shape)
    X = X.reshape(X.shape[0], cnn_input, nSym, 1)
    Y = Y.reshape(Y.shape[0], cnn_input, nSym, 1)
    cnn_model = []
    if cnn_type == 'SRCNN':
        cnn_model = SRCNN_model(nSym, cnn_input)
    elif cnn_type == 'DNCNN':
        cnn_model = SRCNN_model(nSym, cnn_input)

    print(cnn_model.summary())

    model_path = './{}_{}_{}_{}_{}_{}_{}.h5'.format(mobility, channel_model, wi_configuration, modulation_order, scheme,
                                                    cnn_type, training_snr)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    cnn_model.fit(X, Y, batch_size=batch_size, validation_split=val_split,
                  callbacks=callbacks_list, epochs=epoch, verbose=2)

else:
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    wi_configuration = sys.argv[3]
    modulation_order = sys.argv[4]
    scheme = sys.argv[5]
    testing_snr = sys.argv[6]
    cnn_type = sys.argv[7]
    cnn_input = int(sys.argv[8])

    for j in SNR_index:
        mat = loadmat('./{}_{}_{}_{}_{}_CNN_testing_dataset_{}.mat'.format(mobility, channel_model,wi_configuration, modulation_order, scheme, j))
        Dataset = mat['CNN_Datasets']
        Dataset = Dataset[0, 0]
        X = Dataset['Test_X']
        Y = Dataset['Test_Y']
        print('Loaded Dataset Inputs: ', X.shape)
        print('Loaded Dataset Outputs: ', Y.shape)

        XT = X.reshape(X.shape[0], cnn_input, nSym, 1)
        YT = Y.reshape(Y.shape[0], cnn_input, nSym, 1)

        model_path = './{}_{}_{}_{}_{}_{}_{}.h5'.format(mobility, channel_model, wi_configuration, modulation_order,
                                                        scheme, cnn_type, testing_snr)

        cnn_model = load_model(model_path)
        Prediction_Y = cnn_model.predict(XT)
        result_path = './{}_{}_{}_{}_{}_{}_Results_{}.pickle'.format(mobility, channel_model, wi_configuration, modulation_order, scheme, cnn_type, j)
        dest_name = './{}_{}_{}_{}_{}_{}_Results_{}.mat'.format(mobility, channel_model, wi_configuration, modulation_order, scheme, cnn_type, j)
        with open(result_path, 'wb') as f:
            pickle.dump([X, Y, Prediction_Y], f)

        a = pickle.load(open(result_path, "rb"))
        scipy.io.savemat(dest_name, {
            '{}_CNN_test_x_{}'.format(scheme, j): a[0],
            '{}_CNN_test_y_{}'.format(scheme, j): a[1],
            '{}_CNN_corrected_y_{}'.format(scheme, j): a[2]
        })
        print("Data successfully converted to .mat file ")
        os.remove(result_path)






