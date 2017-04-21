import os
import csv
import cv2
import numpy as np
import sklearn
import random
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers import Cropping2D, Dropout, Lambda
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

# data location and training set
# training set assumed in csv file with 2 columns
# img_file_without_path,steering_angel
# use data_augment.py to generate training set
# by default use current directory and assume data file is called training_set.csv
# image folder is assumed to be IMG under the same folder
data_dir = os.getcwd()

# use the final folder generated from data_augment
final_data_folder = '{}/final_data'.format(data_dir)
training_set_file = '{}/driving_log.csv'.format(final_data_folder)
img_folder = '{}/IMG'.format(final_data_folder)

# model files and if load previous model
model_file = 'my_model.h5'
previous_model = 'previous.h5'
pre_load_weights = False

# hyper-parameters
epoch_num = 10
default_batch_size = 256
default_validation_size = 0.4
# more correction for multi-cam if needed
more_correction = 0.0


# add correction if default 0.2 is not enough
def aug_img(file, angel):
    # left
    if file.startswith("left"):
        return angel + more_correction
    # right
    elif file.startswith("right"):
        return angel - more_correction
    # flipped left
    elif file.startswith("flipped_left"):
        return -(-angel + more_correction)
    # flipped right
    elif file.startswith("flipped_right"):
        return -(-angel - more_correction)
    else:
        return angel


# generator
def generator(samples, batch_size=default_batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_file = batch_sample[0]
                name = '{}/{}'.format(img_folder, img_file)
                image = cv2.imread(name)
                angle = float(batch_sample[1])
                angle = aug_img(img_file, angle)
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

if __name__ == '__main__':
    samples = []
    with open(training_set_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=default_validation_size)

    print(len(train_samples))
    print(len(validation_samples))

    # dropout rate
    default_drop_out_rate = 0.5

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=default_batch_size)
    validation_generator = generator(validation_samples, batch_size=default_batch_size)

    # nvidia self-driving car model
    model = Sequential()

    # trim img
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.))
    model.add(BatchNormalization())

    # Nvidia model
    # Conv layer with elu activation and max pooling
    model.add(Convolution2D(24, 5, 5, border_mode='same'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(BatchNormalization())

    model.add(Dropout(default_drop_out_rate))

    # Conv layer with elu activation and max pooling
    model.add(Convolution2D(36, 5, 5, border_mode='same'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(BatchNormalization())

    model.add(Dropout(default_drop_out_rate))

    # Conv layer with elu activation and max pooling
    model.add(Convolution2D(48, 3, 3, border_mode='same'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), border_mode='same'))
    model.add(BatchNormalization())

    model.add(Dropout(default_drop_out_rate))

    # Conv layer with elu activation and max pooling
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), border_mode='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(default_drop_out_rate))

    # FC layer with elu activation
    model.add(Dense(1164))
    model.add(Activation('elu'))

    # FC layer with elu activation
    model.add(Dense(100))
    model.add(Activation('elu'))

    # FC layer with elu activation
    model.add(Dense(50))
    model.add(Activation('elu'))

    # FC layer with elu activation
    model.add(Dense(10))
    model.add(Activation('elu'))

    # Output
    model.add(Dense(1))

    # load other previous model if needed
    if pre_load_weights:
        print("load previous weights!")
        model.load_weights(previous_model)

    # checkpoints. save all models that improve the validation loss
    filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # use absolute error and adam optimizer
    model.compile(loss='mae', optimizer='adam')

    # train model
    model.fit_generator(train_generator,
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        callbacks=callbacks_list,
                        nb_epoch=epoch_num, verbose=1)

    # save final model
    model.save(model_file)
