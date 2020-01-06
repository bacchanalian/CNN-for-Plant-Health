# -*- coding: utf-8 -*-
"""
Created on Mon May 28 01:15:10 2018

@author: Abdifatah
"""
# Disabling unnecessary warning
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# INITIALIZE LIBRARIES
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from PIL import Image
from IPython.display import display

#______________________________________________________

# FIX RANDOM SEED
np.random.seed(7)
batch_size = 7
epochs = 25

# Needs to be optimized
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle = False,
        class_mode='categorical')

train_sam = train_generator.samples
val_sam = validation_generator.samples
test_sam = test_generator.samples

if 1:
    #______________________________________________________
    # MODEL DEFINITION
    model = Sequential()
    
    # What's the jusification for this model?
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(150, 150, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    
    # Optimizer, why?
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    #______________________________________________________
    # MODEL USE
    model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            steps_per_epoch=train_sam // batch_size,
            validation_steps=val_sam // batch_size)
    
    #model.save_weights('attempt1.h5')
    #______________________________________________________
    
    # EVAL
    scorev = model.evaluate_generator(validation_generator, val_sam // batch_size)
    scoret = model.evaluate_generator(test_generator, test_sam // batch_size)
    print("val s: %.2f%%" % (model.metrics_names[1], scorev[1]*100))
    print("test s: %.2f%%" % (model.metrics_names[1], scoret[1]*100))


# =============================================================================
# TESTING
#predic_val = model.predict_generator(validation_generator, 16, verbose = 1)
#predic_test = model.predict_generator(test_generator, 9, verbose = 1)

# PREDICTION
predic_val = model.predict_generator(validation_generator, val_sam // batch_size + 1, verbose = 1)
predic_test = model.predict_generator(test_generator, test_sam // batch_size + 1, verbose = 1)

predictions = np.argmax(predic_test, axis=-1) #multiple categories

label_map = (train_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
predictions = [label_map[k] for k in predictions]

print(predictions)

# LABEL LIST
#dic_t = train_generator.class_indices
#inv_dic = {v: k for k, v in dic_t.items()}
#new dictionary mapping

#dic_v = val_generator.class_indices

# SAVING
#model.save('models/plant_disease_B.h5')
