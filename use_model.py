# -*- coding: utf-8 -*-
"""
Created on Mon May 28 02:27:58 2018

@author: Abdifatah
"""

# SAVING
#model.save('modelname.h5')

import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# LOADING
loaded_model = keras.models.load_model('models/plant_disease_A.h5')

# SETUP
batch_size = 8
epochs = 25

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# GENERATORS
validation_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle = False,
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle = False,
        class_mode='categorical')

# MODEL TEMPLATE
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# PREDICTION
test_generator.reset()
validation_generator.reset()
predic_test = loaded_model.predict_generator(test_generator, steps=np.ceil(test_generator.samples/batch_size), verbose = 1, workers=1)
predic_val = loaded_model.predict_generator(validation_generator, steps=np.ceil(validation_generator.samples/batch_size), verbose = 1, workers=1)

# SCORE
test_generator.reset()
score_t = loaded_model.evaluate_generator(test_generator, steps=np.ceil(test_generator.samples/batch_size), verbose = 1, workers=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score_t[1]*100))

validation_generator.reset()
score_v = loaded_model.evaluate_generator(validation_generator, steps=np.ceil(validation_generator.samples/batch_size), verbose = 1, workers=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score_v[1]*100))


# Detection
from keras.preprocessing import image
from matplotlib import pyplot as plt


image_datagen = ImageDataGenerator(rescale=1./255)
image_generator = image_datagen.flow_from_directory(
        'data/object3',
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle = False,
        class_mode='categorical')

image_generator.reset()
im_pred = loaded_model.predict_generator(image_generator, steps=np.ceil(image_generator.samples/batch_size), verbose = 1, workers=1)


# Test code
#i0 = image.load_img('data/object3/x/rp0.jpg')
#i1 = image.load_img('data/object3/x/rp1.jpg')
#i2 = image.load_img('data/object3/x/rp2.jpg')
#i3 = image.load_img('data/object3/x/rp3.jpg')
#i4 = image.load_img('data/object3/x/rp4.jpg')
#i5 = image.load_img('data/object3/x/rp5.jpg')
#i6 = image.load_img('data/object3/x/rp6.jpg')
#c14 = image.load_img('data/object3/x/c14.jpg')
#
#img = c14
#img_t = image.img_to_array(img)
#img_t = np.expand_dims(img_t, axis=0)
#img_t /= 255.
##pred = loaded_model.predict(img_t, 1);
#pred = np.vstack([pred, loaded_model.predict(img_t, 1)])


# Region Proposal
#im = i1
#M = im.shape[0]//2
#N = im.shape[1]//2
#img_array = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
# change region proposal
# use 2 disease, 2 non disease and 1 empty space
# note that it lacks robustness because of there is no negative training for objects that are not leaves









