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
        'data/detect',
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle = False,
        class_mode='categorical')

image_generator.reset()
im_pred = loaded_model.predict_generator(image_generator, steps=np.ceil(image_generator.samples/batch_size), verbose = 1, workers=1)









