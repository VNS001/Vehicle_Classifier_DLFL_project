# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:59:29 2021

@author: VIRAJ
"""
#%% importing mobilenetv2 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#%%
pre_trained_model = MobileNetV2(input_shape = (224, 224, 3),  # we instantiate the mobilenet pretrained model for our desired input data
                                include_top = False,   # mobilenetV2 has a fully connected at top which we don't want to use as we just need convolution layers , hence top is not included by writing FALSE
                                weights = "imagenet") # as we want to use weights which were btained by training this mobilenetV2 model over imagenet dataset
#%%
for layer in pre_trained_model.layers:
  layer.trainable = False      # we are locking our pretrained model's layers by saying that they are not going to be trained in our model.

#printing our model summary
pre_trained_model.summary()
#%% selecting block_12_add layer as our last layer from mobilenetv2

last_layer = pre_trained_model.get_layer('block_12_add')  # all the layers have names so we can look at the name of that last layer which we want to use from model summary ,
# here we wanted till layer called block_12_add which have 10 by 10 convolution and have slightly more info about features than other 5 by 5 bottom layers
print('last layer output shape: ', last_layer.output_shape) # just to see the shape of last layer which we want to use i.e here block_12_add
last_output = last_layer.output #last output which we'll use will be output of block_12_add layer and pass it to our own fully connected layers which we'll write below

#%% Adding our fully connected layer to the mobilenetV2 last layer i.e block_12_add

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output) # here we have taken output from inception model's mixed7 layer and flattening it for training 
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2 i.e 20% neurons out of 1024 will be randomly deactivated/dropped during training to avoid training.
x = layers.Dropout(0.2)(x)                  
# Add a final softmax layer for classification, 4 neurons for 4 classes - bus car truck two_wheeler (arranged in alphabetical order)
x = layers.Dense  (5, activation='softmax')(x)   # softmax for multi class classification ,as it returns floating probability values for each class that sums upto 1       

model = Model( pre_trained_model.input, x)  # defining our final model consisting of pretrained model's inputs (pre_trained_model.input) and our required dense layer inputs (x)

model.compile(optimizer = RMSprop(learning_rate=0.0001), # we'll use rmsprop as our optimizer with 0.0001 learning rate
              loss = 'categorical_crossentropy',   # as we have multi class classification thus we use loss function as categorical_crossentropy
              metrics = ['accuracy'])        # we want to see accuracy matrix
#%% Adding our data-augmentation parameters to ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "D:/DOCUMENTS/PYTHON/ML Deep learning fuzzy logic elective/ML spyder codes/Dataset_bus_car_truck_bike_scooty/train"
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    fill_mode='nearest')


validation_dir="D:/DOCUMENTS/PYTHON/ML Deep learning fuzzy logic elective/ML spyder codes/Dataset_bus_car_truck_bike_scooty/val"
# very important Note that is the validation data should not be augmented!!!!!!!!
valid_datagen = ImageDataGenerator( rescale = 1./255. ) # we can just rescale it



# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'categorical',  # as loss fn of categorical_crossentropy is used.
                                                    target_size = (224, 224))

validation_generator =  valid_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'categorical', 
                                                          target_size = (224, 224))

#%% model.fit

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 113.95,  # 2279 images = batch_size * steps = 20*113.95 = 2279
            epochs = 5,
            validation_steps = 100,  # 2000 images = batch_size * steps = 20*100 = 2000
            verbose = 1)


#%% plotting training vs validation accuracy curves to check for overfitting

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

#%% predicting by giving images to our model

import numpy as np
img = tf.keras.preprocessing.image.load_img("D:/DOCUMENTS/PYTHON/ML Deep learning fuzzy logic elective/ML spyder codes/car.png", target_size=(224,224))
img_array = np.array(img)/255.
img_array = np.expand_dims(img_array, 0)  # Create batch axis
predictions = model.predict(img_array)
score = predictions[0]
print(predictions)
print(score)
bike = predictions[0][0]
bus = predictions[0][1]
car = predictions[0][2]
scooty = predictions[0][3]
truck = predictions[0][4]

maxim = max(bike,bus,car,scooty,truck)
if(maxim == bike):
    print("its a bike")
if(maxim == bus):
    print("its a bus")
if(maxim == car):
    print("its a car")
if(maxim == scooty):
    print("its a scooty")
if(maxim == truck):
    print("its a truck")


#%% saving our model as a file

keras_file = "vehiclemob_sigmoid.h5"
keras.models.save_model(model , keras_file)  # this will save our h5 file in keras

#so a file named vehicle.h5 will be saved here:  D:/DOCUMENTS/PYTHON/ML Deep learning fuzzy logic elective/ML spyder codes
# what is .h5 extension?
#An H5 file is a data file saved in the Hierarchical Data Format (HDF). 
#It contains multidimensional arrays of scientific data. 

#%%  Converting our model into TensorFlow lite model

from tensorflow import lite
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("vehiclemob_sigmoid.tflite","wb").write(tflite_model)  # wb means write binary 

#%% getting the labels.txt file 

# The class_labels.txt file should just be a plain text file with one label per line, in order of the classes in your training set. For example,
# dog
# cat
# person
# would be your label file for a three-class network where class 0 was "dog", class 1 was "cat", and class 2 was "person". 
#If this is a public classification dataset, you should have that information with the dataset, and if it's your own you'll just have to create such a mapping file. 
#You'd have to do this anyway to associate class numbers with values.

# so in note pad write:
#     bus
#     car
#     truck
#     two_wheeler
    
# and save it as labels.txt file
#%%













