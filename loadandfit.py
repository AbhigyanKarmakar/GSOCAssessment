#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import glob
import cv2 
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import os
import random
from planetaryimage import PDS3Image
from keras.callbacks import ModelCheckpoint


imgDirNorm = './Normal/'
imgDirAnom = './Anomaly/'


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
    return images

def imrotate(img , angle , scale=1):
    (h, w) = img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (h, w))


#DATA AUGMENTATION  (OPTIONAL)
def augment_folder(folder):
    augmented_images = []
    images = load_images_from_folder(folder)

    for i in images:
        augmented_images.append(cv2.flip(i, 1))
        augmented_images.append(cv2.flip(i, 0))
        augmented_images.append(cv2.flip(i, -1))
        rotate90 = imrotate(i,90)
        rotate180 = imrotate(i,180)
        rotate270 = imrotate(i,270)
        augmented_images.append(cv2.flip(rotate90, 1))
        augmented_images.append(cv2.flip(rotate180, 0))
        augmented_images.append(cv2.flip(rotate270, -1))
        augmented_images.append(rotate90)
        augmented_images.append(rotate180)
        augmented_images.append(rotate270)
    
    return augmented_images    


print('LOAD FROM DISK...........')

Anomaly = load_images_from_folder(imgDirAnom)
Normal = load_images_from_folder(imgDirNorm)

print('DISK LOAD COMPLETE!')

print('NORMAL IMAGES : ' , len(Normal))
print('ANOMALOUS IMAGES : ' , len(Anomaly))


print('IMAGE AUGMENT')
augmented_images = augment_folder(imgDirAnom)
Anomaly = Anomaly + augmented_images
print('IMAGE AUGMENT COMPLETE')

print('NORMAL IMAGES : ' , len(Normal))
print('ANOMALOUS IMAGES : ' , len(Anomaly))


print('DATA SHUFFLE')
random.shuffle(Anomaly)
random.shuffle(Normal)
random.shuffle(Anomaly)
random.shuffle(Normal)
print('SHUFFLE COMPLETE')

print('NORMAL IMAGES : ' , len(Normal))
print('ANOMALOUS IMAGES : ' , len(Anomaly))

train_images=[]
train_labels=[]
test_images=[] 
test_labels=[]

print('LABLE CREATION')

ratio = len(Normal)//len(Anomaly)

i=0
j=0

while(i<len(Normal) and j<len(Anomaly)):
    
    if i%ratio:
        if (Normal[i].shape[0]==512):
            Normal[i] = cv2.resize(Normal[i]  , (1024 , 1024))  
        train_images.append(np.reshape(Normal[i],(1024,1024,1)))
        train_labels.append([0])

    if not i%ratio:
        if (Anomaly[j].shape[0]==512):
            Anomaly[j] = cv2.resize(Anomaly[j]  , (1024 , 1024))            
        train_images.append(np.reshape(Anomaly[j],(1024 , 1024 , 1)))
        train_labels.append([1])
        j+=1
        if (Normal[i].shape[0]==512):
            Normal[i] = cv2.resize(Normal[i]  , (1024 , 1024)) 
        train_images.append(np.reshape(Normal[i],(1024 , 1024 , 1)))
        train_labels.append([0])
        

    i+=1
print('LABELING COMPLETE')

class_names = ['Clean', 'Anomaly Present']

train_images=np.array(train_images)/255.0
train_labels=np.array(train_labels)/1.0
test_images=np.array(test_images)/255.0 
test_labels=np.array(test_labels)/1.0

traintestratio = 0.70

cutindex = int(traintestratio*len(train_images))
print ("CUT AT : ",cutindex)
test_images=train_images[cutindex:len(train_images)]
train_images=train_images[0:cutindex]

test_labels=train_labels[cutindex:len(train_labels)]
train_labels=train_labels[0:cutindex]

#print((train_images[0]).shape)
#print(type(train_images))
print(np.array(train_images).shape)
print(np.array(train_labels).shape)
print(len(test_images))
print(len(test_labels))

inputshape = (1024, 1024, 1)
model = models.Sequential()
model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=inputshape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (1, 1), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

for i in range(10):

    print('EPOCH LOAD')

    model = models.load_model(savedModelDir)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=1, 
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('EPOCH SAVED')

    model.save("./savedmodel")
    print("Test Loss : " , test_loss)
    print("Test Accuracy : " , test_acc)



