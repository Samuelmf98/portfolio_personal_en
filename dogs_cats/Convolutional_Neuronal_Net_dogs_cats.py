'''Importamos las librerías necesarias'''
import os
import csv
from parametros.prm_Red_Neuronal_Convolucional_Perro_Gato import folder_train, folder_test, folder_predict
import pandas as pd
import numpy as np
import tensorflow as tf


''''Construir nuestra Red Neuronal Convolucional.
La práctica óptima a la hora de elegir los mejores parámetros es hacer uso del 
GridSearchCV pero debido a los altos costes de tiempo he escogido algunos parámetros
de manera aleatoria para comprobar el rendimiento de la RNA'''
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from keras.layers import Dropout 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


classifier=Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())


classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))   


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])  

'''Preprocesamos las imagenes de que vamos a importar'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)  

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory(folder_train,
                                                    target_size=(64, 64),
                                                    batch_size=64,
                                                    class_mode='binary') #este es optimo porque es binario el resultado,sino habria que cambiarlo

testing_dataset = test_datagen.flow_from_directory(folder_test,
                                                target_size=(64, 64),
                                                batch_size=64,
                                                class_mode='binary')
batch_size=64
classifier.fit_generator(training_dataset,
                        steps_per_epoch=int(8000/batch_size),
                        epochs=100,
                        validation_data=testing_dataset,
                        validation_steps=int(2000/batch_size))




'''Nuestro modelo tiene una precisión del 81%'''

training_dataset.class_indices


'''A continuación comprobaremos la eficacia de la RNC introduciendo imágenes de perros y gatos 
extraídos de internet'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
predict_datagen = ImageDataGenerator(rescale=1./255)

test_images = predict_datagen.flow_from_directory(folder_predict, 
                            target_size=(100, 100), batch_size=1,class_mode='binary', shuffle=False)

for i in range(test_images.samples):
    img_path = os.path.join(folder_predict, test_images.filenames[i])
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img = np.expand_dims(img, axis=0)
    
    prediction = classifier.predict(img,steps=1)
    if prediction>=0.5:
      valor='perro'
    else:
      valor='gato'
    
    plt.imshow(img[0])
    plt.title('{}'.format(prediction)+'{}'.format((valor)))
    plt.show()
    
 
