'''Se importan las librerías necesarias'''
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from parametros.prm_Red_Neuronal_Recurrente_Acciones_Google import csv_folder, csv_google_train, csv_google_test

'''Abrir el archivo CSV utilizando la ruta completa'''
csv_path = os.path.join(csv_folder, csv_google_train)
datos_train = pd.read_csv(csv_path, delimiter=',')

'''Para este estudio se utiliza la columna del precio de apretura de la acción'''
training_set=datos_train.iloc[:,1:2].values 

'''Para el preprocesamiento se hace uso de la normalización'''

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_scaled=sc.fit_transform(training_set)

'''Se crea una estructura con 60 timesteps, es decir, de 3 meses donde el primer valor de
y_train estará formado desde x_train=0 a x_train=59, el segundo valor de y_train estará formado
desde x_train=1 a x_train=60, y así sucesivamente'''
x_train=[]
y_train=[]


for i in range(60,1258):
    x_train.append(training_scaled[i-60:i, 0])   
    y_train.append(training_scaled[i, 0])    
                              
x_train,y_train=np.array(x_train),np.array(y_train) 

'''Se redimensionan los datos para añdir una tercera dimensión'''


x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1], 1)) 




'''Se construye la Red Neuronal Recurrente para resolver el problema de regresión'''
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regresor=Sequential() 

regresor.add(LSTM(units= 50,return_sequences=True,input_shape=(x_train.shape[1],1))) 

regresor.add(LSTM(units= 50,return_sequences=True))
regresor.add(Dropout(0.2))

regresor.add(LSTM(units= 50,return_sequences=True))
regresor.add(Dropout(0.2))

regresor.add(LSTM(units= 50,return_sequences=True))
regresor.add(Dropout(0.2))

regresor.add(LSTM(units= 50)) 
regresor.add(Dropout(0.2))

regresor.add(Dense(units=1) ) 

regresor.compile(optimizer='adam',loss=('mean_squared_error')) #se puede usar adam o RMSprop


regresor.fit(x_train,y_train, 
                 epochs=100,
                 batch_size=32)
   

'''Cargamos los rsultado reales del primer mes de 2017'''
csv_path = os.path.join(csv_folder, csv_google_test)
datos_test = pd.read_csv(csv_path, delimiter=',')
datos_test=datos_test.iloc[:,1:2].values 

'''Se realizan predicciones para comprobar si se asemejan a los datos que se acaban de importar.
Previamente se transforman los datos manera que el dataframe contiene los últimos 3 meses (60 días) de 2016 
y el primer mes de 2017 (20  días)'''


total=np.concatenate((training_set,datos_test),axis=0)


inputs=total[len(total)-len(datos_test)-60:]

'''Se otorga dimensión y se escalan los datos'''
inputs=inputs.reshape(-1,1)

inputs=sc.transform(inputs) 

'''Se vuelve a crear la estrucutura anterior'''
x_test=[]

for i in range(60,80):  
    x_test.append(inputs[i-60:i, 0])  
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1], 1)) 

predicted_stock=regresor.predict(x_test)

predicted_stock=sc.inverse_transform(predicted_stock) 

#Comprobemos con los valores reales
plt.plot(datos_test,color='red',label='Precio real de la accion de google')
plt.plot(predicted_stock,color='blue',label='Precio predicho de la accion de google')
plt.title('Prediccion con una RNN del precio de la accion de google en Enero de 2017')
plt.xlabel('Fecha')
plt.ylabel('Precio de la accion')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(datos_test, predicted_stock))
rmse=rmse/800