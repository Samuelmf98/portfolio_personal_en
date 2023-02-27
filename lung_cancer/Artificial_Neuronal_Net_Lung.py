'''Importamos las librerías necesarias'''
import os
import csv
from parametros.prm_Red_Neuronal_Artificial_Lung import csv_folder, csv_file
import pandas as pd
import numpy as np

'''Abrir el archivo CSV utilizando la ruta completa'''
csv_path = os.path.join(csv_folder, csv_file)
df = pd.read_csv(csv_path, delimiter=',')

'''Transformamos la columna LUNG_CANCER para que devuelva 1 en caso de que tenga cancer y 0 en caso negativo'''
cancer=[]

for elemento in df['LUNG_CANCER']:
    if elemento=='YES':
        valor=1
        cancer.append(valor)
    else:
        valor=0
        cancer.append(valor)
        
df['LUNG_CANCER']=cancer        

'''Transformamos la columna GENDER para que devuelva 1 si es hombre y 0 si es mujer'''        
df['GENDER']=df['GENDER'].replace('M',1)    
df['GENDER']=df['GENDER'].replace('F',0)   
    
'''Establecemos nuestros predictores y target'''
predictores=df.iloc[: ,0:15].values
target=df.iloc[: ,15].values


''''Dividimos los datos en train y test'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(predictores, target,
                                                  test_size=0.2, random_state=0)

''''Estandarizamos las variables'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test) 

''''Construir nuestra Red Neuronal Artificial.
La práctica óptima a la hora de elegir los mejores parámetros es hacer uso del 
GridSearchCV pero debido a los altos costes de tiempo he escogido algunos parámetros
de manera aleatoria para comprobar el rendimiento de la RNA'''
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from keras.layers import Dropout 

classifier = Sequential()

classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu',input_dim=15)) 
classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu')) 
classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size=32,epochs=150)

'''Nuestro modelo tiene una precisión del 96.3%'''

'''Guardamos las predicciones'''

y_pred=classifier.predict(x_test) 


'''Establecemos el umbral.El cliente que se encuentre por encima del 3% de probabilidad 
de padecer cáncer de pulmón será catalogado como True'''

y_pred=(y_pred>0.03) 

'''Por último vamos a realizar una predicción con un paciente cuyos datos vamos a inventarnos'''
new_prediction = classifier.predict(sc.transform(np.array([[1,57,1, 1, 2, 2, 2, 1, 1,1,1,1,2,2,2]])))
print(new_prediction)
print(new_prediction > 0.03)

'''A continuación aplicaremos validación cruzada, de esta forma podemos obtener la precisión
real de nuestra red sin caer en la aletoriedad'''

def build_classifier():
    classifier = Sequential()

    classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu',input_dim=15)) 
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu')) 
    classifier.add(Dropout(rate=0.1))

    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier, batch_size=25,epochs=150)
accuracies=cross_val_score(classifier,x_train,y_train,cv=10)
print(accuracies)
print(accuracies.mean())
print(accuracies.std())

'''Nuestra RNA nos arroja una precisión final del 90.7%'''


'''A continuación, facilito el código necesario para encontrar los mejores parámetros 
y así maximizar el accuracy de la RNA'''

from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()

    classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu',input_dim=15)) 
    classifier.add(Dropout(rate=0.1)) 

    classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu')) 
    classifier.add(Dropout(rate=0.1)) 

    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier) 

parametros={'batch_size':[25,32],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']}

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parametros, scoring='accuracy',cv=10)

grid_search=grid_search.fit(x_train,y_train)

mejores_parametros= grid_search.best_params_
print(mejores_parametros)

mejor_precision=grid_search.best_score_
print(mejor_precision)
