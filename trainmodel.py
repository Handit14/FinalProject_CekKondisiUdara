import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input,Dense, Activation, Dropout,LSTM

df = pd.read_csv('ISPU_after_preprocessing.csv')

X=df.iloc[:,:5]
y=df.iloc[:,5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

y_train_c=to_categorical(y_train, 4)
y_test_c=to_categorical(y_test, 4)

ann = Sequential()
ann.add(Input(shape=(X_train.shape[1:])))
ann.add(Dense( 32, activation = 'relu' ))
ann.add(Dense( 16, activation = 'relu' ))
ann.add(Dense( 4, activation = 'softmax'))

ann.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
ann.fit(X_train, y_train_c, validation_data=(X_test, y_test_c), epochs=25,  batch_size=32)
ann.save('ann.h5')