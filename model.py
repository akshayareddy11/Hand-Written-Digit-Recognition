import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

(X_train, y_train),(X_test,y_test)=mnist.load_data()
X_train.shape,y_train.shape,X_test.shape,y_test.shape

def plot_input_img(i):
  plt.imshow(X_train[0],cmap='binary')
  plt.axes('off')
  plt.title(y_train[i])
  plt.show()

X_train=X_train.astype(np.float32)/255
X_test=X_test.astype(np.float32)/255


X_train=np.expand_dims(X_train,-1)
X_test=np.expand_dims(X_test,-1)
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

np.squeeze(X_train)
X_train = X_train.reshape((60000, 28, 28, 1))

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu', input_shape=(28, 28, 1)))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer="adam",loss= keras.losses.categorical_crossentropy,metrics=['accuracy'])
from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_acc',min_delta=0.01,patience=4,verbose=1)
mc=ModelCheckpoint('./bestmodel.h5',monitor='val_acc',verbose=1,save_best_only=True)
cb=[es,mc]
his = model.fit(X_train,y_train,epochs =50, validation_split=0.3)
model_S=keras.models.load_model('Handwritten Digit Recognition In Python Source Code/mnist.h5')
