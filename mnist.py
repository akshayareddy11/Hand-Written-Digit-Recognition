import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split from sklearn.neighbors import KNeighborsClassifier from sklearn.metrics import accuracy_score
def show_digit(no): 
  pred_no=model.predict([X_test[no]]) 
  img_array=X_test[no].reshape((28,28))
  plt.figure(figsize=(3,3)) 
  plt.title(f"predicted number={pred_no}") 
  plt.imshow(img_array)
df=pd.read_csv("mnist.csv") X=df.values[:,1:]
y=df.values[:,0] X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=KNeighborsClassifier(n_neighbors=5) model.fit(X_train,y_train)
y_pred=model.predict(X_test) print(y_pred)
a=accuracy_score(y_test,y_pred) print(f"the accuracy is:{a}")
show_digit(20)
plt.plot(y_test,y_pred) plt.show()
