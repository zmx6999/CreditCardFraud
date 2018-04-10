import pandas
from sklearn import model_selection,metrics
from sklearn.preprocessing import StandardScaler
from keras import models,layers,optimizers
import numpy
data = pandas.read_csv('creditcard.csv')
print(data.shape)
print(data.head())
print(data.isnull().values.any())

vc = pandas.value_counts(data['Class']).values
n = int(vc[0]/vc[1])

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
y = data['Class']
x = data.drop(['Time','Class'],axis=1)
x0 = x[y == 0]
x1 = x[y == 1]
x1 = numpy.tile(x1,(n,1))
y0 = y[y == 0]
y1 = y[y == 1]
y1 = numpy.tile(y1,n)
x = numpy.concatenate((x0,x1))
y = numpy.concatenate((y0,y1))
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=.2,random_state=42)

sequential = models.Sequential()
sequential.add(layers.Dense(14,activation='tanh',input_shape=(x.shape[1],)))
sequential.add(layers.Dense(7,activation='relu'))
sequential.add(layers.Dense(7,activation='tanh'))
sequential.add(layers.Dense(1,activation='sigmoid'))

sequential.compile(optimizer=optimizers.adam(),loss='binary_crossentropy',metrics=['accuracy'])
#sequential.fit(x_train,y_train,validation_data=(x_test,y_test),class_weight={0:1,1:n},epochs=1)
'''
             precision    recall  f1-score   support

          0       1.00      0.97      0.98     56864
          1       0.05      0.91      0.09        98
'''
sequential.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=1)
y_pred = sequential.predict_classes(x_test)
print(metrics.classification_report(y_test,y_pred))
'''
             precision    recall  f1-score   support

          0       0.99      0.99      0.99     56720
          1       0.99      0.99      0.99     56920
'''