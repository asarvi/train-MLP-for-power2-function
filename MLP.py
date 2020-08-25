
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ProgbarLogger
import numpy as np
import matplotlib.pyplot as plt

n_samples = 500

def powerTwo(x):
 return x*x
np.random.seed(10)
X = np.random.uniform(-2, 2, n_samples)
Y = powerTwo(X)
# tarin MLP
model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=20, batch_size=2, verbose=1, validation_split = 0.1, callbacks =[ProgbarLogger()])
#try new samples
new_samples = 300
new_X = np.random.uniform(-2, 2, new_samples)
new_Y = powerTwo(new_X)
pred_Y = model.predict(new_X)
plt.plot(new_X,new_Y,'go',label="X*X")
plt.plot(new_X,pred_Y,'yo',label="PredictedX*X")
plt.show()
print("green shows original X^2 and yellow shows predicted X^2")

