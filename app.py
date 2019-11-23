import os
import shutil
import numpy as np
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

dataframe = read_csv("iris-custom.data", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4].astype(int)

Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y, 
    test_size=.2,
    random_state=42
)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_train,
    Y_train,
    test_size=.2,
    random_state=42
)

if (not os.path.isdir("data/")):
    os.mkdir("data")

# NPZ file creation
X_file = open("data/x_test.npz", "wb")
Y_file = open("data/y_test.npz", "wb")
print(np.asarray(Y))
np.savez(X_file, np.asarray(X_test))
np.savez(Y_file, np.asarray(Y_test))
X_file.close()
Y_file.close()

if (os.path.isfile("data/x_norm.npz")):
    shutil.os.remove("data/x_norm.npz")
shutil.copy("data/x_test.npz", "data/x_norm.npz")

# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)

def create_baseline():
    model = Sequential()
    # model.add(Dense(units=4, activation="relu", input_dim=4, kernel_initializer='normal'))
    # model.add(Dense(units=4, activation="relu", kernel_initializer='normal'))
    model.add(Dense(units=2, activation="softmax", input_dim=4, kernel_initializer='normal'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return lambda : model

model_returner = create_baseline()
 
# Test ve train sayılarını bul
# estimator = KerasClassifier(build_fn=model_returner, epochs=50, batch_size=5, verbose=0)
# kfold = StratifiedKFold(n_splits=50, shuffle=True)
# results = cross_val_score(estimator, X, Y, cv=kfold)

model_returner().fit(X_train, Y_train, batch_size=50, epochs=20, verbose=2, validation_data=(X_val, Y_val))

model_returner().save("iris-model.h5")

print(Y_test)
print(model_returner().predict(X_test))

loss, acc = model_returner().evaluate(X_train, Y_train, verbose=0)
print("loss: %.2f%%; acc: %.2f%%" % (loss*100, acc*100))

