import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing

dataset1 = pd.read_csv('combine_final.csv')
X_train = dataset1.iloc[:-1,3:-2].values
y_train = dataset1.iloc[:-1,-2].values

print(X_train)
print(y_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

print(X_train)


# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=6, activation='softmax'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)



