import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   width_shift_range=0.4,
                                   height_shift_range=0.4,
                                   vertical_flip=True,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('images/train',
                                                 target_size=(48, 48),
                                                 color_mode='grayscale',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('images/validation',
                                            target_size=(48, 48),
                                            color_mode='grayscale',
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=True)
cnn = tf.keras.models.Sequential()

# model -1
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='elu', input_shape=[48, 48, 1],
                               kernel_initializer='he_normal', padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='elu', input_shape=[48, 48, 1],
                               kernel_initializer='he_normal', padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

# model -2
cnn.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='elu', kernel_initializer='he_normal', padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='elu', kernel_initializer='he_normal', padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='elu', kernel_initializer='he_normal',
                               padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='elu', kernel_initializer='he_normal', padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='elu', kernel_initializer='he_normal',
                               padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='elu', kernel_initializer='he_normal', padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

# block -5
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(64, kernel_initializer='he_normal', activation='elu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Dense(64, kernel_initializer='he_normal', activation='elu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Dense(7, kernel_initializer='he_normal', activation='softmax'))
print(cnn.summary())

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

