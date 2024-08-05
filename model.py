from keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers import Adam, Adamax

SHAPE = (128, 128, 3)


def init_model():
    cap_model = Xception(weights='imagenet', include_top=False, pooling='avg',
                         input_shape=SHAPE)

    cap_model.trainable = False
    model = Sequential()
    model.add(cap_model)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.45))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
