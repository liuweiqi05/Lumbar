import os
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
import proc_df as pf
import model as ml
import label_encode as le
import df_generator as dg


batch_size = 100

x_train_path = 'X_train.npy'
x_val_path = 'X_val.npy'
y_train_path = 'y_train.npy'
y_val_path = 'y_val.npy'

if os.path.exists(x_train_path) and os.path.exists(x_val_path) and os.path.exists(y_train_path) and os.path.exists(y_val_path):
    X_train = np.load(x_train_path)
    X_val = np.load(x_val_path)
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)
else:
    train_labels = pd.read_csv('D:/Lumbar/train.csv')
    train_coords = pd.read_csv('D:/Lumbar/train_label_coordinates.csv')
    image_dir = 'D:/Lumbar/train_images'

    X, y = pf.proce_data(train_labels, train_coords, image_dir)
    print(f'Shape of X: {X.shape}')
    print(f'Shape of y: {y.shape}')

    X_train, X_val, y_train, y_val = le.label_ec(X, y)

    np.save(x_train_path, X_train)
    np.save(x_val_path, X_val)
    np.save(y_train_path, y_train)
    np.save(y_val_path, y_val)

train_g = dg.data_generator(X_train, y_train, batch_size)
val_g = dg.data_generator(X_val, y_val, batch_size)
train_steps = len(X_train) // batch_size
val_steps = len(X_val) // batch_size


model = ml.init_model()

history = model.fit(train_g, epochs=10, steps_per_epoch=train_steps, validation_data=val_g, validation_steps=val_steps)
val_loss, val_acc = model.evaluate(val_g, steps=val_steps)

print(f'Validation Accuracy: {val_acc:.4f}')
