import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
import proc_df as pf
import model as ml
import label_encode as le
from sklearn.model_selection import train_test_split
import df_generator as dg


train_labels = pd.read_csv('D:/Lumbar/train.csv')
train_coords = pd.read_csv('D:/Lumbar/train_label_coordinates.csv')
batch_size = 100
image_dir = 'D:/Lumbar/train_images'

# X_train, X_val = train_test_split(train_labels, test_size=0.2, random_state=42)

X, y = pf.proce_data(train_labels, train_coords, image_dir)
print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')
X_train, X_val, y_train, y_val = le.label_ec(X, y)

train_g = dg.data_generator(X_train, y_train, batch_size)
val_g = dg.data_generator(X_val, y_val, batch_size)
train_steps = len(X_train) // batch_size
val_steps = len(X_val) // batch_size
# X_t, y_t = pf.proce_data(X_train, train_coords, image_dir, 32)
#
# y_t = le.label_ec(X_t, y_t)
# X_v, y_v = pf_1.proce_data(X_val, train_coords, image_dir)
# y_v = le.label_ec(X_v, y_v)
# train_steps = len(X_t) // 32
# val_steps = len(y_t) // 32

model = ml.init_model()
# history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))
# val_loss, val_acc = model.evaluate(X_val, y_val)
history = model.fit(train_g, epochs=10, steps_per_epoch=train_steps, validation_data=val_g, validation_steps=val_steps)
val_loss, val_acc = model.evaluate(val_g, steps=val_steps)
# history = model.fit(X_t, y_t, epochs=10, steps_per_epoch=train_steps, validation_data=(X_v, y_v))
# val_loss, val_acc = model.evaluate(X_v, y_v)
print(f'Validation Accuracy: {val_acc:.4f}')
