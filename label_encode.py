import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def label_ec(X, y):
    if X.size > 0 and y.size > 0:
        severity_mapping = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        y_encoded = np.array([severity_mapping[severity] for severity in y])
        y_categorical = to_categorical(y_encoded, num_classes=3)

        X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

        # X_train = np.expand_dims(X_train, axis=-1)
        # X_val = np.expand_dims(X_val, axis=-1)

        return X_train, X_val, y_train, y_val
