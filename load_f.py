import pydicom
import cv2
import numpy as np


def load_dicom(path):
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_rgb = cv2.resize(image_rgb, (128, 128))
    image_rgb = image_rgb / 255.0

    return image_rgb.astype(np.float16)


def check_dicom(path):
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    image = cv2.resize(image, (128, 128))
    a = [len(cv2.split(image)), len(image.shape)]
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    b = [len(cv2.split(image)), len(image.shape)]

    return a, b
