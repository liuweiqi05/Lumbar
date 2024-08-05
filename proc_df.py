import os
import numpy as np
import pandas as pd
import load_f as lf


def normalize_condition(condition):
    return condition.replace('_', ' ').title()


def normalize_level(level):
    return level.replace('_', '/').upper()


def proce_data(train_labels, train_coords, image_dir):
    images = []
    labels = []
    for _, row in train_labels.iterrows():
        study_id = row['study_id']
        for col in train_labels.columns[1:]:
            condition_level = col.split('_')
            condition = '_'.join(condition_level[:-2])
            level = '_'.join(condition_level[-2:])
            severity = row[col]
            if pd.isna(severity):
                continue
            normalized_condition = normalize_condition(condition)
            normalized_level = normalize_level(level)

            coords = train_coords[
                (train_coords['study_id'] == study_id) & (train_coords['condition'] == normalized_condition) & (
                        train_coords['level'] == normalized_level)]

            if not coords.empty:
                series_id = coords.iloc[0]['series_id']
                instance_number = coords.iloc[0]['instance_number']
                img_path = f'{image_dir}/{study_id}/{series_id}/{instance_number}.dcm'

                if os.path.exists(img_path):
                    image = lf.load_dicom(img_path)
                    images.append(image)
                    labels.append(severity)
                else:
                    print(f'Image not found: {img_path}')
            else:
                print(f'No coordinates found for {study_id}, {normalized_condition}, {normalized_level}')

    return np.array(images, dtype=np.float16), np.array(labels)
