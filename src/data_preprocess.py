import os
import json
import numpy as np
import pandas as pd
from collections import Counter

# Directory paths for database, results and scoring program
DB_ID = 'mimic_iv'
SAMPLE_DATA_DIR = 'sample_data'
BASE_DATA_DIR = 'data/mimic_iv'
RESULT_DIR = 'sample_result_submission/'
SCORE_PROGRAM_DIR = 'scoring_program/'

# File paths for the dataset and labels
TABLES_PATH = os.path.join('data', DB_ID, 'tables.json')               # JSON containing database schema
TRAIN_DATA_PATH = os.path.join(BASE_DATA_DIR, 'train', 'data.json')    # JSON file with natural language questions for training data
TRAIN_LABEL_PATH = os.path.join(BASE_DATA_DIR, 'train', 'label.json')  # JSON file with corresponding SQL queries for training data
VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, 'valid', 'data.json')    # JSON file for validation data
DB_PATH = os.path.join('data', DB_ID, f'{DB_ID}.sqlite')               # Database path
# Set directory for the new splitted data
NEW_TRAIN_DIR = os.path.join(SAMPLE_DATA_DIR, '__train')
NEW_VALID_DIR = os.path.join(SAMPLE_DATA_DIR, '__valid')
NEW_TEST_DIR = os.path.join(BASE_DATA_DIR, 'valid')

from utils.data_io import read_json as read_data
from utils.data_io import write_json as write_data

if __name__ == "__main__":
    # Load train and validation sets
    train_data = read_data(TRAIN_DATA_PATH)
    train_label = read_data(TRAIN_LABEL_PATH)
    valid_data = read_data(VALID_DATA_PATH)

    # Quick summary of the dataset
    print(f"Train data: {len(train_data['data'])} entries, Train labels: {len(train_label)} entries")
    print(f"Valid data: {len(valid_data['data'])} entries")

    # Explore keys and data structure
    # print(train_data.keys())
    # print(train_data['version'])
    # print(train_data['data'][0])

    # Explore the label structure
    # print(train_label.keys())
    # print(train_label[list(train_label.keys())[0]])

    from sklearn.model_selection import train_test_split

    # Define stratification criteria for consistent distribution between answerable and unanswerable questions
    stratify = ['unans' if train_label[id_]=='null' else 'ans' for id_ in list(train_label.keys())]

    # Split the original training data into new training and validation sets, while maintaining the distribution
    new_train_keys, new_valid_keys = train_test_split(
        list(train_label.keys()),
        train_size=0.9,
        random_state=42,
        stratify=stratify
    )

    # Initialize containers for the new training and validation sets
    new_train_data = []
    new_train_label = {}
    new_valid_data = []
    new_valid_label = {}

    # Sort each sample into the new training or validation set as determined by the split
    for sample in train_data['data']:
        if sample['id'] in new_train_keys:
            new_train_data.append(sample)
            new_train_label[sample['id']] = train_label[sample['id']]
        elif sample['id'] in new_valid_keys:
            new_valid_data.append(sample)
            new_valid_label[sample['id']] = train_label[sample['id']]
        else:
            # If a sample is neither in the train nor valid keys, raise an error
            raise ValueError(f"Error: Sample with ID {sample['id']} has an invalid split.")

    # Structure the new datasets in a JSON-compatible format
    new_train_data = {'version': f'{DB_ID}_sample', 'data': new_train_data}
    new_valid_data = {'version': f'{DB_ID}_sample', 'data': new_valid_data}

    # Display the size of the new training and validation sets for verification
    print(f"New Train data: {len(new_train_data['data'])} entries, New Train labels: {len(new_train_label)} entries, Unanswerable: {sum(value == 'null' for value in new_train_label.values())}")
    print(f"New Valid data: {len(new_valid_data['data'])} entries, New Valid labels: {len(new_valid_label)} entries, Unanswerable: {sum(value == 'null' for value in new_valid_label.values())}")

    # Save the new datasets to JSON files for later use
    write_data(os.path.join(NEW_TRAIN_DIR, "data.json"), new_train_data)
    write_data(os.path.join(NEW_TRAIN_DIR, "label.json"), new_train_label)
    write_data(os.path.join(NEW_VALID_DIR, "data.json"), new_valid_data)
    write_data(os.path.join(NEW_VALID_DIR, "label.json"), new_valid_label)