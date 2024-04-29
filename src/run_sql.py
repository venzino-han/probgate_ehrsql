from scoring_program.scoring_utils import execute_all

import os
import torch
import json
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

file_name = ''
file_path = f'{file_name}.json'

# Directory paths for database, results and scoring program
DB_ID = 'mimic_iv'
# File paths for the dataset and labels
DB_PATH = os.path.join('data', DB_ID, f'{DB_ID}.sqlite')               # Database path


with open(file_path, 'r') as file:
    data = json.load(file)


print(list(data.items())[:20])
answer = execute_all(data, db_path=DB_PATH, tag='real')
answer

store_file=f"submission/answer_{file_name}.json"
with open(store_file, 'w') as file:
    json.dump(answer, file, indent=4)

# zip the file
import zipfile
with zipfile.ZipFile(f"{store_file}.zip", 'w') as zipf:
    zipf.write(store_file, os.path.basename(store_file))

