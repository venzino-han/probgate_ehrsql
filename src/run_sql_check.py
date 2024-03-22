
from scoring_program.scoring_utils import execute_all

import os
import torch
import json
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

DB_PATH = './data/mimic_iv/mimic_iv.sqlite'

if __name__ == "__main__":
    file_name = 'prediction_77'
    file_name = 'prediction'

    file_path = f'./sample_result_submission/{file_name}.json'

    with open(file_path, 'r') as file:
        data = json.load(file)


    print(list(data.items())[:20])
    answer = execute_all(data, db_path=DB_PATH, tag='real')

    print(list(answer.items())[:20])

    count = 0
    for k, v in answer.items():
        if "error" in v or v==[] or v=="[]":
            data[k] = "null"
            count += 1
        else:
            continue

    print(count)

    # Save the merged data to a new JSON file
    with open(f"sample_result_submission/sql_check_{file_name}.json", 'w') as file:
        json.dump(data, file, indent=4)

    # store_file=f"sample_result_submission/answer_{file_name}.json"
    # with open(store_file, 'w') as file:
    #     json.dump(answer, file, indent=4)

    # # zip the file
    # import zipfile
    # with zipfile.ZipFile(f"{store_file}.zip", 'w') as zipf:
    #     zipf.write(store_file, os.path.basename(store_file))


