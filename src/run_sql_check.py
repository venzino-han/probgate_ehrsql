
from scoring_program.scoring_utils import execute_all

import os
import torch
import json
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from scoring_program.postprocessing import post_process_sql

DB_PATH = './data/mimic_iv/mimic_iv.sqlite'

if __name__ == "__main__":
    submission= './submission/'
    file_name = 'log_prob_420_bottom_10'

    # file_path = f'./sample_result_submission/{file_name}.json'
    file_path = f'{submission}{file_name}.json'

    with open(file_path, 'r') as file:
        data = json.load(file)

    for sql in list(data.items())[:5]:
        print(sql)

    for k, v in data.items():
        data[k] = post_process_sql(v)

    answer = execute_all(data, db_path=DB_PATH, tag='real')

    for ans in list(answer.items())[:5]:
        print(ans)

    count = 0
    for k, v in answer.items():
        if "error" in v or v==[] or v=='[]' or v=="['']":
            print("ERROR", k," ", v)
            data[k] = "null"
            count += 1
        else:
            continue

    print(count)

    # Save the merged data to a new JSON file
    with open(f"{submission}/sql_check_{file_name}.json", 'w') as file:
        json.dump(data, file, indent=4)
