import json

def merge_json(json_file1, json_file2, output_file):
    # Read data from the first JSON file
    with open(json_file1, 'r') as file:
        data1 = json.load(file)

    # Read data from the second JSON file
    with open(json_file2, 'r') as file:
        data2 = json.load(file)

    # Merge the data
    merged_data = data1

    count = 0
    for k, v in data2.items():
        if k in data1 and data1[k] != 'null' and v=='null':
            merged_data[k] = 'null'
            count += 1
        elif k not in data1:
            merged_data[k] = 'null'
            count += 1
        elif k in data1 and data1[k] == 'answerable':
            merged_data[k] = v
            count += 1
    
    print(count)

    # Save the merged data to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(merged_data, file, indent=4)

# Example usage
# json_file1 = './sample_result_submission/prediction_74.json'
json_file1 = './sample_result_submission/sql_check_prediction_80.json'
json_file2 = './sample_result_submission/binary_prediction.json'
output_file = './sample_result_submission/output_file.json'
merge_json(json_file1, json_file2, output_file)

