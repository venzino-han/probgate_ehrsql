import json
import random
import torch
from torch.utils.data import Dataset


def encode_file(tokenizer, text, max_length, truncation=True, padding=True, return_tensors="pt"):
    """
    Tokenizes the text and returns tensors.
    """
    return tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )


class T5Dataset(Dataset):
    """
    A dataset class for the T5 model, handling the conversion of natural language questions to SQL queries.
    """
    def __init__(
        self,
        tokenizer,
        data_dir,
        is_test=False,
        max_source_length=256, # natural langauge question
        max_target_length=512, # SQL
        db_id='mimiciii', # NOTE: `mimic_iv` will be used for codabench
        tables_file=None,
        exclude_unans=False, # exclude unanswerable questions b/c they have no valid sql.
        random_seed=0,
        append_schema_info=False,
    ):

        super().__init__()
        self.tokenizer = tokenizer
        self.db_id = db_id
        self.is_test = is_test # this option does not include target label
        self.random = random.Random(random_seed) # initialized for schema shuffling
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # Load data from JSON files
        with open(f'{data_dir}/data.json') as json_file:
            data = json.load(json_file)["data"]

        label = {}
        if not self.is_test:
            with open(f'{data_dir}/label.json') as json_file:
                label = json.load(json_file)

        self.db_json = None
        if tables_file:
            with open(tables_file) as f:
                self.db_json = json.load(f)

        # Process and encode the samples from the loaded data
        ids = []
        questions = []
        labels = []
        weights = []
        for sample in data:

            # id
            if exclude_unans:
                if sample["id"] in label and label[sample["id"]] == "null":
                    continue
            ids.append(sample['id'])

            if sample["id"] in label and label[sample["id"]] == "null":
                weights.append(4.0)
            else:
                weights.append(1.0)
            
            # question
            question = self.preprocess_sample(sample, append_schema_info)
            questions.append(question)

            # label
            if not self.is_test:
                labels.append(label[sample["id"]])

        self.ids = ids
        question_encoded = encode_file(tokenizer, questions, max_length=self.max_source_length)
        self.source_ids, self.source_mask = question_encoded['input_ids'], question_encoded['attention_mask']
        if not self.is_test:
            label_encoded = encode_file(tokenizer, labels, max_length=self.max_target_length)
            self.target_ids = label_encoded['input_ids']

            # TODO : check null case token
            # higher probability of unanswerable questions
            # self.null_token_id = self.tokenizer.convert_tokens_to_ids("null")
            # self.weights = torch.ones(len(self.source_ids))
            # self.weights[self.target_ids[:, 0] == self.null_token_id] = 2.0
            # print null token ratio 
            print(f"null token ratio: {sum(weights) / len(weights)}")
            self.weights = torch.tensor(weights)/sum(weights)

        

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        if self.is_test:
            return {
                "id": self.ids[index],
                "source_ids": self.source_ids[index],
                "source_mask": self.source_mask[index]
            }
        else:
            return {
                "id": self.ids[index],
                "source_ids": self.source_ids[index],
                "source_mask": self.source_mask[index],
                "target_ids": self.target_ids[index]
            }

    def preprocess_sample(self, sample, append_schema_info=False):
        """
        Processes a single data sample, adding schema description to the question.
        """
        question = "Question: " + sample["question"]
        
        if append_schema_info:
            if self.db_json:
                tables_json = [db for db in self.db_json if db["db_id"] == self.db_id][0]
                schema_description = self.get_schema_description(tables_json)
                question += f"\nSchema: {schema_description}"
            return question
        else:
            return question

    def get_schema_description(self, tables_json, shuffle_schema=False):
        """
        Generates a textual description of the database schema.
        """
        table_names = tables_json["table_names_original"]
        if shuffle_schema:
            self.random.shuffle(table_names)

        columns = [
            (column_name[0], column_name[1].lower(), column_type.lower())
            for column_name, column_type in zip(tables_json["column_names_original"], tables_json["column_types"])
        ]

        schema_description = [""]
        for table_index, table_name in enumerate(table_names):
            table_columns = [column[1] for column in columns if column[0] == table_index]
            if shuffle_schema:
                self.random.shuffle(table_columns)
            column_desc = " , ".join(table_columns)
            schema_description.append(f"{table_name.lower()} : {column_desc}")

        return " [SEP] ".join(schema_description)

    def collate_fn(self, batch, return_tensors='pt', padding=True, truncation=True):
        """
        Collate function for the DataLoader.
        """
        ids = [x["id"] for x in batch]
        input_ids = torch.stack([x["source_ids"] for x in batch]) # BS x SL
        masks = torch.stack([x["source_mask"] for x in batch]) # BS x SL
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)

        if self.is_test:
            return {
                "source_ids": source_ids,
                "source_mask": source_mask,
                "id": ids,
            }
        else:
            target_ids = torch.stack([x["target_ids"] for x in batch]) # BS x SL
            target_ids = trim_batch(target_ids, pad_token_id)
            return {
                "source_ids": source_ids,
                "source_mask": source_mask,
                "target_ids": target_ids,
                "id": ids,
            }

def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """
    Trims padding from batches of tokenized text.
    """
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
