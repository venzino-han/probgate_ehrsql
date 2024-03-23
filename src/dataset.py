import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from scoring_program.scoring_utils import execute_sql_wrapper
from sql_utils import DB_PATH



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


class MultiLabelSBERT(torch.nn.Module):
    def __init__(self,):
        super(MultiLabelSBERT, self).__init__()
        self.linear1 = torch.nn.Linear(768, 64)
        self.linear2 = torch.nn.Linear(64, 7)
        self.droptout = torch.nn.Dropout(0.1)
        self.leakyrelu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.droptout(x)
        x = self.linear2(x)
        return x

from sentence_transformers import SentenceTransformer

TABLE_NAME_2_TABLE_ID={
        'patients': 0,
        'admissions': 1,
        'd_icd_diagnoses': 2,
        'd_icd_procedures': 3,
        'd_labitems': 4,
        'd_items': 5,
        'diagnoses_icd': 6,
        'procedures_icd': 7,
        'labevents': 8,
        'prescriptions': 9
 }

def generate_table_vectors_from_questions(questions, device='cuda:0'):
    sbert = SentenceTransformer('bert-base-nli-mean-tokens').to(device)
    table_predict_model = MultiLabelSBERT().to(device)
    table_predict_model.load_state_dict(torch.load('best_model_0.018.pth'))
    sbert_embeddings = sbert.encode(questions)

    table_vectors = []
    for emb in sbert_embeddings:
        table_vector = table_predict_model(emb.to(device))
        table_vectors.append(table_vector.cpu().detach())
    table_vectors = torch.stack(table_vectors)
    return table_vectors.numpy()

def get_schema_from_table_vectors(table_vectors, schema_dict):
    schema_texts = []
    for i in range(table_vectors.shape[0]):
        schema_text = 'Schema:\n'
        for j in range(table_vectors.shape[1]):
            if table_vectors[i][j] > 0.5:
                schema_text += schema_dict[j] + '\n'
        schema_texts.append(schema_text)

    return schema_texts




class T5Dataset(Dataset):
    """
    A dataset class for the T5 model, handling the conversion of natural language questions to SQL queries.
    """
    def __init__(
        self,
        tokenizer,
        data_dir,
        is_test=False,
        max_source_length=512, # natural langauge question
        max_target_length=512, # SQL
        db_id='mimic_iv', # NOTE: `mimic_iv` will be used for codabench
        tables_file=None,
        exclude_unans=False, # exclude unanswerable questions b/c they have no valid sql.
        random_seed=0,
        append_schema_info=False,
        answerable_or_not_binary=False,
        null_sample_ratio=0.5,
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

        self.schema_description = self._get_schema_description_from_file("schema.txt")

        split = 'train' if 'train' in data_dir else 'valid'
        if is_test:
            split = 'test'
        with open(f"{split}_schema_dict.json") as f:
            self.schema_dict = json.load(f)

        self.db_json = None
        if tables_file:
            with open(tables_file) as f:
                self.db_json = json.load(f)

        # Process and encode the samples from the loaded data
        ids = []
        questions = []
        labels = []
        
        for sample in data:
            sample_id = sample["id"]
            ids.append(sample_id)            
            question = self.preprocess_sample(sample_id, sample, append_schema_info)
            questions.append(question)

            if not self.is_test:
                sample_label = label.get(sample_id, "null")
                labels.append(sample_label)

        if exclude_unans:
            questions = [q for q, l in zip(questions, labels) if l != "null"]
            labels = [l for l in labels if l != "null"]
            ids = [i for i, l in zip(ids, labels) if l != "null"]

        self.sql_result_labels = [ execute_sql_wrapper(0, label, DB_PATH, "real")[-1] for label in labels]
        # print(self.sql_result_labels)[:10]
        question_encoded = encode_file(tokenizer, questions, max_length=self.max_source_length)
        self.source_ids, self.source_mask = question_encoded['input_ids'], question_encoded['attention_mask']
        
        if not self.is_test: # only for training, validation
            if answerable_or_not_binary:
                print("="*80)
                print("binary classification mode.")
                labels = ["answerable" if label != "null" else "null" for label in labels]
            self.weights = torch.ones(len(labels))
            # adjust sample weight 
            # original_null_sample_ratio = sum([1 for l in labels if l == "null"]) / len(labels)
            # print(f"null weigth : {null_sample_ratio / original_null_sample_ratio}")
            # if null_sample_ratio != original_null_sample_ratio:
            #     weights = [1.0 if l != "null" else null_sample_ratio / original_null_sample_ratio for l in labels]
            #     weights = np.array(weights)
            #     weights = weights / weights.sum()

            label_encoded = encode_file(tokenizer, labels, max_length=self.max_target_length)
            self.target_ids = label_encoded['input_ids']
            # self.weights = torch.tensor(weights)/sum(weights)

        self.questions = questions
        self.labels = labels
        self.ids = ids

        self.id_to_question = {i: q for i, q in zip(ids, questions)}
        
    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        if self.is_test:
            return {
                "id": self.ids[index],
                "source_ids": self.source_ids[index],
                "source_mask": self.source_mask[index],
            }
        else:
            return {
                "id": self.ids[index],
                "source_ids": self.source_ids[index],
                "source_mask": self.source_mask[index],
                "target_ids": self.target_ids[index],
                "sql_result_labels": self.sql_result_labels[index]
            }

    def preprocess_sample(self, key, sample, append_schema_info=False):
        """
        Processes a single data sample, adding schema description to the question.
        """
        question = "Question: " + sample["question"]
        
        if append_schema_info:
            # question += f"\nSchema: {self.schema_description}"
            question += f"\n{self.schema_dict.get(key, '')}"
            return question
        else:
            return question

    def _get_schema_description_from_file(self, schema_file_path):
        with open(schema_file_path) as f:
            schema = f.read().strip()
        return schema
    

    def _get_schema_description(self, tables_json, shuffle_schema=False):
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

        return " | ".join(schema_description)


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
                "sql_result_labels": [x["sql_result_labels"] for x in batch],
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

