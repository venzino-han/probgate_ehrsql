"""
1. Run the generated SQL queries 
2. get query results and check correctness
3. save the query results and correctness to csv file
"""

import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForSequenceClassification, T5ForConditionalGeneration, get_linear_schedule_with_warmup


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.y = torch.tensor(labels, dtype=torch.float)

        print(type(texts), type(texts[0]))
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        self.source_ids = encoding["input_ids"]
        self.source_mask = encoding["attention_mask"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        return {
            "source_ids": self.source_ids[idx],
            "source_mask": self.source_mask[idx],
            "labels": self.y[idx], 
        }

    def collate_fn(self, batch, return_tensors='pt', padding=True, truncation=True):
        """
        Collate function for the DataLoader.
        """
        input_ids = torch.stack([x["source_ids"] for x in batch]) # BS x SL
        masks = torch.stack([x["source_mask"] for x in batch]) # BS x SL
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "labels": torch.stack([x["labels"] for x in batch])
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




def add_default_args(parser):
    """
    Define and set default arguments for the script.
    """
    parser.add_argument("--db_id", type=str, default="mimic_iv", help="database name")  # NOTE: `mimic_iv` will be used for codabench
    parser.add_argument("--question_file_path", type=str, help="question file path", default="./data/valid.json")
    parser.add_argument("--sql_file_path", type=str, help="generated SQL file path", default="./sample_result_submission/valid_sql.json")
    parser.add_argument("--output_file_path", type=str, default="./sql_validations/validation_results.csv", help="output csv file path")
    
    return parser

if __name__ == "__main__":
    # ARGS_STR = """
    #     --question_file_path="./data/valid.json" \
    #     --sql_file_path="./sample_result_submission/valid_sql.json" \
    #     --output_file_path="./sql_validations/validation_results.csv" \
    #     """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    # args = parser.parse_args(ARGS_STR.split("\n"))
    args = parser.parse_args()


    print("load dataset")
    train_df = pd.read_csv('./sql_validations/train_qsc_text2sql_0.0031.csv')
    val_df = pd.read_csv('./sql_validations/val_qsc_text2sql_0.0031.csv')
    # test_df = pd.read_csv('./sql_validations/val_question_sql_correctness_0.0114.csv')
    test_df = pd.read_csv('./sample_result_submission/prediction_80_qsc.csv')

    print(len(train_df), len(val_df))
    print(len(test_df))
    # replace nan with "null"
    train_df['generated_sql'] = train_df['generated_sql'].fillna("null")
    val_df['generated_sql'] = val_df['generated_sql'].fillna("null")
    test_df['generated_sql'] = test_df['generated_sql'].fillna("null")
    print(len(train_df), len(val_df))
    print(len(test_df))


    # merge question and sql columns
    train_df['question_sql'] = train_df['question'] + '\n SQL: ' + train_df['generated_sql']
    val_df['question_sql'] = val_df['question'] + '\n SQL: ' + val_df['generated_sql']
    test_df['question_sql'] = test_df['question'] + '\n SQL: ' + test_df['generated_sql']

    # Load pre-trained T5 tokenizer
    # model_name = "sangryul/Flan-T5-XL-text2sql-spider"
    model_name = "sangryul/Flan-T5-XL-text2sql-spider"
    model_name = "gaussalgo/T5-LM-Large-text2sql-spider"
    # model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Create custom dataset
    max_length = 512  # Define maximum sequence length
    train_dataset = CustomDataset(
        texts=train_df["question_sql"].tolist(),
        labels=train_df["correctness"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    val_dataset = CustomDataset(
        texts=val_df["question_sql"].tolist(),
        labels=val_df["correctness"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    test_dataset = CustomDataset(
        texts=test_df["question_sql"].tolist(),
        labels=test_df["correctness"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Create DataLoader with seedable sampler
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )

    # Define T5 model for sequence classification
    model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # to bfloat16
    model = model.to(torch.bfloat16)

    # load parameters
    model.load_state_dict(torch.load("sql_val_model_0.83.pth"))

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=len(train_dataloader) * 30
    )

    # Training loop
    print("start training")
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            input_ids = batch["source_ids"].to(device)
            attention_mask = batch["source_mask"].to(device).to(torch.bfloat16)
            labels = batch["labels"].to(device).to(torch.bfloat16)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} completed")
        
        # Validation loop
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        for batch in tqdm(val_dataloader):
            input_ids = batch["source_ids"].to(device)
            attention_mask = batch["source_mask"].to(device).to(torch.bfloat16)
            labels = batch["labels"].to(device).to(torch.bfloat16)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_losses.append(outputs.loss.item())
                val_preds.append(outputs.logits.float())
                val_labels.append(labels.float())

        val_loss = np.mean(val_losses)
        print(f"Validation loss: {val_loss:.4f}")

        val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
        val_labels = torch.cat(val_labels, dim=0).cpu().numpy()

        # calculate accuracy, precision, recall, f1, auc from val_labels, val_preds
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        accuracy = accuracy_score(val_labels, (val_preds>0.5).astype(int))
        precision = precision_score(val_labels, (val_preds>0.5).astype(int))
        recall = recall_score(val_labels, (val_preds>0.5).astype(int))
        f1 = f1_score(val_labels, (val_preds>0.5).astype(int))
        auc = roc_auc_score(val_labels, val_preds)

        print(f"""
        Validation accuracy: {accuracy:.4f}
        Validation precision: {precision:.4f}
        Validation recall: {recall:.4f}
        Validation f1: {f1:.4f}
        Validation AUC: {auc:.4f}
        """)

        # save the model if the validation loss is the best so far
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = auc
            torch.save(model.state_dict(), f"sql_val_model_{best_auc:.4f}.pth")

            #pred for test 
            test_preds = []
            for batch in tqdm(test_dataloader):
                input_ids = batch["source_ids"].to(device)
                attention_mask = batch["source_mask"].to(device).to(torch.bfloat16)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    test_preds.append(outputs.logits.float())

            test_preds = torch.cat(test_preds, dim=0).cpu().numpy()
            test_df['correctness'] = test_preds

            test_df.to_csv(f'./sql_validations/prediction_{epoch}_{best_auc:.2f}.csv', index=False)
