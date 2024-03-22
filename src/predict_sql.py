from transformers import T5Tokenizer
from data_preprocess import NEW_TRAIN_DIR,  NEW_VALID_DIR,  NEW_TEST_DIR,  RESULT_DIR,  TABLES_PATH,  DB_ID, DB_PATH
from dataset import T5Dataset

import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from t5_model_train import *
from sql_utils import generate_sql

from scoring_program.scoring_utils import execute_all, reliability_score, penalize
from scoring_program.postprocessing import post_process_sql


def get_sql_correctness(dataloader, model, tokenizer, args, test=False):
    generated_sqls = generate_sql(tokenizer, model, dataloader, args)
    id2pred_sql = {sample['id']: post_process_sql(sample['pred']) for sample in generated_sqls}
    
    pred_result = execute_all(id2pred_sql, db_path=DB_PATH, tag='pred')
    
    # count error in pred_result and real_result
    error_count = 0
    for key, val in pred_result.items():
        if type(val)==str:
            if val.startswith('error'):
              error_count += 1
    print(f"pred error_count: {error_count}")

    if not test:
        id2real_sql = {sample['id']: post_process_sql(sample['real']) for sample in generated_sqls}
        real_result = execute_all(id2real_sql, db_path=DB_PATH, tag='real')
    
        error_count = 0
        for key, val in real_result.items():
            if type(val)==str:
                if val.startswith('error'):
                    error_count += 1
        print(f"real error_count: {error_count}")
        correctness = [1 if pred_result[id_] == real_result[id_] else 0 for id_ in pred_result]
    else:
        correctness = [0 for id_ in pred_result]

    generated_sqls = [ id2pred_sql[id_] for id_ in pred_result]
    ids = [id_ for id_ in pred_result]

    return ids, correctness, generated_sqls

if __name__ == "__main__":
    for best_val_loss in [
        # 0.0114,
        # 0.0101,
        # 0.0045,
        0.0031,
        # 0.0003,
        ]:

        exp_name = "t5-text2sql"
        model_name = "gaussalgo/T5-LM-Large-text2sql-spider"
        # model_name = "sangryul/Flan-T5-XL-text2sql-spider"
        ARGS_STR = f"""
        --exp_name={exp_name} \
        --model_name={model_name} \
        --train_data_dir={NEW_TRAIN_DIR} \
        --valid_data_dir={NEW_VALID_DIR} \
        --test_data_dir={NEW_TEST_DIR} \
        --tables_file={TABLES_PATH} \
        --train_epochs=20 \
        --train_batch_size=32 \
        --valid_batch_size=32 \
        --test_batch_size=32 \
        --gradient_accumulation_steps=1 \
        --bf16=1\
        --use_schema_info=1\
        --db_id=mimic_iv\
        --max_target_length=512\
        """

        # Parse arguments
        parser = argparse.ArgumentParser()
        parser = add_default_args(parser)
        args = parser.parse_args(ARGS_STR.split())

        # Configure CUDA settings
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Set random seed for reproducibility
        set_seed(args)

        # Determine device for training and set model save path
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.n_gpu = torch.cuda.device_count()
        args.save_model_path = os.path.join(args.output_dir, args.exp_name)

        # Initialize T5 model and set device
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        model = model.to(args.device)

        # Convert model to bfloat16 precision if required
        if args.bf16:
            print("bfloat16 precision will be used")
            model = model.to(torch.bfloat16)
        # load custom tokens
        with open('schema_tokens.txt', 'r') as f:
            custom_tokens = f.read().split('\n')
        # Initialize tokenizer with additional SQL tokens
        add_tokens = ["<", "<=", "<>"] #+ custom_tokens

        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        tokenizer.add_tokens(add_tokens)

        # Resize model token embeddings
        model.resize_token_embeddings(len(tokenizer))

        # Define parameters for dataset preparation
        dataset_kwargs = dict(
            db_id=args.db_id,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            tables_file=args.tables_file,
        )

        # Initialize datasets for different phases
        train_dataset = T5Dataset(tokenizer, args.train_data_dir, is_test=False, exclude_unans=False, **dataset_kwargs)
        valid_dataset = T5Dataset(tokenizer, args.valid_data_dir, is_test=False, exclude_unans=False, **dataset_kwargs)
        test_dataset = T5Dataset(tokenizer, args.test_data_dir, is_test=True, exclude_unans=False, **dataset_kwargs)

        print(f"Train dataset: {len(train_dataset)}")
        print(f"Valid dataset: {len(valid_dataset)}")
        print(f"Test dataset: {len(test_dataset)}")

        # Create DataLoader instances for batch processing
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn, 
                                #   shuffle=True,
                                sampler=sampler,
                                )
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, collate_fn=valid_dataset.collate_fn, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

        # Load existing model or initialize optimizer and scheduler
        if args.load_checkpoint_path:
            model, optimizer, scheduler, args, step, best_metric = load_model(model, args.load_checkpoint_path, args, train_loader)
        else:
            step, best_metric = 0, -1
            optimizer, scheduler = set_optim(model, train_loader, args)


        # Load the best-performing model checkpoint

        model, _, _, args, step, best_metric = load_model(
            model,
            os.path.join(args.save_model_path, f'checkpoint_best_{best_val_loss:.4f}.pth.tar'),
            args,
            train_loader,
        )

        print("Start SQL generation and validation")
        model_name = args.model_name.replace("/", "_")
        
        # val_ids, val_correctness, val_generated_sqls = get_sql_correctness(valid_loader, model, tokenizer, args)
        # val_questions = [valid_dataset.id_to_question[id_] for id_ in val_ids]
        
        # print(len(val_ids), len(val_questions), len(val_correctness), len(val_generated_sqls))

        # val_question_sql_correctness_df = pd.DataFrame({
        #     'id': val_ids, # 'id' is the key in the scoring program
        #     'question': val_questions,
        #     'correctness': val_correctness,
        #     'generated_sql': val_generated_sqls,
        # })
        # val_question_sql_correctness_df.to_csv(f'./sql_validations/val_qsc_{model_name}_{best_val_loss:.4f}.csv', index=False)

        # train_ids, train_correctness, train_generated_sqls = get_sql_correctness(train_loader, model, tokenizer, args)
        # train_questions = [train_dataset.id_to_question[id_] for id_ in train_ids]

        # print(f"""
        #     train_ids, {len(train_ids)}, 
        #     train_questions, {len(train_questions)}, 
        #     train_correctness, {len(train_correctness)}, 
        #     train_generated_sqls, {len(train_generated_sqls)}
        # """
        # )

        # train_question_sql_correctness_df = pd.DataFrame({
        #     'id': train_ids, # 'id' is the key in the scoring program
        #     'question': train_questions,
        #     'correctness': train_correctness,
        #     'generated_sql': train_generated_sqls,
        # })
        # train_question_sql_correctness_df.to_csv(f'./sql_validations/train_qsc_{model_name}_{best_val_loss:.4f}.csv', index=False)

        test_ids, test_correctness, test_generated_sqls = get_sql_correctness(test_loader, model, tokenizer, args, test=True)
        test_questions = [test_dataset.id_to_question[id_] for id_ in test_ids]

        print(f"""
            test_ids, {len(test_ids)},
            test_questions, {len(test_questions)},
            test_correctness, {len(test_correctness)},
            test_generated_sqls, {len(test_generated_sqls)}
        """
        )

        test_question_sql_correctness_df = pd.DataFrame({
            'id': test_ids, # 'id' is the key in the scoring program
            'question': test_questions,
            'correctness': test_correctness,
            'generated_sql': test_generated_sqls,
        })
        test_question_sql_correctness_df.to_csv(f'./sql_validations/test_qsc_{model_name}_{best_val_loss:.4f}.csv', index=False)


