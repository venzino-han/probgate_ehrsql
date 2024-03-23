from transformers import T5Tokenizer
from configs import NEW_TRAIN_DIR,  NEW_VALID_DIR,  NEW_TEST_DIR,  RESULT_DIR,  TABLES_PATH,  DB_ID, DB_PATH
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
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForSequenceClassification, get_linear_schedule_with_warmup

from train_utils import get_default_args, set_seed, set_optim, load_model, generate_sql, get_threshold
from classifier_train_utils import train, predict_binary

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# exp_name = "text2sql"
# model_name = "gaussalgo/T5-LM-Large-text2sql-spider"
# TRAIN_BATCH_SIZE = 8
# VAL_BATCH_SIZE = 16
# TEST_BATCH_SIZE = 16
# NULL_SAMPLE_RATIO = 0.1
# USE_SCHEMA = 0
# best_val_loss = 0.0530
# RUN_TRAIN = False

# model_name = "sangryul/Flan-T5-XL-text2sql-spider"
# model_name = "./flan-t5-xl-extended-schema"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# exp_name = "flan-t5-small"
# model_name = "google/flan-t5-small"
# TRAIN_BATCH_SIZE = 16
# VAL_BATCH_SIZE = 16
# TEST_BATCH_SIZE = 16
# NULL_SAMPLE_RATIO = 0.1
# USE_SCHEMA = 0
# best_val_loss = 0.9524
# RUN_TRAIN = False

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
exp_name = "flan-t5-small_null30_schema_classify_50"
model_name = "google/flan-t5-small"
TRAIN_BATCH_SIZE = 12
VAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
NULL_SAMPLE_RATIO = 0.5
USE_SCHEMA = 1
RUN_TRAIN = True
best_val_loss = None
EPOCHS = 10

ADD_CUSTOM_TOKENS = True

# --exclude_unans=1
if __name__ == "__main__":

    ARGS_STR = f"""
    --exp_name={exp_name} \
    --model_name={model_name} \
    --train_data_dir={NEW_TRAIN_DIR} \
    --valid_data_dir={NEW_VALID_DIR} \
    --test_data_dir={NEW_TEST_DIR} \
    --tables_file={TABLES_PATH} \
    --train_epochs={EPOCHS} \
    --train_batch_size={TRAIN_BATCH_SIZE} \
    --valid_batch_size={VAL_BATCH_SIZE} \
    --test_batch_size={TEST_BATCH_SIZE} \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --report_every_step=10 \
    --eval_every_step=10 \
    --bf16=1\
    --db_id=mimic_iv\
    --use_schema_info={USE_SCHEMA}\
    --max_target_length=512\
    """
    # --binary_classification=1
    # Define and parse command line arguments for model configuration
    # exp_name='t5-baseline'
    # model_name='t5-base'
    # --load_checkpoint_path=outputs/t5-text2sql/checkpoint_best_0.0003.pth.tar

    # Parse arguments
    parser = get_default_args()
    args = parser.parse_args(ARGS_STR.split())

    # Configure CUDA settings

    # Set random seed for reproducibility
    set_seed(args)

    # Determine device for training and set model save path
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.save_model_path = os.path.join(args.output_dir, args.exp_name)

    
    print("="*50)
    args_dict = vars(args)
    for k, v in args_dict.items():
        print(f"{k} : {v}")
    print("="*50)


    # Initialize T5 model and set device
    model = T5ForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    # model = AutoModel.from_pretrained(args.model_name)
    model = model.to(args.device)

    # Convert model to bfloat16 precision if required
    if args.bf16:
        print("bfloat16 precision will be used")
        model = model.to(torch.bfloat16)
    
    # load custom tokens
    with open('schema_tokens.txt', 'r') as f:
        custom_tokens = f.read().split('\n')

    # Initialize tokenizer with additional SQL tokens
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    add_tokens = ["<", "<=", "<>"] + custom_tokens
    add_tokens = custom_tokens
    if ADD_CUSTOM_TOKENS:
        tokenizer.add_tokens(add_tokens)

    # Resize model token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Define parameters for dataset preparation
    dataset_kwargs = dict(
        db_id=args.db_id,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        tables_file=args.tables_file,
        append_schema_info=args.use_schema_info,
        binary_classification=True,
    )

    print(dataset_kwargs)

    # Initialize datasets for different phases
    train_dataset = T5Dataset(
        tokenizer, 
        args.train_data_dir, 
        is_test=False, 
        exclude_unans=args.exclude_unans, 
        null_sample_ratio=NULL_SAMPLE_RATIO,
        **dataset_kwargs)
    valid_dataset = T5Dataset(tokenizer, args.valid_data_dir, is_test=False, exclude_unans=False, **dataset_kwargs)
    # valid_dataset_exclude_unans = T5Dataset(tokenizer, args.valid_data_dir, is_test=False, exclude_unans=True, **dataset_kwargs)
    test_dataset = T5Dataset(tokenizer, args.test_data_dir, is_test=True, exclude_unans=False, **dataset_kwargs)

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Valid dataset: {len(valid_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")

    # get labels and questions
    train_labels = train_dataset.labels
    train_questions = train_dataset.questions
    valid_labels = valid_dataset.labels
    valid_questions = valid_dataset.questions
    test_labels = test_dataset.labels
    test_questions = [ 'null' for _ in range(len(test_labels))]

    # print(train_labels[:5])
    # print(train_questions[:5])

    labels = train_labels + valid_labels + test_labels
    questions = train_questions + valid_questions + test_questions

    print(len(labels), len(questions))

    # save as csv
    df = pd.DataFrame({'questions': questions, 'labels': labels})
    df.to_csv('questions_labels.csv', index=False)

    # Create DataLoader instances for batch processing
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))
    valid_sampler = WeightedRandomSampler(valid_dataset.weights, len(valid_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn, 
                            #   shuffle=True,
                            sampler=sampler,
                            )
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, collate_fn=valid_dataset.collate_fn, 
                            sampler=valid_sampler,
                            )
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

    # Load existing model or initialize optimizer and scheduler
    if args.load_checkpoint_path:
        model, optimizer, scheduler, args, step, best_metric = load_model(model, args.load_checkpoint_path, args, train_loader)
    else:
        step, best_metric = 0, -1
        optimizer, scheduler = set_optim(model, train_loader, args)



    from scoring_program.scoring_utils import execute_all, reliability_score, penalize
    from scoring_program.postprocessing import post_process_sql

    # best_val_loss = 0.0000

    # Load the best-performing model checkpoint

    if best_val_loss is not None:    
        model, optimizer, scheduler, args, step, best_metric = load_model(
            model,
            os.path.join(args.save_model_path, f'checkpoint_best_{best_val_loss:.4f}.pth.tar'),
            args,
            train_loader,
        )

    # Start the training process
    if RUN_TRAIN:
        best_val_loss = train(
            tokenizer=tokenizer,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            best_metric=best_metric,
            args=args,
        )

    model, optimizer, scheduler, args, step, best_metric = load_model(
        model,
        os.path.join(args.save_model_path, f'checkpoint_best_{best_val_loss:.4f}.pth.tar'),
        args,
        train_loader,
    )

    # Perform inference on the validation set
    # valid_eval = generate_sql(tokenizer, model, valid_loader, args)
    binary_preds, binary_labels, _ = predict_binary(tokenizer, model, valid_loader, args)
    

    """  
    answerable --> 1 
    unanswerable --> 0
    """
    print(f"Binary Preds: {binary_preds[:5]}")
    print(f"Binary Labels: {binary_labels[:5]}")
    # binary classification score
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
    # binary_preds, binary_labels = 
    binary_accuracy = accuracy_score(binary_labels>0.5, binary_preds>0.5)
    # binary_recall = recall_score(binary_labels, binary_preds)
    # binary_precision = precision_score(binary_labels, binary_preds)
    # binary_f1 = f1_score(binary_labels, binary_preds)
    binary_auc = roc_auc_score(binary_labels, binary_preds)

    print(f"""
            Binary Classification Scores: 
            Accuracy: {binary_accuracy:.3f}, 
            AUC: {binary_auc:.3f},
            """)
    
    test_binary_preds, _, question_ids = predict_binary(tokenizer, model, test_loader, args, get_label=False)

    # save json file
    import json

    test_binary_preds = {qid: pred for qid, pred in zip(question_ids, test_binary_preds)}
    with open('test_binary_preds.json', 'w') as f:
        json.dump(test_binary_preds, f)
