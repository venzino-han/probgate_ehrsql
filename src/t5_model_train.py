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
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from train_utils import get_default_args, set_seed, set_optim, load_model, train, generate_sql, get_threshold

# --exclude_unans=1
if __name__ == "__main__":

    exp_name = "t5_text2sql_schema"
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
    --train_batch_size=4 \
    --valid_batch_size=8 \
    --test_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --report_every_step=10 \
    --eval_every_step=10 \
    --bf16=1\
    --db_id=mimic_iv\
    --use_schema_info=1\
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    add_tokens = ["<", "<=", "<>"] + custom_tokens
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
        append_schema_info=args.use_schema_info,
        answerable_or_not_binary=args.binary_classification,
    )

    print(dataset_kwargs)

    # Initialize datasets for different phases
    train_dataset = T5Dataset(tokenizer, args.train_data_dir, is_test=False, exclude_unans=args.exclude_unans, **dataset_kwargs)
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

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn, 
                              shuffle=True,
                            # sampler=sampler,
                            )
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, collate_fn=valid_dataset.collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

    # Load existing model or initialize optimizer and scheduler
    if args.load_checkpoint_path:
        model, optimizer, scheduler, args, step, best_metric = load_model(model, args.load_checkpoint_path, args, train_loader)
    else:
        step, best_metric = 0, -1
        optimizer, scheduler = set_optim(model, train_loader, args)


    # Start the training process
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

    from scoring_program.scoring_utils import execute_all, reliability_score, penalize
    from scoring_program.postprocessing import post_process_sql

    # best_val_loss = 0.0000

    # Load the best-performing model checkpoint
    model, optimizer, scheduler, args, step, best_metric = load_model(
        model,
        os.path.join(args.save_model_path, f'checkpoint_best_{best_val_loss:.4f}.pth.tar'),
        args,
        train_loader,
    )

    # Perform inference on the validation set
    # valid_eval = generate_sql(tokenizer, model, valid_loader, args)
    valid_eval = generate_sql(tokenizer, model, valid_loader, args)
    
    # binary classification score
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
    binary_preds = np.array([1 if sample['pred'] == 'null' else 0 for sample in valid_eval])
    binary_labels = np.array([1 if sample['real'] == 'null' else 0 for sample in valid_eval])
    binary_accuracy = accuracy_score(binary_labels, binary_preds)
    binary_recall = recall_score(binary_labels, binary_preds)
    binary_precision = precision_score(binary_labels, binary_preds)
    binary_f1 = f1_score(binary_labels, binary_preds)
    binary_auc = roc_auc_score(binary_labels, binary_preds)

    print(f"""
            Binary Classification Scores: 
            Accuracy: {binary_accuracy:.3f}, 
            AUC: {binary_auc:.3f},
            """)


    # Post-process SQL queries for evaluation
    label = {sample['id']: post_process_sql(sample['real']) for sample in valid_eval}
    label_y = {sample['id']: post_process_sql(sample['pred']) for sample in valid_eval}
    id2maxent = {sample['id']: max(sample['entropy']) for sample in valid_eval}  # NOTE: Abstain strategy not used here

    # Calculate the Reliability Score (RS) across all queries
    real_dict = {id_: post_process_sql(label[id_]) for id_ in label}
    pred_dict = {id_: post_process_sql(label_y[id_]) for id_ in label_y}
    assert set(real_dict) == set(pred_dict), "IDs do not match"

    real_result = execute_all(real_dict, db_path=DB_PATH, tag='real')
    pred_result = execute_all(pred_dict, db_path=DB_PATH, tag='pred')

    scores, score_dict = reliability_score(real_result, pred_result, return_dict=True)
    accuracy0 = penalize(scores, penalty=0)
    accuracy5 = penalize(scores, penalty=5)
    accuracy10 = penalize(scores, penalty=10)
    accuracyN = penalize(scores, penalty=len(scores))

    print(f"RS Scores: RS0: {accuracy0:.3f}, RS5: {accuracy5:.3f}, RS10: {accuracy10:.3f}, RSN: {accuracyN:.3f}")


    # Calculate threshold for filtering unanswerable queries
    threshold = get_threshold(id2maxent, score_dict)
    print(f"Threshold for filtering: {threshold}")

    # Apply threshold to filter out uncertain predictions
    val_label_y = {sample['id']: 'null' if threshold < max(sample['entropy']) else post_process_sql(sample['pred']) for sample in valid_eval}

    # Recalculate RS with filtered predictions
    real_dict = {id_: post_process_sql(label[id_]) for id_ in label}
    pred_dict = {id_: post_process_sql(val_label_y[id_]) for id_ in val_label_y}

    scores_filtered = reliability_score(real_dict, pred_dict)

    accuracy0_filtered = penalize(scores_filtered, penalty=0)
    accuracy5_filtered = penalize(scores_filtered, penalty=5)
    accuracy10_filtered = penalize(scores_filtered, penalty=10)
    accuracyN_filtered = penalize(scores_filtered, penalty=len(scores))

    # Output the refined RS scores with abstention
    # filter unanswerable queries
    print(f"RS Score with filtered: RS0: {accuracy0_filtered:.3f}, RS5: {accuracy5_filtered:.3f}, RS10: {accuracy10_filtered:.3f}, RSN: {accuracyN_filtered:.3f}")

    # Conduct inference on the test set (For now, we use original validation set as test data)
    test_eval = generate_sql(tokenizer, model, test_loader, args)

    # Apply the threshold to uncertain predictions
    label_y = {sample['id']: 'null' if threshold < max(sample['entropy']) else post_process_sql(sample['pred']) for sample in test_eval}

    label_with_entropy = {sample['id']: (post_process_sql(sample['pred']), max(sample['entropy'])) for sample in test_eval}

    # label_y = {sample['id']: sample['pred'] for sample in test_eval}

    import locale; locale.getpreferredencoding = lambda: "UTF-8" # if necessary
    from utils.data_io import write_json as write_label

    # Save the filtered predictions to a JSON file
    os.makedirs(RESULT_DIR, exist_ok=True)
    SCORING_OUTPUT_DIR = os.path.join(RESULT_DIR, 'prediction.json')
    write_label(SCORING_OUTPUT_DIR, label_y)


    SCORING_OUTPUT_DIR = os.path.join(RESULT_DIR, f'{exp_name}_prediction_with_entropy.json')
    write_label(SCORING_OUTPUT_DIR, label_with_entropy)

    # Verify the file creation
    print("Listing files in RESULT_DIR:")
    print(os.listdir(RESULT_DIR))
    """
    # Change to the directory containing the prediction file
    %cd {RESULT_DIR}
    # Compress the prediction.json file into a ZIP archive
    !zip predictions.zip prediction.json
    """

    # zip the prediction file
    import zipfile
    with zipfile.ZipFile(os.path.join(RESULT_DIR, f'predictions_{exp_name}_{best_val_loss:.4f}.zip'), 'w') as z:
        z.write(SCORING_OUTPUT_DIR, arcname='prediction.json')


