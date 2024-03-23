
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
from sql_utils import get_sql_query_result_from_batch

def get_default_args():
    """
    Define and set default arguments for the script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_id", type=str, default="mimiciii", help="database name")  # NOTE: `mimic_iv` will be used for codabench
    parser.add_argument("--train_data_dir", type=str, help="train data path")
    parser.add_argument("--valid_data_dir", type=str, help="valid data path")
    parser.add_argument("--test_data_dir", type=str, help="test data path")
    parser.add_argument("--tables_file", type=str, help="table schema path")

    parser.add_argument("--output_dir", type=str, default="outputs", help="output directory")
    parser.add_argument("--output_file", type=str, default="prediction_raw.json", help="output file name")

    parser.add_argument("--binary_classification", type=bool, default=False)

    # basic parameters
    parser.add_argument("--exp_name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_checkpoint_path", type=str, default=None)
    parser.add_argument("--load_checkpoint_path", type=str, default=None)

    # training parameters
    parser.add_argument("--train_batch_size", type=int, default=12)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=786)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=str, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    parser.add_argument("--report_every_step", type=int, default=1000)
    parser.add_argument("--eval_every_step", type=int, default=-1000)
    parser.add_argument("--save_every_epoch", type=bool, default=False)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--use_schema_info", type=bool, default=False)
    parser.add_argument("--exclude_unans", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    # generation parameters
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=1)
    return parser


def set_seed(args):
    """
    Ensure reproducibility by setting the seed for random number generation.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, optimizer, scheduler, step, best_metric, args, name="last"):
    """
    Save model checkpoints during or after training.
    """
    os.makedirs(args.save_model_path, exist_ok=True)

    save_file_path = os.path.join(args.save_model_path, f"checkpoint_{name}.pth.tar")
    state_dict = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": args,
        "best_metric": best_metric,
    }
    torch.save(state_dict, save_file_path)
    print(f"Model checkpoint '{name}' saved successfully to {save_file_path}.")


def load_model(model, load_model_path, args, train_loader, reset_optim=False):
    """
    Load a saved model checkpoint.
    """
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    prev_args = checkpoint["args"]
    args = update_args(new_args=args, prev_args=prev_args)

    step = checkpoint["step"]
    best_metric = checkpoint["best_metric"]
    if not reset_optim:
        optimizer, scheduler = set_optim(model, train_loader, args)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer, scheduler = set_optim(args, model)

    return model, optimizer, scheduler, args, step, best_metric


def update_args(new_args, prev_args):
    """
    Update training arguments with the values saved in the checkpoint.
    """
    for arg in vars(prev_args):
        if arg not in new_args:
            setattr(new_args, arg, getattr(prev_args, arg))
    return new_args


def set_optim(model, train_loader, args):
    """
    Initialize the optimizer and learning rate scheduler for the model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_epsilon)
    t_total = (len(train_loader.dataset) // (args.train_batch_size * max(1, args.n_gpu))) * args.train_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return optimizer, scheduler


def train(tokenizer, model, train_loader, optimizer, step=0, valid_loader=None, best_metric=-1, scheduler=None, args=None):
    """
    Conduct the training process for a given model.
    """
    train_loss_list = []
    batch_idx = 0

    if best_metric == -1:
        best_metric = np.inf

    early_stop_count = 0
    # Main training loop
    for epoch in range(1, args.train_epochs + 1):
        if early_stop_count > 3:
            print("Early stopping activated.")
            break
        early_stop_count += 1
        model.train()  # Set the model to training mode
        
        c = 0
        for batch in tqdm(train_loader):
            c += 1
            # Extract and send batch data to the specified device
            source_ids = batch["source_ids"].to(args.device)
            attention_mask = batch["source_mask"].to(args.device)
            labels = batch["target_ids"].to(args.device)
            sql_results_labels = batch["sql_result_labels"]

             # Making padded ids (pad=0) are set to -100, which means ignore for loss calculation
            labels[labels[:,:]==tokenizer.pad_token_id] = -100
            labels = labels.to(args.device)

            # Forward pass and calculate loss
            loss = model(input_ids=source_ids, attention_mask=attention_mask, labels=labels)[0]

            # Generate sql & run sql

            # print(sql_results[:5])
            # print(sql_results_labels[:5])

            # additional loss weigth for sql result
            # 10 times weight for sql result wrong or query for null answer
            # 1 times weight for sql result correct
            wieght = 1

            if epoch > 8:
                sql_results = get_sql_query_result_from_batch(tokenizer, model, batch, args)
                batch_size = len(sql_results)
                wieght = 0
                for sql_result, sql_result_label in zip(sql_results, sql_results_labels):
                    if sql_result == sql_result_label:
                        wieght += 1/batch_size
                    elif (sql_result == 'null') and (sql_result_label != 'null'):
                        wieght += 1/batch_size
                    else:
                        wieght += 10/batch_size

            # Normalize loss to account for gradient accumulation
            loss = torch.mean(loss)*wieght / args.gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation logic
            if batch_idx % args.gradient_accumulation_steps == 0:
                # Clip gradients to avoid exploding gradient problem
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # Update model parameters
                model.zero_grad()  # Reset gradients
                step += 1

            train_loss_list.append(loss.item())

            # Get the current learning rate from scheduler or optimizer
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]

            # Log training progress
            # if batch_idx % (args.report_every_step * args.gradient_accumulation_steps) == 0:
                # print(log)
            log = f"epoch: {epoch} (step: {step}) | "
            log += f"train loss: {sum(train_loss_list)/len(train_loss_list):.6f} | "
            log += f"lr: {lr:.6f}"
            
            if scheduler:
                scheduler.step()  # Update learning rate

            # if c == 10:
            #     break

        train_loss_list = []
        print(log)
        
        # Validation step
        # if valid_loader and batch_idx % (args.eval_every_step * args.gradient_accumulation_steps) == 0:
        model.eval()  # Set the model to evaluation mode
        valid_loss_list = []
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                ids = batch["source_ids"].to(args.device)
                mask = batch["source_mask"].to(args.device)
                labels = batch["target_ids"].to(args.device)

                labels[labels[:,:]==tokenizer.pad_token_id] = -100
                labels = labels.to(args.device)

                valid_loss = model(input_ids=ids, attention_mask=mask, labels=labels)[0]
                valid_loss_list.append(valid_loss.item())

            # Calculate average validation loss
            valid_loss = sum(valid_loss_list) / len(valid_loss_list)

            log = f"epoch: {epoch} (step: {step})"
            log += f" | valid_loss: {valid_loss:.6f}"
            
            if best_metric > valid_loss:
                early_stop_count = 0
                best_metric = valid_loss
                save_model(model, optimizer, scheduler, step, best_metric, args, name=f"best_{best_metric:.4f}")

            model.train()  # Set the model back to training mode

            batch_idx += 1

            # Clear CUDA cache if it's a good time
            if batch_idx % (args.eval_every_step * args.gradient_accumulation_steps) == 0:
                torch.cuda.empty_cache()
                gc.collect()  # Trigger Python garbage collection

        print(log)
        # Save a checkpoint at the end of each epoch if specified in args
        if args.save_every_epoch:
            save_model(model, optimizer, scheduler, epoch, best_metric, args, name=f"{epoch}")
    
    # for batch in tqdm(valid_loader):
    #     model.train() 
    #     # Extract and send batch data to the specified device
    #     source_ids = batch["source_ids"].to(args.device)
    #     attention_mask = batch["source_mask"].to(args.device)
    #     labels = batch["target_ids"].to(args.device)
    #     sql_results_labels = batch["sql_result_labels"]

    #         # Making padded ids (pad=0) are set to -100, which means ignore for loss calculation
    #     labels[labels[:,:]==tokenizer.pad_token_id] = -100
    #     labels = labels.to(args.device)

    #     # Forward pass and calculate loss
    #     loss = model(input_ids=source_ids, attention_mask=attention_mask, labels=labels)[0]

    #     # Generate sql & run sql
    #     sql_results = get_sql_query_result_from_batch(tokenizer, model, batch, args)

    #     # print(sql_results[:5])
    #     # print(sql_results_labels[:5])

    #     # additional loss weigth for sql result
    #     # 10 times weight for sql result wrong or query for null answer
    #     # 1 times weight for sql result correct
    #     batch_size = len(sql_results)
    #     wieght = 0
    #     for sql_result, sql_result_label in zip(sql_results, sql_results_labels):
    #         if sql_result == sql_result_label:
    #             wieght += 1/batch_size
    #         elif (sql_result == 'null') and (sql_result_label != 'null'):
    #             wieght += 1/batch_size
    #         else:
    #             wieght += 10/batch_size

    #     # Normalize loss to account for gradient accumulation
    #     loss = torch.mean(loss)*wieght / args.gradient_accumulation_steps
    #     loss.backward()

    #     # Gradient accumulation logic
    #     if batch_idx % args.gradient_accumulation_steps == 0:
    #         # Clip gradients to avoid exploding gradient problem
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #         optimizer.step()  # Update model parameters
    #         model.zero_grad()  # Reset gradients
    #         step += 1
    model.eval()
    return best_metric

def generate_sql(tokenizer, model, eval_loader, args):
    # Set the model to evaluation mode. This turns off certain layers like dropout.
    model.eval()

    # Disable gradient calculations for efficiency, as they are not needed in evaluation.
    with torch.no_grad():
        out_eval = [None for _ in range(len(eval_loader.dataset))]

        sample_idx = 0
        # Iterate over batches of data in the evaluation dataset.
        for batch in tqdm(eval_loader):
            # Extract relevant data from the batch.
            ids = batch["id"]
            source_ids = batch["source_ids"].to(args.device)
            # attention_mask = batch["source_mask"].to(args.device)

            # Generate predictions using the model.
            generation_output = model.generate(
                input_ids=source_ids,
                max_length=args.max_target_length,
                num_beams=args.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # Move the generated sequences to the CPU if using CUDA.
            preds = generation_output["sequences"].cpu() if args.device == "cuda" else generation_output["sequences"]

            # Process logits and calculate probabilities and entropies.
            logits = torch.stack(generation_output["scores"], dim=1)[:: int(args.num_beams / args.num_samples)]
            logits = logits.cpu() if args.device == "cuda" else logits
            probs = torch.softmax(logits, dim=2).float()
            log_probs = torch.log_softmax(logits, dim=2).float()
            entropies = (torch.sum(probs * log_probs, axis=2) * (-1)).numpy()

            # Determine if the current batch is for testing or training.
            is_test = True
            if "target_ids" in batch:
                is_test = False
                reals = batch["target_ids"]

            # Initialize lists to store predictions, probabilities, and entropies.
            pred_list = []
            entropy_list = []

            # Process each prediction in the batch.
            for idx in range(len(preds)):
                pred = preds[idx]
                pred_tensor = preds[idx][1:]
                entropy_truncated = entropies[idx].tolist()

                # Truncate the prediction at the end-of-sequence token, if present.
                if tokenizer.eos_token_id in pred_tensor:
                    pred_eos_idx = torch.nonzero(pred_tensor == tokenizer.eos_token_id)[0].item()
                    entropy_truncated = entropy_truncated[: pred_eos_idx + 1]

                pred_list.append(pred)
                entropy_list.append(entropy_truncated)

            # Construct the output results for each prediction.
            for idx in range(len(preds)):
                result = {
                    "id": ids[idx],
                    "question": tokenizer.decode(source_ids[idx], skip_special_tokens=True),
                    "pred": tokenizer.decode(pred_list[idx], skip_special_tokens=True),
                    "entropy": entropy_list[idx],
                }

                # Include the real target output if it's training data.
                if not is_test:
                    result["real"] = tokenizer.decode(reals[idx], skip_special_tokens=True)
                
                out_eval[sample_idx] = result
                sample_idx += 1

            # Clear cache after processing each batch
            torch.cuda.empty_cache()
            gc.collect()

        return out_eval
    

def get_threshold(id2maxent, score_dict):
    """
    Determine the optimal threshold for filtering based on maximum entropy and scores.
    """
    values = []
    scores = []
    for key, val in id2maxent.items():
        values.append(val)
        scores.append(score_dict[key])

    sorted_indices = np.argsort(values)
    sorted_values = np.array(values)[sorted_indices]
    sorted_scores = np.array(scores)[sorted_indices]

    max_score, threshold = 0, -1
    count = 0
    for idx in range(len(sorted_scores)):
        cum_score = sum(sorted_scores[:idx+1])
        if cum_score > max_score:
            count += 1
            max_score, threshold = cum_score, sorted_values[idx-1]
    print(f"Number of queries: {len(values)}, Number of queries filtered: {count}")
    return threshold  # We abstain if maxent is greater than this threshold.

