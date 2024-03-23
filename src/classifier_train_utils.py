
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

from train_utils import save_model

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2)

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
            labels = batch["y"].to(args.device)

            # Forward pass and calculate loss
            output = model(input_ids=source_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

            # loss = output.loss
            pred = output.logits.squeeze(-1)
            # if pred=1 and y=0, 10 times more important
            # if pred=0 and y=1, 1 times more important
            weights = torch.ones_like(pred, dtype=torch.bfloat16).to(args.device)
            weights[(pred > 0.5) & (labels < 0.5)] *= 10.4

            loss = weighted_mse_loss(pred, labels, weights).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()  # Update model parameters
            model.zero_grad()  # Reset gradients
            step += 1

            train_loss_list.append(loss.item())

            # Get the current learning rate from scheduler or optimizer
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]

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
                labels = batch["y"].to(args.device)

                output = model(input_ids=ids, attention_mask=mask, labels=labels)
                pred = output.logits.squeeze(-1)
                # if pred=1 and y=0, 10 times more important
                # if pred=0 and y=1, 1 times more important
                weights = torch.ones_like(pred, dtype=torch.bfloat16).to(args.device)
                weights[(pred > 0.5) & (labels < 0.5)] *= 10.4

                loss = weighted_mse_loss(pred, labels, weights).mean()
                valid_loss_list.append(loss.item())

            # Calculate average validation loss
            valid_loss = sum(valid_loss_list) / len(valid_loss_list)

            log = f"epoch: {epoch} (step: {step})"
            log += f" | valid_loss: {valid_loss:.6f}"
            
            if best_metric > valid_loss:
                early_stop_count = 0
                best_metric = valid_loss
                save_model(model, optimizer, scheduler, step, best_metric, args, name=f"best_{best_metric:.4f}")

            model.train()  # Set the model back to training mode

            torch.cuda.empty_cache()
            gc.collect()  # Trigger Python garbage collection

        print(log)
        # Save a checkpoint at the end of each epoch if specified in args
        if args.save_every_epoch:
            save_model(model, optimizer, scheduler, epoch, best_metric, args, name=f"{epoch}")
    
    model.eval()
    return best_metric


def predict_binary(tokenizer, model, dataloader, args, get_label=True):

    model.eval()
    preds = []
    labels = []
    question_ids = []

    for batch in dataloader:
        qid = batch["id"]
        ids = batch["source_ids"].to(args.device)
        mask = batch["source_mask"].to(args.device)
        if get_label:
            label = batch["y"].to(args.device)
        else:
            label = None

        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=mask, labels=label, return_dict=True)
            preds += outputs.logits.float().cpu().tolist()
            question_ids += qid
            if get_label:
                labels += label.float().cpu().tolist()
    preds, labels = np.array(preds).flatten(), np.array(labels)
    return preds, labels, question_ids
