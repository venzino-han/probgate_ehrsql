import torch 
import gc
from tqdm import tqdm
from scoring_program.scoring_utils import execute_sql_wrapper
import os

DB_ID = 'mimic_iv'
# File paths for the dataset and labels
DB_PATH = os.path.join('data', DB_ID, f'{DB_ID}.sqlite') # Database path


def get_sql_query_result_from_batch(tokenizer, model, batch, args):
    # Set the model to evaluation mode. This turns off certain layers like dropout.
    model.eval()

    # Disable gradient calculations for efficiency, as they are not needed in evaluation.
    with torch.no_grad():

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

        preds = generation_output["sequences"].cpu() if args.device == "cuda" else generation_output["sequences"]

        # Process logits and calculate probabilities and entropies.
        logits = torch.stack(generation_output["scores"], dim=1)[:: int(args.num_beams / args.num_samples)]
        logits = logits.cpu() if args.device == "cuda" else logits
        probs = torch.softmax(logits, dim=2).float()
        log_probs = torch.log_softmax(logits, dim=2).float()
        entropies = (torch.sum(probs * log_probs, axis=2) * (-1)).numpy()

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
        results = ["null" for _ in range(len(preds))]
        for idx in range(len(preds)):
            sql = tokenizer.decode(pred_list[idx], skip_special_tokens=True)
            # excute sql
            if sql == "null":
                continue
            else:
                query_result = execute_sql_wrapper(0, sql, DB_PATH, 'pred')[-1]

            results[idx] = query_result

        # Clear cache after processing each batch
        torch.cuda.empty_cache()
        gc.collect()
        return results
    


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