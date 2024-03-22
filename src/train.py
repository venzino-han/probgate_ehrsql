import torch


exp_name = "t5-text2sql"
model_name = "gaussalgo/T5-LM-Large-text2sql-spider"
ARGS_STR = f"""
--exp_name={exp_name} \
--model_name={model_name} \
--train_data_dir={NEW_TRAIN_DIR} \
--valid_data_dir={NEW_VALID_DIR} \
--test_data_dir={NEW_TEST_DIR} \
--tables_file={TABLES_PATH} \
--train_epochs=20 \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-3 \
--report_every_step=10 \
--eval_every_step=10 \
--bf16=1\
--use_schema_info=1\
--db_id=mimic_iv\
--exclude_unans=1
"""
if __name__ == "__main__":

        # Define and parse command line arguments for model configuration
        # exp_name='t5-baseline'
        # model_name='t5-base'
        # --load_checkpoint_path=outputs/t5-text2sql/checkpoint_best_0.0003.pth.tar

        # Parse arguments
        parser = argparse.ArgumentParser()
        parser = add_default_args(parser)
        args = parser.parse_args(ARGS_STR.split())



if __name__ == "__main__":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_epsilon)
    t_total = (len(train_loader.dataset) // (args.train_batch_size * max(1, args.n_gpu))) * args.train_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)