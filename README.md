# ProbGate at EHRSQL 2024

This repository contains the code and data about the project:
[ProbGate at EHRSQL 2024: Enhancing SQL Query Generation Accuracy through Probabilistic Threshold Filtering and Error Handling](https://arxiv.org/abs/2404.16659)


This is part of the shared tasks at [NAACL 2024 - Clinical NLP](https://clinical-nlp.github.io/2024).

- Task overview: [https://sites.google.com/view/ehrsql-2024](https://sites.google.com/view/ehrsql-2024)



## Install
Tested on `python 3.9.18`
```
cd src
pip install -r requirements.txt
```
### presequites

Please see below link for the detailed information of the baseline code if you needed. This repository is based on below repository(initial setting, scoring).

`Code` and `Dataset` : [https://github.com/glee4810/ehrsql-2024](https://github.com/glee4810/ehrsql-2024)

## Code
### Fine-Tuning
Please run `./src/chatgpt_sql_finetuning.ipynb`. you need your own `<your open ai api key>`. For the experiments, we use OPENAI `gpt-3.5-turbo-0125` model. If it is deperacated use another model. See https://platform.openai.com/docs/guides/fine-tuning.

Suggestion : for the better result you can use LLM specialed for Text2SQL task like [sqlcoder](https://huggingface.co/defog/sqlcoder-7b-2)

### Inference
Please run `./src/chatgpt_sql_inference.ipynb`. you need your own `<your open ai api key>`. You can use either `valid` or `test` dataset. Please see `src/data/mimic_iv/` folder. The output format is either `pkl` or `json` file.

For the convenience, we share `log_probability_final_test.json` and `log_probability_test_new.pickle` file after training and inference with `test` dataset.

### ProbGate
Please run `./src/filter_log_probability.ipynb` for ProbGate. Like we described in the original paper. we set `k` into 425, `t` into 10.

### Grammatical Errors Filtering
Please run `./src/run_sql_check.py` for the GEF. After this, you can get your final answer json file.

### (Optional) Execute SQL statement
Please run `./src/run_sql.py` to get the inferenced answer of executing generated sql statement(if it is answerable) with your final json file.


## Citation
Please cite with below link. Also, If you have any question, contact to the corresponding author. Thank you
```
@article{kim2024probgate,
  title={ProbGate at EHRSQL 2024: Enhancing SQL Query Generation Accuracy through Probabilistic Threshold Filtering and Error Handling},
  author={Kim, Sangryul and Han, Donghee and Kim, Sehyun},
  journal={arXiv preprint arXiv:2404.16659},
  year={2024}
}
```
