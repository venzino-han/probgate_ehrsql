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
@inproceedings{kim-etal-2024-probgate,
    title = "{P}rob{G}ate at {EHRSQL} 2024: Enhancing {SQL} Query Generation Accuracy through Probabilistic Threshold Filtering and Error Handling",
    author = "Kim, Sangryul  and
      Han, Donghee  and
      Kim, Sehyun",
    editor = "Naumann, Tristan  and
      Ben Abacha, Asma  and
      Bethard, Steven  and
      Roberts, Kirk  and
      Bitterman, Danielle",
    booktitle = "Proceedings of the 6th Clinical Natural Language Processing Workshop",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.clinicalnlp-1.65",
    pages = "687--696",
    abstract = "Recently, deep learning-based language models have significantly enhanced text-to-SQL tasks, with promising applications in retrieving patient records within the medical domain. One notable challenge in such applications is discerning unanswerable queries. Through fine-tuning model, we demonstrate the feasibility of converting medical record inquiries into SQL queries. Additionally, we introduce an entropy-based method to identify and filter out unanswerable results. We further enhance result quality by filtering low-confidence SQL through log probability-based distribution, while grammatical and schema errors are mitigated by executing queries on the actual database.We experimentally verified that our method can filter unanswerable questions, which can be widely utilized even when the parameters of the model are not accessible, and that it can be effectively utilized in practice.",
}

```
