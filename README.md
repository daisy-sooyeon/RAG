# Detecting Unanswerable Questions with RAG

## Description

Creating a classification model for detecting questions that are **unanswerable** for a RAG model to respond 


## Used Data

squad, hotpotqa(distractor, fullwiki)



## How to get the classification model


### 1. Download the required packages with **requirements.txt**

```
pip install -r requirements.txt
pip install --upgrade transformers
pip install --upgrade sentence-transformers
```




### 2. Run *01_prepare_data.py* and create **combined_datasets.json**

combined_datasets.json: the full dataset used throughout this study




### 3. Run **02_context_pool.py** and create **full_context.json**

full_context.json: the context pool used for retrieval




### 4. Run **03_classifier_data_{model_name}.py** and create **qa_results_{n}.json**

```
hf_token = 'hf_XX' # insert your personal Hugging Face token and run
```

{model_name} input
- llama_3_1_8b(meta-llama/Llama-3.1-8B-Instruct)
- llama_2_7b(meta-llama/Llama-2-7b-chat-hf)
- llama_2_13b(meta-llama/Llama-2-13b-chat-hf)
- gemma_2_2b(google/gemma-2-2b-it)

set n = 1, 2, 3, 4 in the same order as the models listed above

qa_results_{n}.json: a final dataset used to train the classifier model




### 5. Run **04_result_counter.py**

```
file_path = qa_results_{n}.json # change n by 1, 2, 3, or 4
```

> code summarizing the final dataset created




### 6. Run **05_classifier.py**

```
file_path = 'qa_results_{n}.json' # change n to train on each dataset
```

> code training a classifier model




### 7. Run **06_compare.py**

```
file_path_1 = 'qa_results_{n}.json' # change n
file_path_2 = 'qa_results_{m}.json' # change m
```

> code to compare the retrieval counts between the two models for improved interpretability 
