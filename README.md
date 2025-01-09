# Detecting Unanswerable Questions with RAG

### 1. requirements.txt 요구되는 패키지 다운로드

```
pip install -r requirements.txt
pip install --upgrade transformers
pip install --upgrade sentence-transformers
```



### 2. 01_prepare_data.py 실행 -> combined_datasets.json 생성

combined_datasets.json: 연구에 사용할 데이터셋



### 3. 02_context_pool.py 실행 -> full_context.json 생성

full_context.json: retrieval을 수행할 context pool



### 4. 03_classifier_data_{model_name}.py 실행

```
hf_token = 'hf_XX' # 개인 huggingface token 넣어서 실행
```

model_name: llama_3_1_8b(meta-llama/Llama-3.1-8B-Instruct), llama_2_7b(meta-llama/Llama-2-7b-chat-hf), llama_2_13b(meta-llama/Llama-2-13b-chat-hf), gemma_2_2b(google/gemma-2-2b-it)

qa_results_{n}.json 파일 생성 -> 위 모델 순서대로 n=1, 2, 3, 4

> classifier를 훈련시킬 데이터셋 qa_results_{n}.json 생성



### 5. 04_result_counter.py 실행

```
file_path = qa_results_{n}.json # n 변경
```

생성한 데이터셋의 결과 요약하는 코드



### 6. 05_classifier.py 실행

```
file_path = 'qa_results_{n}.json' # n 변경하며 각 데이터셋에 대해 훈련 진행
```

> classifier 모델 훈련



### 7. 06_compare.py 실행

```
file_path_1 = 'qa_results_{n}.json' # n 변경
file_path_2 = 'qa_results_{m}.json' # m 변경
```

> 두 모델의 retrieval 횟수 결과에 어떤 차이가 있는지를 확인하기 위한 코드