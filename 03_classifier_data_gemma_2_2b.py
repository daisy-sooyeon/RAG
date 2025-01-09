import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

file_path = 'combined_datasets.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

file_path = 'full_context.json'

with open(file_path, 'r', encoding='utf-8') as f:
    all_contexts = json.load(f)


model_id = "google/gemma-2-2b-it"
hf_token = "hf_" ### token
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_token)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float32, "use_auth_token": hf_token},
    device_map = 'auto'
)
    
top_k = 7
chunk_size = 50

def chunk_text(text, chunk_size=100):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

chunked_passages = list(set([chunk for passage in all_contexts for chunk in chunk_text(passage, chunk_size)]))
tokenized_corpus = [word_tokenize(doc.lower()) for doc in chunked_passages]
bm25 = BM25Okapi(tokenized_corpus)

def initialize_retrieval():
    """ Resets the available indices for a new query. """
    global available_indices
    available_indices = set(range(len(all_contexts))) 

def retrieve_contexts(query, k, retrieved_indices):
    tokenized_query = word_tokenize(query.lower())
    doc_scores = bm25.get_scores(tokenized_query)
    sorted_doc_indices = np.argsort(doc_scores)[::-1]  

    new_contexts = []
    for idx in sorted_doc_indices:
        if idx not in retrieved_indices and len(new_contexts) < k:
            new_contexts.append(chunked_passages[idx])  
            retrieved_indices.add(idx)
    
    return new_contexts

def answer_question(tokenizer, question, context, num_retrievals=0):
    messages = [
        {"role": "user", "content": f"You are a QA bot who answers the question of the user only with the information provided in the context. Context: {context}, Question: {question}"},
    ]
    
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )

    answer = outputs[0]['generated_text'][-1]["content"]
    print(answer)

    return answer, num_retrievals + 1 

retrieved_indices = set()

# 질문-답변 세션
results = []

for qa in data:
    if qa['dataset'] == 'squad':
        if len(qa['answers']['text']) > 0:
            gt = qa['answers']['text'][0]
        else:
            continue
    else:
        gt = qa['answer']

    num_retrievals = 0
    initialize_retrieval()
    question = qa['question']

    k = 0
    retrieved_indices.clear()
    contexts = ''

    while k < 8:
        retrieved_contexts = retrieve_contexts(question, 5, retrieved_indices)
        context = " ".join(retrieved_contexts) 
        contexts += context
        answer, num_retrievals = answer_question(tokenizer, question, contexts, num_retrievals)
        if gt.lower() in answer.lower():
            break
        else:
            k += 1
    if num_retrievals > 7:
        num_retrievals = 'unanswerable'
    results.append({"question": question, "answer": answer, "ground truth": gt, "retrievals": num_retrievals, "dataset": qa['dataset']})
    print(results)
    
    with open("qa_results_4.json", "w") as f:
        json.dump(results, f)