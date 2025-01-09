import json
from collections import Counter

file_path = 'qa_results_{n}.json' # 데이터셋 변경

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

retrieval_counts = Counter()

for dt in data:
    retrieval_counts[dt['retrievals']] += 1

print(retrieval_counts)