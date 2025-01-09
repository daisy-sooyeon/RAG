import json
from collections import Counter

file_path_1 = 'qa_results_{n}.json'
file_path_2 = 'qa_results_{m}.json'

with open(file_path_1, 'r', encoding='utf-8') as f:
    data_1 = json.load(f)

with open(file_path_2, 'r', encoding='utf-8') as f:
    data_2 = json.load(f)

compare = []

for i, data1 in enumerate(data_1):
    if i < len(data_2) and data1['retrievals'] == 'unanswerable':
        for j, data2 in enumerate(data_2):
            if (data2['retrievals'] != 'unanswerable') and (data2['question'] == data1['question']):
                compare.append([data2['question'], data1['answer'], data2['answer'], data1['ground truth'], data2['retrievals']])
                print([data2['question'], data1['answer'], data2['answer'], data2['ground truth'], data2['retrievals'], data1['dataset']])
            #if (data2['question'] == data1['question']):
                #compare.append(data2['retrievals'])

#unique_counts = Counter(compare)
#print(unique_counts)

## 주석처리된 부분은 data_1에서 unanswerable한 질문들이 data_2에서 어떤 label로 되어 있는지를 count
## 주석처리되지 않은 부분은 data_1에서 unanswerable한 질문들이 data_2에서 답변된 경우 어떻게 답변이 되어있는지 확인