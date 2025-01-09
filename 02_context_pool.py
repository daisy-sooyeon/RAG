import json

file_path = 'combined_datasets.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_contexts = []
for qa in data:
    if qa['dataset'] == 'squad':
        all_contexts.append(qa['context'])
    else:
        for element in qa['context']['sentences']:
            for sentence in element:
                all_contexts.append(sentence)

with open('full_context.json', 'w') as f:
    json.dump(all_contexts, f)

print("JSON file has been saved as 'full_context.json'")