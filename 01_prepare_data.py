from datasets import load_dataset
import json
import random
import numpy as np
import aiohttp

timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour total timeout

squad_dataset = load_dataset("squad_v2")
# print(squad_dataset['train'][0])
# {'id': '56be85543aeaaa14008c9063', 'title': 'Beyoncé', 'context': 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".', 'question': 'When did Beyonce start becoming popular?', 'answers': {'text': ['in the late 1990s'], 'answer_start': [269]}}
hotpotqa_dataset_distractor = load_dataset("hotpotqa/hotpot_qa", "distractor")
# print(hotpotqa_dataset_distractor['train'][0])
# {'id': , 'question': , 'answer': "Arthur's Magazine", 'type': 'comparison', 'level': 'medium', 'supporting_facts': {'title': ["Arthur's Magazine", 'First for Women'], 'sent_id': [0, 0]}, 'context':
hotpotqa_dataset_fullwiki = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
# print(hotpotqa_dataset_fullwiki['train'][0])

squad = []
hotpotqa_distractor = []
hotpotqa_fullwiki = []

for item in squad_dataset['train']:
    item['dataset'] = 'squad'
    squad.append(item)

for item in hotpotqa_dataset_distractor['train']:
    item['dataset'] = 'hotpotqa_distractor'
    hotpotqa_distractor.append(item)

for item in hotpotqa_dataset_fullwiki['train']:
    item['dataset'] = 'hotpotqa_fullwiki'
    hotpotqa_fullwiki.append(item)


np.random.seed(42)
np.random.shuffle(squad)
np.random.shuffle(hotpotqa_distractor)
np.random.shuffle(hotpotqa_fullwiki)

# 데이터셋을 리스트로 변환 (모든 데이터를 포함)
combined_data = squad[:200] + hotpotqa_distractor[:200] + hotpotqa_fullwiki[:200]

# 리스트를 JSON 문자열로 변환
combined_json = json.dumps(combined_data)

with open('combined_datasets.json', 'w') as f:
    f.write(combined_json)

print("JSON file has been saved as 'combined_datasets.json'")