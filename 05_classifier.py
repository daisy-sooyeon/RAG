from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import json
from functools import partial
from transformers import DataCollatorWithPadding


# {"text": [질문 list], "labels": [classifier label]} 형식으로 formatting
def format_data(data):
    formatted_data = {"text": [], "labels": []}
    for entry in data:
        formatted_data["text"].append(entry['question'])
        
        if isinstance(entry['retrievals'], int):
            formatted_data["labels"].append(0)
        else:
            formatted_data["labels"].append(1)
            
    return formatted_data


def preprocess_function(tokenizer, examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
    tokenized_inputs['labels'] = [int(label) if isinstance(label, str) and label.isdigit() else label for label in examples['labels']]
    return tokenized_inputs

def evaluate_model(classifier, dataset):
    output = []
    predictions = []
    for question in dataset["text"]:
        result = classifier(question)
        output.append(result)
        pred_label = int(result[0]['label'].split('_')[-1]) 
        predictions.append(pred_label)

    cm = confusion_matrix([int(label) for label in dataset['labels']], predictions)
    accuracy = accuracy_score([int(label) for label in dataset['labels']], predictions)
    precision = precision_score([int(label) for label in dataset['labels']], predictions, average='macro')
    recall = recall_score([int(label) for label in dataset['labels']], predictions, average='macro')

    return cm, accuracy, precision, recall

def main():
	file_path = 'qa_results_{n}.json'

	with open(file_path, 'r', encoding='utf-8') as f:
		data_loaded = json.load(f)

	# train, test split (랜덤하게 8:2)
	data_loaded = format_data(data_loaded)
	full_dataset = Dataset.from_dict(data_loaded)
	train, test = train_test_split(full_dataset, test_size=0.2, random_state=42)  # random_state ensures reproducibility
	train = Dataset.from_dict(train)
	test = Dataset.from_dict(test)
	
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
	model = AutoModelForSequenceClassification.from_pretrained(
		'bert-base-uncased', 
		num_labels=2
	)

	preprocess_with_tokenizer = partial(preprocess_function, tokenizer)

	tokenized_datasets = train.map(preprocess_with_tokenizer, batched=True)
	tokenized_test = test.map(preprocess_with_tokenizer, batched=True)

	training_args = TrainingArguments(
		output_dir="./results",
		evaluation_strategy="epoch",
		learning_rate=2e-5,
		per_device_train_batch_size=8,
		weight_decay=0.01,
		save_total_limit=1,
		num_train_epochs=3,
		report_to="none"
	)
	
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_datasets,
		eval_dataset=tokenized_test, 
		tokenizer=tokenizer,
		data_collator=data_collator
	)
	
	trainer.train()
	
	trainer.evaluate()
	model.save_pretrained("./trained_model")
	tokenizer.save_pretrained("./trained_model")
	
	# ------------ 훈련된 모델 가져오기 -------------
	
	from transformers import pipeline
	from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
	
	classifier = pipeline("text-classification", model="./trained_model", tokenizer="./trained_model")
	
	cm, accuracy, precision, recall = evaluate_model(classifier, test)
	
	print("Confusion Matrix:\n", cm)
	print("Accuracy:", accuracy)
	print("Precision:", precision)
	print("Recall:", recall)

if __name__ == "__main__":
    main()