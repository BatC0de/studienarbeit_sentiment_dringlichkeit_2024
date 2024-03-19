import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset

# daten laden
csv_file_path = 'E:/stud/Dataset_4/gpt_generated_1200.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Teilen des Datensatzes in Trainings-, Validierungs- und Testsets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['data'], df['label'], test_size=0.4)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenisierung
train_encodings = tokenizer(train_texts.to_list(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.to_list(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.to_list(), truncation=True, padding=True)


class UrgentRequestsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Datensätze erstellen
train_dataset = UrgentRequestsDataset(train_encodings, train_labels.to_list())
val_dataset = UrgentRequestsDataset(val_encodings, val_labels.to_list())
test_dataset = UrgentRequestsDataset(test_encodings, test_labels.to_list())

# Modell und TrainingArguments bleiben gleich
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# TrainingArguments definieren
training_args = TrainingArguments(
    output_dir='./results/gen12',          
    num_train_epochs=12,
    fp16=True,              
    save_strategy="epoch",
    load_best_model_at_end=True, 
    per_device_train_batch_size=6,  
    per_device_eval_batch_size=6,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./results/gen12/logs',            
    evaluation_strategy="epoch",
    eval_steps=100,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer initialisieren mit compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics  # Hier fügen wir die Metrikberechnung hinzu
)

trainer.train()

# Testset evaluieren, um die Metriken auf dem Testset zu erhalten
results = trainer.evaluate(test_dataset)
print(results)
