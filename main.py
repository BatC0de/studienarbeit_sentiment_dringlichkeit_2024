import pandas as pd
import sys
import numpy as np
import pymysql
from sqlalchemy import create_engine, text  
from transformers import pipeline,BertTokenizer, BertModel, AutoTokenizer,AutoModelForSequenceClassification
from joblib import load
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from decimal import Decimal
import re


#Funktionen zur Vorverarbeitung
def remove_crap(text):
    text = re.sub(r'\bxx+\b', '', text)
    text = re.sub(r'\b(?!48\b|72\b|24\b)\d+\b', '', text)
    return text

#Datenbankverbindung
db_user = 'root'
db_pass = ''
db_name = 'complaintsDB'
db_host = 'localhost'  
db_port = 3306 

#SQLAlchemy
engine = create_engine(f'mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}')

#Auswahl von Datensätzen
#query = "SELECT ID, data FROM complaintsdata WHERE transmission_counter = 0 ORDER BY RAND() LIMIT 20"

query = "SELECT ID, data FROM complaintsdata WHERE label = 3 ORDER BY RAND() LIMIT 5"
df = pd.read_sql_query(query, engine)

#IDs der ausgewählten Datensätze für das Update sammeln
selected_ids = df['ID'].tolist()
selected_records = df['data'].tolist()

# Verbindung vom Engine erhalten und Transaktion starten sowie +1 transmission_counter --> nötig, damit datensätze nicht mehrfach eingelesen werden
# with engine.begin() as connection:  # Verwenden von engine.begin() für automatisches Commit/Rollback
#     # Erhöhen des transmission_counter um 1 für die ausgewählten Datensätze
#     update_query = text(f"UPDATE complaintsdata SET transmission_counter = transmission_counter + 1 WHERE ID IN ({','.join([str(id) for id in selected_ids])})")
#     connection.execute(update_query)

data_dict = df.to_dict('records')  #'records' gibt eine Liste von Dicts zurück, wobei jeder Dict einen Datensatz repräsentiert
data_list = df['data'].tolist()  #zu liste konvertieren
#print(data_list)

#tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

#zusammenfassungs-pipeline
summarizer = pipeline(
    "summarization", 
    num_beams=5, 
    do_sample=True, 
    no_repeat_ngram_size=3,
    max_length=1024,
    device=0,
    batch_size=8
)#, model="facebook/bart-large-cnn")
summary = summarizer(data_list, max_length=70, min_length=1, do_sample=False, length_penalty=1.0)
zusammenfassungs_dict = dict(zip(selected_ids, summary)) #--> umwandlung in ein dict

zusammenfassungs_dict = dict(zip(selected_ids, selected_records)) #--> umwandlung in ein dict

#print(zusammenfassungs_dict) #-->dictionary

#sentimentanalyse-pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model='ProsusAI/finbert') #tokenizing nicht nötig durch pipeline
max_length = 510
# Durchführen der Sentiment-Analyse und Speichern der Ergebnisse in einem neuen Dictionary
output_dict = {}
for item in data_dict:
    tokens = tokenizer.tokenize(item['data'])
    if len(tokens) > max_length:
        # Behalte nur die ersten `max_length` Tokens
        # Da von hinten gekürzt werden soll, nehmen wir die letzten 512 Tokens statt der ersten
        tokens = tokens[-max_length:]
    # Konvertiere die Tokens zurück in einen String für die Analyse
    truncated_text = tokenizer.convert_tokens_to_string(tokens)
    
    # Führe die Sentiment-Analyse mit dem gekürzten Text durch
    sentiment_result = sentiment_analyzer(truncated_text)[0]
    
    # Speichere das Ergebnis im output_dict
    output_dict[item['ID']] = {
        'score': sentiment_result['score'],
        'rating': sentiment_result['label']
    }

#print(output_dict) #-->dictionary
# print("Das item: ")
# print(data_dict)

#SVM-dringlichkeitsanalyse
# Laden der Pipeline
pipeline = load('E:/stud/SVM/Modellgenerationen/Generation_10/svm_pipeline_model.joblib')
#anwenden der vektoren auf die liste
predictions = pipeline.predict(data_list)

vorhersage_dict = dict(zip(selected_ids, predictions))

print(vorhersage_dict) #--> dictionary


#TRANSFORMER-dringlichkeitsanalyse
model_path = 'E:/stud/results/gen10/checkpoint-645'

# Modell und Tokenizer laden
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(data_list, truncation=True, padding=True, return_tensors="pt")

# Vorhersagen
with torch.no_grad():
    outputs = model(**encodings)

# Die Logits aus den Ausgaben extrahieren
logits = outputs.logits

# Wahrscheinlichkeiten berechnen
probabilities = torch.softmax(logits, dim=1)

# Wahrscheinlichkeiten oder Klassenlogits ausgeben
#print(probabilities)
probabilities_list = probabilities.tolist()

#umwandlung in dezimal (damit ich es besser lesen konnte))
decimal_probabilities = [[Decimal(str(val)) for val in row] for row in probabilities_list]

# print(decimal_probabilities) #---> liste
# print(probabilities_list) 
# print(type(probabilities_list)) #---> liste


#umwandlung von dem ganzen spaß in ein dictionary
results_dict = {}

#durchlaufen der Output-Liste 
for id, values in zip(selected_ids, decimal_probabilities):
    results_dict[id] = {
        'urgencyindex_false': float(values[0]),  # Konvertierung zu float
        'urgencyindex_true': float(values[1])    # Konvertierung zu float
    }

print(results_dict) #--> dictionary

#import in die neue datenbanktabelle:
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'db': 'complaintsDB',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

#Verbindung zur Datenbank herstellen
connection = pymysql.connect(**db_config)

try:
    with connection.cursor() as cursor:
        
        sql = """
        INSERT INTO results (
            complaintsdata_id, summary, sentimentscore,sentimentscore_rating, svm_urgency, urgencyindex_false, urgencyindex_true
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        #neues dictionary was die anderen zusammenfasst
        for complaintsdata_id, data in zusammenfassungs_dict.items():
            sentimentscore = output_dict.get(complaintsdata_id, {}).get('score', None)
            sentimentscore_rating = output_dict.get(complaintsdata_id, {}).get('rating', None)
            svm_urgency = vorhersage_dict.get(complaintsdata_id, None)
            urgencyindex_false = results_dict.get(complaintsdata_id, {}).get('urgencyindex_false', None)
            urgencyindex_true = results_dict.get(complaintsdata_id, {}).get('urgencyindex_true', None)
            
            print("complaintsdata_id:", complaintsdata_id)
            print("sentimentscore:", sentimentscore)
            print("sentimentscore_rating:", sentimentscore_rating)
            print("svm_urgency:", svm_urgency)
            print("urgencyindex_false:", urgencyindex_false)
            print("urgencyindex_true:", urgencyindex_true)

            #SQL-Statement ausführen mit allen Werten
            cursor.execute(sql, (
                complaintsdata_id, 
                data['summary_text'], 
                sentimentscore,
                sentimentscore_rating, 
                svm_urgency, 
                urgencyindex_false, 
                urgencyindex_true
            ))
    
    
    connection.commit()
finally:
   
    connection.close()