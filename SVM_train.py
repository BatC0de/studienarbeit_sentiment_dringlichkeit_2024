import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from joblib import dump, load
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re


df = pd.read_csv("E:/stud/Dataset_4/gpt_generated_1200.csv", delimiter=';')

#mit pandas daten extrahieren
X = df['data']
y = df['label']

#Aufteilen der Daten in Trainings- und Testdatensätze (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Weiteres Aufteilen der Trainingsdaten in eigentliche Trainings- und Validierungsdatensätze (80% Training, 20% Validierung vom ursprünglichen Trainingssatz)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

#einfache regex um daten zu filten
def remove_crap(text):
    text = re.sub(r'\bxx+\b', '', text)
    text = re.sub(r'\b(?!48\b|72\b|24\b)\d+\b', '', text)
    return text


#Pipeline mit TF-IDF Vektorisierung und SVM Klassifikator
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=remove_crap, max_features=800,stop_words='english',lowercase=True,ngram_range=(1,5),sublinear_tf=True, max_df=0.8)),
    ('svm', SVC(kernel='linear', class_weight='balanced', C=1.5,probability=True))
])

#Kreuzvalidierung mit der Pipeline
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("Kreuzvalidierung Genauigkeiten für jede Falte: ", scores)
print("Durchschnittliche Genauigkeit aus der Kreuzvalidierung: ", scores.mean())

#Training
pipeline.fit(X_train, y_train)

#Ausgabe der erfassten Features
feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
print("Feature-Namen (Wörter/Token) vom TfidfVectorizer verarbeitet:", feature_names)


#Vorhersagen auf den Validierungsdaten
y_val_pred = pipeline.predict(X_val)
print("Validierungsdaten Bewertung:")
print(classification_report(y_val, y_val_pred))

#Vorhersagen auf den Testdaten
y_test_pred = pipeline.predict(X_test)
print("Testdaten Bewertung:")
print(classification_report(y_test, y_test_pred))

# Angenommen, new_data ist eine Liste von neuen Textdaten, die Sie klassifizieren möchten
new_data = ["es ist schwierig etwas zu schreiben, das nicht in einer 1 endet", "my account information has been stolen, i need immediate support on this case"]

# Vektorisierung der neuen Daten mit der Pipeline und Vorhersage
new_data_predictions = pipeline.predict(new_data)
print(new_data_predictions)

#Speichern des Pipeline-Modells
dump(pipeline, 'svm_pipeline_model_test.joblib')

# # Speichern des Modells
# dump(model, 'svm_model_7_gpt.joblib')

# # Optional: Speichern des TfidfVectorizer
# dump(vectorizer, 'tfidf_vectorizer_7_gpt.joblib')



#visualizer
# Verteilung der Labels im gesamten Datensatz anzeigen
# sns.countplot(x=y)
# plt.title('Verteilung der Klassen')
# plt.xlabel('Klasse')
# plt.ylabel('Anzahl')
# plt.show()


# Kreuzvalidierung Genauigkeiten visualisieren
# plt.bar(np.arange(len(scores)), scores)
# plt.xlabel('Fold')
# plt.ylabel('Genauigkeit')
# plt.title('Kreuzvalidierung Genauigkeiten')
# plt.show()

# print(f"Durchschnittliche Genauigkeit: {np.mean(scores)}")


#Wichtigste Features anzeigen lassen
# tfidf = pipeline.named_steps['tfidf']
# importances = tfidf.idf_
# indices = np.argsort(importances)[::-1]

# top_n = 20
# top_features = [feature_names[i] for i in indices[:top_n]]
# top_scores = [importances[i] for i in indices[:top_n]]

# plt.figure(figsize=(10, 8))
# sns.barplot(x=top_scores, y=top_features)
# plt.title('Top 20 Wichtigste Features im TF-IDF Vektorisierer')
# plt.xlabel('IDF Score')
# plt.show()