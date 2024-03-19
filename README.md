# studienarbeit_sentiment_dringlichkeit_2024
Im folgenden eine Auflistung der enthaltenen Dateien und Inhalte

main:
Enthält den nötigen Programmcode um die Daten aus der Datenbank zu laden und sie in die Pipeline zu senden. Nach jeder Ausführung sind die neuen Einträge im Webservice sichtbar

CREATE TABLE:
Definitionen der zur Speicherung der Daten notwendigen Tabellen

transformer_train:
Enthält den Code, um ein Transformermodell zu trainieren

svm_train:
Enthält den Code, um die SVM zu trainieren

webservice:
Einfacher Webservice, der den Score berechnet und die Darstellung als Tabelle aufbaut sowie eine Priorität zuordnet
