from flask import Flask, render_template, request
import mysql.connector

app = Flask(__name__)

#Datenbank
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="complaintsDB"
)

# Cursor erstellen
mycursor = mydb.cursor()

@app.route('/')
def index():
    #Score berechnen, einträge ziehen
    mycursor.execute("""
    SELECT id, svm_urgency + urgencyindex_true + CASE WHEN sentimentscore_rating = 'negative' THEN sentimentscore ELSE 0 END AS Score
    FROM results
    ORDER BY id DESC
    LIMIT 20
    """)
    score_updates = mycursor.fetchall()

    #Score
    for update in score_updates:
        update_query = """
        UPDATE results
        SET score = %s
        WHERE id = %s
        """
        mycursor.execute(update_query, (update[1], update[0]))
    
    mydb.commit()

    #aktualisierten Daten für die Anzeige
    mycursor.execute("""
    SELECT id, complaintsdata_id, summary, sentimentscore, urgencyindex_false, urgencyindex_true, svm_urgency, sentimentscore_rating, score AS Score
    FROM (
        SELECT * 
        FROM results 
        ORDER BY id DESC 
        LIMIT 20
    ) AS latest_results
    ORDER BY Score DESC
    """)
    results = mycursor.fetchall()

    
    column_names = [i[0] for i in mycursor.description]

    return render_template('index.html', results=results, column_names=column_names)

@app.route('/add-complaint', methods=['POST'])
def add_complaint():
    complaint_text = request.form['complaintText']
    insert_query = """
    INSERT INTO complaintsdata (data, label)
    VALUES (%s, 3)
    """
    mycursor.execute(insert_query, (complaint_text,))
    mydb.commit()
    
    return "Beschwerde erfolgreich eingereicht!"

if __name__ == '__main__':
    app.run(debug=True)
