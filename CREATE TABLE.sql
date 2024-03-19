CREATE TABLE complaintsdata_all (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    data TEXT,
    transmission_counter INT default 0
);

CREATE TABLE results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    complaintsdata_id INT,
    summary TEXT,
    sentimentscore DECIMAL(20,18),
    urgencyindex_false DECIMAL(20,18),
    urgencyindex_true DECIMAL(20,18),
    svm_urgency INT,
    sentimentscore_rating TEXT,
    score DECIMAL(20,18),
    FOREIGN KEY (complaintsdata_id) REFERENCES complaintsdata(ID)
);