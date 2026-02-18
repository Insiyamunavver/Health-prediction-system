import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# 🔐 Neon Connection
conn = psycopg2.connect(
    host="ep-falling-pond-aiqqf317-pooler.c-4.us-east-1.aws.neon.tech",
    database="neondb",
    user="neondb_owner",
    password="npg_Ak8glI7WEemP",
    port="5432"
)

cur = conn.cursor()

# 🧱 Create table
cur.execute("""
DROP TABLE IF EXISTS healthcare;

CREATE TABLE healthcare (
    Patient_ID INT,
    Age INT,
    Gender TEXT,
    Symptoms TEXT,
    Symptom_Count INT,
    Disease TEXT
);
""")

conn.commit()

# 📂 Load CSV
df = pd.read_csv("dataset/Healthcare.csv")

# 🚀 FAST Batch Insert
data_tuples = [tuple(row) for row in df.to_numpy()]

execute_values(
    cur,
    """
    INSERT INTO healthcare
    (Patient_ID, Age, Gender, Symptoms, Symptom_Count, Disease)
    VALUES %s
    """,
    data_tuples
)

conn.commit()

# 🔚 Close connection
cur.close()
conn.close()

print("✅ Data uploaded to Neon Postgres successfully!")
