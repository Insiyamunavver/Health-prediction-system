import psycopg2
import pandas as pd


def load_data():

    conn = psycopg2.connect(
        host="ep-falling-pond-aiqqf317-pooler.c-4.us-east-1.aws.neon.tech",
        database="neondb",
        user="neondb_owner",
        password="npg_Ak8glI7WEemP",
        port="5432"
    )

    query = "SELECT * FROM healthcare;"
    df = pd.read_sql(query, conn)

    conn.close()

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\n✅ Data loaded from Neon Postgres!")
