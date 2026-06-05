import psycopg2
import pandas as pd
from config import DB_CONFIG

names = [
    "Aaron Sinclair Bartley",
    "Abdullahi Saleh",
    "Ponki Miah",
    "Angeline Sook Yi Lewis",
    "Emran Uddin",
]

conn = psycopg2.connect(**DB_CONFIG)
results = []
for name in names:
    # Split into parts and search flexibly (handles middle names / order variations)
    parts = name.split()
    like_clauses = " AND ".join(
        f"UPPER(driver_full_name) LIKE '%{p.upper()}%'" for p in parts
    )
    df = pd.read_sql(f"""
        SELECT DISTINCT dim_driver_id, driver_full_name
        FROM rep_fact_driver_performance
        WHERE {like_clauses}
        LIMIT 5
    """, conn)
    if df.empty:
        # Fallback: first + last name only
        like_clauses = (
            f"UPPER(driver_full_name) LIKE '%{parts[0].upper()}%' "
            f"AND UPPER(driver_full_name) LIKE '%{parts[-1].upper()}%'"
        )
        df = pd.read_sql(f"""
            SELECT DISTINCT dim_driver_id, driver_full_name
            FROM rep_fact_driver_performance
            WHERE {like_clauses}
            LIMIT 5
        """, conn)
    df["searched_for"] = name
    results.append(df)

conn.close()
out = pd.concat(results, ignore_index=True)
print(out.to_string(index=False))
