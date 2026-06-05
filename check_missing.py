import psycopg2

conn = psycopg2.connect(
    host='dev-odysse.postgres.database.azure.com', port=5432,
    dbname='unifieddwh', user='odysys', password='Odysys@2026', sslmode='require'
)
cur = conn.cursor()

missing_ids = [128, 180, 230, 228, 182, 195]
names = {128:'Monier Janabi', 180:'Abdi Saeed', 230:'Anish Chaudhry',
         228:'MHD Amir Aljaghsi', 182:'Mohamed Warsame', 195:'Brijenkumar Patel'}

# 1. rep_fact_trips — any status
cur.execute("""
    SELECT dim_driver_id, status, COUNT(*)
    FROM rep_fact_trips WHERE dim_driver_id = ANY(%s)
    GROUP BY dim_driver_id, status ORDER BY dim_driver_id, COUNT(*) DESC
""", (missing_ids,))
rows = cur.fetchall()
print('rep_fact_trips (all statuses):')
if rows:
    for r in rows:
        print(f'  {names.get(r[0], r[0]):<22} status={r[1]}  count={r[2]}')
else:
    print('  (no rows at all for these 6 drivers)')

# 2. All trip-related tables
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema='public' AND table_name LIKE '%trip%'
    ORDER BY table_name
""")
print('\nAll trip-related tables:')
for r in cur.fetchall(): print(' ', r[0])

# 3. Bolt/Uber raw tables
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema='public'
      AND (table_name LIKE '%bolt%' OR table_name LIKE '%uber%' OR table_name LIKE '%autocab%')
    ORDER BY table_name
""")
print('\nPlatform-specific tables:')
for r in cur.fetchall(): print(' ', r[0])

# 4. Check silver_bolt_orderhistory for these drivers
cur.execute("""
    SELECT table_name, column_name FROM information_schema.columns
    WHERE table_schema='public' AND column_name LIKE '%driver%'
      AND table_name IN (
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public' AND (table_name LIKE '%bolt%' OR table_name LIKE '%silver%')
      )
    ORDER BY table_name, column_name
    LIMIT 30
""")
print('\nDriver-related columns in bolt/silver tables:')
for r in cur.fetchall(): print(f'  {r[0]}.{r[1]}')

cur.close()
conn.close()
