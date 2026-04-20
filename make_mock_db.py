import pandas as pd
import sqlite3

print("Loading CSV...")
df = pd.read_csv('data.csv')

print("Creating local SQLite database...")
# This creates a local file named 'mock_database.db' in your folder
conn = sqlite3.connect('mock_database.db')

# Save the data into a table with the exact name your AI Agent expects
df.to_sql('ai_campaign_performance_mart', conn, if_exists='replace', index=False)
conn.close()

print("Success! mock_database.db has been created.")