# migrate_add_validation.py — One-time migration script
#
# Run this ONCE to add the `is_validated` column to the queries table.
# Command: python migrate_add_validation.py

import mysql.connector
import config

conn = mysql.connector.connect(
    host=config.DB_HOST,
    port=config.DB_PORT,
    user=config.DB_USER,
    password=config.DB_PASSWORD,
    database=config.DB_NAME,
)
cursor = conn.cursor()

try:
    # First check if the column already exists
    cursor.execute("""
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = 'queries'
        AND COLUMN_NAME = 'is_validated'
    """, (config.DB_NAME,))

    (count,) = cursor.fetchone()

    if count == 0:
        # Column doesn't exist — add it
        cursor.execute("""
            ALTER TABLE queries
            ADD COLUMN is_validated BOOLEAN DEFAULT FALSE
        """)
        conn.commit()
        print("✅ Migration complete: 'is_validated' column added to queries table.")
    else:
        print("ℹ Column 'is_validated' already exists — nothing to do.")

except Exception as e:
    print(f"❌ Migration failed: {e}")

finally:
    cursor.close()
    conn.close()