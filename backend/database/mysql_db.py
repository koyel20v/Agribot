# database/mysql_db.py — MySQL connection pool + table initialisation

import mysql.connector
from mysql.connector import pooling
import config

# ── Connection pool (reuse connections, don't open a new one per request) ──
_pool = pooling.MySQLConnectionPool(
    pool_name="agrigenius_pool",
    pool_size=5,
    host=config.DB_HOST,
    port=config.DB_PORT,
    user=config.DB_USER,
    password=config.DB_PASSWORD,
    database=config.DB_NAME,
)


def get_connection():
    """Return a connection from the pool. Always call .close() when done."""
    return _pool.get_connection()


def init_db():
    """
    Create required tables if they don't already exist.
    Called once at app startup.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # ── users table ──────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    INT          AUTO_INCREMENT PRIMARY KEY,
            name       VARCHAR(100) NOT NULL,
            email      VARCHAR(150) UNIQUE NOT NULL,
            password   VARCHAR(255) NOT NULL,
            role       ENUM('farmer', 'officer') NOT NULL,
            created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── queries table ─────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            query_id    INT       AUTO_INCREMENT PRIMARY KEY,
            user_id     INT       NOT NULL,
            question    TEXT      NOT NULL,
            ai_response TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()