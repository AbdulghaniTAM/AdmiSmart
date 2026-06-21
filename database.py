import psycopg2
from psycopg2.extras import DictCursor
from config import DB_URL

def get_connection():
    """
    Returns a PostgreSQL database connection using DB_URL from config.
    Can be used in a context manager: 'with get_connection() as conn:'
    """
    try:
        return psycopg2.connect(DB_URL)
    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to PostgreSQL database at {DB_URL}.")
        print("Please verify that your database server is running and the DB_URL is correct.")
        raise e


