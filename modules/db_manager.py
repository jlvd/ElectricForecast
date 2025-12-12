import duckdb
import pandas as pd
import os

DB_PATH = "energia.duckdb"

def init_db():
    """Initializes the DuckDB database and creates the main table if it doesn't exist."""
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS demanda_historica (
                fecha DATE PRIMARY KEY,
                valor DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    finally:
        conn.close()

def save_data(df, overwrite=False):
    """
    Saves a dataframe to the database.
    Expected df columns: 'fecha', 'valor'.
    """
    conn = duckdb.connect(DB_PATH)
    try:
        if overwrite:
            # Upsert logic or delete overlap could be here, but for simplicity in this MVP:
            # We will clear data for the incoming range if overwrite is requested?
            # Or simplified: Delete all and replace if 'overwrite' meant full reload?
            # Proposal said: "Si ya existe, pregunta sí desea sobrescribir".
            # Let's assume overwrite means replacing overlapping dates.
            
            # Identify date range
            min_date = df['fecha'].min()
            max_date = df['fecha'].max()
            
            # Delete overlap
            conn.execute("DELETE FROM demanda_historica WHERE fecha BETWEEN ? AND ?", [min_date, max_date])
            
        # Append new data (using INSERT OR IGNORE to skip duplicates if overwrite=False)
        # Note: 'fecha' is PK, so duplicates will fail or be ignored.
        # We use appender or simple insert.
        
        # We need to ensure df matches schema
        # DuckDB's append usually matches by position or name.
        # Let's simple insert via DF
        
        # If not overwrite, we want to skip existing keys.
        # INSERT OR IGNORE INTO...
        
        # Register df as view for SQL ops
        conn.register('df_view', df)
        
        if overwrite:
             conn.execute("INSERT INTO demanda_historica (fecha, valor) SELECT fecha, valor FROM df_view")
        else:
             conn.execute("INSERT OR IGNORE INTO demanda_historica (fecha, valor) SELECT fecha, valor FROM df_view")

    except Exception as e:
        raise e
    finally:
        conn.close()

def load_data():
    """Loads all data sorted by date."""
    conn = duckdb.connect(DB_PATH)
    try:
        df = conn.execute("SELECT fecha, valor FROM demanda_historica ORDER BY fecha ASC").fetchdf()
        return df
    finally:
        conn.close()

def clear_data():
    """Deletes all data from the main table."""
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM demanda_historica")
    finally:
        conn.close()

def get_db_stats():
    """Returns basic stats about the DB."""
    conn = duckdb.connect(DB_PATH)
    try:
        count = conn.execute("SELECT COUNT(*) FROM demanda_historica").fetchone()[0]
        if count > 0:
            last_date = conn.execute("SELECT MAX(fecha) FROM demanda_historica").fetchone()[0]
        else:
            last_date = None
        return {"count": count, "last_date": last_date}
    finally:
        conn.close()

