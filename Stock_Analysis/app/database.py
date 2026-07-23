import sqlite3
import os
from datetime import datetime
from app.config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            open_price REAL NOT NULL,
            high_price REAL NOT NULL,
            low_price REAL NOT NULL,
            adj_close REAL NOT NULL,
            volume REAL NOT NULL,
            predicted_close REAL NOT NULL,
            trend TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_prediction(company_name, open_price, high_price, low_price, adj_close, volume, predicted_close, trend, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prediction_history 
        (company_name, open_price, high_price, low_price, adj_close, volume, predicted_close, trend, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        company_name, 
        float(open_price), 
        float(high_price), 
        float(low_price), 
        float(adj_close), 
        float(volume), 
        float(predicted_close), 
        trend, 
        float(confidence),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()

def get_history(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM prediction_history ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    # Convert Row objects to list of dicts for easy JSON serialization
    history = []
    for r in rows:
        history.append({
            'id': r['id'],
            'company_name': r['company_name'],
            'open_price': r['open_price'],
            'high_price': r['high_price'],
            'low_price': r['low_price'],
            'adj_close': r['adj_close'],
            'volume': r['volume'],
            'predicted_close': r['predicted_close'],
            'trend': r['trend'],
            'confidence': r['confidence'],
            'timestamp': r['timestamp']
        })
    return history

def delete_history_item(item_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prediction_history WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prediction_history')
    conn.commit()
    conn.close()
