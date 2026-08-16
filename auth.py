import sqlite3
import bcrypt
import os

DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    if not username or not password:
        return False, "Username and password are required."
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if user exists
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if c.fetchone():
        conn.close()
        return False, "Username already exists."
    
    # Hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed.decode('utf-8')))
        conn.commit()
        return True, "Account created successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def verify_user(username, password):
    if not username or not password:
        return False, "Username and password are required."
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return False, "Invalid username or password."
        
    stored_hash = row[0].encode('utf-8')
    if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
        return True, "Login successful."
    return False, "Invalid username or password."

# Initialize DB on import
init_db()
