import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def reset_db():
    dns = os.environ.get("dns")
    if not dns:
        print("Error: Missing 'dns' in environment.")
        return

    print(f"Connecting to database: {dns}")
    try:
        conn = psycopg2.connect(dns)
        cur = conn.cursor()
        
        print("Dropping table 'xpu_entries'...")
        cur.execute("DROP TABLE IF EXISTS xpu_entries;")
        
        conn.commit()
        cur.close()
        conn.close()
        print("✓ Table dropped successfully. Next run will recreate it with correct dimension.")
        
    except Exception as e:
        print(f"Error resetting DB: {e}")

if __name__ == "__main__":
    reset_db()
