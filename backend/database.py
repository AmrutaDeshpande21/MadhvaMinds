import asyncpg
import json
from datetime import datetime

DATABASE_URL = "postgres://madhva:password@127.0.0.1:5432/incident_db"

async def init_db():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS cameras (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                rtsp_url TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id BIGSERIAL PRIMARY KEY,
                camera_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity SMALLINT NOT NULL,
                confidence REAL NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                metadata JSONB
            );
        ''')
        await conn.close()
        print("PostgreSQL Database schema initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize DB (ensure Docker is running): {e}")

async def save_incident(incident_data):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute('''
            INSERT INTO incidents (camera_id, event_type, severity, confidence, started_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
        ''', 
        incident_data['camera_id'],
        incident_data['type'],
        incident_data['severity'],
        incident_data['confidence'],
        datetime.fromtimestamp(incident_data['timestamp']),
        json.dumps(incident_data)
        )
        await conn.close()
        print(f"Incident saved to DB: {incident_data['type']}")
    except Exception as e:
        print(f"DB Save Error: {e}")

async def get_incidents(limit=50):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        records = await conn.fetch('''
            SELECT id, camera_id, event_type, severity, confidence, started_at
            FROM incidents
            ORDER BY started_at DESC
            LIMIT $1
        ''', limit)
        await conn.close()
        
        # Convert records to list of dicts
        return [dict(r) for r in records]
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        return []
