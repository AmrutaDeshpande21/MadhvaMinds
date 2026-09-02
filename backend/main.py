import asyncio
import base64
import json
import multiprocessing as mp
import threading
import os
import httpx
from dotenv import load_dotenv
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from vision_pipeline import VisionPipeline
from camera_worker import camera_worker_process
import database

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = FastAPI(title="Incident Intelligence Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
frame_queue = None
camera_process = None
pipeline = None
latest_frame = None

def run_vision_pipeline(in_queue):
    global latest_frame
    try:
        pipeline = VisionPipeline(in_queue=in_queue)
        frame_generator = pipeline.process_frames()
        for frame_bytes in frame_generator:
            latest_frame = frame_bytes
    except Exception as e:
        print(f"Vision Pipeline Error: {e}")

async def send_telegram_alert(alert_data):
    """Sends a Telegram message if credentials exist and severity is high."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
        
    if alert_data.get("severity", 0) < 5:
        return # Only ping phone for severity 5 events

    msg_text = (
        f"🚨 <b>MADHVAMINDS ALERT</b> 🚨\n\n"
        f"<b>Type:</b> {alert_data['type']}\n"
        f"<b>Camera:</b> {alert_data['camera_id']}\n"
        f"<b>Confidence:</b> {alert_data['confidence'] * 100:.1f}%\n\n"
        f"<i>Action Required Immediately.</i>"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg_text,
        "parse_mode": "HTML"
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=5.0)
            if resp.status_code != 200:
                print(f"Failed to send Telegram alert: {resp.text}")
            else:
                print("Telegram notification dispatched successfully!")
    except Exception as e:
        print(f"Telegram dispatch error: {e}")

@app.on_event("startup")
async def startup_event():
    global frame_queue, camera_process
    
    await database.init_db()
    
    frame_queue = mp.Queue(maxsize=30)
    camera_source = 0 # Change this to video path if needed
    camera_process = mp.Process(target=camera_worker_process, args=(camera_source, frame_queue))
    camera_process.start()
    
    threading.Thread(target=run_vision_pipeline, args=(frame_queue,), daemon=True).start()
    print("Backend started with Telegram support enabled.")

@app.on_event("shutdown")
async def shutdown_event():
    global camera_process
    if camera_process and camera_process.is_alive():
        camera_process.terminate()

@app.get("/")
async def root():
    return {"message": "Phase 4 Backend is running"}

@app.get("/api/incidents")
async def get_historical_incidents(limit: int = 50):
    incidents = await database.get_incidents(limit)
    return {"incidents": incidents}

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    redis_client = aioredis.from_url("redis://localhost")
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("alerts:raw")
    
    try:
        while True:
            if latest_frame:
                b64_image = base64.b64encode(latest_frame).decode('utf-8')
                payload = {
                    "type": "feed",
                    "image": f"data:image/jpeg;base64,{b64_image}",
                    "alerts": [] 
                }
                await websocket.send_json(payload)
            
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.01)
            if message:
                alert_data = json.loads(message['data'])
                
                # 1. Async Postgres Save
                asyncio.create_task(database.save_incident(alert_data))
                
                # 2. Async Telegram Notification (if configured)
                asyncio.create_task(send_telegram_alert(alert_data))
                
                # 3. WebSocket push
                await websocket.send_json({
                    "type": "feed",
                    "image": f"data:image/jpeg;base64,{b64_image}" if latest_frame else None,
                    "alerts": [alert_data]
                })
                
            await asyncio.sleep(0.03) 
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        await pubsub.unsubscribe("alerts:raw")
        await redis_client.close()
