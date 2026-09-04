import asyncio
import base64
import json
import multiprocessing as mp
import threading
import os
import queue
import httpx
import glob
import cv2
import time
from dotenv import load_dotenv
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from vision_pipeline import VisionPipeline
from camera_worker import camera_worker_process
import database

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

from fastapi.responses import FileResponse

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
alert_queue = None
camera_process = None
pipeline = None
latest_frame = None


def run_vision_pipeline(in_queue, out_alert_queue):
    global latest_frame
    try:
        pipeline = VisionPipeline(in_queue=in_queue, alert_queue=out_alert_queue)
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
    global frame_queue, alert_queue, camera_process
    
    await database.init_db()
    
    frame_queue = mp.Queue(maxsize=30)
    alert_queue = queue.Queue(maxsize=100)
    camera_source = 0 # Change this to video path if needed
    camera_process = mp.Process(target=camera_worker_process, args=(camera_source, frame_queue))
    camera_process.start()
    
    threading.Thread(target=run_vision_pipeline, args=(frame_queue, alert_queue), daemon=True).start()
    print("Backend started with Telegram support enabled.")

@app.on_event("shutdown")
async def shutdown_event():
    global camera_process
    if camera_process and camera_process.is_alive():
        camera_process.terminate()

from simulation import global_dataset_simulation

def on_simulation_alert(alert_data):
    if alert_queue is not None:
        try:
            alert_queue.put(alert_data)
        except Exception:
            pass

@app.get("/")
async def root():
    return {"message": "Phase 4 Backend is running"}

@app.get("/api/incidents")
async def get_historical_incidents(limit: int = 50):
    incidents = await database.get_incidents(limit)
    return {"incidents": incidents}

# Real UCF-Crime Video Dataset Endpoints
DATASET_VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset", "ucf-crime-videos"))
os.makedirs(DATASET_VIDEO_DIR, exist_ok=True)
app.mount("/dataset/videos", StaticFiles(directory=DATASET_VIDEO_DIR), name="dataset_videos")

@app.get("/api/dataset/videos")
async def list_dataset_videos():
    video_files = sorted(glob.glob(os.path.join(DATASET_VIDEO_DIR, "*.mp4")))
    result = []
    for vf in video_files:
        fn = os.path.basename(vf)
        cap = cv2.VideoCapture(vf)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()

        is_normal = "Normal" in fn
        if is_normal:
            category = "Normal"
        else:
            raw_c = fn.split('_')[0]
            category = ''.join([c for c in raw_c if not c.isdigit()]) or "Anomaly"

        result.append({
            "filename": fn,
            "path": vf,
            "url": f"http://localhost:8000/dataset/videos/{fn}",
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "resolution": f"{w}x{h}",
            "category": category,
            "is_anomaly": not is_normal
        })

    return {
        "count": len(result),
        "video_dir": DATASET_VIDEO_DIR,
        "videos": result
    }

@app.post("/api/dataset/analyze-video")
async def analyze_dataset_video(payload: dict):
    filename = payload.get("filename")
    if not filename:
        return {"error": "filename parameter required"}

    video_path = os.path.join(DATASET_VIDEO_DIR, filename)
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {filename}"}

    try:
        from violence.test_real_video import run_real_video_inference
        res = run_real_video_inference(video_path)

        # Log anomaly incident to DB if detected
        if res.get("is_anomaly"):
            now_ts = time.time()
            alert_payload = {
                "id": int(now_ts * 1000),
                "type": res.get("prediction", "Anomaly Detected"),
                "event_type": res.get("event_type"),
                "severity": 5,
                "confidence": res.get("confidence") / 100.0,
                "timestamp": now_ts,
                "camera_id": "UCF-CRIME RAW MP4",
                "source": f"UCF-Crime MP4 ({filename})",
                "sample": filename,
                "timestamp_seconds": res.get("timestamp_seconds"),
                "status": "ACTIVE",
                "is_dataset_simulation": True
            }
            if alert_queue:
                try:
                    alert_queue.put(alert_payload)
                except Exception:
                    pass

        return {"success": True, "analysis": res}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/simulation/start")
async def start_simulation():
    success = global_dataset_simulation.start(alert_callback=on_simulation_alert)
    status = global_dataset_simulation.get_status()
    return {"success": success, "message": "Dataset simulation started" if success else "Simulation already running", "status": status}


@app.post("/api/simulation/stop")
async def stop_simulation():
    success = global_dataset_simulation.stop()
    status = global_dataset_simulation.get_status()
    return {"success": success, "message": "Dataset simulation stopped" if success else "Simulation not running", "status": status}

@app.post("/api/simulation/replay")
async def replay_simulation():
    success = global_dataset_simulation.replay(alert_callback=on_simulation_alert)
    status = global_dataset_simulation.get_status()
    return {"success": success, "message": "Dataset simulation replayed", "status": status}

@app.get("/api/simulation/status")
async def get_simulation_status():
    return global_dataset_simulation.get_status()

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    pubsub = None
    redis_client = None
    use_redis = True
    try:
        redis_client = aioredis.from_url("redis://localhost", socket_timeout=1.0)
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("alerts:raw")
    except Exception as e:
        use_redis = False
        print(f"Notice: Redis Pub/Sub fallback active (Redis not connected: {e})")

    try:
        while True:
            current_frame = global_dataset_simulation.latest_frame if (global_dataset_simulation.running and global_dataset_simulation.latest_frame) else None
            sim_status = global_dataset_simulation.get_status()

            if current_frame:

                b64_image = base64.b64encode(current_frame).decode('utf-8')
                payload = {
                    "type": "feed",
                    "image": f"data:image/jpeg;base64,{b64_image}",
                    "alerts": [],
                    "simulation": sim_status
                }
                await websocket.send_json(payload)
            
            pending_alerts = []
            if use_redis and pubsub:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.01)
                    if message and 'data' in message:
                        alert_data = json.loads(message['data'])
                        pending_alerts.append(alert_data)
                except Exception:
                    pass

            if not pending_alerts and alert_queue and not alert_queue.empty():
                try:
                    while not alert_queue.empty():
                        pending_alerts.append(alert_queue.get_nowait())
                except Exception:
                    pass

            for alert_data in pending_alerts:
                # 1. Async Postgres Save
                asyncio.create_task(database.save_incident(alert_data))
                
                # 2. Async Telegram Notification (if configured)
                asyncio.create_task(send_telegram_alert(alert_data))
                
                # 3. WebSocket push
                await websocket.send_json({
                    "type": "feed",
                    "image": f"data:image/jpeg;base64,{b64_image}" if current_frame else None,
                    "alerts": [alert_data],
                    "simulation": sim_status
                })
                
            await asyncio.sleep(0.03) 


            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        if pubsub:
            try:
                await pubsub.unsubscribe("alerts:raw")
            except Exception:
                pass
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                pass

