# 🚨 MadhvaMinds - Real-Time Incident Intelligence Platform

Every CCTV camera today is a passive recorder. By the time someone reviews the footage, the incident has already happened. **MadhvaMinds** turns passive cameras into an active, real-time safety layer using state-of-the-art computer vision and a distributed architecture.

Built for the Yuva Hackathon 2.0.

## 🧠 Core Architecture (4-Module Pipeline)

MadhvaMinds employs a 4-module AI pipeline capable of sub-200ms latency inference on CPU/Edge devices:

1. **Module A (Intrusion Detection):** Uses YOLOv8 (optimized to ONNX) and Shapely polygon intersections to instantly detect unauthorized access into restricted zones.
2. **Module B (Pose-Based Fall Detection):** Uses MediaPipe Pose estimation to mathematically track a person's torso collapse angle and vertical centroid velocity to detect falls and medical emergencies.
3. **Module C (Violence & Fight Detection):** Utilizes a lightweight spatial-proximity gate to measure bounding box centroid distances. If subjects breach the proximity threshold with erratic motion, a Severity 5 alert is triggered.
4. **Module D (Fire & Smoke Detection):** Employs HSV color-space thresholding and temporal debouncing across 30-frame sliding windows to detect fire without heavy CNN overhead.

## 🏗 Tech Stack

* **Backend:** FastAPI (Python 3.11), asynchronous event loop.
* **Frontend:** Next.js (React), TailwindCSS, TypeScript.
* **Message Broker:** Redis Pub/Sub (for sub-millisecond alert fan-out).
* **Database:** PostgreSQL with `asyncpg` (for permanent historical incident logging).
* **Machine Learning:** Ultralytics YOLOv8 (ONNX runtime), MediaPipe, OpenCV.
* **Notifications:** Telegram Bot API (Instant mobile push notifications).

## 🚀 Getting Started

To run the full distributed architecture locally, you will need **Docker** and **Python 3.11+**.

### 1. Start the Infrastructure
Spin up the Redis message broker and PostgreSQL database using Docker:
```bash
docker-compose up -d
```

### 2. Configure Telegram (Optional but Recommended)
To receive mobile notifications for Severity 5 incidents (Fires, Falls, Violence), rename `backend/.env.template` to `backend/.env` and add your Telegram Bot Token and Chat ID.

### 3. Start the Backend
The backend spawns a multi-process queue to bypass the Python GIL and streams alerts over WebSockets.
```bash
cd backend
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 4. Start the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to view the live dashboard and the Historical Police Analysis database!

## 📊 Benchmarks
* **Inference Pipeline:** Runs concurrently with YOLOv8 + MediaPipe + Heuristics.
* **Latency:** End-to-end (Frame Capture → WebSocket dispatch) consistently under **150ms**.
* **Scaling:** Horizontal scaling enabled via Redis Pub/Sub decoupling.
