# AI-Powered Real-Time Incident Intelligence Platform
## Technical Implementation & Architecture Plan

---

## Phase 1 — System Architecture & Tech Stack Justification

> **Hackathon Scope Note:** The architecture below represents the complete, production-grade vision (4 concurrent ML modules). For the 4-day hackathon implementation, we will focus exclusively on **Module A (Intrusion)** and **Module B (Fall Detection)** to guarantee a flawless live demo, while presenting the full architecture to the judges.

### 1.1 ASCII System Architecture Diagram

```
                                   ┌────────────────────────────────────────────┐
                                   │              CCTV / RTSP SOURCES            │
                                   │   (N cameras, H.264/H.265 streams)          │
                                   └───────────────────┬──────────────────────--┘
                                                        │ RTSP/RTMP
                                                        ▼
                          ┌─────────────────────────────────────────────────┐
                          │        INGEST LAYER (per-camera process)         │
                          │  OpenCV/FFmpeg VideoCapture (GStreamer backend)  │
                          │  - Frame grab @ source FPS                       │
                          │  - Adaptive frame-skip / keyframe sampling       │
                          └───────────────────┬───────────────────────────--┘
                                               ▼
                          ┌─────────────────────────────────────────────────┐
                          │         DECODE + QUEUE (Redis Streams / mp.Queue)│
                          │  - Ring buffer, drop-oldest backpressure policy  │
                          │  - Frame metadata: cam_id, ts, seq               │
                          └───────────────────┬───────────────────────────--┘
                                               ▼
                ┌──────────────────────────────┴──────────────────────────────┐
                ▼                    ▼                    ▼                   ▼
      ┌─────────────────┐ ┌──────────────────┐ ┌───────────────────┐ ┌────────────────┐
      │ Module A         │ │ Module B          │ │ Module C            │ │ Module D        │
      │ Intrusion /      │ │ Fall / Medical    │ │ Violence / Fight    │ │ Fire & Smoke    │
      │ Line-Crossing    │ │ (Pose Est.)       │ │ (Det + Temporal)    │ │ (YOLO det.)     │
      │ YOLOv8 + Shapely │ │ MediaPipe/RTMPose │ │ YOLOv8 + CNN-LSTM   │ │ Fine-tuned YOLO │
      └────────┬─────────┘ └────────┬─────────-┘ └─────────┬──────────┘ └────────┬───────-┘
                └───────────────────┴──────────────────────┴─────────────────────┘
                                               ▼
                          ┌─────────────────────────────────────────────────┐
                          │     SEVERITY + CONFIDENCE FUSION ENGINE           │
                          │  - Per-event confidence normalization             │
                          │  - Severity scoring (1-5 scale)                   │
                          └───────────────────┬───────────────────────────--┘
                                               ▼
                          ┌─────────────────────────────────────────────────┐
                          │     TEMPORAL DEBOUNCING ENGINE                    │
                          │  - Sliding window vote over N frames              │
                          │  - Hysteresis (rising/falling thresholds)         │
                          │  - Cooldown per (camera, event_type)              │
                          └───────────────────┬───────────────────────────--┘
                                               ▼
                          ┌─────────────────────────────────────────────────┐
                          │        BACKEND API (FastAPI, async)               │
                          │  - Event persistence → PostgreSQL/PostGIS         │
                          │  - Redis Pub/Sub fan-out                          │
                          │  - Webhook/SMS/Telegram dispatcher (async workers)│
                          └───────────────────┬───────────────────────────--┘
                                               ▼
                          ┌─────────────────────────────────────────────────┐
                          │        WEBSOCKET BROADCAST LAYER                  │
                          │  FastAPI WebSocket / Redis Pub-Sub bridge         │
                          └───────────────────┬───────────────────────────--┘
                                               ▼
                          ┌─────────────────────────────────────────────────┐
                          │        FRONTEND DASHBOARD (Next.js)               │
                          │  - Live camera grid + overlay boxes               │
                          │  - Incident timeline, heatmaps, alert feed        │
                          └─────────────────────────────────────────────────┘
```

### 1.2 Tech Stack & Justification

| Layer | Technology | Why it wins for latency + hackathon impact |
|---|---|---|
| Language/runtime | Python 3.11+ | Fastest path to CV/DL ecosystem; 3.11's speed improvements matter under real-time load |
| Video I/O | OpenCV + FFmpeg (GStreamer backend) | Battle-tested RTSP handling; hardware-accelerated decode (VAAPI/NVDEC) drops CPU load |
| Object detection | YOLOv8 (Ultralytics) | Best speed/accuracy tradeoff for real-time; exports cleanly to ONNX/TensorRT |
| Tracking | ByteTrack | Lightweight, no re-ID network needed, handles occlusion well — critical for crowd/fight scenes |
| Pose estimation | MediaPipe Pose (or RTMPose for higher accuracy) | Runs on CPU in real time; sufficient keypoint accuracy for fall-angle math |
| Temporal violence classifier | Compact CNN-LSTM or 3D-CNN (e.g., MoViNet-A0) | Small footprint, still captures short-term motion dynamics without full 3D-CNN cost |
| Serving/inference optimization | ONNX Runtime / TensorRT | 2-4x throughput gain, essential to hit the <200ms budget across parallel models |
| Backend API | FastAPI (async, uvicorn/gunicorn workers) | Native async, auto OpenAPI docs (great for judge Q&A), WebSocket support built in |
| Queueing / pub-sub | Redis (Streams + Pub/Sub) | Sub-millisecond in-memory ops; doubles as both frame buffer and alert bus |
| Database | PostgreSQL + PostGIS | Relational integrity for incidents; PostGIS enables geofenced/zone queries later |
| Frontend | Next.js (React) | SSR for fast first paint during demo, easy WebSocket client integration |
| Alerting | Webhook + Twilio (SMS) + Telegram Bot API | All three are trivially demoable live, reinforcing "production-grade" narrative |
| Containerization | Docker Compose | One-command judge reproducibility |

**Design principle:** every model runs in its own process (Python `multiprocessing`, not threads) to sidestep the GIL, with each process pinned to a model+camera combination. This is what makes "parallel model processing streams" actually parallel rather than a marketing phrase.

---

## Phase 2 — Data Sourcing, Extraction & Preprocessing

### 2.1 Dataset Acquisition Plan

| Task | Dataset | Notes |
|---|---|---|
| Violence/fighting | UCF-Crime, RWF-2000, Hockey Fight Dataset | UCF-Crime is weakly labeled (video-level) — use its MIL framing for pretraining, then fine-tune on RWF-2000 (clip-level labels) for the final classifier |
| Falls | UR Fall Detection Dataset, Le2i Fall Dataset | Provides synchronized depth/RGB in UR Fall — use RGB stream only for CCTV realism |
| Fire/smoke | FireNet dataset, Kaggle Fire & Smoke datasets, FASDD | Combine to cover both aerial and indoor CCTV-style angles |
| Crowd density | ShanghaiTech Crowd Counting (Part A/B), Mall dataset | Used to calibrate density-per-pixel heuristics rather than full crowd-counting network (compute budget) |
| Intrusion / general person detection | COCO-person subset, CrowdHuman | For robust person detector pretraining before zone-specific fine-tuning |

### 2.2 Frame Extraction & Preprocessing

- **Extraction rate:** 5 FPS for training-clip generation from long untrimmed video (temporal redundancy is high at native 25-30 FPS); inference runs at full available FPS post-deployment.
- **Resolution target:** 640×640 for YOLO input (letterboxed, aspect-preserving padding); pose model uses cropped 256×256 person patches from the detector's bounding boxes.
- **Augmentations:**
  - Photometric: brightness/contrast jitter, gamma shift to simulate low-light CCTV, synthetic sensor noise
  - Motion blur (kernel-based, simulating compression artifacts / fast subject motion)
  - Horizontal flip (safe for most classes; **excluded** for asymmetric text/sign-dependent frames if any)
  - Random JPEG re-compression to mimic CCTV encoder artifacts
  - Night/IR simulation: desaturate + add grain for datasets that lack native night footage

### 2.3 Scene-Level Split Strategy (Leakage Prevention)

The most common mistake in surveillance ML: splitting by clip instead of by **scene/camera source**, which leaks near-duplicate frames across train/val/test and inflates reported accuracy.

```python
# Pseudocode: scene-aware split
from collections import defaultdict
import random

def scene_aware_split(video_records, train=0.7, val=0.15, test=0.15, seed=42):
    scenes = defaultdict(list)
    for rec in video_records:
        scenes[rec["scene_id"]].append(rec)  # scene_id = camera/location fingerprint

    scene_ids = list(scenes.keys())
    random.Random(seed).shuffle(scene_ids)

    n = len(scene_ids)
    train_ids = set(scene_ids[: int(n * train)])
    val_ids = set(scene_ids[int(n * train): int(n * (train + val))])
    test_ids = set(scene_ids[int(n * (train + val)):])

    return (
        [r for sid in train_ids for r in scenes[sid]],
        [r for sid in val_ids for r in scenes[sid]],
        [r for sid in test_ids for r in scenes[sid]],
    )
```

---

## Phase 3 — Core Modular AI Pipeline

### Module A — Dynamic Polygon Intrusion Detection

Uses per-camera configurable polygons (drawn in the dashboard, stored as normalized coordinates) and `cv2.pointPolygonTest` / Shapely for robust point-in-polygon checks, including partial overlap for bounding boxes.

```python
from shapely.geometry import Polygon, box

def check_intrusion(bbox, restricted_polygon: Polygon, overlap_thresh=0.15):
    x1, y1, x2, y2 = bbox
    person_box = box(x1, y1, x2, y2)
    if not person_box.intersects(restricted_polygon):
        return False, 0.0
    inter_area = person_box.intersection(restricted_polygon).area
    overlap_ratio = inter_area / person_box.area
    return overlap_ratio >= overlap_thresh, overlap_ratio
```

For **dynamic line crossing** (e.g., perimeter breach direction), track centroid position relative to the line's signed distance across consecutive frames using ByteTrack IDs, and flag a crossing when the sign flips.

### Module B — Pose-Based Fall Detection

Core heuristic: sudden torso-angle collapse combined with near-zero vertical velocity after impact (to distinguish a fall from bending/crouching).

**Torso angle** (relative to vertical) from shoulder and hip midpoints:

```
θ = atan2(|hip_mid.x - shoulder_mid.x|, |hip_mid.y - shoulder_mid.y|) × (180/π)
```

- Standing: θ ≈ 0–20°
- Fall event candidate: θ crosses > 60° within a short window (≤ 1s) AND torso centroid vertical velocity drops near zero for ≥ N subsequent frames (post-impact stillness)

```python
import numpy as np

def torso_angle(shoulder_mid, hip_mid):
    dx = abs(hip_mid[0] - shoulder_mid[0])
    dy = abs(hip_mid[1] - shoulder_mid[1]) + 1e-6
    return np.degrees(np.arctan2(dx, dy))

def is_fall(angle_history, centroid_history, angle_thresh=60, still_frames=8, vel_thresh=2.0):
    if angle_history[-1] < angle_thresh:
        return False
    # velocity over last `still_frames`
    recent = centroid_history[-still_frames:]
    if len(recent) < still_frames:
        return False
    velocities = [np.linalg.norm(np.array(recent[i]) - np.array(recent[i-1]))
                  for i in range(1, len(recent))]
    return np.mean(velocities) < vel_thresh
```

### Module C — Proximity-Gated Violence Detection (Deferred for Hackathon)

Two-stage gate to save compute: only run the expensive temporal classifier when two or more person bounding boxes are in close proximity (IoU-adjacent or centroid distance below threshold), avoiding wasted inference on empty or sparse scenes.

```python
def proximity_gate(boxes, dist_thresh_px=120):
    centroids = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            d = ((centroids[i][0]-centroids[j][0])**2 + (centroids[i][1]-centroids[j][1])**2) ** 0.5
            if d < dist_thresh_px:
                return True
    return False
```

When gated `True`, feed a rolling buffer of the last ~16 frames (cropped to the region of interest, resized to 112×112 or 224×224) into a compact CNN-LSTM or MoViNet-A0 to classify `fight` vs `normal` with a temporal confidence score. Gating alone typically cuts unnecessary temporal-model calls by 70-90% in normal-density scenes.

### Module D — Fire & Smoke Detection (Deferred for Hackathon)

Fine-tuned YOLOv8 on FireNet/FASDD-style data. Two classes (`fire`, `smoke`) with a lower confidence floor than person detection (fire has fuzzier boundaries) but a **longer required consecutive-frame streak** to compensate for false positives from red/orange lighting or steam.

### 3.1 Temporal Debouncing Algorithm

The shared mitigation layer across all four modules — prevents flicker/false positives from single noisy frames.

```python
from collections import deque
import time

class Debouncer:
    def __init__(self, window=15, rise_ratio=0.6, fall_ratio=0.2, cooldown_sec=10):
        self.window = window
        self.rise_ratio = rise_ratio   # fraction of positive frames to trigger ALERT
        self.fall_ratio = fall_ratio   # fraction to clear ALERT (hysteresis)
        self.cooldown_sec = cooldown_sec
        self.buffers = {}     # key: (cam_id, event_type) -> deque
        self.state = {}       # key -> bool (currently alerting)
        self.last_alert_ts = {}

    def push(self, key, positive: bool):
        buf = self.buffers.setdefault(key, deque(maxlen=self.window))
        buf.append(1 if positive else 0)
        if len(buf) < self.window:
            return False  # not enough evidence yet

        ratio = sum(buf) / len(buf)
        currently_alerting = self.state.get(key, False)

        if not currently_alerting and ratio >= self.rise_ratio:
            now = time.time()
            if now - self.last_alert_ts.get(key, 0) > self.cooldown_sec:
                self.state[key] = True
                self.last_alert_ts[key] = now
                return True  # fire new alert
        elif currently_alerting and ratio <= self.fall_ratio:
            self.state[key] = False

        return False
```

Hysteresis (different rise/fall thresholds) prevents rapid alert/clear oscillation at the decision boundary — a common judge-visible flaw in naive threshold implementations.

---

## Phase 4 — Real-Time Backend & Data Ingestion

> **Hackathon Execution Strategy:** To minimize risk, we will initially build a **monolithic** Python pipeline (processing frames, running YOLO/MediaPipe sequentially, and triggering alerts). Once the end-to-end flow is stable with the Next.js frontend, we will refactor into the `multiprocessing` architecture below if time permits.

### 4.1 Multi-Process Frame Queue

```python
import multiprocessing as mp
import cv2

def camera_worker(cam_id, rtsp_url, out_queue: mp.Queue, max_queue=30):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    seq = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)  # reconnect
            continue
        if out_queue.qsize() >= max_queue:
            try:
                out_queue.get_nowait()   # drop-oldest backpressure
            except Exception:
                pass
        out_queue.put({"cam_id": cam_id, "seq": seq, "frame": frame, "ts": cv2.getTickCount()})
        seq += 1
```

Each detection module runs as its own consumer process reading from a per-camera `mp.Queue` (or Redis Streams for cross-host scaling), enabling true parallelism across CPU cores/GPUs.

### 4.2 PostgreSQL Schema (with PostGIS)

```sql
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    rtsp_url TEXT NOT NULL,
    location GEOGRAPHY(POINT, 4326),
    zone_polygon GEOGRAPHY(POLYGON, 4326),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE incidents (
    id BIGSERIAL PRIMARY KEY,
    camera_id UUID REFERENCES cameras(id),
    event_type TEXT NOT NULL,       -- 'fight' | 'fall' | 'intrusion' | 'fire' | 'crowd'
    severity SMALLINT NOT NULL CHECK (severity BETWEEN 1 AND 5),
    confidence REAL NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ,
    snapshot_url TEXT,
    metadata JSONB
);

CREATE INDEX idx_incidents_camera_time ON incidents (camera_id, started_at DESC);
CREATE INDEX idx_incidents_type_severity ON incidents (event_type, severity);
CREATE INDEX idx_incidents_metadata_gin ON incidents USING GIN (metadata);
```

### 4.3 Redis Pub/Sub Layout

- `alerts:raw` — internal channel, debounced events land here first
- `alerts:{camera_id}` — per-camera channel, subscribed to by the WebSocket bridge for scoped dashboard views
- `alerts:broadcast` — global channel for the "all incidents" admin view
- Alert payload: `{event_id, cam_id, type, severity, confidence, ts, snapshot_url}`

### 4.4 WebSocket Push Architecture

FastAPI WebSocket endpoint subscribes to Redis Pub/Sub on connect and forwards messages verbatim to connected dashboard clients — keeps the hot path (detection → dashboard) to a single Redis hop plus one WebSocket frame, which is what keeps end-to-end latency under budget.

---

## Phase 5 — Testing, Optimization & Benchmarks

### 5.1 ONNX / TensorRT Conversion

```bash
# Export YOLOv8 to ONNX, then to TensorRT engine
yolo export model=best.pt format=onnx opset=17 dynamic=True
trtexec --onnx=best.onnx --saveEngine=best.trt --fp16 --workspace=4096
```

Expect roughly 2-4x throughput improvement over native PyTorch inference on the same GPU, with FP16 typically costing negligible accuracy for detection-confidence thresholds already tuned with margin.

### 5.2 Edge-Case Testing Matrix

| Condition | Test approach | Pass criteria |
|---|---|---|
| Low light / night | IR-simulated + genuinely dark clips | Detection recall doesn't drop more than ~15% vs daylight baseline |
| Dense crowds | ShanghaiTech Part B-style footage | Person detector maintains tracking IDs through partial occlusion |
| Stream drops / reconnect | Kill RTSP mid-stream | Worker auto-reconnects within 3s, no crash, queue recovers cleanly |
| Camera angle extremes | Oblique/high-mount angles | Pose keypoints degrade gracefully rather than producing false fall triggers |
| Rapid lighting change | Simulated flicker (fluorescent) | Debouncer's hysteresis absorbs transient false positives |

### 5.3 Benchmarks to Present to Judges

- **End-to-end latency:** frame capture → alert dispatched (target: sub-200ms for the CV pipeline hop; total including network/SMS delivery quoted separately)
- **Sustained FPS per camera** under full 4-module parallel load
- **False-alarm rate before/after debouncing** — this delta is one of the most persuasive judge-facing numbers, so instrument it explicitly and log both raw and debounced trigger counts
- **Accuracy on held-out scene-disjoint test split** (precision/recall/F1 per event type)

---

## Phase 6 — Hackathon Presentation & Demo Script

### 6.1 Three-Minute Live Presentation Script

1. **0:00-0:25 — Hook:** Open on the problem, not the tech: "Every CCTV camera today is a passive recorder. By the time someone reviews the footage, the incident already happened." Frame the platform as turning passive cameras into an active safety layer.
2. **0:25-0:55 — Architecture in one breath:** Show the architecture diagram for ~10 seconds, name the four detection modules in a single sentence, then pivot immediately to the live demo — judges remember demos, not slides.
3. **0:55-2:00 — Live demo:**
   - Trigger a real fall on a local webcam feed → show the dashboard alert appear within ~1 second, including severity score and Telegram notification arriving on a visible phone.
   - Cut to a pre-recorded benchmark stream (UCF-Crime/RWF-2000 clip) to show a violence detection trigger with the temporal confidence overlay — this demonstrates the system works on real-world footage, not just staged demo conditions.
   - Show the debouncing effect: briefly toggle a "raw" vs "debounced" view to visually prove false-positive suppression.
4. **2:00-2:40 — Numbers that matter:** State the three benchmark numbers rehearsed cold: end-to-end latency, sustained FPS, false-alarm reduction percentage.
5. **2:40-3:00 — Close:** One sentence on scalability (camera count) and one sentence on the business/deployment angle (software-only, retrofits existing CCTV — no hardware swap required). End on the hook restated as solved.

### 6.2 Live Demo Setup Guide

- Run the local webcam as `camera_id=demo-1` alongside 1-2 pre-recorded benchmark streams looped via `ffmpeg -re -stream_loop -1 -i clip.mp4 -f rtsp rtsp://localhost:8554/demo2` (using a lightweight local RTSP server like `mediamtx`) so both feeds appear identically as "cameras" in the dashboard.
- Pre-stage a phone with Telegram open and notifications visible to the audience/judges for the alert payoff moment.
- Rehearse the physical fall trigger (a controlled, safe "stumble" motion) beforehand so the pose-angle threshold reliably fires on the first take — do not leave this to chance live.

### 6.3 Anticipated Judge Q&A

**"How does this scale to 100+ cameras?"**
Horizontal scaling: each camera's ingestion + inference is an independent process/pod; Redis Streams and PostgreSQL are the only shared state, both of which scale via standard clustering (Redis Cluster, Postgres read replicas / partitioned incident tables by camera_id). GPU inference batches across cameras when using a shared model server (e.g., Triton Inference Server) rather than one model instance per camera, which is the actual bottleneck-avoidance strategy at scale.

**"Edge deployment vs. cloud processing?"**
Frame decode + lightweight modules (intrusion polygon check, motion gating) run at the edge (NVIDIA Jetson-class device) to cut bandwidth; heavier temporal classification can either run at the edge with a quantized model or be offloaded to a central GPU cluster over a compressed stream, depending on available edge compute. Frame the tradeoff as bandwidth-vs-compute rather than claiming one is strictly better.

**"How do you avoid false positives at 3am with poor lighting?"**
Point to the low-light augmentation strategy in Phase 2, the debouncing hysteresis in Phase 3, and note that severity scoring is deliberately conservative in low-confidence conditions rather than binary alert/no-alert.

---

### 6.4 💡 Hackathon Tip (For the Presentation)

**Remember to tell the judges:**
> *"We designed a comprehensive 4-module architecture to solve this problem holistically (show the diagram). However, for the sake of the 4-day hackathon timeframe and to guarantee a robust, real-time live demo today, we fully implemented the Intrusion and Fall Detection modules live."*
