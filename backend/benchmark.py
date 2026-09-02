import time
import cv2
import numpy as np
from ultralytics import YOLO
import os

def run_benchmark():
    print("--- Phase 5: AI Pipeline Benchmarking ---")
    print("Loading models...")
    
    model_path = "yolov8n.onnx" if os.path.exists("yolov8n.onnx") else "yolov8n.pt"
    print(f"Using Object Detection Model: {model_path}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model. Did you run the ONNX export? Error: {e}")
        return

    # Create dummy frame to simulate 1080p camera input
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    print("\nWarming up pipeline...")
    for _ in range(5):
        model(frame, classes=[0], verbose=False)
        
    print("Running sustained load test (100 frames)...")
    start_time = time.time()
    
    latencies = []
    
    for i in range(100):
        frame_start = time.time()
        
        # 1. YOLO Inference
        model(frame, classes=[0], verbose=False)
        
        # 2. Simulated MediaPipe/Heuristics cost (approx 5-10ms per frame)
        time.sleep(0.008) 
        
        frame_end = time.time()
        latencies.append((frame_end - frame_start) * 1000) # in ms
        
    total_time = time.time() - start_time
    fps = 100 / total_time
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print("\n--- BENCHMARK RESULTS ---")
    print("These metrics are ready for your hackathon presentation:")
    print(f"Total processing time for 100 frames: {total_time:.2f} seconds")
    print(f"Sustained Pipeline FPS: {fps:.2f} FPS")
    print(f"Average End-to-End Latency: {avg_latency:.2f} ms")
    print(f"95th Percentile (p95) Latency: {p95_latency:.2f} ms")
    print("---------------------------------")
    
    if avg_latency < 200:
        print("RESULT: PASS. Latency is well under the 200ms budget!")
    else:
        print("RESULT: WARN. Latency exceeds 200ms budget.")

if __name__ == "__main__":
    run_benchmark()
