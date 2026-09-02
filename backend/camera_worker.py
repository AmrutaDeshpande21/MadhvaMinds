import cv2
import multiprocessing as mp
import time

def camera_worker_process(source, out_queue, max_queue=30):
    """
    Dedicated process to grab frames and push to a queue, bypassing the GIL.
    Implements drop-oldest backpressure.
    """
    print(f"Starting camera worker for source: {source}")
    cap = cv2.VideoCapture(source)
    seq = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Reconnect or loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(source)
                continue
            
        # Backpressure: drop oldest if queue is full
        if out_queue.qsize() >= max_queue:
            try:
                out_queue.get_nowait() 
            except Exception:
                pass
                
        out_queue.put({
            "cam_id": "demo-cam-1",
            "seq": seq,
            "frame": frame,
            "ts": time.time()
        })
        seq += 1
