import cv2
import os
import random
from collections import defaultdict

def extract_frames(video_path, output_dir, target_fps=5, resolution=(640, 640)):
    """
    Extract frames from a video file at a target FPS (useful for ML training data generation).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0:
        source_fps = 30 # fallback assumption
        
    frame_skip = int(source_fps / target_fps)
    if frame_skip < 1:
        frame_skip = 1

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            # Resize while preserving aspect ratio (letterboxing)
            h, w = frame.shape[:2]
            scale = min(resolution[0]/w, resolution[1]/h)
            nh, nw = int(h*scale), int(w*scale)
            
            resized = cv2.resize(frame, (nw, nh))
            
            # Create a blank black canvas
            canvas = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
            # Center the image
            y_offset = (resolution[1] - nh) // 2
            x_offset = (resolution[0] - nw) // 2
            canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
            
            filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(filename, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_count += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

def scene_aware_split(video_records, train=0.7, val=0.15, test=0.15, seed=42):
    """
    Splits video datasets by scene/camera source to prevent data leakage 
    across training, validation, and test sets.
    """
    scenes = defaultdict(list)
    for rec in video_records:
        # Assuming rec is a dict with a 'scene_id' key
        scenes[rec.get("scene_id", "unknown")].append(rec)

    scene_ids = list(scenes.keys())
    random.Random(seed).shuffle(scene_ids)

    n = len(scene_ids)
    train_ids = set(scene_ids[: int(n * train)])
    val_ids = set(scene_ids[int(n * train): int(n * (train + val))])
    test_ids = set(scene_ids[int(n * (train + val)):])

    train_split = [r for sid in train_ids for r in scenes[sid]]
    val_split = [r for sid in val_ids for r in scenes[sid]]
    test_split = [r for sid in test_ids for r in scenes[sid]]

    return train_split, val_split, test_split

if __name__ == "__main__":
    import numpy as np # required for extract_frames resizing
    print("Dataset utilities loaded. Ready for preprocessing.")
