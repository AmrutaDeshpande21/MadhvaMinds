import os
import sys
import argparse
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from violence_classifier import ViolenceInference


class VideoFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ResNet50 backbone truncated before FC classification head -> outputs 2048-D features
        resnet = models.resnet50(pretrained=True)
        resnet.fc = torch.nn.Identity()
        self.extractor = resnet.to(self.device)
        self.extractor.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frame_feature(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor_in = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.extractor(tensor_in).cpu().numpy().squeeze(0) # (2048,)
        return feat

    def process_video_temporal_windows(self, video_path, num_windows=32):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            cap.release()
            raise RuntimeError("Empty video file.")

        step_indices = np.linspace(0, total_frames - 1, num_windows, dtype=int)
        features_list = []
        timestamps_list = []

        for frame_idx in step_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                feat = np.zeros((2048,), dtype=np.float32)
            else:
                feat = self.extract_frame_feature(frame)

            features_list.append(feat)
            timestamps_list.append(round(float(frame_idx / fps), 2))

        cap.release()
        feature_matrix = np.array(features_list, dtype=np.float32) # (32, 2048)
        
        return {
            "duration": round(float(duration), 2),
            "fps": round(float(fps), 2),
            "resolution": f"{w}x{h}",
            "total_frames": total_frames,
            "feature_matrix": feature_matrix,
            "timestamps": timestamps_list
        }


def run_real_video_inference(video_path):
    print(f"\n==========================================")
    print(f"REAL UCF-CRIME VIDEO TEMPORAL INFERENCE")
    print(f"==========================================")
    print(f"Video Path: {video_path}")

    extractor = VideoFeatureExtractor()
    meta = extractor.process_video_temporal_windows(video_path, num_windows=32)

    classifier = ViolenceInference()
    full_pred = classifier.predict_violence_sequence(meta["feature_matrix"])

    raw_score = full_pred["raw_score"]
    is_anom = full_pred["is_anomaly"]
    conf = round(full_pred["confidence"] * 100.0, 1)

    # Evaluate sliding temporal sub-windows across video duration
    feats = meta["feature_matrix"]
    timestamps = meta["timestamps"]
    duration = meta["duration"]

    win_len = 16
    stride = 4
    sub_window_results = []

    for start_idx in range(0, 32 - win_len + 1, stride):
        sub_feats = feats[start_idx : start_idx + win_len]
        res = classifier.predict_violence_sequence(sub_feats)
        t_start = timestamps[start_idx]
        t_end = timestamps[min(start_idx + win_len - 1, 31)]
        
        sub_window_results.append({
            "start_time": t_start,
            "end_time": t_end,
            "raw_score": res["raw_score"],
            "confidence": res["confidence"],
            "is_anomaly": res["is_anomaly"]
        })

    # Temporal Debouncing: Merge contiguous sub-window anomaly detections
    intervals = []
    current_interval = None

    for sub in sub_window_results:
        if sub["is_anomaly"]:
            if current_interval is None:
                current_interval = {
                    "event_type": "fight",
                    "label": "FIGHTING DETECTED",
                    "start_time": sub["start_time"],
                    "end_time": sub["end_time"],
                    "peak_confidence": round(sub["confidence"] * 100.0, 1),
                    "peak_raw_score": sub["raw_score"],
                    "is_anomaly": True
                }
            else:
                current_interval["end_time"] = sub["end_time"]
                current_interval["peak_confidence"] = max(current_interval["peak_confidence"], round(sub["confidence"] * 100.0, 1))
                current_interval["peak_raw_score"] = max(current_interval["peak_raw_score"], sub["raw_score"])
        else:
            if current_interval is not None:
                intervals.append(current_interval)
                current_interval = None

    if current_interval is not None:
        intervals.append(current_interval)

    # If full sequence is an anomaly but no sub-window triggered, add peak interval
    if is_anom and not intervals:
        feats_norms = [float(np.linalg.norm(f)) for f in feats]
        peak_idx = int(np.argmax(feats_norms))
        peak_t = timestamps[peak_idx] if timestamps else 0.0
        intervals.append({
            "event_type": "fight",
            "label": "FIGHTING DETECTED",
            "start_time": peak_t,
            "end_time": min(duration, round(peak_t + 10.0, 2)),
            "peak_confidence": conf,
            "peak_raw_score": raw_score,
            "is_anomaly": True
        })

    fn = os.path.basename(video_path)
    gt_category = "Normal" if "Normal" in fn else fn.split('_')[0]

    result = {
        "video": fn,
        "duration": meta["duration"],
        "fps": meta["fps"],
        "resolution": meta["resolution"],
        "ground_truth": gt_category,
        "input_shape": list(meta["feature_matrix"].shape),
        "prediction": "FIGHTING DETECTED" if is_anom else "NORMAL ACTIVITY",
        "event_type": "fight" if is_anom else "normal",
        "is_anomaly": is_anom,
        "confidence": conf,
        "raw_score": raw_score,
        "intervals": intervals,
        "detections": intervals # Isolated detections belonging ONLY to this video
    }

    print(f"\nVIDEO: {fn} ({meta['duration']}s)")
    print(f"PREDICTION: {result['prediction']} | CONFIDENCE: {conf}% | IS ANOMALY: {is_anom}")
    print(f"DEBOUNCED INTERVALS COUNT: {len(intervals)}")
    for inv in intervals:
        print(f"  - [{inv['start_time']}s - {inv['end_time']}s]: {inv['label']} ({inv['peak_confidence']}%)")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Real UCF-Crime Video Anomaly Detection Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to raw UCF-Crime MP4 video file")
    args = parser.parse_args()

    run_real_video_inference(args.video)
