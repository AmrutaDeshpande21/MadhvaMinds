import os
import sys
import glob
import time
import json
import threading
import numpy as np
import cv2
from datetime import datetime

try:
    from backend.violence_classifier import ViolenceInference
except ImportError:
    from violence_classifier import ViolenceInference

DATASET_FEATURE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset", "ucf-crime-i3d"))

class UCFCrimeI3DSimulationEngine:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.dataset_name = "UCF-Crime I3D Feature Dataset"
        self.dataset_path = DATASET_FEATURE_DIR
        self.npy_files = []
        self.current_idx = 0
        self.total_samples = 0
        
        self.current_sample_name = "None"
        self.progress_pct = 0.0
        
        self.total_processed = 0
        self.total_anomalies = 0
        self.total_normal = 0
        self.total_correct = 0
        self.accuracy_pct = 100.0
        
        self.category_counts = {}
        self.latest_prediction = None
        
        self.detector = ViolenceInference()
        self.latest_frame = None
        self.alert_callback = None
        
        self.discover_dataset_files()

    def discover_dataset_files(self):
        files = []
        if os.path.exists(DATASET_FEATURE_DIR):
            files = sorted(glob.glob(os.path.join(DATASET_FEATURE_DIR, "**", "*.npy"), recursive=True))
            # Interleave anomaly and normal samples for a balanced demonstration flow
            anom = [f for f in files if "Normal" not in os.path.basename(f)]
            norm = [f for f in files if "Normal" in os.path.basename(f)]
            
            interleaved = []
            max_len = max(len(anom), len(norm))
            for i in range(max_len):
                if i < len(anom):
                    interleaved.append(anom[i])
                if i < len(norm):
                    interleaved.append(norm[i])
            files = interleaved

        self.npy_files = files
        self.total_samples = len(files)
        print(f"UCFCrimeI3DSimulationEngine discovered {self.total_samples} .npy feature samples in {DATASET_FEATURE_DIR}")


    def start(self, alert_callback=None):
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.alert_callback = alert_callback
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("UCF-Crime I3D Feature Simulation Engine started.")
            return True

    def stop(self):
        with self.lock:
            if not self.running:
                return False
            self.running = False
            print("UCF-Crime I3D Feature Simulation Engine stopped.")
            return True

    def replay(self, alert_callback=None):
        with self.lock:
            self.current_idx = 0
            self.total_processed = 0
            self.total_anomalies = 0
            self.total_normal = 0
            self.total_correct = 0
            self.accuracy_pct = 100.0
            self.category_counts = {}
            self.progress_pct = 0.0
            
        if not self.running:
            return self.start(alert_callback=alert_callback)
        return True

    def get_status(self):
        with self.lock:
            return {
                "status": "RUNNING" if self.running else "READY" if self.total_processed == 0 else "STOPPED",
                "running": self.running,
                "dataset_name": self.dataset_name,
                "dataset_path": self.dataset_path,
                "current_sample": self.current_sample_name,
                "current_idx": self.current_idx + 1,
                "total_samples": self.total_samples,
                "progress": round(self.progress_pct, 1),
                "total_processed": self.total_processed,
                "anomalies": self.total_anomalies,
                "normal": self.total_normal,
                "accuracy": round(self.accuracy_pct, 1),
                "category_counts": self.category_counts,
                "latest_prediction": self.latest_prediction
            }

    def _extract_ground_truth_category(self, filename):
        base = os.path.basename(filename)
        if "Normal" in base:
            return "Normal", False
        
        prefix = base.split('_')[0]
        category = ''.join([c for c in prefix if not c.isdigit()])
        if not category:
            category = "Anomaly"
        return category, True

    def _render_i3d_feature_analysis_frame(self, sample_name, arr_shape, dtype_str, mean_energy, pred_res, gt_category, is_correct):
        """
        Renders a high-fidelity AI temporal feature waveform and tensor diagnostics panel
        based strictly on the actual 2048-D feature values loaded from the .npy sample.
        """
        h, w = 540, 960
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Background gradient and grid lines
        img[:, :] = (15, 18, 24) # Dark slate background
        for y in range(80, h - 60, 50):
            cv2.line(img, (40, y), (w - 40, y), (30, 35, 45), 1)

        # Header Title Banner
        cv2.rectangle(img, (0, 0), (w, 55), (10, 12, 18), -1)
        cv2.putText(img, "UCF-CRIME I3D TEMPORAL FEATURE ANALYSIS", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 225, 255), 2)
        cv2.putText(img, "SOURCE: PRE-EXTRACTED .NPY TENSORS (NO WEBCAM)", (w - 450, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # File Metadata Section
        cv2.putText(img, f"SAMPLE: {sample_name}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 2)
        cv2.putText(img, f"SHAPE: {arr_shape} | DTYPE: {dtype_str} | DIM: 2048-D", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Waveform Plot Box
        plot_x1, plot_y1, plot_w, plot_h = 40, 140, w - 80, 230
        cv2.rectangle(img, (plot_x1, plot_y1), (plot_x1 + plot_w, plot_y1 + plot_h), (25, 30, 42), -1)
        cv2.rectangle(img, (plot_x1, plot_y1), (plot_x1 + plot_w, plot_y1 + plot_h), (50, 60, 80), 1)
        cv2.putText(img, "I3D TEMPORAL FEATURE ENERGY SIGNAL WAVEFORM (t=0..T)", (plot_x1 + 15, plot_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # Draw Actual Temporal Feature Waveform Curve
        T = len(mean_energy)
        if T > 1:
            pts = []
            max_val = max(1e-5, np.max(mean_energy))
            min_val = np.min(mean_energy)
            rng = max(1e-5, max_val - min_val)

            for t_idx in range(T):
                x = int(plot_x1 + 20 + (t_idx / (T - 1)) * (plot_w - 40))
                norm_v = (mean_energy[t_idx] - min_val) / rng
                y = int(plot_y1 + plot_h - 20 - norm_v * (plot_h - 50))
                pts.append((x, y))

            # Draw lines connecting energy points
            line_color = (0, 100, 255) if pred_res["is_anomaly"] else (0, 255, 128)
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i+1], line_color, 2)
                cv2.circle(img, pts[i], 3, (255, 255, 255), -1)
            if pts:
                cv2.circle(img, pts[-1], 3, (255, 255, 255), -1)

        # Inference Prediction & Evaluation Panel
        panel_y = plot_y1 + plot_h + 20
        cv2.rectangle(img, (40, panel_y), (w - 40, h - 30), (20, 24, 34), -1)
        cv2.rectangle(img, (40, panel_y), (w - 40, h - 30), (60, 70, 90), 1)

        is_anom = pred_res["is_anomaly"]
        conf = pred_res["confidence"] * 100.0
        
        pred_title = f"🔴 VIOLENCE / ANOMALY DETECTED ({conf:.1f}%)" if is_anom else f"🟢 NORMAL ACTIVITY ({conf:.1f}%)"
        pred_color = (0, 0, 255) if is_anom else (0, 255, 0)

        cv2.putText(img, f"DATASET GROUND TRUTH: {gt_category.upper()}", (60, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"MODEL PREDICTION: {pred_title}", (60, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)

        eval_str = "✓ CORRECT EVALUATION" if is_correct else "✗ MODEL INCORRECT"
        eval_color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(img, eval_str, (w - 280, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, eval_color, 2)

        # Encode to JPG
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    def _run_loop(self):
        while self.running:
            if not self.npy_files:
                self.discover_dataset_files()
                if not self.npy_files:
                    time.sleep(1.0)
                    continue

            if self.current_idx >= len(self.npy_files):
                self.current_idx = 0 # Cycle through dataset samples

            npy_path = self.npy_files[self.current_idx]
            sample_name = os.path.basename(npy_path)
            gt_category, gt_is_anomaly = self._extract_ground_truth_category(sample_name)

            with self.lock:
                self.current_sample_name = sample_name
                self.progress_pct = ((self.current_idx + 1) / self.total_samples) * 100.0

            # 1. Load actual .npy feature array from filesystem
            try:
                raw_arr = np.load(npy_path).astype(np.float32)
            except Exception as e:
                print(f"Error loading {sample_name}: {e}")
                self.current_idx += 1
                continue

            arr_shape = str(raw_arr.shape)
            dtype_str = str(raw_arr.dtype)

            # Extract 1D temporal energy profile across time steps for waveform visualization
            if raw_arr.ndim == 3 and raw_arr.shape[1] == 10:
                # Shape (T, 10, 2048) -> average 10 crops -> (T, 2048) -> mean across 2048 -> (T,)
                mean_energy = raw_arr.mean(axis=(1, 2))
            elif raw_arr.ndim == 2:
                mean_energy = raw_arr.mean(axis=1)
            else:
                mean_energy = np.ones((32,), dtype=np.float32)

            # 2. Perform authentic PyTorch model inference on loaded feature matrix
            pred_res = self.detector.predict_violence_sequence(raw_arr)
            pred_is_anomaly = pred_res["is_anomaly"]
            conf = float(pred_res["confidence"])
            
            is_correct = (pred_is_anomaly == gt_is_anomaly)

            # Render feature waveform frame for visual analysis canvas
            frame_bytes = self._render_i3d_feature_analysis_frame(
                sample_name, arr_shape, dtype_str, mean_energy, pred_res, gt_category, is_correct
            )
            self.latest_frame = frame_bytes

            # 3. Update Dataset Statistics
            with self.lock:
                self.total_processed += 1
                if is_correct:
                    self.total_correct += 1

                self.accuracy_pct = (self.total_correct / self.total_processed) * 100.0

                if pred_is_anomaly:
                    self.total_anomalies += 1
                    self.category_counts[gt_category] = self.category_counts.get(gt_category, 0) + 1
                else:
                    self.total_normal += 1

                self.latest_prediction = {
                    "sample": sample_name,
                    "shape": arr_shape,
                    "ground_truth": gt_category,
                    "gt_is_anomaly": gt_is_anomaly,
                    "prediction": "VIOLENCE / ANOMALY DETECTED" if pred_is_anomaly else "NORMAL ACTIVITY",
                    "is_anomaly": pred_is_anomaly,
                    "confidence": conf,
                    "is_correct": is_correct,
                    "accuracy": round(self.accuracy_pct, 1)
                }

            # 4. Emit WebSocket alert event ONLY if model predicted an anomaly
            if pred_is_anomaly and self.alert_callback:
                now_ts = time.time()
                alert_data = {
                    "id": int(now_ts * 1000),
                    "type": f"{gt_category} Anomaly Detected",
                    "event_type": gt_category.lower(),
                    "severity": 5 if gt_category in ["Fighting", "Assault", "Explosion", "Robbery"] else 4,
                    "confidence": conf,
                    "timestamp": now_ts,
                    "camera_id": "UCF-CRIME I3D SOURCE",
                    "source": "UCF-Crime I3D Dataset",
                    "sample": sample_name,
                    "ground_truth": gt_category,
                    "is_correct": is_correct,
                    "status": "ACTIVE",
                    "is_dataset_simulation": True
                }
                try:
                    self.alert_callback(alert_data)
                except Exception as e:
                    print(f"I3D Simulation alert error: {e}")

            # Hold sample view for 1.2 seconds for realistic evaluation speed
            time.sleep(1.2)
            self.current_idx += 1

        with self.lock:
            self.current_sample_name = "Stopped"

global_dataset_simulation = UCFCrimeI3DSimulationEngine()
