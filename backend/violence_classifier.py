import os
import json
import torch
import numpy as np

try:
    from backend.train_violence_model import TemporalViolenceClassifier
except ImportError:
    from train_violence_model import TemporalViolenceClassifier

class ViolenceInference:
    def __init__(self, model_path=None, meta_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.dirname(__file__)
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "best_violence_model.pt")
        if meta_path is None:
            meta_path = os.path.join(base_dir, "models", "best_violence_model_meta.json")

        self.model_path = model_path
        self.meta_path = meta_path
        
        self.in_features = 2048
        self.sequence_length = 32
        self.confidence_threshold = 0.5
        
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                    self.in_features = meta.get("in_features", 2048)
                    self.sequence_length = meta.get("sequence_length", 32)
            except Exception as e:
                print(f"Warning: Could not read metadata: {e}")

        self.model = TemporalViolenceClassifier(
            in_features=self.in_features,
            hidden_dim=256,
            lstm_hidden=128,
            dropout=0.0
        ).to(self.device)

        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                print(f"Loaded Violence Classifier model from: {self.model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model checkpoint: {e}")
                self.model.eval()
        else:
            print(f"Notice: Model file not found at {self.model_path}. Initialized fallback classifier.")
            self.model.eval()

    def preprocess_features(self, raw_features):
        """
        Accepts numpy array or torch tensor of shape (T, 10, F), (T, F), or (B, T, F).
        Returns tensor of shape (1, sequence_length, in_features).
        """
        if isinstance(raw_features, torch.Tensor):
            arr = raw_features.cpu().numpy()
        else:
            arr = np.array(raw_features, dtype=np.float32)

        if arr.ndim == 3 and arr.shape[1] == 10:
            # (T, 10, F) -> average 10 crops -> (T, F)
            arr = arr.mean(axis=1)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            # (1, T, F)
            arr = arr.squeeze(0)

        if arr.ndim != 2:
            arr = arr.reshape(-1, self.in_features)

        T, F = arr.shape
        if T >= self.sequence_length:
            indices = np.linspace(0, T - 1, self.sequence_length, dtype=int)
            sampled = arr[indices]
        else:
            sampled = np.zeros((self.sequence_length, F), dtype=np.float32)
            sampled[:T] = arr

        tensor_in = torch.tensor(sampled, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor_in

    def predict_violence_sequence(self, raw_features):
        """
        Returns JSON-serializable event dictionary:
        {
            "event_type": "fight",
            "confidence": 0.93,
            "is_anomaly": True
        }
        or
        {
            "event_type": "normal",
            "confidence": 0.91,
            "is_anomaly": False
        }
        """
        tensor_in = self.preprocess_features(raw_features)

        with torch.no_grad():
            logits = self.model(tensor_in)
            confidence = torch.sigmoid(logits).item()

        is_anomaly = bool(confidence >= self.confidence_threshold)
        event_type = "fight" if is_anomaly else "normal"
        report_conf = confidence if is_anomaly else (1.0 - confidence)

        return {
            "event_type": event_type,
            "confidence": round(float(report_conf), 4),
            "is_anomaly": is_anomaly,
            "raw_score": round(float(confidence), 4)
        }

if __name__ == "__main__":
    detector = ViolenceInference()
    dummy_input = np.random.randn(50, 10, 2048).astype(np.float32)
    res = detector.predict_violence_sequence(dummy_input)
    print("Inference Test Result:", json.dumps(res, indent=2))
