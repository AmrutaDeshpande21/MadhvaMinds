from ultralytics import YOLO
import time

def export_model():
    print("Loading PyTorch YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")
    
    print("Exporting model to ONNX format for accelerated CPU/GPU inference...")
    start_time = time.time()
    
    # Export the model
    # dynamic=True allows variable input sizes if needed
    path = model.export(format="onnx", dynamic=True)
    
    duration = time.time() - start_time
    print(f"Model successfully exported to: {path}")
    print(f"Export took {duration:.2f} seconds.")
    print("You can now update the vision pipeline to use 'yolov8n.onnx'.")

if __name__ == "__main__":
    export_model()
