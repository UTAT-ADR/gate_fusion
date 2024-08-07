from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")

# Train the model
results = model.train(data="0807_pencil/data.yaml", epochs=300, imgsz=640, device=0, workers=20, plots=True)