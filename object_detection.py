import torch
import torchvision.transforms as transforms
from PIL import Image
from yolov5 import YOLOv5

# Load YOLOv5 model
model = YOLOv5("yolov5s.pt")

def detect_objects_yolo(image_file):
    image = Image.open(image_file)
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        results = model(image)
    
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return detections
 
