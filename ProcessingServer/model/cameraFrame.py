import datetime
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f" Torch cuda available: {torch.cuda.is_available()}")
torch.device(device)
model1 = YOLO('yolov8l.pt').to(device)
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model2 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

model2.to(device)
class CameraFrame:
    def __init__(self, image_data, timestamp=None):
        self.__image_data = image_data  # The raw JPEG data as a byte array
        self.__timestamp = timestamp if timestamp else datetime.datetime.now()
        # Check if CUDA is available, and set the device accordingly
    
    def process_objects(self):
        """
        Process the JPEG data.
        This could involve decoding the JPEG, analyzing the image, etc.
        """

        nparr = np.frombuffer(self.__image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert your image to a PyTorch tensor and send it to the device
        img_tensor = torch.from_numpy(img).to(device)

        # Add batch dimension if required (YOLO models typically expect batches)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        # Perform inference
        results = model1.predict(source=img)
        return results  # Implement your processing logic here


    
    def process_floor(self):
        """
        Process the JPEG data for floor estimation.
        This involves decoding the JPEG, converting it into a format suitable for the depth estimation model, etc.
        """

        # Decode the JPEG image data
        nparr = np.frombuffer(self.__image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the numpy array (image) to a PIL Image, as required by the transformers pipeline
        img_pil = Image.fromarray(img)

        inputs = processor(images=img_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model2(**inputs)
        logits = outputs.logits # shape (batch_size, num_labels, height/4, width/4)

        return logits
    
    def toCvImage(self):
        nparr = np.frombuffer(self.__image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
