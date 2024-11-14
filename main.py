import torch
import cv2
import pyttsx3
import paho.mqtt.client as mqtt
from time import time

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# MQTT Setup (Optional)
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/best.pt', force_reload=True)  # Use your custom model

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for Raspberry Pi Camera Module

def preprocess_frame(frame):
    # Resize and normalize the frame for the model
    frame_resized = cv2.resize(frame, (640, 640))  # YOLOv5 expects 640x640 input
    return frame_resized / 255.0  # Normalize pixel values

def detect_traffic_signs(frame):
    # Preprocess the frame
    frame_preprocessed = preprocess_frame(frame)
    # Convert to tensor and add batch dimension
    frame_tensor = torch.tensor([frame_preprocessed]).permute(0, 3, 1, 2)

    # Run model inference
    results = model(frame_tensor)
    labels = results.xyxyn[0][:, -1].numpy()  # Get class labels
    coords = results.xyxyn[0][:, :-1].numpy()  # Get bounding box coordinates
    return labels, coords

def display_results(frame, labels, coords):
    for i, label in enumerate(labels):
        x_min, y_min, x_max, y_max, confidence = coords[i]

        if confidence > 0.5:  # Confidence threshold
            # Draw bounding box and label
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            label_name = model.names[int(label)]
            cv2.putText(frame, f"{label_name} {confidence:.2f}", 
                        (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Text-to-speech output
            engine.say(label_name)
            engine.runAndWait()

            # Send detection via MQTT
            client.publish("traffic_sign_detection", label_name)

    return frame

while cap.isOpened():
    start = time()
    ret, frame = cap.read()
    if not ret:
        break
    
    labels, coords = detect_traffic_signs(frame)
    frame = display_results(frame, labels, coords)

    # Show the frame
    cv2.imshow("Traffic Sign Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    end = time()
    fps = 1 / (end - start)
    print(f"FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()
