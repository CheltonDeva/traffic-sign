from importlib.resources import path
from time import time
from networkx import selfloop_edges
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pyttsx3
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/best.pt', force_reload=True)
engine = pyttsx3.init()
cap = cv2.VideoCapture(0)

def score_frame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    frame = [torch.tensor(frame)]
    results = selfloop_edges.model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    cord = results.xyxyn[0][:, :-1].numpy()
    return labels, cord

while cap.isOpened():
    start=time()
    ret, frame = cap.read()
    result = model(frame)
    labels = result.xyxyn[0][:, -1].numpy()
    cord = result.xyxyn[0][:, :-1].numpy()
    n = len(labels)
    for i in range(n):
        row = cord[i]
        
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.5: 
            continue
        classes = model.names
        answer = classes[labels[i]]
        print(answer)
        engine.say(answer)
    engine.runAndWait()
    #for x in result:
     #   print(x)
    #print(result.type())
    
    cv2.imshow('Screen', frame)
    if cv2.waitKey(10) & 0xff == ord('x'):
        break
    if cv2.getWindowProperty("Screen", cv2.WND_PROP_VISIBLE) <1:
        break
    
    end = time()
    fps = 1/(end-start)
    print(fps)

cap.release()
cv2.destroyAllWindows() 