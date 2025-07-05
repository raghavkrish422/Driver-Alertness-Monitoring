import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/last.pt', force_reload=True)
buffer = []

cap = cv2.VideoCapture("http://192.168.43.233:81/stream")
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    key = str(results).split('\n')[0].split(' ')[-1]
    
    if len(buffer) != 5:
        buffer.append(key)
    else:
        buffer = buffer[1:] + [key]
    
    if len(set(buffer)) == 1 and len(buffer) == 5 and buffer[0] == "drowsy":
        print("DROWSY")
        buffer = []
        '''
        mp3_file_path = os.path.join(folder_path, random_mp3_file)
        pygame.mixer.music.load(mp3_file_path)
        pygame.mixer.music.play()
        '''
        time.sleep(5)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        #pygame.mixer.music.stop()
        break
cap.release()
cv2.destroyAllWindows()