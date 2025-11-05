'''

Vision Guide v4.0
All rights reserved.
By Aditya Shankar Nath.

SETUP INSTRUCTIONS : 

Dependencies (IN THIS ORDER ONLY) : 
(not running the GPU) 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip3 install opencv-python ultralytics

pip3 install pyttsx3

sudo apt-get install espeak-ng

To run off GPU : Install ultralytics and opencv first, then pyttsx3.
                 The GPU version of Torch will install itself.

Instructions : 
python3 live_object_detector.py # run code
's' to hear the summary
'q' to quit

Virtual Environment (for self) : 
python3 -m venv venv
source venv/bin/activate
deactivate (when done)

'''


import cv2
from ultralytics import YOLO
import pyttsx3
import time
from collections import Counter

# Initialize the text-to-speech engine
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Failed to initialize TTS engine: {e}")
    engine = None

# Load the model.
# v4 is currently the most powerful one in storage.
# Previous generations are available in sub-directories.
model = YOLO('image_model_v4.pt')

# Open a connection to the default camera
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
except Exception as e:
    print(f"Error accessing webcam: {e}")
    exit()

# Cooldown period for speech output (in seconds)
SPEECH_COOLDOWN = 5 
last_speech_time = 0

print("Press 's' to hear a summary of detected objects.")
print("Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the current frame
    results = model(frame)

    # Get the names of all detected objects
    detected_objects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            detected_objects.append(label)

    # --- Draw bounding boxes and labels on the frame ---
    annotated_frame = results[0].plot()

    # --- Display the resulting frame ---
    cv2.imshow('VisionGuide v4.0', annotated_frame)

    # --- Keyboard controls ---
    key = cv2.waitKey(1) & 0xFF

    # If 's' is pressed, generate and speak the summary
    if key == ord('s'):
        current_time = time.time()
        if current_time - last_speech_time > SPEECH_COOLDOWN:
            if detected_objects:
                # Count the occurrences of each object
                object_counts = Counter(detected_objects)
                
                # Create the summary string
                summary_parts = []
                for obj, count in object_counts.items():
                    plural = 's' if count > 1 else ''
                    summary_parts.append(f"{count} {obj}{plural}")
                
                summary_text = "I can see " + ", ".join(summary_parts) + "."
                
                print(f"Generated Summary: {summary_text}")

                # Speak the summary
                if engine:
                    engine.say(summary_text)
                    engine.runAndWait()
                
                last_speech_time = current_time
            else:
                print("No objects detected to summarize.")
                if engine:
                    engine.say("I do not see any objects.")
                    engine.runAndWait()
        else:
            print("Speech is on cooldown to prevent spamming.")

    # If 'q' is pressed, break the loop
    if key == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()