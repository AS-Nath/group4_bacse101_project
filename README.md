# Vision Guide

The visually impaired have for long struggled under the yoke of mundane, non-smart so-called 'blind sticks'.
This project aims to change that and give the blind / visually impaired a more exciting perspective on life. 
This repository contains the backbone of the final model.

### Setup Instructions (available in the source)
If running Linux, create a virtual environment before running any ```pip3 install``` commands.

**Dependencies** (not running the GPU): 

1. ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```

2. ```pip3 install opencv-python ultralytics```

3. ```pip3 install pyttsx3```

4. ```sudo apt-get install espeak-ng``` (Linux only)

**To run off GPU** : 
Install ```ultralytics``` and ```opencv``` first, then ```pyttsx3```.
The GPU version of Torch will install itself.

### Usage Instructions 
```python3 live_object_detector.py```

Press ```s``` to hear a summary of all objects detected.

Press ```q``` to quit the program. 