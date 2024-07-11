import cv2
import numpy as np
from PIL import Image
from transparent_background import Remover

# Load model
remover = Remover()  # default setting
remover = Remover(mode='fast', jit=True)  # custom setting
# remover = Remover(mode='base-nightly') # nightly release checkpoint

# Usage for image
img = Image.open('/Users/toheed/PanduAI/backend/temp/man_image.jpg').convert('RGB')  # read image

# Process image with different types
process_types = [
    'rgba', 'map', 'green', 'white', 'blur',
]

for process_type in process_types:
    
    out = remover.process(img, type=process_type)
    
    out.save(f'output_{process_type}.png')  # save result with a unique filename

# Usage for video
cap = cv2.VideoCapture('/Users/toheed/PanduAI/backend/temp/man_on_phone.mp4')  # video reader for input
fps = cap.get(cv2.CAP_PROP_FPS)
writer = None

while cap.isOpened():
    ret, frame = cap.read()  # read video

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame).convert('RGB')

    if writer is None:
        writer = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (img.width, img.height))  # video writer for output

    out = remover.process(img, type='white')  # same as image, except for 'rgba' which is not for video.
    writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR))

cap.release()
writer.release()
