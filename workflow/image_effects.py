import cv2
import numpy as np
import requests
import base64
import os
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, ImageSequenceClip, VideoClip
from utils.util import  effect_images, transition_images
import random
import re

def sort_files_numerically(files):
    # Function to extract the numeric part of the filename and sort accordingly
    def extract_number(f):
        match = re.search(r'_(\d+)', f)
        return int(match.group(1)) if match else float('inf')
    return sorted(files, key=extract_number)

def generate_fade_in_out(video1, video2, output_video_path, frames=30, fps=10, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        for _ in range(frame_repeat):
            out.write(frame1)

    if not cap2.isOpened():
        print("Error: Unable to open the second video.")
        return

    for i in range(frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            frame1 = cv2.resize(frame1, (width, height))
        else:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)

        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
        else:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        alpha = i / frames
        beta = 1.0 - alpha
        fade_frame = cv2.addWeighted(frame1, beta, frame2, alpha, 0.0)

        for _ in range(frame_repeat):
            out.write(fade_frame)

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()

def pan_image_horizontal(image, crop_percent=0.75, pan_step=1, direction='left_to_right'):

    h, w = image.shape[:2]
    
    # Compute the new width after cropping
    new_w = int(w * crop_percent)
    
    # Pan from left to right
    if direction == 'left_to_right':
        left = pan_step
        right = left + new_w
        panned_image = image[:, left:right]
    
    # Pan from right to left
    elif direction == 'right_to_left':
        right = w - pan_step
        left = right - new_w
        panned_image = image[:, left:right]
    
    else:
        raise ValueError("Invalid direction. Use 'left_to_right' or 'right_to_left'.")
    
    # Resize the cropped image back to the original width
    panned_image = cv2.resize(panned_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return panned_image

def pan_image_vertical(image, crop_percent=0.75, pan_step=1, direction='top_to_bottom'):
 
    h, w = image.shape[:2]
    
    # Compute the new height after cropping
    new_h = int(h * crop_percent)
    
    # Pan from top to bottom
    if direction == 'top_to_bottom':
        top = pan_step
        bottom = top + new_h
        panned_image = image[top:bottom, :]
    
    # Pan from bottom to top
    elif direction == 'bottom_to_top':
        bottom = h - pan_step
        top = bottom - new_h
        panned_image = image[top:bottom, :]
    
    else:
        raise ValueError("Invalid direction. Use 'top_to_bottom' or 'bottom_to_top'.")
    
    # Resize the cropped image back to the original height
    panned_image = cv2.resize(panned_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return panned_image

def pan_image_diagonal(image, crop_percent=0.75, pan_step=1, direction='top_left_to_bottom_right'):

    h, w = image.shape[:2]

    # Compute the new width and height after cropping
    new_h = int(h * crop_percent)
    new_w = int(w * crop_percent)

    # Diagonal panning logic
    if direction == 'top_left_to_bottom_right':
        top = pan_step
        left = pan_step
        bottom = top + new_h
        right = left + new_w
    elif direction == 'bottom_right_to_top_left':
        bottom = h - pan_step
        right = w - pan_step
        top = bottom - new_h
        left = right - new_w
    elif direction == 'bottom_left_to_top_right':
        bottom = h - pan_step
        left = pan_step
        top = bottom - new_h
        right = left + new_w
    elif direction == 'top_right_to_bottom_left':
        top = pan_step
        right = w - pan_step
        bottom = top + new_h
        left = right - new_w
    else:
        raise ValueError("Invalid direction. Use 'top_left_to_bottom_right', 'bottom_right_to_top_left', 'bottom_left_to_top_right', or 'top_right_to_bottom_left'.")

    panned_image = image[top:bottom, left:right]
    
    # Resize the cropped image back to the original dimensions
    panned_image = cv2.resize(panned_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return panned_image

def zoom_in(image, zoom_step=1.1):
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Compute the bounding box of the zoomed-in image
    new_w, new_h = int(w / zoom_step), int(h / zoom_step)
    left, right = max(center_x - new_w // 2, 0), min(center_x + new_w // 2, w)
    top, bottom = max(center_y - new_h // 2, 0), min(center_y + new_h // 2, h)
    
    # Crop and resize the image
    cropped = image[top:bottom, left:right]
    zoomed_in = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed_in

def generate_zoom_video(input_image_path, output_video_path, zoom_step=1.005, frames=100, fps=30, frame_repeat=1):
    # Load the image
    image = cv2.imread(input_image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {input_image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    # Check if the VideoWriter is opened successfully
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return
    
    current_image = image.copy()
    for i in range(frames):
        for _ in range(frame_repeat):
            out.write(current_image)
        current_image = zoom_in(current_image, zoom_step)
    
    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_zoom_out_video(image_path, output_video_path, zoom_step=1.1, frames=100, fps=30, frame_repeat=1):
      # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    current_image = image.copy()
    zoomed_images = []
    for i in range(frames):
        current_image = zoom_in(current_image, zoom_step)
        zoomed_images.append(current_image)
    
    # Write the frames in reverse order
    for current_image in reversed(zoomed_images):
        for _ in range(frame_repeat):
            out.write(current_image)
    
    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_pan_left_to_right_video(image_path, output_video_path, zoom_step=1.1, crop_percent=0.75, frames=60, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Zoom in on the image
    current_image = image.copy()
    current_image = zoom_in(current_image, zoom_step)

    # Calculate the step size for panning
    pan_range = w - int(w * crop_percent)
    step_size = pan_range // frames
    
    # Pan from left to right
    for i in range(frames):
        for _ in range(frame_repeat):
            pan_step = i * step_size
            panned_image = pan_image_horizontal(current_image, crop_percent=crop_percent, pan_step=pan_step, direction='left_to_right')
            out.write(panned_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_pan_right_to_left_video(image_path, output_video_path, zoom_step=1.1, crop_percent=0.75, frames=60, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Zoom in on the image
    current_image = image.copy()
    current_image = zoom_in(current_image, zoom_step)

    # Calculate the step size for panning
    pan_range = w - int(w * crop_percent)
    step_size = pan_range // frames
    
    # Pan from right to left
    for i in range(frames):
        for _ in range(frame_repeat):
            pan_step = i * step_size
            panned_image = pan_image_horizontal(current_image, crop_percent=crop_percent, pan_step=pan_step, direction='right_to_left')
            out.write(panned_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_pan_top_to_bottom_video(image_path, output_video_path, zoom_step=1.1, crop_percent=0.75, frames=60, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Zoom in on the image
    current_image = image.copy()
    current_image = zoom_in(current_image, zoom_step)

    # Calculate the step size for panning
    pan_range = h - int(h * crop_percent)
    step_size = pan_range // frames
    
    # Pan from top to bottom
    for i in range(frames):
        for _ in range(frame_repeat):
            pan_step = i * step_size
            panned_image = pan_image_vertical(current_image, crop_percent=crop_percent, pan_step=pan_step, direction='top_to_bottom')
            out.write(panned_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_pan_bottom_to_top_video(image_path, output_video_path, zoom_step=1.1, crop_percent=0.75, frames=60, fps=10, frame_repeat=1):
    print(type (image_path))
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Zoom in on the image
    current_image = image.copy()
    current_image = zoom_in(current_image, zoom_step)

    # Calculate the step size for panning
    pan_range = h - int(h * crop_percent)
    step_size = pan_range // frames
    
    # Pan from bottom to top
    for i in range(frames):
        for _ in range(frame_repeat):
            pan_step = i * step_size
            panned_image = pan_image_vertical(current_image, crop_percent=crop_percent, pan_step=pan_step, direction='bottom_to_top')
            out.write(panned_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_pan_top_left_to_bottom_right_video(image_path, output_video_path, zoom_step=1.1, crop_percent=0.75, frames=60, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Zoom in on the image
    current_image = image.copy()
    current_image = zoom_in(current_image, zoom_step)

    # Calculate the step size for panning
    pan_range = min(h, w) - int(min(h, w) * crop_percent)
    step_size = pan_range // frames
    
    # Pan from top left to bottom right
    for i in range(frames):
        for _ in range(frame_repeat):
            pan_step = i * step_size
            panned_image = pan_image_diagonal(current_image, crop_percent=crop_percent, pan_step=pan_step, direction='top_left_to_bottom_right')
            out.write(panned_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_pan_bottom_right_to_top_left_video(image_path, output_video_path, zoom_step=1.1, crop_percent=0.75, frames=60, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Zoom in on the image
    current_image = image.copy()
    current_image = zoom_in(current_image, zoom_step)

    # Calculate the step size for panning
    pan_range = min(h, w) - int(min(h, w) * crop_percent)
    step_size = pan_range // frames
    
    # Pan from bottom right to top left
    for i in range(frames):
        for _ in range(frame_repeat):
            pan_step = i * step_size
            panned_image = pan_image_diagonal(current_image, crop_percent=crop_percent, pan_step=pan_step, direction='bottom_right_to_top_left')
            out.write(panned_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_pan_bottom_left_to_top_right_video(image_path, output_video_path, zoom_step=1.1, crop_percent=0.75, frames=60, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Zoom in on the image
    current_image = image.copy()
    current_image = zoom_in(current_image, zoom_step)

    # Calculate the step size for panning
    pan_range = min(h, w) - int(min(h, w) * crop_percent)
    step_size = pan_range // frames
    
    # Pan from bottom left to top right
    for i in range(frames):
        for _ in range(frame_repeat):
            pan_step = i * step_size
            panned_image = pan_image_diagonal(current_image, crop_percent=crop_percent, pan_step=pan_step, direction='bottom_left_to_top_right')
            out.write(panned_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_pan_top_right_to_bottom_left_video(image_path, output_video_path, zoom_step=1.1, crop_percent=0.75, frames=60, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Zoom in on the image
    current_image = image.copy()
    current_image = zoom_in(current_image, zoom_step)

    # Calculate the step size for panning
    pan_range = min(h, w) - int(min(h, w) * crop_percent)
    step_size = pan_range // frames
    
    # Pan from top right to bottom left
    for i in range(frames):
        for _ in range(frame_repeat):
            pan_step = i * step_size
            panned_image = pan_image_diagonal(current_image, crop_percent=crop_percent, pan_step=pan_step, direction='top_right_to_bottom_left')
            out.write(panned_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_zoom_in_rotate_clockwise_video(image_path, output_video_path, zoom_step=1.1, angle_step=5, frames=30, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Generate frames for zoom in and rotate clockwise
    for i in range(frames):
        for _ in range(frame_repeat):
            angle = -i * angle_step  # Negative for clockwise rotation
            zoom = zoom_step ** i
            M = cv2.getRotationMatrix2D(center, angle, zoom)
            rotated_zoomed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            out.write(rotated_zoomed_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_zoom_in_rotate_anticlockwise_video(image_path, output_video_path, zoom_step=1.1, angle_step=5, frames=30, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Generate frames for zoom in and rotate anticlockwise
    for i in range(frames):
        for _ in range(frame_repeat):
            angle = i * angle_step  # Positive for anticlockwise rotation
            zoom = zoom_step ** i
            M = cv2.getRotationMatrix2D(center, angle, zoom)
            rotated_zoomed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            out.write(rotated_zoomed_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_zoom_out_rotate_anticlockwise_video(image_path, output_video_path, zoom_step=0.9, angle_step=5, frames=30, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Generate frames for zoom out and rotate anticlockwise
    for i in range(frames):
        for _ in range(frame_repeat):
            angle = i * angle_step  # Positive for anticlockwise rotation
            zoom = zoom_step ** i
            M = cv2.getRotationMatrix2D(center, angle, zoom)
            rotated_zoomed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            out.write(rotated_zoomed_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_zoom_out_rotate_clockwise_video(image_path, output_video_path, zoom_step=0.9, angle_step=5, frames=30, fps=10, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Generate frames for zoom out and rotate clockwise
    for i in range(frames):
        for _ in range(frame_repeat):
            angle = -i * angle_step  # Negative for clockwise rotation
            zoom = zoom_step ** i
            M = cv2.getRotationMatrix2D(center, angle, zoom)
            rotated_zoomed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            out.write(rotated_zoomed_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

def generate_ken_burns_effect_video(image_path, output_video_path, start_zoom=1.0, end_zoom=2.0, pan_direction='right', frames=60, fps=30, frame_repeat=1):
     # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h, w = image.shape[:2]

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Calculate the step size for zooming
    zoom_step = (end_zoom - start_zoom) / frames

    # Generate the Ken Burns effect frames
    for i in range(frames):
        for _ in range(frame_repeat):
            # Calculate current zoom level
            current_zoom = start_zoom + i * zoom_step

            # Calculate the region of interest (ROI) for cropping
            roi_w = int(w / current_zoom)
            roi_h = int(h / current_zoom)

            # Calculate the top-left corner of the ROI for panning
            if pan_direction == 'right':
                x_offset = int((w - roi_w) * (i / frames))
                y_offset = 0
            elif pan_direction == 'left':
                x_offset = int((w - roi_w) * ((frames - i) / frames))
                y_offset = 0
            elif pan_direction == 'down':
                x_offset = 0
                y_offset = int((h - roi_h) * (i / frames))
            elif pan_direction == 'up':
                x_offset = 0
                y_offset = int((h - roi_h) * ((frames - i) / frames))
            else:
                x_offset = int((w - roi_w) * (i / frames))
                y_offset = int((h - roi_h) * (i / frames))

            # Crop and resize the image to simulate zoom and pan
            cropped_image = image[y_offset:y_offset + roi_h, x_offset:x_offset + roi_w]
            resized_image = cv2.resize(cropped_image, (w, h))

            out.write(resized_image)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

# MoviePY tansitions

def crossfade_transition(clip1, clip2, duration=0.5):
    return concatenate_videoclips([clip1.crossfadeout(duration), clip2.crossfadein(duration)], method="compose")

def fadeinout_transition(clip1, clip2, duration=0.5):
    return concatenate_videoclips([clip1.fadeout(duration), clip2.fadein(duration)], method="compose")

def slide_transition(clip1, clip2, duration=0.5):
    return concatenate_videoclips([clip1, clip2.set_start(clip1.duration).set_position(lambda t: ('center', -clip2.h + (clip2.h / duration) * t))], method="compose")

def generate_wipe_bottom_to_top(video1, video2, output_video_path, frames=7, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        for _ in range(frame_repeat):
            out.write(frame1)

    if not cap2.isOpened():
        print("Error: Unable to open the second video.")
        return

    for i in range(frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            frame1 = cv2.resize(frame1, (width, height))
        else:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)

        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
        else:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        y = int(height * (i / frames))
        wipe_frame = np.zeros_like(frame1)
        wipe_frame[y:, :] = frame1[:height-y, :]
        wipe_frame[:y, :] = frame2[height-y:, :]

        for _ in range(frame_repeat):
            out.write(wipe_frame)

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()

def generate_wipe_left_to_right(video1, video2, output_video_path, frames=7, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        for _ in range(frame_repeat):
            out.write(frame1)

    if not cap2.isOpened():
        print("Error: Unable to open the second video.")
        return

    for i in range(frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            frame1 = cv2.resize(frame1, (width, height))
        else:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)

        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
        else:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        x = int(width * (i / frames))
        wipe_frame = np.zeros_like(frame1)
        wipe_frame[:, :x] = frame2[:, :x]
        wipe_frame[:, x:] = frame1[:, :width-x]

        for _ in range(frame_repeat):
            out.write(wipe_frame)

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()

def generate_wipe_right_to_left(video1, video2, output_video_path, frames=7, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        for _ in range(frame_repeat):
            out.write(frame1)

    if not cap2.isOpened():
        print("Error: Unable to open the second video.")
        return

    for i in range(frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            frame1 = cv2.resize(frame1, (width, height))
        else:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)

        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
        else:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        x = int(width * (i / frames))
        wipe_frame = np.zeros_like(frame1)
        wipe_frame[:, :width-x] = frame1[:, x:]
        wipe_frame[:, width-x:] = frame2[:, :x]

        for _ in range(frame_repeat):
            out.write(wipe_frame)

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()

def generate_horizontal_stripes11s(video1, video2, transition_frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Apply the horizontal stripes transition
    num_stripes = 8
    stripe_height = height // num_stripes
    transition_step = max(1, transition_frames // num_stripes)

    for frame_idx in range(transition_frames):
        # Create a mask
        mask = np.zeros((height, width), dtype=np.uint8)

        stripes_to_reveal = frame_idx // transition_step
        for y in range(stripes_to_reveal):
            mask[y * stripe_height:(y + 1) * stripe_height, :] = 255

        mask = cv2.merge([mask, mask, mask])

        img1_part = cv2.bitwise_and(last_frame_first_video, cv2.bitwise_not(mask))
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_vertical_stripes(video1, video2, transition_frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Apply the vertical stripes transition
    num_stripes = 8
    stripe_width = width // num_stripes
    transition_step = max(1, transition_frames // num_stripes)

    for frame_idx in range(transition_frames):
        # Create a mask
        mask = np.zeros((height, width), dtype=np.uint8)

        stripes_to_reveal = frame_idx // transition_step
        for x in range(stripes_to_reveal):
            mask[:, x * stripe_width:(x + 1) * stripe_width] = 255

        mask = cv2.merge([mask, mask, mask])

        img1_part = cv2.bitwise_and(last_frame_first_video, cv2.bitwise_not(mask))
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_box_inward(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the inward box transition effect
    for i in range(frames):
        # Calculate the size of the revealing box
        box_size = int((1 - (i / frames)) * width)  # Width and height decrease
        x_start = (width - box_size) // 2
        y_start = (height - box_size) // 2

        # Create a mask with a rectangle
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y_start:y_start + box_size, x_start:x_start + box_size] = 255
        mask = cv2.merge([mask, mask, mask])  # Ensure mask has 3 channels

        img1_part = cv2.bitwise_and(last_frame_first_video, cv2.bitwise_not(mask))
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_box_outward(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the outward box transition effect
    for i in range(frames):
        # Calculate the size of the revealing box
        box_size = int((i / frames) * width)  # Width and height increase
        x_start = (width - box_size) // 2
        y_start = (height - box_size) // 2

        # Create a mask with a rectangle
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y_start:y_start + box_size, x_start:x_start + box_size] = 255
        mask = cv2.merge([mask, mask, mask])  # Ensure mask has 3 channels

        img1_part = cv2.bitwise_and(last_frame_first_video, cv2.bitwise_not(mask))
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_horizontal_sliding_door(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the horizontal sliding door transition effect
    for i in range(frames):
        offset = int((i / frames) * (width // 2))

        left_part = last_frame_first_video[:, :width // 2 - offset]
        right_part = last_frame_first_video[:, width // 2 + offset:]
        middle_part = first_frame_second_video_resized[:, width // 2 - offset:width // 2 + offset]

        combined_frame = np.hstack((left_part, middle_part, right_part))

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_vertical_sliding_door(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the vertical sliding door transition effect
    for i in range(frames):
        offset = int((i / frames) * (height // 2))

        top_part = last_frame_first_video[:height // 2 - offset, :]
        bottom_part = last_frame_first_video[height // 2 + offset:, :]
        middle_part = first_frame_second_video_resized[height // 2 - offset:height // 2 + offset, :]

        combined_frame = np.vstack((top_part, middle_part, bottom_part))

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_diagonal_sliding_door_tl_br(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the diagonal sliding door transition effect
    for i in range(frames):
        offset_x = int((i / frames) * (width // 2))
        offset_y = int((i / frames) * (height // 2))

        mask = np.zeros_like(last_frame_first_video)

        # Top-left part
        mask[:height // 2 - offset_y, :width // 2 - offset_x] = last_frame_first_video[:height // 2 - offset_y, :width // 2 - offset_x]
        
        # Bottom-right part
        mask[height // 2 + offset_y:, width // 2 + offset_x:] = last_frame_first_video[height // 2 + offset_y:, width // 2 + offset_x:]

        # Middle part
        mask[height // 2 - offset_y:height // 2 + offset_y, width // 2 - offset_x:width // 2 + offset_x] = first_frame_second_video_resized[height // 2 - offset_y:height // 2 + offset_y, width // 2 - offset_x:width // 2 + offset_x]

        for _ in range(frame_repeat):
            frames_list.append(mask)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_diagonal_sliding_door_bl_tr(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the diagonal sliding door transition effect
    for i in range(frames):
        offset_x = int((i / frames) * (width // 2))
        offset_y = int((i / frames) * (height // 2))

        mask = np.zeros_like(last_frame_first_video)

        # Bottom-left part
        mask[height // 2 + offset_y:, :width // 2 - offset_x] = last_frame_first_video[height // 2 + offset_y:, :width // 2 - offset_x]
        
        # Top-right part
        mask[:height // 2 - offset_y, width // 2 + offset_x:] = last_frame_first_video[:height // 2 - offset_y, width // 2 + offset_x:]

        # Middle part
        mask[height // 2 - offset_y:height // 2 + offset_y, width // 2 - offset_x:width // 2 + offset_x] = first_frame_second_video_resized[height // 2 - offset_y:height // 2 + offset_y, width // 2 - offset_x:width // 2 + offset_x]

        for _ in range(frame_repeat):
            frames_list.append(mask)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_minimize_to_topleft(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the minimizing to top-left transition effect
    for i in range(frames):
        scale = 1 - (i / frames)
        scaled_frame = cv2.resize(last_frame_first_video, (int(width * scale), int(height * scale)))

        mask = np.zeros_like(last_frame_first_video)
        mask[:int(height * scale), :int(width * scale)] = scaled_frame

        combined_frame = np.where(mask == 0, first_frame_second_video_resized, mask).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_minimize_to_topright(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the minimizing to top-right transition effect
    for i in range(frames):
        scale = 1 - (i / frames)
        scaled_frame = cv2.resize(last_frame_first_video, (int(width * scale), int(height * scale)))

        mask = np.zeros_like(last_frame_first_video)
        mask[:int(height * scale), width - int(width * scale):] = scaled_frame

        combined_frame = np.where(mask == 0, first_frame_second_video_resized, mask).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_minimize_to_bottomleft(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the minimizing to bottom-left transition effect
    for i in range(frames):
        scale = 1 - (i / frames)
        scaled_frame = cv2.resize(last_frame_first_video, (int(width * scale), int(height * scale)))

        mask = np.zeros_like(last_frame_first_video)
        mask[height - int(height * scale):, :int(width * scale)] = scaled_frame

        combined_frame = np.where(mask == 0, first_frame_second_video_resized, mask).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_minimize_to_bottomright(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the minimizing to bottom-right transition effect
    for i in range(frames):
        scale = 1 - (i / frames)
        scaled_frame = cv2.resize(last_frame_first_video, (int(width * scale), int(height * scale)))

        mask = np.zeros_like(last_frame_first_video)
        mask[height - int(height * scale):, width - int(width * scale):] = scaled_frame

        combined_frame = np.where(mask == 0, first_frame_second_video_resized, mask).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_halfstripe_horizontal(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the two stripe horizontal transition effect
    for i in range(frames):
        offset = int(width * (i / frames))
        top_half = last_frame_first_video[:height // 2, :]
        bottom_half = last_frame_first_video[height // 2:, :]
        
        top_frame = np.zeros_like(first_frame_second_video_resized)
        bottom_frame = np.zeros_like(first_frame_second_video_resized)
        
        top_frame[:height // 2, :width - offset] = first_frame_second_video_resized[:height // 2, offset:]
        bottom_frame[height // 2:, :width - offset] = first_frame_second_video_resized[height // 2:, offset:]

        combined_frame = top_frame + bottom_frame

        combined_frame = np.where(combined_frame == 0, last_frame_first_video, combined_frame).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_halfstripe_vertical(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the two stripe vertical transition effect
    for i in range(frames):
        offset = int(height * (i / frames))
        left_half = last_frame_first_video[:, :width // 2]
        right_half = last_frame_first_video[:, width // 2:]
        
        left_frame = np.zeros_like(first_frame_second_video_resized)
        right_frame = np.zeros_like(first_frame_second_video_resized)
        
        left_frame[:height - offset, :width // 2] = first_frame_second_video_resized[offset:, :width // 2]
        right_frame[:height - offset, width // 2:] = first_frame_second_video_resized[offset:, width // 2:]

        combined_frame = np.hstack((left_frame, right_frame))

        combined_frame = np.where(combined_frame == 0, last_frame_first_video, combined_frame).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_wipe_top_to_bottom(clip1, clip2, frames=7, fps=30, frame_repeat=1):
    width, height = clip1.size

    def make_frame(t):
        if t < clip1.duration:
            return clip1.get_frame(t)
        else:
            t_transition = t - clip1.duration
            frame1 = clip1.get_frame(clip1.duration - 1 / fps)
            frame2 = clip2.get_frame(0)
            
            i = int((t_transition * fps) / frame_repeat)
            if i >= frames:
                return frame2
            
            y = int(height * (i / frames))

            wipe_frame = np.zeros_like(frame1)
            wipe_frame[:height - y, :] = frame1[y:, :]
            wipe_frame[height - y:, :] = frame2[:y, :]

            return wipe_frame

    duration = clip1.duration + (frames / fps)
    wipe_clip = VideoClip(make_frame, duration=duration).set_fps(fps)

    final_clip = concatenate_videoclips([wipe_clip, clip2.set_start(clip1.duration + (frames / fps))])

    return final_clip

def generate_outward_vignette_transition(clip1, clip2, frames=18, fps=30, frame_repeat=1):
    width, height = clip1.w, clip1.h

    frames_list = []

    # Get the last frame of the first video
    last_frame_first_video = clip1.get_frame(clip1.duration)

    # Get the first frame of the second video
    first_frame_second_video = clip2.get_frame(0)

    # Create the outward vignette transition frames
    for i in range(frames):
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (width//2, height//2), int((width//2) * (i+1) / frames), 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = cv2.merge([mask, mask, mask])

        img1_part = cv2.bitwise_and(last_frame_first_video, mask)
        inverted_mask = cv2.bitwise_not(mask)
        img2_part = cv2.bitwise_and(first_frame_second_video, inverted_mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Create the transition clip
    transition_clip = ImageSequenceClip(frames_list, fps=fps)

    # Concatenate the clips
    final_clip = concatenate_videoclips([clip1, transition_clip, clip2])

    return final_clip

def generate_inward_vignette_transition(video1, video2, frames=18, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture the last frame of the first video
    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the inward vignette transition frames
    for i in range(frames):
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (width//2, height//2), int((width//2) * (frames-i) / frames), 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = cv2.merge([mask, mask, mask])

        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        inverted_mask = cv2.bitwise_not(mask)
        img1_part = cv2.bitwise_and(last_frame_first_video, inverted_mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Add all frames from the second video
    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_horizontal_stripes(clip1, clip2, frames=20, fps=30, frame_repeat=1):
    width, height = clip1.size

    def make_frame(t):
        if t < clip1.duration:
            return clip1.get_frame(t)
        else:
            t_transition = t - clip1.duration
            frame1 = clip1.get_frame(clip1.duration - 1 / fps)
            frame2 = clip2.get_frame(0)
            
            frame_idx = int(t_transition * fps)
            if frame_idx >= frames:
                return frame2
            
            num_stripes = 8
            stripe_height = height // num_stripes
            transition_step = max(1, frames // num_stripes)

            mask = np.zeros((height, width), dtype=np.uint8)
            stripes_to_reveal = frame_idx // transition_step
            for y in range(stripes_to_reveal):
                mask[y * stripe_height:(y + 1) * stripe_height, :] = 255

            mask = cv2.merge([mask, mask, mask])
            img1_part = cv2.bitwise_and(frame1, cv2.bitwise_not(mask))
            img2_part = cv2.bitwise_and(frame2, mask)
            blended_image = cv2.add(img1_part, img2_part)

            return blended_image

    duration = clip1.duration + (frames / fps)
    wipe_clip = VideoClip(make_frame, duration=duration).set_fps(fps)

    final_clip = concatenate_videoclips([wipe_clip, clip2.set_start(clip1.duration + (frames / fps))])

    return final_clip









def add_transitions(clips, fade_duration=0.5):
    # Apply fade in and fade out transitions to every other clip
    clips_with_transitions = []
    for i, clip in enumerate(clips):
        if i % 2 == 1:  # Apply transitions to every other clip (1, 3, 5, ...)
            clip = clip.crossfadein(fade_duration).crossfadeout(fade_duration)
        clips_with_transitions.append(clip)

    # Concatenate the video clips with crossfade transitions
    final_clip = concatenate_videoclips(clips_with_transitions, method="compose")
    return final_clip

def apply_effect_to_generated_image(input_image_folder, output_video_path, effect_function, **effect_params):
    for filename in os.listdir(input_image_folder):
        # Construct the full file path
        file_path = os.path.join(input_image_folder, filename)

        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # Read the image
            image = cv2.imread(file_path)

            # Apply the effect function to the image
            effect_function(image, output_video_path, **effect_params)
            print(f"Effect applied and video saved as {output_video_path}")

def process_multiple_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.mp4'
            output_path = os.path.join(output_folder, output_filename)
            # Effects
            effect_function_name = random.choice(list(effect_images.values()))
            effect_function = globals()[effect_function_name]
            print("Applying effects:", effect_function)
            effect_function(input_path, output_path)

    # # Delete all files in the input folder
    # image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # for f in image_files:
    #     os.remove(f)

def process_image(input_path, output_folder, effect_function, **effect_kwargs):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.basename(input_path)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        output_filename = os.path.splitext(filename)[0] + '.mp4'
        output_path = os.path.join(output_folder, output_filename)
        effect_function(input_path, output_path, **effect_kwargs)
    else:
        print(f"Unsupported file format: {filename}")

def process_single_image(input_image_path, output_folder, effect_function_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.basename(input_image_path)
    output_filename = os.path.splitext(filename)[0] + '.mp4'
    output_path = os.path.join(output_folder, output_filename)

    # Apply effects
    # effect_function_name = random.choice(list(effect_images.values()))
    effect_function = globals()[effect_function_name]
    print("Applying effects:", effect_function_name)
    effect_function(input_image_path, output_path)
    
    # remove image
    # os.remove(input_image_path)

def stitch_videos_with_transition1(input_video_folder, output_video_path, transition_function, transition_params):
    # Get list of videos in the input folder
    video_files = sorted([f for f in os.listdir(input_video_folder) if os.path.isfile(os.path.join(input_video_folder, f))])
    
    if not video_files:
        print(f"Error: No video files found in the folder {input_video_folder}")
        return

    # Initialize the video writer
    first_video_path = os.path.join(input_video_folder, video_files[0])
    cap = cv2.VideoCapture(first_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open or find the video {first_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    transition_video_paths = []

    for i in range(len(video_files) - 1):
        current_video = os.path.join(input_video_folder, video_files[i])
        next_video = os.path.join(input_video_folder, video_files[i + 1])
        transition_video_path = os.path.join(input_video_folder, f'image_{i}.mp4')
        transition_video_paths.append(transition_video_path)

        # Apply the transition effect between current and next video
        transition_function(current_video, next_video, transition_video_path, **transition_params)

        # Read current video and write frames to the output video
        cap = cv2.VideoCapture(current_video)
        if not cap.isOpened():
            print(f"Error: Could not open or find the video {current_video}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

        # Read transition video and write frames to the output video
        cap = cv2.VideoCapture(transition_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open or find the transition video {transition_video_path}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    # Write the last video
    last_video = os.path.join(input_video_folder, video_files[-1])
    cap = cv2.VideoCapture(last_video)
    if not cap.isOpened():
        print(f"Error: Could not open or find the video {last_video}")
        out.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()

    # Release the video writer
    out.release()

    # Remove temporary transition videos
    for transition_video_path in transition_video_paths:
        try:
            os.remove(transition_video_path)
        except OSError as e:
            print(f"Error deleting transition video {transition_video_path}: {e}")

    print(f"Stitched video saved as {output_video_path}")

def stitch_videos(input_folder, output_folder, output_filename, fade_duration=1):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all video files in the input folder
    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Sort the video files (optional, depending on how you want to order the clips)
    video_files.sort()

    # Load all video clips
    clips = [VideoFileClip(f) for f in video_files]

    # Add transitions between the clips
    final_clip = add_transitions(clips, fade_duration)

    # final_clip = concatenate_videoclips(clips, method='compose')

    # Define the output path
    output_path = os.path.join(output_folder, output_filename)

    # Write the final video to the output path
    final_clip.write_videofile(output_path, codec='libx264')

    # Close all the clips
    for clip in clips:
        clip.close()

    # Delete all files in the input folder
    for f in video_files:
        os.remove(f)

def stitch_videos_with_transition(video_paths, output_path, transition_function_name):

    video_files = [os.path.join(video_paths, f) for f in os.listdir(video_paths) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    # Sort video files numerically
    sorted_video_files = sort_files_numerically(video_files)
    print("Sorted video files:", sorted_video_files)

    if not video_files:
        raise ValueError("No video files found in the provided folder.")
    
    clips = [VideoFileClip(video_file) for video_file in sorted_video_files]
    clip1 = clips[0]
    
    for i in range(1, len(clips)):
        # Effects
        # print("Applying random transition effect..."    )
        # transition_function_name = random.choice(list(transition_images.values()))
        print(f"Applying transition:", {transition_function_name})
        transition_function = globals()[transition_function_name]
        final_clip = fadeinout_transition(clip1, clips[i])
        # final_clip = transition_function(str(clip1), str(clips[i]))
    
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Delete all files in the input folder
    # for f in video_files:
    #     os.remove(f)

#
# engine_id = "stable-diffusion-v1-6"
# api_key = os.getenv("STABILITY_SECRET_KEY")
# generate_image_engine(engine_id, api_key, prompt)

#input_image_path = 'wallpaper.jpg'
#input_image_path1 = 'wallpaper.jpg'
#input_image_path2 = 'screenshot.png'
#output_zoom_in_path = 'zoom_in_video.mp4'
#output_zoom_out_path = 'zoom_out_video.mp4'
#output_pan_left_to_right_path = 'pan_left_to_right_video.mp4'
#output_pan_right_to_left_path = 'pan_right_to_left_video.mp4'
#output_pan_top_to_bottom_path = 'pan_top_to_bottom_video.mp4'
#output_pan_bottom_to_top_path = 'pan_bottom_to_top_video.mp4'
#output_pan_top_left_to_bottom_right_path = 'pan_top_left_to_bottom_right_video.mp4'
#output_pan_bottom_right_to_top_left_path = 'pan_bottom_right_to_top_left_video.mp4'
#output_pan_bottom_left_to_top_right_path = 'pan_bottom_left_to_top_right_video.mp4'
#output_pan_top_right_to_bottom_left_path = 'pan_top_right_to_bottom_left_video.mp4'
#output_zoom_in_rotate_clockwise_path = 'zoom_in_rotate_clockwise_video.mp4'
#output_zoom_in_rotate_anticlockwise_path = 'zoom_in_rotate_anticlockwise_video.mp4'
#output_zoom_out_rotate_anticlockwise_path = 'zoom_out_rotate_anticlockwise_video.mp4'
#output_zoom_out_rotate_clockwise_path = 'zoom_out_rotate_clockwise_video.mp4'
#output_outward_vignette_path = 'outward_vignette_video.mp4'
#output_inward_vignette_path = 'inward_vignette_video.mp4'
#output_ken_burns_effect_path = 'ken_burns_effect_video.mp4'




#generate_zoom_video(input_image_path, output_zoom_in_path, zoom_step=1.005, frame_repeat=1)

#generate_zoom_out_video(input_image_path, output_zoom_out_path, zoom_step=1.005, frame_repeat=1)

#generate_pan_left_to_right_video(input_image_path, output_pan_left_to_right_path, zoom_step=1.005, crop_percent=0.75, frames=100, fps=30, frame_repeat=1)

#generate_pan_right_to_left_video(input_image_path, output_pan_right_to_left_path, zoom_step=1.005, crop_percent=0.75, frames=100, fps=30, frame_repeat=1)

#generate_pan_top_to_bottom_video(input_image_path, output_pan_top_to_bottom_path, zoom_step=1.005, crop_percent=0.75, frames=100, fps=30, frame_repeat=1)

#generate_pan_bottom_to_top_video(input_image_path, output_pan_bottom_to_top_path, zoom_step=1.1, crop_percent=0.75, frames=100, fps=30, frame_repeat=1)

#generate_pan_top_left_to_bottom_right_video(input_image_path, output_pan_top_left_to_bottom_right_path, zoom_step=1.005, crop_percent=0.75, frames=100, fps=30, frame_repeat=1)

#generate_pan_bottom_right_to_top_left_video(input_image_path, output_pan_bottom_right_to_top_left_path, zoom_step=1.005, crop_percent=0.75, frames=100, fps=30, frame_repeat=1)

#generate_pan_bottom_left_to_top_right_video(input_image_path, output_pan_bottom_left_to_top_right_path, zoom_step=1.005, crop_percent=0.75, frames=100, fps=30, frame_repeat=1)

#generate_pan_top_right_to_bottom_left_video(input_image_path, output_pan_top_right_to_bottom_left_path, zoom_step=1.005, crop_percent=0.75, frames=100, fps=30, frame_repeat=1)

#generate_zoom_in_rotate_clockwise_video(input_image_path, output_zoom_in_rotate_clockwise_path, zoom_step=1.005, angle_step=5, frames=100, fps=30, frame_repeat=1)

#generate_zoom_in_rotate_anticlockwise_video(input_image_path, output_zoom_in_rotate_anticlockwise_path, zoom_step=1.005, angle_step=5, frames=100, fps=30, frame_repeat=1)

#generate_zoom_out_rotate_anticlockwise_video(input_image_path, output_zoom_out_rotate_anticlockwise_path, zoom_step=0.995, angle_step=5, frames=100, fps=30, frame_repeat=1)

#generate_zoom_out_rotate_clockwise_video(input_image_path, output_zoom_out_rotate_clockwise_path, zoom_step=0.995, angle_step=5, frames=100, fps=30, frame_repeat=1)

#generate_outward_vignette_transition_video(input_image_path1, input_image_path2, output_outward_vignette_path, frames=100, fps=30, frame_repeat=1)

#generate_inward_vignette_transition_video(input_image_path1, input_image_path2, output_inward_vignette_path, frames=100, fps=30, frame_repeat=1)

#generate_ken_burns_effect_video(input_image_path, output_ken_burns_effect_path, start_zoom=1.0, end_zoom=2.0, pan_direction='right', frames=100, fps=30, frame_repeat=1)

#generate_ken_burns_effect_video(input_image_path, output_ken_burns_effect_path, start_zoom=1.0, end_zoom=2.0, pan_direction='left', frames=100, fps=30, frame_repeat=1)