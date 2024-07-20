import cv2
from moviepy.editor import VideoFileClip
import numpy as np


# Define a sepia filter
def sepia(frame):
    frame = frame.astype(float)
    frame[..., 0] *= 1.07  # Red channel
    frame[..., 1] *= 0.74  # Green channel
    frame[..., 2] *= 0.43  # Blue channel
    frame = frame.clip(0, 255)
    return frame.astype('uint8')

# Define a grayscale filter
def grayscale(frame):
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    return np.stack((gray,)*3, axis=-1).astype('uint8')

# Define an invert colors filter
def invert_colors(frame):
    return (255 - frame).astype('uint8')

# Edge detection
def edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Sharpen effect
def sharpen(frame):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def munstead_filter(frame):
    # Increase contrast
    contrast_factor = 1.5
    mean = np.mean(frame, axis=(0, 1), keepdims=True)
    contrast_frame = np.clip((frame - mean) * contrast_factor + mean, 0, 255)

    # Add a slight vignette
    rows, cols = frame.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    vignette_frame = np.copy(contrast_frame)
    
    for i in range(3):
        vignette_frame[:, :, i] = vignette_frame[:, :, i] * mask

    # Add a purple hue
    purple_hue = np.zeros_like(vignette_frame)
    purple_hue[..., 0] = 1.2  # Adjusting the Red channel
    purple_hue[..., 2] = 1.1  # Adjusting the Blue channel
    final_frame = np.clip(vignette_frame * purple_hue, 0, 255).astype('uint8')

    return final_frame

def cool_tone(frame):
    frame = frame.astype(float)
    frame[..., 0] *= 0.9  # Red channel
    frame[..., 1] *= 0.95  # Green channel
    frame[..., 2] *= 1.1  # Blue channel
    return np.clip(frame, 0, 255).astype('uint8')

def warm_tone(frame):
    frame = frame.astype(float)
    frame[..., 0] *= 1.1  # Red channel
    frame[..., 1] *= 0.9  # Green channel
    frame[..., 2] *= 0.9  # Blue channel
    return np.clip(frame, 0, 255).astype('uint8')

def high_contrast(frame):
    factor = 2.0  # Increase this value for more contrast
    mean = np.mean(frame, axis=(0, 1), keepdims=True)
    frame = (frame - mean) * factor + mean
    return np.clip(frame, 0, 255).astype('uint8')

def solarize(frame, threshold=128):
    solarized_frame = np.where(frame < threshold, frame, 255 - frame)
    return solarized_frame.astype('uint8')

def posterize(frame, bits=4):
    shift = 8 - bits
    return ((frame >> shift) << shift).astype('uint8')

def emboss(frame):
    kernel = np.array([[ -2, -1, 0],
                       [ -1,  1, 1],
                       [  0,  1, 2]])
    return cv2.filter2D(frame, -1, kernel)

def cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def sepia_alternative(frame):
    frame = frame.astype(float)
    tr = 0.393 * frame[..., 2] + 0.769 * frame[..., 1] + 0.189 * frame[..., 0]
    tg = 0.349 * frame[..., 2] + 0.686 * frame[..., 1] + 0.168 * frame[..., 0]
    tb = 0.272 * frame[..., 2] + 0.534 * frame[..., 1] + 0.131 * frame[..., 0]
    sepia_frame = np.stack((tb, tg, tr), axis=-1)
    return np.clip(sepia_frame, 0, 255).astype('uint8')


# Load your video
# video = VideoFileClip("/Users/toheed/PanduAI/backend/workflow/video/generate_pan_left_to_right_video.mp4")

# # Apply the filter
# sepia_video = video.fl_image(sharpen)

# # Save the result
# sepia_video.write_videofile("output_video_filter_sharpen.mp4", codec="libx264")

# Apply the  filter
def apply_filter(video_path, output_path):
    video = VideoFileClip(video_path)
    filtered_video = video.fl_image(posterize)
    filtered_video.write_videofile(output_path, codec="libx264")


# Apply the Munstead filter to a video
apply_filter("/Users/toheed/PanduAI/backend/workflow/video/generate_pan_left_to_right_video.mp4", "output_video_posterize.mp4")