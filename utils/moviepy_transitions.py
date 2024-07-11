from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, CompositeVideoClip
from moviepy.video.tools.drawing import color_gradient


# Load your video clips
clip1 = VideoFileClip("/Users/toheed/PanduAI/backend/assets/video1.mp4")
clip2 = VideoFileClip("/Users/toheed/PanduAI/backend/assets/video2.mp4")

# Define the transition duration
transition_duration = 3  # duration in seconds

# Create a crossfade transition
clip1 = clip1.set_end(clip1.duration - transition_duration)
clip2 = clip2.set_start(transition_duration)

# Create a crossfade effect
crossfade_clip = clip1.crossfadeout(transition_duration).set_start(clip1.duration - transition_duration)

# Combine the clips with the crossfade effect
final_clip = concatenate_videoclips([clip1, crossfade_clip, clip2], method="compose")

# Write the final video to a file
final_clip.write_videofile("./outputs/output_video_with_transition1.mp4", codec="libx264")

# fadeIn fadeOut

clip1 = VideoFileClip("/Users/toheed/PanduAI/backend/assets/video1.mp4").fadeout(1)
clip2 = VideoFileClip("/Users/toheed/PanduAI/backend/assets/video2.mp4").fadein(1)

final_clip = concatenate_videoclips([clip1, clip2])

final_clip.write_videofile("output_fadein_fadeout.mp4", codec="libx264")


# Custom zoom and slide transition
def zoom_and_slide_transition(clip1, clip2, duration, slide_direction='left'):
    w, h = clip1.size

    # Define the sliding position function for clip1
    def slide_out(t):
        if slide_direction == 'left':
            return ('center', -w * (t / duration))
        elif slide_direction == 'right':
            return ('center', w * (t / duration))
        elif slide_direction == 'top':
            return (0, -h * (t / duration))
        elif slide_direction == 'bottom':
            return (0, h * (t / duration))
    
    # Clip1 slides out
    sliding_clip1 = clip1.set_position(slide_out).set_duration(duration)
    
    # Clip2 zooms in
    zooming_clip2 = clip2.resize(lambda t: 1 + 0.5 * (t / duration)).set_start(duration).set_duration(duration)
    
    # Extend clip1's duration to accommodate the transition
    clip1_extended = clip1.set_duration(clip1.duration + duration)
    
    # Create the composite video clip
    final_clip = CompositeVideoClip([clip1_extended, sliding_clip1, zooming_clip2.set_start(clip1.duration)])
    
    return final_clip


# Apply the custom zoom and slide transition
final_clip = zoom_and_slide_transition(clip1, clip2, transition_duration, slide_direction='left')

# Write the final video to a file
final_clip.write_videofile("./outputs/output_zoom_and_slide.mp4", codec="libx264")


def cross_wrap(clip1, clip2, duration):
    w, h = clip1.size
    
    def slide_out(t):
        return ('center', -w * (t / duration))
    
    def slide_in(t):
        return ('center', w * (1 - t / duration))

    # Create sliding clips
    sliding_clip1 = clip1.set_position(slide_out).set_duration(duration)
    sliding_clip2 = clip2.set_position(slide_in).set_start(clip1.duration).set_duration(duration)
    
    # Extend clip1's duration to accommodate the transition
    clip1_extended = clip1.set_duration(clip1.duration + duration)
    
    # Create the composite video clip
    final_clip = CompositeVideoClip([clip1_extended, sliding_clip1, sliding_clip2])
    
    return final_clip

# Load your video clips
clip1 = VideoFileClip("/Users/toheed/PanduAI/backend/assets/video1.mp4")
clip2 = VideoFileClip("/Users/toheed/PanduAI/backend/assets/video2.mp4")

transition_duration = 1  # duration in seconds

# Apply the cross-wrap transition
final_clip = cross_wrap(clip1, clip2, transition_duration)

# Write the final video to a file
final_clip.write_videofile("./outputs/output_cross_wrap.mp4", codec="libx264")
