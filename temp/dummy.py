# from pathlib import Path
# from openai import OpenAI
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
from dotenv import load_dotenv

load_dotenv() 




# client = OpenAI(api_key=OPEN_AI_SECRET_KEY)

# def generate_speech(text, output_file_path, model="tts-1", voice="alloy"):
#     response = client.audio.speech.create(
#         model=model,
#         voice=voice,
#         input=text
#     )
#     response.stream_to_file(Path(output_file_path))

# def transcribe_audio(file_path, model="whisper-1"):
#     with open(file_path, "rb") as audio_file:
#         transcription = client.audio.transcriptions.create(
#             model=model,
#             file=audio_file
#         )
#     return transcription.text

# # Example usage:
# generate_speech("Today is a wonderful day to build something people love!", "./audio_outputs/speech.mp3")
# transcription_text = transcribe_audio("./audio_outputs/speech.mp3")
# print(transcription_text)


# from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip

# def create_video_with_audio_and_text(video_path, audio_path, output_path, text, font='Amiri-regular', fontsize=75, text_color='white', bg_color=(0, 0, 0), bg_opacity=0.6, text_duration=5):
#     # Load the video and audio files
#     video = VideoFileClip(video_path)
#     audio = AudioFileClip(audio_path)

#     w, h = video.size

#     # Add the audio to the video (this will stop when the audio ends)
#     video = video.set_audio(audio)

#     # Create a text clip with a black semi-opaque background
#     txt = TextClip(text, font=font, color=text_color, fontsize=fontsize)
#     txt_col = txt.on_color(size=(video.w + txt.w, txt.h - 10), color=bg_color, pos=(6, 'center'), col_opacity=bg_opacity)

#     # Animate the text clip
#     txt_mov = txt_col.set_pos(lambda t: (max(w / 30, int(w - 0.5 * w * t)), max(5 * h / 6, int(100 * t))))

#     # Create the final composite video clip
#     final = CompositeVideoClip([video, txt_mov])

#     # Export the final video
#     final.subclip(0, text_duration).write_videofile(output_path, codec='libx264', audio_codec='aac')

# Example usage
# create_video_with_audio_and_text(
#     video_path="/Users/toheed/PanduAI/backend/assets/video1.mp4",
#     audio_path="/Users/toheed/PanduAI/backend/audio_outputs/speech.mp3",
#     output_path="./results/final_video5.mp4",
#     text="Today is a wonderful day to build something people love!"
# )



# print(TextClip.list('font'))

# test_data = {
        
#             "video": {
#                 "title": "Nature Exploration",
#                 "description": "A journey through diverse landscapes showcasing nature's beauty.",
#                 "prompt": "Create a video montage of various natural landscapes, including forests, mountains, rivers, and wildlife."
#             },
#             "audio": {
#                 "background_music": "Gentle acoustic guitar with nature sounds.",
#                 "prompt": "Produce an audio track featuring soothing acoustic guitar layered with sounds of birds chirping and gentle river flow."
#             },
#             "text_script": {
#                 "content": "Join us as we explore the breathtaking beauty of nature. From towering mountains to serene rivers, witness the wonders of our planet.",
#                 "voiceover_prompt": "Generate a calm and engaging voiceover narrating the beauty of natural landscapes and wildlife."
#             },
#         },

# subtitle animations

  # min_duration = min(video.duration, audio.duration)

    # video = video.subclip(0, min_duration)
    # audio = audio.subclip(0, min_duration)

    # w, h = video.size

    # Add the audio to the video (this will stop when the audio ends)
    # duration = video.duration

    # txt = (TextClip(text, fontsize=50, font='Amiri-regular',
    #             color='white')
    #    .set_duration(duration))

    # # Transparent text background
    # txt_col = txt.on_color(size=(video.w + txt.w, txt.h - 10),
    #                    color=(0, 0, 0), pos=(6, 'center'), col_opacity=0.8)

    # # Animate the text clip
    # txt_mov = txt_col.set_pos(lambda t: (
    #     int(w - 0.5 * w * t * text_speed_factor),
    #     max(5.4 * h / 6, int(0 * t))
    # ))
# import cv2
# import numpy as np
# from PIL import Image
# from transparent_background import Remover

# # Load model
# remover = Remover()  # default setting
# remover = Remover(mode='fast', jit=True)  # custom setting
# # remover = Remover(mode='base-nightly') # nightly release checkpoint

# # Usage for image
# # img = Image.open('/Users/toheed/PanduAI/backend/temp/man_image.jpg').convert('RGB')  # read image

# # # Process image with different types
# # process_types = [
# #     'rgba', 'map', 'green', 'white', 'blur',
# # ]

# # for process_type in process_types:
    
# #     out = remover.process(img, type=process_type)
    
# #     out.save(f'output_{process_type}.png')  # save result with a unique filename

# # Usage for video
# cap = cv2.VideoCapture('/Users/toheed/PanduAI/backend/temp/man_on_phone.mp4')  # video reader for input
# fps = cap.get(cv2.CAP_PROP_FPS)
# writer = None

# while cap.isOpened():
#     ret, frame = cap.read()  # read video

#     if not ret:
#         break

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame).convert('RGB')

#     if writer is None:
#         writer = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (img.width, img.height))  # video writer for output

#     out = remover.process(img, type='white')  # same as image, except for 'rgba' which is not for video.
#     writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR))

# cap.release()
# writer.release()

# import cv2
# import numpy as np
# import requests
# import base64
# import os
# from PIL import Image
# from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
# import io

# def generate_wipe_right_to_left(video1, video2, frames=30, fps=10, frame_repeat=1):
#     cap1 = cv2.VideoCapture(video1)
#     cap2 = cv2.VideoCapture(video2)

#     width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     # Use a memory buffer to store the video
#     out = cv2.VideoWriter('appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! mp4mux ! filesink location=/dev/stdout', cv2.CAP_GSTREAMER, fourcc, fps, (width, height))

#     while True:
#         ret, frame1 = cap1.read()
#         if not ret:
#             break
#         for _ in range(frame_repeat):
#             out.write(frame1)

#     if not cap2.isOpened():
#         print("Error: Unable to open the second video.")
#         out.release()
#         cap1.release()
#         cap2.release()
#         return None

#     cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset cap1 to the start

#     for i in range(frames):
#         ret1, frame1 = cap1.read()
#         ret2, frame2 = cap2.read()

#         if not ret1 and not ret2:
#             break

#         if ret1:
#             frame1 = cv2.resize(frame1, (width, height))
#         else:
#             frame1 = np.zeros((height, width, 3), dtype=np.uint8)

#         if ret2:
#             frame2 = cv2.resize(frame2, (width, height))
#         else:
#             frame2 = np.zeros((height, width, 3), dtype=np.uint8)

#         x = int(width * (i / frames))
#         wipe_frame = np.zeros_like(frame1)
#         wipe_frame[:, :width-x] = frame1[:, x:]
#         wipe_frame[:, width-x:] = frame2[:, :x]

#         for _ in range(frame_repeat):
#             out.write(wipe_frame)

#     while True:
#         ret, frame2 = cap2.read()
#         if not ret:
#             break
#         frame2 = cv2.resize(frame2, (width, height))
#         for _ in range(frame_repeat):
#             out.write(frame2)

#     cap1.release()
#     cap2.release()
#     out.release()

#     # Read the video from the memory buffer
#     with open('/dev/stdout', 'rb') as f:
#         video_data = f.read()

#     return video_data


# final_clip = generate_wipe_right_to_left("/Users/toheed/PanduAI/backend/temp/man_on_phone.mp4","/Users/toheed/PanduAI/backend/temp/man_walking.mp4")
# print(final_clip)
# final_clip.write_videofile("/Users/toheed/PanduAI/backend/temp/dummy_video.mp4", codec="libx264", audio_codec="aac")

data = """
{
    "scenes": [
        {
            "script": "Picture the Garden of Eden, an idyllic paradise, with all the wonders of creation. Nature thrives unchecked, an epitome of serenity and tranquility, untouched by human rebellion.",
            "images": [
                {
                    "sound_effects": [],
                    "prompt": "A serene view of the Garden of Eden with flourishing flora and fauna, pristine water bodies reflecting the clear sky",
                    "effects_animation": "generate_pan_bottom_to_top_video",
                    "style_preset": "photographic"
                }
            ],
            "transition": [
                {
                    "sound_effects": [],
                    "transition": "fadeinout_transition"
                }
            ]
        },
        {
            "script": "Suddenly, a sly serpent slithers amidst this beauty, beguiling Eve with his deceptive words.",
            "images": [
                {
                    "sound_effects": ["snake_hiss"],
                    "prompt": "A serpent approaching Eve in the Garden of Eden, its eyes gleaming with misleading intentions",
                    "effects_animation": "generate_pan_left_to_right_video",
                    "style_preset": "photographic"
                }
            ],
            "transition": [
                {
                    "sound_effects": [],
                    "transition": "crossfade_transition"
                }
            ]
        },
        {
            "script": "Tempted by the serpent, she reaches out for the forbidden fruit, a grim twist in the tale of mankind's innocence.",
            "images": [
                {
                    "sound_effects": ["anticipatory_sound"],
                    "prompt": "Eve's hand reaching out for the forked fruit, her face filled with curiosity and fear mingling together",
                    "effects_animation": "generate_pan_bottom_to_top_video",
                    "style_preset": "photographic"
                }
            ],
            "transition": [
                {
                    "sound_effects": [],
                    "transition": "fadeinout_transition"
                }
            ]
        },
        {
            "script": "Adam, succumbing to the same temptation, bites into the fruit, and the innocence of mankind is lost.",
            "images": [
                {
                    "sound_effects": ["bite_crunch"],
                    "prompt": "Adam biting into the fruit, his expression changing from unflinching trust to painful realization",
                    "effects_animation": "generate_pan_bottom_to_top_video",
                    "style_preset": "photographic"
                }
            ],
            "transition": [
                {
                    "sound_effects": [],
                    "transition": "crossfade_transition"
                }
            ]
        },
        {
            "script": "As they fully grasp the gravity of their disobedience, they find themselves fallen, forever marked by their fateful choice.",
            "images": [
                {
                    "sound_effects": [],
                    "prompt": "Adam and Eve visibly distraught, the heavy weight of regret pressing down upon them",
                    "effects_animation": "generate_pan_left_to_right_video",
                    "style_preset": "photographic"
                }
            ],
            "transition": [
                {
                    "sound_effects": [],
                    "transition": "fadeinout_transition"
                }
            ]
        }
    ],
    "bgm_prompt": "A slow poignant melody, using string instruments to create a somber and reflective atmosphere that underscores the gravity of the events",
    "style_preset": "photographic",
    "subtitle_styles": {
        "size": null,
        "color": "white",
        "fontsize": 74,
        "bg_color": "black",
        "font": "Arial",
        "stroke_color": "black",
        "stroke_width": 4,
        "method": "caption",
        "kerning": null,
        "align": "center",
        "interline": null,
        "transparent": true,
        "remove_temp": true,
        "print_cmd": null
    }
}
"""


import json

data = {
    "scenes": [
        {
            "script": "In the beginning, there was nothing but darkness. Then, a spark of light appeared, marking the birth of the universe.",
            "images": [
                {
                    "sound_effects": ["deep hum", "spark"],
                    "prompt": "A vast, dark void with a sudden spark of light in the center",
                    "effects_animation": "slow zoom into the light",
                    "style_preset": "cinematic"
                }
            ],
            "transition": [
                {
                    "sound_effects": ["whoosh"],
                    "transition": "fade to bright light"
                }
            ]
        },
        {
            "script": "From this light, stars and galaxies began to form, spreading across the cosmos.",
            "images": [
                {
                    "sound_effects": ["ethereal chimes", "space ambience"],
                    "prompt": "Stars and galaxies forming and expanding in space",
                    "effects_animation": "time-lapse of galaxies forming",
                    "style_preset": "ethereal"
                }
            ],
            "transition": [
                {
                    "sound_effects": ["soft transition"],
                    "transition": "cross dissolve to starry sky"
                }
            ]
        },
        {
            "script": "Among these galaxies, our Milky Way galaxy formed, and within it, a small solar system emerged.",
            "images": [
                {
                    "sound_effects": ["calm cosmic sounds"],
                    "prompt": "Milky Way galaxy with a focus on the solar system",
                    "effects_animation": "zoom into the solar system",
                    "style_preset": "realistic"
                }
            ],
            "transition": [
                {
                    "sound_effects": ["swish"],
                    "transition": "wipe to solar system"
                }
            ]
        },
        {
            "script": "In this solar system, the Earth took shape, a blue and green jewel among the stars.",
            "images": [
                {
                    "sound_effects": ["ocean waves", "wind"],
                    "prompt": "Earth forming, showing blue oceans and green lands",
                    "effects_animation": "rotation of Earth",
                    "style_preset": "vivid"
                }
            ],
            "transition": [
                {
                    "sound_effects": ["soft whoosh"],
                    "transition": "fade to Earth close-up"
                }
            ]
        },
        {
            "script": "Life began to flourish on Earth, from simple organisms to complex ecosystems.",
            "images": [
                {
                    "sound_effects": ["nature sounds", "birds chirping"],
                    "prompt": "Evolution of life on Earth, from single-celled organisms to diverse wildlife",
                    "effects_animation": "montage of different life forms",
                    "style_preset": "documentary"
                }
            ],
            "transition": [
                {
                    "sound_effects": ["nature transition"],
                    "transition": "cross dissolve to diverse life"
                }
            ]
        },
        {
            "script": "Human beings emerged, developing civilizations and shaping the world as we know it today.",
            "images": [
                {
                    "sound_effects": ["crowd noise", "construction sounds"],
                    "prompt": "Evolution of human civilization, from ancient times to modern cities",
                    "effects_animation": "time-lapse of human development",
                    "style_preset": "historical"
                }
            ],
            "transition": [
                {
                    "sound_effects": ["modern transition"],
                    "transition": "fade to present day"
                }
            ]
        }
    ],
    "bgm_prompt": "Epic orchestral music with a mix of ethereal and cosmic sounds, transitioning to a more uplifting and inspiring tone as the video progresses, matching the creation and evolution of the world.",
    "subtitle_styles": {
        "font": "Arial",
        "size": "18px",
        "color": "white"
    }
}

from uuid import UUID, uuid4
video_task_id = uuid4()
print(video_task_id)

# print(json.dumps(data, indent=4))