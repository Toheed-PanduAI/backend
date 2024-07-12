import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI
import warnings
from pathlib import Path
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, TextClip, CompositeVideoClip, CompositeAudioClip
import moviepy.audio.fx.all as afx
from moviepy.editor import concatenate_videoclips, ColorClip
from moviepy.video.tools.drawing import color_split
from . import image_effects
from . import audio_prompts
# import image_effects
# import audio_prompts
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, play
import scripts.speech_synthesis as speech_synthesis
import random
from utils.util import text_styles, effect_images
import json
from dotenv import load_dotenv
import threading
import string
from database import db, aws
import shutil
from config import secret_config

load_dotenv() 

OPEN_AI_SECRET_KEY = secret_config.OPEN_AI_SECRET_KEY
ELEVEN_LABS_SECRET_KEY = secret_config.ELEVEN_LABS_SECRET_KEY
STABILITY_SECRET_KEY = secret_config.STABILITY_SECRET_KEY

warnings.filterwarnings("ignore", category=DeprecationWarning)  

client = OpenAI(api_key=OPEN_AI_SECRET_KEY)

elevan_labs_client = ElevenLabs(
  api_key= ELEVEN_LABS_SECRET_KEY
)

subtitle_styles = {
    "size": None,
    "color": "white",
    "fontsize": 74,
    "bg_color": "black",
    "font": "Arial",
    "stroke_color": "black",
    "stroke_width": 4,
    "method": "caption",
    "kerning": None,
    "align": "center",
    "interline": None,
    "transparent": True,
    "remove_temp": True,
    "print_cmd": None
        }

static_instructions = """
You are an expert in creating video content, namely the script, generating prompts for images, background music, and sound effects.

Generate a JSON containing the following aspects. These aspects in the JSON will be used for creating a video of 40-60 seconds (which means around 8 to 12 images) in length in a 9:16 aspect ratio.

The output should be in this JSON format:

{
    "scenes": [
        {
            "script": " ",
            "images": [
                {
                    "sound_effects": [],
                    "prompt": " ",
                    "effects_animation": " "
                }
            ],
            "transition": [
                {
                    "sound_effects": [],
                    "transition": " "
                }
            ]
        }
    ],
    "bgm_prompt": " ",
    "style_preset": " "
}

Here's a breakdown of each aspect:

1. **Script**:
   - The "script" will be what the narration is while each image is shown.
   - It should be engaging and informative, covering all key points related to the topic.
   - Each script must be at least 20 words long and should form a coherent story when combined.
   - If the script content is longer, split it to create another list with the remaining part of the script and a new prompt for the image.

2. **Prompt**:
   - The "prompt" will be the description of the image that will be shown on the screen.
   - The image prompt must be highly descriptive and very detailed to get the most suitable image for the script.
   - All images must be consistent with each other so that they do not look out of place when combined together in the video.

3. **Effects Animation**:
   - The "effects_animation" will be the animation effect that will be applied to the image.
   - Choose one of the following effects which you think is suitable for each image. Each image will have this effect and they will combine to form a video, so choose the effects in a manner where it should look appealing when they are combined together:
     1. "generate_pan_bottom_to_top_video" 
     (Pans image from bottom to top slowly)
     2. "generate_pan_top_to_bottom_video" 
     (Pans image from top to bottom slowly)
     3. "generate_pan_left_to_right_video" 
     (Pans image from left to right slowly)
     4. "generate_pan_right_to_left_video" 
     (Pans image from right to left slowly)
     5. "generate_pan_top_right_to_bottom_left_video" 
     (Pans image from top right to bottom left slowly)
     6. "generate_pan_bottom_left_to_top_right_video" 
     (Pans image from bottom left to top right slowly)
     7. "generate_pan_bottom_right_to_top_left_video" 
     (Pans image from bottom right to top left slowly)
     8. "generate_pan_top_left_to_bottom_right_video" (
     Pans image from top left to bottom right slowly)

4. **Transition**:
   - The "transition" will be the transition effect that will be applied between the images.
   - Each "images" should have one transition
   - Even the last image should have a transition effect to the end screen.
   - Choose one of the following which you think is suitable based on the images before and after the transition:
     1. "crossfade_transition"
     2. "fadeinout_transition"
     3. "slide_transition"

5. **Sound Effects**:
   - The "sound_effects" will be the sound effects that will be played during the transition or when the image is shown.
   - Generate a prompt for sound effects based on the script and image if required wherever a sound effect would sound good. Do not generate a sound effect for every image. It can be left blank if there is no suitable effect.
   - Along with the prompt for the sound effect, give a duration in seconds to determine how long the sound effect should last. It should vary between 1 to 3 seconds. For example, a "whoosh" sound effect while the transition is occurring, or a "birds chirping" effect if the image has birds in it, or even a "drum roll" sound effect when the script has a joke in it.

6. **Background Music (BGM)**:
   - The "bgm_prompt" will be the description of the background music that will be played throughout the video.
   - The music should enhance the key visuals in the images and match the overall tone and mood of the video.
   - The bgm prompt should be very detailed and specific to get the most suitable bgm as per the visuals and script of the video. The bgm prompt must have 400 characters or less.

7. **Style Preset**:
   - Choose a suitable style_preset for the images from the following list. This style will be applied to all images within the video. Choose a style based on the script which will provide suitable visuals. This style must be relevant and not random:
     1. "3d-model" 
     (Realistic 3D rendering)
     2. "analog-film" 
     (Classic film aesthetic)
     3. "anime" 
     (Japanese animation style)
     4. "cinematic" 
     (Movie-like quality)
     5. "comic-book" 
     (Illustrated comic appearance)
     6. "digital-art" 
     (Modern digital creation)
     7. "enhance" 
     (Improved details and clarity)
     8. "fantasy-art" 
     (Imaginative and mythical)
     9. "isometric line-art" 
     (3D view with no perspective distortion)
     10. "low-poly" 
     (Simplified polygonal shapes)
     11. "modeling-compound" 
     (Sculpted and textured)
     12. "neon-punk" 
     (Bright, glowing, and futuristic)
     13. "origami" 
     (Paper-folding style)
     14. "photographic" 
     (Realistic photography)
     15. "pixel-art" 
     (Pixelated and retro)
     16. "tile-texture" 
     (Repeating pattern design)
     17. "line-art" 
     (Clean lines and minimal shading)

### Example Generation:

Generate a detailed transcript for a video on the topic: [Topic]. The title of the video is "[Title]". The transcript should be engaging and informative, covering all key points related to the topic. This transcript should be broken down into "scripts" for each image and placed in the "script" object. Each script must be 15 words long. All of the scripts should make sense and relatable when combined together and viewed as a whole. The scripts must form a story.

If the details are given for each object then fill them in accordingly in the list after modifying them according to the instructions:

- The script should be mind-capturing and engaging for the audience.
- The images should match the script.
- Each script should be at least 15 words long.
- If the script content is longer, then create another list with the remaining part of the script and a new prompt for the image.
- If the script is provided after this section in this prompt, do not use the same script for the entire scene object. Modify the script and make it longer for a 40-60 second video. Then, split this to create multiple "scene" objects.
- Each script must be at least 5 seconds long. For reference, 20 words average around 6 seconds.
- Each script must only have 1 image. If the script is long enough for more than 1 image, split the script and make a new script object with a new image.
- Each scene should have a transition between images.

Generate prompts for the images based on the title alone and others required in the object format.

Make sure each script is at least 15 words long and fits well within a 40-60 second video timeframe.

The output must have at least 8 images and at most 12 images.

Based on the "script", estimate the duration when it is converted to audio. This script should be long enough to cover one image while each image has to last 5 seconds along with the transition. For reference, 20 words average around 6 seconds.

Generate the full JSON object based on these instructions.
"""

progress = {"percentage": 0}

video_id = ""

def update_progress(step, total_steps):
    global progress
    progress["percentage"] = int((step / total_steps) * 100)

def get_progress(id):
    global video_id
    if video_id == id:
        return progress
    return None

def setVideoID(id):
    global video_id
    video_id = id

def delete_files_in_folder(folder_path):
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"The folder {folder_path} does not exist.")
            return
        
        # Iterate over the files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file and delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Check if it's a directory and delete it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        
        print(f"All files in the folder {folder_path} have been deleted.")
    except Exception as e:
        print(f"Error occurred: {e}")

def remove_punctuation(word):
    # Create a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    # Use the translate method to remove the punctuation
    return word.translate(translator)

def generate_prompts(static_instructions, user_prompt, number_of_prompts=1, prompt_type="JSON generation"):
    complete_prompt = f"""
            {static_instructions}

            ---

            {user_prompt}
        """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Generate a json output only."},
            {"role": "user", "content": complete_prompt}
        ]
    )
    return response.choices[0].message.content

def generate_subtitle_clips(data, video):
    if data is None or 'words' not in data or 'word_start_times' not in data or 'word_end_times' not in data:
        raise ValueError("Invalid or missing data structure for subtitles.")
    
    clips = []
    y_position = video.size[1] * 0.75
    # font_style = random.choice(list(text_styles.values()))
   

    for word, start, end in zip(data['words'], data['word_start_times'], data['word_end_times']):
        text_clip = TextClip(
            txt=word,
            size=subtitle_styles['size'],
            color=subtitle_styles['color'], 
            fontsize=subtitle_styles['fontsize'],
            bg_color=subtitle_styles['bg_color'],
            font=subtitle_styles['font'],
            stroke_color=subtitle_styles['stroke_color'],
            stroke_width=subtitle_styles['stroke_width'],
            method=subtitle_styles['method'],
            kerning=subtitle_styles['kerning'],
            align=subtitle_styles['align'],
            interline=subtitle_styles['interline'],
            transparent=subtitle_styles['transparent'],
            remove_temp=subtitle_styles['remove_temp'],
            print_cmd=subtitle_styles['print_cmd']
        ).set_position(('center', y_position)).set_start(start).set_duration(end - start)

        clips.append(text_clip)
    return clips

def create_video_with_audio_and_text(video_path, audio_path, output_path, subtitle_data, fadeout_duration=1, text=None, image_path=None, position=('center', 'center'), 
                  font='Arial', font_size=24, color='white', opacity=0.5, padding=10, bgm_volume=0.5):
   
    # Load the video and audio files
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    print(f"audio duration: {audio.duration}")

    audio = afx.audio_fadeout(audio, fadeout_duration)

    video = video.subclip(0, audio.duration)
    audio = audio.subclip(0, audio.duration)

    video = video.set_audio(audio)

    if text:
        watermark = (TextClip(text, fontsize=font_size, font=font, color=color)
                     .set_opacity(opacity)
                     .set_pos(position)
                     .set_duration(video.duration)
                      .margin(left=padding, right=padding, top=padding, bottom=padding, opacity=0))
    elif image_path:
        watermark = (ImageClip(image_path)
                     .set_duration(video.duration)
                     .set_opacity(opacity)
                     .set_pos(position)
                      .margin(left=padding, right=padding, top=padding, bottom=padding, opacity=0))
        
        
    subtitle_clips = generate_subtitle_clips(subtitle_data, video)

    # Create the final composite video clip with the watermark
    final = CompositeVideoClip([video, watermark] + subtitle_clips)

    # Export the final video
    final.write_videofile(output_path, codec='libx264', audio_codec='aac')

def add_bgm(video_path, bgm_path, output_path, bgm_volume=0.15, extra_duration=1, fadeout_duration=1):
    # Load the video file
    video = VideoFileClip(video_path)
    audio = video.audio

    # Load the BGM audio file and set its volume
    bgm_audio = AudioFileClip(bgm_path).volumex(bgm_volume)

    # Loop the background music to match the extended video's duration
    bgm_audio = afx.audio_loop(bgm_audio, duration=video.duration)

    # Apply fade-out effect to the BGM
    bgm_audio = afx.audio_fadeout(bgm_audio, fadeout_duration)
    audio = afx.audio_fadeout(audio, fadeout_duration)

    # Combine the original audio with the BGM
    if video.audio is not None:
        combined_audio = CompositeAudioClip([audio, bgm_audio])
    else:
        combined_audio = bgm_audio

    # Set the combined audio to the extended video
    video = video.set_audio(combined_audio)

    # Export the final video
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')

def generate_video_data(user_prompt):
    generated_prompts = generate_prompts(static_instructions, user_prompt)
    video_data = json.loads(generated_prompts)
    return video_data

# generate_video_data("Creation of the world, In the beginning, God created the heavens and the earth. Over six days, God created light, sky, land, vegetation, stars, sea creatures, birds, animals, and mankind. On the seventh day, He rested. The background audio should be very magestic and instrumental")

def generate_video(video_data):
    # Paths
    image_name = 0
    scene_name = 0
    output_format="jpeg"
    output_image_folder_generation = '/Users/toheed/PanduAI/backend/workflow/Images'
    input_video_folder_transition = '/Users/toheed/PanduAI/backend/workflow/Scenes'
    output_video_folder_effects = '/Users/toheed/PanduAI/backend/workflow/Effects'
    bgm_path = f"/Users/toheed/PanduAI/backend/workflow/BGM/bgm.mp3"
    output_filename_stiched = f"/Users/toheed/PanduAI/backend/workflow/Transitions/stitched_video_{image_name}.mp4"
    final_video_output_path= f"/Users/toheed/PanduAI/backend/workflow/Final_Video/final_video.mp4"

    total_steps = len(video_data["scenes"]) + 3  
    current_step = 0


    # Generate BGM
    bgm_prompt = video_data["bgm_prompt"]
    bgm_data = speech_synthesis.generate_and_save_sound(bgm_prompt, bgm_path)
    current_step += 1
    update_progress(current_step, total_steps)


    for scene in video_data["scenes"]:
        current_step += 1
        update_progress(current_step, total_steps)

        scene_name += 1
        output_path_scene= f"/Users/toheed/PanduAI/backend/workflow/Scenes/scene_with_watermark_{scene_name}.mp4"
        scene_input_video = f"/Users/toheed/PanduAI/backend/workflow/Effects/image_{scene_name}.mp4"
        audio_path = f"/Users/toheed/PanduAI/backend/workflow/Audio/audio{scene_name}.mp3"

        script = scene["script"]
        images = scene["images"]
        transition = scene["transition"]
        # subtitle_styles = video_data["subtitle_styles"]
        style_preset = video_data["style_preset"]

        # Generate audio
        print(f"Generating audio")
        subtitle_data = speech_synthesis.generate_tts_with_timestamps(script, audio_path)
        # subtitle_data = {
        #     'words': ['Born', 'and', 'raised', 'in!', 'the', 'charming,', 'south,', 'I', 'can', 'add', 'a', 'tou,ch', 'of', 'sweet', 'southern', 'hospitality', 'to', 'your', 'audiobooks', 'and', 'podcasts'],
        #     'word_start_times': [0.0, 0.383, 0.499, 0.801, 0.894, 0.998, 1.405, 1.869, 1.997, 2.148, 2.334, 2.403, 2.659, 2.775, 3.123, 3.483, 4.226, 4.342, 4.516, 5.074, 5.19],
        #     'word_end_times': [0.348, 0.453, 0.766, 0.859, 0.964, 1.335, 1.811, 1.916, 2.101, 2.299, 2.357, 2.612, 2.705, 3.053, 3.437, 4.168, 4.296, 4.458, 5.039, 5.143, 6.223]
        # }
        # Remove Punctuation from words 
        subtitle_data['words'] = [remove_punctuation(word) for word in subtitle_data['words']]
        print(subtitle_data)
    
        for image in images:
            image_name += 1
            prompt = image["prompt"]
            effect_function_name = image["effects_animation"]
            sound_effects = image["sound_effects"]
            
            # Generate images
            audio_prompts.generate_multiple_images_with_stability_engine(prompt, style_preset, image_name = image_name, output_dir = output_image_folder_generation)

            input_image_path = os.path.join(output_image_folder_generation, f"image_{image_name}.{output_format}")
            # Effects
            print(input_image_path)
            image_effects.process_single_image(input_image_path, output_video_folder_effects, effect_function_name)
        
        
        # Compile scene
        create_video_with_audio_and_text(
            video_path= scene_input_video,
            audio_path= audio_path,
            output_path= output_path_scene,
            subtitle_data= subtitle_data,
            fadeout_duration=1,
            text="Pandu AI", 
            position=("right", "top"), 
            font="Helvetica-Bold", 
            font_size=50, 
            color="gray", 
            opacity=0.8,
            padding=10,
            bgm_volume=0.15
            )
        

    # Transition
    print("Transition")
    transition = transition[0]["transition"]
    print(transition)
    image_effects.stitch_videos_with_transition(input_video_folder_transition, output_filename_stiched, transition)
    current_step += 1
    update_progress(current_step, total_steps)

    # Add BGM
    add_bgm(output_filename_stiched, bgm_path, final_video_output_path, bgm_volume=0.08)
    current_step += 1
    update_progress(current_step, total_steps)

def main(user_input):
    try:
        db.video_tasks_collection.update_one(
            {"video_task_id": video_id},
            {"$set": {"task_status": "in progress"}}
        )
        print("Generating prompts")
        video_data = generate_video_data(user_input)
        print(video_data)
        print("Generating video")
        generate_video(video_data)
        
        # Upload the final video to S3 and get the CDN URL
        video_uuid, series_uuid = aws.upload_final_video_to_s3()
        cdn_url = aws.get_cdn_url_video(series_uuid, video_uuid)

        # Update task status in database as completed
        print("Uploading to S3")
        db.video_tasks_collection.update_one(
            {"video_task_id": video_id},
            {"$set": {"video_url": str(cdn_url), "task_status": "completed"}}
        )

        # Upload all the prompt, images, bgm and audio files to S3
        aws.upload_to_s3(video_data, series_uuid)

        # Delete all files in the Images, effects, audio, bgm and final video folder
        print("Deleting temperory files")
        delete_files_in_folder(str(secret_config.Image_folder_path))
        delete_files_in_folder(str(secret_config.Audio_folder_path))
        delete_files_in_folder("/Users/toheed/PanduAI/backend/workflow/BGM")
        delete_files_in_folder("/Users/toheed/PanduAI/backend/workflow/Final_Video")
        delete_files_in_folder(str(secret_config.Effect_folder_path))
        delete_files_in_folder(str(secret_config.Scenes_folder_path))
        delete_files_in_folder(str(secret_config.Transition_folder_path))

    except Exception as e:
       
        print("Deleting temperory files")
        delete_files_in_folder(str(secret_config.Image_folder_path))
        delete_files_in_folder(str(secret_config.Audio_folder_path))
        delete_files_in_folder("/Users/toheed/PanduAI/backend/workflow/BGM")
        delete_files_in_folder("/Users/toheed/PanduAI/backend/workflow/Final_Video")
        delete_files_in_folder(str(secret_config.Effect_folder_path))
        delete_files_in_folder(str(secret_config.Scenes_folder_path))
        delete_files_in_folder(str(secret_config.Transition_folder_path))
        
        # Handle any errors or exceptions
        print(f"Error generating video: {e}")

        # Update task status in database as failed
        db.video_tasks_collection.update_one(
            {"video_task_id": video_id},
            {"$set": {"task_status": "failed"}}
        )



# Example usage
# folder_path = "/path/to/your/folder"
# delete_files_in_folder(folder_path)

# video_data = generate_video_data(user_input)
# print(video_data)

# generate_video(video_data)

# threading.Thread(target=generate_video, args=(video_data,)).start()


# Title: Creation of the World
# Narrator: 'In the beginning, God created the heavens and the earth. Over six days, God created light, sky, land, vegetation, stars, sea creatures, birds, animals, and mankind. On the seventh day, He rested.'
# God commanding creation, light piercing darkness, animals in their habitats, Adam and Eve.
# Characters: God - The omnipotent creator. Adam - The first man. Eve - The first woman.
# Background Audio: Majestic, awe-inspiring orchestral music.


# user_prompt = """
# Creation of the world, In the beginning, God created the heavens and the earth. Over six days, God created light, sky, land, vegetation, stars, sea creatures, birds, animals, and mankind. On the seventh day, He rested. The background audio should be very magestic and instrumental

# """