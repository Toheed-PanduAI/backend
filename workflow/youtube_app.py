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
# from . import image_effects
# from . import audio_prompts
import image_effects
import audio_prompts
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
from config import secret_config, credits_config
import requests
from pytube import YouTube

# Load environment variables
load_dotenv()

OPEN_AI_SECRET_KEY = secret_config.OPEN_AI_SECRET_KEY
ELEVEN_LABS_SECRET_KEY = secret_config.ELEVEN_LABS_SECRET_KEY
STABILITY_SECRET_KEY = secret_config.STABILITY_SECRET_KEY
SERP_API_KEY = secret_config.SERP_API_KEY

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
You are an expert in creating video content, namely the script and background music

Generate a JSON containing the following aspects. These aspects in the JSON will be used for creating a video of 40-60 seconds duration in a 9:16 aspect ratio.

The output should be in this JSON format:

{
    "script": " ", 
    "bgm_prompt": " "
}

Here's a breakdown of each aspect:

1. **Script**:
   - The script must be very interesting based on the topic given below and must be engaging to the audience. It will be the transcript for a youtube short or instagram reel.
  -  The script should be 40 to 60 seconds long. (20 words will be around 6 seconds for reference.)


2. **Background Music (BGM)**:
   - The "bgm_prompt" will be the description of the background music that will be played throughout the video.
   - The music should enhance the key visuals in the images and match the overall tone and mood of the video.
   - The bgm prompt should be very detailed and specific to get the most suitable bgm as per the visuals and script of the video. The bgm prompt must have 400 characters or less.


### Example Generation:

Generate a detailed transcript for a video on the topic: [Topic]. The title of the video is "[Title]". The transcript should be engaging and informative, covering all key points related to the topic. 

If the details are given for each object then fill them in accordingly in the list after modifying them according to the instructions:

- The script should be mind-capturing and engaging for the audience.
- If the script is provided after this section in this prompt, do not use the same script for the entire scene object. Modify the script and make it longer for a 40-60 second video.

Generate a suitable bgm prompt based on the script. This bgm prompt should match the tone of the script and should be detailed for audio generation

Generate the full JSON object based on these instructions for the following topic.
"""

progress = {"percentage": 0}

credits_used = 0

video_data = {
    "video_id": "",
    "user_id": ""
}

def update_progress(step, total_steps):
    global progress
    progress["percentage"] = int((step / total_steps) * 100)

def get_progress(id):
    global video_data
    if video_data["video_id"] == id:
        return progress
    return None

def setVideoID(id):
    global video_data
    video_data["video_id"] = id

def setUserID(id):
    global video_data
    video_data["user_id"] = id

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

def youtube_search(query: str):
    api_key = SERP_API_KEY 
    url = "https://serpapi.com/search"
    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to get data from API: {response.text}")
        raise Exception(f"Failed to get data from API: {response.text}")
    return response.json()

def get_youtube_url(query: str):
    result = youtube_search(query)
    video_results = result.get('video_results', [])

    # Filter and extract links where the title contains "no copyright"
    filtered_links = [
        video['link']
        for video in video_results
        if "no copyright" in video.get('title', '').lower()  # Case insensitive search
    ]

    if filtered_links:
        return filtered_links[0]
    else:
        raise Exception("No videos found with 'no copyright' in the title.")

def download_and_resize_video(video_url, temp_folder, output_path):
    # Create folders if they don't exist
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # Download the video
    video = YouTube(video_url)
    stream = video.streams.filter(res="720p").first()
    if not stream:
        raise Exception("1080p resolution is not available for this video.")
   
    temp_video_path = os.path.join(temp_folder, "temp_video.mp4")
    stream.download(output_path=temp_folder, filename="temp_video.mp4")

    # Load the video
    video_clip = VideoFileClip(temp_video_path)
    video_clip = video_clip.subclip(0, 60)

    # Calculate the target width and height for 9:16 aspect ratio
    target_height = 1024
    target_width = 1792

    target_aspect_ratio = 0.57

    if target_width / target_height > target_aspect_ratio:
        # The video is wider than the target aspect ratio, resize by height
        new_height = target_height
        new_width = int(new_height * target_aspect_ratio)
    else:
        # The video is taller than the target aspect ratio, resize by width
        new_width = target_width
        new_height = int(new_width / target_aspect_ratio)

    resized_clip = video_clip.resize(newsize=(new_width, new_height))

    resized_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # # Calculate the target width and height for 9:16 aspect ratio
    # target_height = video_clip.size[1]
    # target_width = int(target_height * 9 / 16)

    # # Calculate the amount to crop from the sides
    # crop_amount = (video_clip.size[0] - target_width) / 2

    # # Crop the video
    # cropped_video = video_clip.crop(x1=crop_amount, x2=video_clip.size[0] - crop_amount)

    # # Resize the video to maintain the aspect ratio
    # resized_video = cropped_video.resize(newsize=(target_width, target_height))

    # # Save the resized video in the output folder
    # output_path = os.path.join(output_folder, "youtube_video.mp4")
    # resized_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

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

def create_video_with_audio_and_text(video_path, audio_path, bgm_path, output_path, subtitle_data, fadeout_duration=1, text=None, image_path=None, position=('center', 'center'), 
                  font='Arial', font_size=24, color='white', opacity=0.5, padding=10, bgm_volume=0.5):
   
    # Load the video and audio files
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    bgm_audio = AudioFileClip(bgm_path)

    print(f"audio duration: {audio.duration}")

    audio = afx.audio_fadeout(audio, fadeout_duration)

    video = video.subclip(0, audio.duration)
    audio = audio.subclip(0, audio.duration)

    bgm_audio = afx.volumex(bgm_audio, bgm_volume)

    video = video.set_audio(None)

    bgm_audio = afx.audio_loop(bgm_audio, duration=audio.duration)

    combined_audio = CompositeAudioClip([bgm_audio.set_duration(video.duration), audio.set_duration(video.duration)])

    video = video.set_audio(combined_audio)

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

def generate_video(video_data, query):
    # Paths
    bgm_path = "/Users/toheed/PanduAI/backend/workflow/BGM/bgm.mp3"
    audio_path = "/Users/toheed/PanduAI/backend/workflow/Audio/audio.mp3"
    final_video_output_path= "/Users/toheed/PanduAI/backend/workflow/Final_Video/final_video.mp4"
    downloaded_video_file = "/Users/toheed/PanduAI/backend/workflow/YoutubeVideo/youtube_video.mp4"
    temp_folder = "/Users/toheed/PanduAI/backend/workflow/TempVideos"
    total_steps = 3
    current_step = 0

    # Get video from youtube
    video_url = get_youtube_url(query)
    download_and_resize_video(video_url, temp_folder=temp_folder, output_path=downloaded_video_file)

    # Generate BGM
    bgm_prompt = video_data["bgm_prompt"]
    bgm_data = speech_synthesis.generate_and_save_sound(bgm_prompt, bgm_path)
    current_step += 1
    update_progress(current_step, total_steps)

    # Generate audio
    print(f"Generating audio")
    subtitle_data = speech_synthesis.generate_tts_with_timestamps(video_data["script"], audio_path)
    subtitle_data['words'] = [remove_punctuation(word) for word in subtitle_data['words']]
    print(subtitle_data)

    print("Creating video with audio and text")
    # Compile the video with audio and text
    create_video_with_audio_and_text(
            video_path= downloaded_video_file,
            audio_path= audio_path,
                bgm_path = bgm_path,
            output_path= final_video_output_path,
            subtitle_data= subtitle_data,
            fadeout_duration=0.5,
            text="Pandu AI", 
            position=("right", "top"), 
            font="Helvetica-Bold", 
            font_size=50, 
            color="gray", 
            opacity=0.8,
            padding=10,
            bgm_volume=0.15
            )
        

    # Add BGM
    # add_bgm(final_video_output_path, bgm_path, final_video_output_path, bgm_volume=0.08)
    # current_step += 1
    # update_progress(current_step, total_steps)

def main(user_input):
    try:
        # db.video_tasks_collection.update_one(
        #     {"video_task_id": video_data["video_id"]},
        #     {"$set": {"task_status": "in progress"}}
        # )
        print("Generating prompts")
        video_data = generate_video_data(user_input["prompt"])
        print(video_data)

        # credits_used += credits_config.CREDIT_COSTS["open_ai"]
        # print(f"Credits used: {credits_used}")
        # credits_config.deduct_credits()

        print("Generating video")
        generate_video(video_data, user_input["query"])
        
        # Upload the final video to S3 and get the CDN URL
        # video_uuid, series_uuid = aws.upload_final_video_to_s3()
        # cdn_url = aws.get_cdn_url_video(series_uuid, video_uuid)

        # Update task status in database as completed
        print("Uploading to S3")
        # db.video_tasks_collection.update_one(
        #     {"video_task_id": video_data["video_id"]},
        #     {"$set": {"video_url": str(cdn_url), "task_status": "completed"}}
        # )

        # Upload all the prompt, images, bgm and audio files to S3
        # aws.upload_to_s3(video_data, series_uuid)

        # Delete all files in the Images, effects, audio, bgm and final video folder
        # print("Deleting temperory files")
        # delete_files_in_folder(str(secret_config.Image_folder_path))
        # delete_files_in_folder(str(secret_config.Audio_folder_path))
        # delete_files_in_folder("/Users/toheed/PanduAI/backend/workflow/BGM")
        # delete_files_in_folder("/Users/toheed/PanduAI/backend/workflow/Final_Video")
        # delete_files_in_folder(str(secret_config.Effect_folder_path))
        # delete_files_in_folder(str(secret_config.Scenes_folder_path))
        # delete_files_in_folder(str(secret_config.Transition_folder_path))

        # Set progress to 0 after completion
        progress["percentage"] = 0

    except Exception as e:
       
        # print("Deleting temperory files")
        # delete_files_in_folder(str(secret_config.Image_folder_path))
        # delete_files_in_folder(str(secret_config.Audio_folder_path))
        # delete_files_in_folder("/Users/toheed/PanduAI/backend/workflow/BGM")
        # delete_files_in_folder("/Users/toheed/PanduAI/backend/workflow/Final_Video")
        # delete_files_in_folder(str(secret_config.Effect_folder_path))
        # delete_files_in_folder(str(secret_config.Scenes_folder_path))
        # delete_files_in_folder(str(secret_config.Transition_folder_path))
        
        # Set progress to 0 after completion
        progress["percentage"] = 0
        # Handle any errors or exceptions
        print(f"Error generating video: {e}")

        # Update task status in database as failed
        # db.video_tasks_collection.update_one(
        #     {"video_task_id": video_data["video_id"]},
        #     {"$set": {"task_status": "failed"}}
        # )


user_input = {
    "prompt": "Most powerful dragons in the house of the dragon series",
    "query": "game of thrones battle no copyright background"
}

main(user_input)


# query = "minecraft video no copyright background"
# video_url = get_youtube_url(query)
# download_and_resize_video(video_url, temp_folder="Temp videos", output_folder="Video Backdrops")

