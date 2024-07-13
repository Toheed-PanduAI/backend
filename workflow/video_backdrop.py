import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
import warnings
from pathlib import Path
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, TextClip, CompositeVideoClip, CompositeAudioClip
import moviepy.audio.fx.all as afx
from moviepy.video.tools.drawing import color_split
import image_effects
import audio_prompts
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, play
import scripts.speech_synthesis as speech_synthesis
import random
from dotenv import load_dotenv
import requests
from pytube import YouTube
from utils.util import text_styles, effect_images
from dotenv import load_dotenv

load_dotenv() 

OPEN_AI_SECRET_KEY = os.getenv('OPEN_AI_SECRET_KEY')
ELEVEN_LABS_SECRET_KEY = os.getenv('ELEVEN_LABS_SECRET_KEY')
STABILITY_SECRET_KEY = os.getenv('STABILITY_SECRET_KEY')

warnings.filterwarnings("ignore", category=DeprecationWarning)  

client = OpenAI(api_key=OPEN_AI_SECRET_KEY)

elevan_labs_client = ElevenLabs(
  api_key= ELEVEN_LABS_SECRET_KEY
)

effects = random.choice(list(effect_images.values()))

filenumber = 40

user_series = {
    "user_prompt": "Create a series of Minecraft tutorial videos. For each video, focus on a specific topic such as building techniques, Redstone contraptions, survival tips, and exploring different biomes. Provide clear, step-by-step instructions and tips for both beginners and advanced players. Use engaging visuals of in-game footage to demonstrate each concept. Highlight common mistakes and how to avoid them. Incorporate creative builds and challenges to inspire viewers. Ensure each video is informative, entertaining, and designed to help viewers improve their Minecraft skills and creativity. Use royalty-free background music to keep the content engaging without copyright issues.",
    "videos": [],      
    "audios": [], 
    "images": [[]],    
    "background_musics": [],          
    "subtitles": [],                
    "duration": 60,                  
    "font": "",                        
    "style": "",                      
    "theme": "",                 
    "audience": "",                
    "platform": [],  
    "total_output_videos": 2,
    "video_transiton": {"name":"generate_inward_vignette_transition_video"},
    "image_effects": {"name": effects},
    "query" : "minecraft video no copyright background"
}

def generate_prompts(prompt, number_of_prompts, user_series, prompt_type):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Create {number_of_prompts} similar but not identical prompts for {prompt_type}, each enhanced with more detail and creativity."},
            {"role": "user", "content": prompt}
        ]
    )

    prompts = response.choices[0].message.content.strip().split('\n\n')
   
    user_series[prompt_type].extend(prompts)

def generate_image_prompts(prompt, number_of_prompts, prompt_type, duration=60, image_time=5):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Create {number_of_prompts} similar but not identical prompts for {prompt_type}, each enhanced with more detail and creativity. There are time stamps so each image should be unique, I am creating a video of {duration} seconds so each image will be displaced for {image_time} seconds."},
            {"role": "user", "content": prompt}
        ]
    )

    prompts = response.choices[0].message.content.strip().split('\n\n')
    return prompts

def generate_audio_scripts(video_prompts):
    for prompt in video_prompts:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Create a voiceover script for a video about: {prompt}. Ensure the output is just a paragraph."},
                {"role": "user", "content": prompt}
            ]
        )

        audio_script = response.choices[0].message.content.strip()
        user_series["audios"].append(audio_script)

def generate_bgm_prompts(video_prompts):
    for prompt in video_prompts:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Create a background music prompt for a video about: {prompt}. Provide creative and suitable prompt that matched this video theme. Generate a prompt to pass to a background music generation tool. The prompt can include instruments and suitable music and starting and. Make it loopable. Ensure the output is just a prompt. It should not exceed 400 characters, it should be maximum of 400 characters"},
                {"role": "user", "content": prompt}
            ]
        )

        bgm_prompt = response.choices[0].message.content.strip()
        user_series["background_musics"].append(bgm_prompt)

def get_audio_duration(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_in_seconds = len(audio) / 1000  # Length of audio in seconds
        return duration_in_seconds

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def enhance_prompts(prompts):

    enhanced_prompts = {}

    for key, prompt in prompts.items():
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Enhance this prompt with more detail and creativity."},
                {"role": "user", "content": prompt}
            ]
        )

        enhanced_prompts[key] = response.choices[0].message.content

    return enhanced_prompts

def get_prompts(test_data):
    prompts = {
        "video_prompt": test_data["video"]["prompt"],
        "background_music": test_data["audio"]["prompt"],
        "voiceover_prompt": test_data["text_script"]["voiceover_prompt"]
    }
    return prompts

def youtube_search(query: str):
    
    api_key = os.getenv("SERPAPI_KEY")  # Assuming the SERPAPI_KEY is stored in the .env file
    url = "https://serpapi.com/search"

    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
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

    # Calculate the amount to crop from the sides
    # crop_amount = (video_clip.size[0] - target_width) / 2

    # # Crop the video
    # cropped_video = video_clip.crop(x1=crop_amount, x2=video_clip.size[0] - crop_amount)

    # # Resize the video to maintain the aspect ratio
    # resized_video = cropped_video.resize(newsize=(target_width, target_height))

    # Save the resized video in the output folder
    # output_path = os.path.join(output_folder, "resized_video.mp4")

    resized_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def generate_audio(text, output_file_path, model="tts-1", voice="alloy"):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    response.stream_to_file(Path(output_file_path))

def transcribe_audio(file_path, model="whisper-1"):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file
        )
    return transcription.text

def match_durations(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    min_duration = min(video.duration, audio.duration)

    video = video.subclip(0, min_duration)
    audio = audio.subclip(0, min_duration)

    video = video.set_audio(audio)

    video.write_videofile(output_path, codec='libx264', audio_codec='aac')

def generate_subtitle_clips(data, video):
    if data is None or 'words' not in data or 'word_start_times' not in data or 'word_end_times' not in data:
        raise ValueError("Invalid or missing data structure for subtitles.")
    
    clips = []
    font_style = random.choice(list(text_styles.values()))
    y_position = video.size[1] * 0.75
    
    for word, start, end in zip(data['words'], data['word_start_times'], data['word_end_times']):
        text_clip = TextClip(
            txt=word,
            size=font_style['size'],
            color=font_style['color'], 
            fontsize=font_style['fontsize'],
            bg_color=font_style['bg_color'],
            font=font_style['font'],
            stroke_color=font_style['stroke_color'],
            stroke_width=font_style['stroke_width'],
            method=font_style['method'],
            kerning=font_style['kerning'],
            align=font_style['align'],
            interline=font_style['interline'],
            transparent=font_style['transparent'],
            remove_temp=font_style['remove_temp'],
            print_cmd=font_style['print_cmd']
        ).set_position(('center', y_position)).set_start(start).set_duration(end - start)

        clips.append(text_clip)
    return clips

def add_watermark(video_path, output_path, audio_path, bgm_path, subtitle_data, text=None, image_path=None, position=('center', 'center'), 
                  font='Arial', font_size=24, color='white', opacity=0.5, padding=10, bgm_volume=0.5):
    """
    Add a watermark to a video.
    
    :param video_path: Path to the input video file.
    :param output_path: Path to save the output video file.
    :param text: Text to use as a watermark (optional).
    :param image_path: Path to an image to use as a watermark (optional).
    :param position: Position of the watermark in the video ('center', 'center' by default).
    :param font: Font type for the text watermark.
    :param font_size: Font size for the text watermark.
    :param color: Font color for the text watermark.
    :param opacity: Opacity level for the watermark (0 to 1).
    """
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    bgm_audio = AudioFileClip(bgm_path)
    
    min_duration = min(video.duration, audio.duration)

    video = video.subclip(0, min_duration)
    audio = audio.subclip(0, min_duration)

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
    else:
        raise ValueError("Either text or image_path must be provided for the watermark.")

    subtitle_clips = generate_subtitle_clips(subtitle_data, video)

    # Composite the video with the watermark
    final_video = CompositeVideoClip([video, watermark] + subtitle_clips)

    # Write the result to a file
    final_video.write_videofile(output_path, codec='libx264', fps=video.fps, audio_codec='aac')

def create_video_with_audio_and_text(video_path, audio_path, bgm_path, output_path, subtitle_data, text=None, image_path=None, position=('center', 'center'), 
                  font='Arial', font_size=24, color='white', opacity=0.5, padding=10, bgm_volume=0.5):
   
    # Load the video and audio files
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    bgm_audio = AudioFileClip(bgm_path)

    min_duration = min(video.duration, audio.duration)

    video = video.subclip(0, min_duration)
    audio = audio.subclip(0, min_duration)

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

# Generae the prompt for series
generate_prompts(user_series["user_prompt"], user_series["total_output_videos"], user_series, "videos")

generate_audio_scripts(user_series["videos"])

generate_bgm_prompts(user_series["videos"])


# # Audio and BGM
bgm_path = f"/Users/toheed/PanduAI/backend/workflow/bgm/bgm{filenumber}.mp3"
audio_path = f"/Users/toheed/PanduAI/backend/workflow/audio/audio{filenumber}.mp3"

subtitle_data = speech_synthesis.generate_tts_with_timestamps(user_series["audios"][0], audio_path)
# bgm_data = speech_synthesis.generate_and_save_sound(user_series["background_musics"][0], bgm_path)


output_path= f"/Users/toheed/PanduAI/backend/workflow/result/result_final_with_watermark_{filenumber}.mp4"
video_path= f"/Users/toheed/PanduAI/backend/workflow/video/stitched_video{filenumber}.mp4"


# Get youtube video for background  

video_url = get_youtube_url(user_series["query"])
download_and_resize_video(video_url, temp_folder="Temp videos", output_path=video_path)


# Compile everything
create_video_with_audio_and_text(
    video_path= video_path,
    audio_path=f"/Users/toheed/PanduAI/backend/workflow/audio/audio{filenumber}.mp3",
    bgm_path=f"/Users/toheed/PanduAI/backend/workflow/bgm/bgm{filenumber}.mp3",
    output_path= output_path,
    subtitle_data=subtitle_data,
    text="Pandu AI", 
    position=("right", "top"), 
    font="Helvetica-Bold", 
    font_size=50, 
    color="gray", 
    opacity=0.8,
    padding=10,
    bgm_volume=0.15
)



