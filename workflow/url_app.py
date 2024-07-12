#!/usr/bin/env python
import sys
import ssl
from openai import OpenAI
from bs4 import BeautifulSoup
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
import warnings
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, CompositeAudioClip
import moviepy.audio.fx.all as afx
import image_effects
import audio_prompts
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, play
import scripts.speech_synthesis as speech_synthesis
import random
from utils.util import text_styles, effect_images
from dotenv import load_dotenv
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

effects = random.choice(list(effect_images.values()))

filenumber = 22

user_series = {
    "user_prompt": "Create a compelling video exploring the fundamentals of cryptocurrency titled 'Exploring Cryptocurrency: A Beginner's Guide.' Begin with an introduction highlighting the growing relevance of digital currencies and their impact on global finance. Discuss the origins of cryptocurrency, how it operates through blockchain technology, and its broader implications for financial transactions.",
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
    "total_output": 2,
    "video_transiton": {"name":"generate_inward_vignette_transition_video"},
    "image_effects": {"name": effects},
}

def generate_prompts_with_url(prompt, url_data, number_of_prompts, user_series, prompt_type):
    
    # Combine the prompt and url_data details
    combined_prompt = f"{prompt}\n\nAdditional Details and topics:\n{url_data}"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Create {number_of_prompts} similar but not identical prompts for {prompt_type}, each enhanced with more detail and creativity."},
            {"role": "user", "content": combined_prompt}
        ]
    )
    
    prompts = response.choices[0].message.content.strip().split('\n\n')
    
    user_series[prompt_type].extend(prompts)

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

def generate_image_prompts(prompt, number_of_prompts, prompt_type):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Create {number_of_prompts} similar but not identical prompts for {prompt_type}, each enhanced with more detail and creativity."},
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
                {"role": "system", "content": f"Create a background music prompt for a video about: {prompt}. Provide creative and suitable prompt that matched this video theme. Generate a prompt to pass to a background music generation tool. The prompt can include instruments and suitable music and starting and. Make it loopable. Ensure the output is just a prompt. It should not exceed a maximum number of 400 characters"},
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
            font=font_style['font'],
            bg_color=font_style['bg_color'],
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

def create_video_with_audio_and_text(video_path, audio_path, bgm_path, output_path, text, text_speed_factor, subtitle_data, bgm_volume=0.5):
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

    subtitle_clips = generate_subtitle_clips(subtitle_data, video)

    # Create the final composite video clip
    final = CompositeVideoClip([video] + subtitle_clips)

    # Export the final video
    final.write_videofile(output_path, codec='libx264', audio_codec='aac')

def get_scraped_data(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a highly skilled assistant that extracts structured information from provided HTML data."},
            {"role": "user", "content": prompt}
        ]
    )

    response = response.choices[0].message.content.strip().split('\n\n')
   
    return response

def fetch_data(url):
    import sys
    import ssl

    # Modify SSL context to bypass SSL certificate verification
    ssl._create_default_https_context = ssl._create_unverified_context

    proxy_handler = {
        'http': 'http://brd-customer-hl_ddda9220-zone-pandu_ai_country_based_premium:72z7l80d5f7t@brd.superproxy.io:22225',
        'https': 'http://brd-customer-hl_ddda9220-zone-pandu_ai_country_based_premium:72z7l80d5f7t@brd.superproxy.io:22225'
    }

    if sys.version_info[0] == 2:
        # Python 2 specific imports and code
        import six
        from six.moves.urllib import request

        # Set up a proxy handler
        opener = request.build_opener(request.ProxyHandler(proxy_handler))

    elif sys.version_info[0] == 3:
        # Python 3 specific imports and code
        import urllib.request

        # Set up a proxy handler
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxy_handler))

    # Fetch and return the data
    response = opener.open(url)
    return response.read().decode('utf-8')

# Url scraping
# url = 'https://www.accuweather.com/en/in/bengaluru/204108/weather-forecast/204108'
# scraped_data = fetch_data(url)

with open('/Users/toheed/PanduAI/backend/workflow/index.html', 'r', encoding='utf-8') as file:
    scraped_data = file.read()

soup = BeautifulSoup(scraped_data, 'html.parser')

# Extract data: example extracting all paragraphs and headings
paragraphs = soup.find_all('p')
headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
lists = soup.find_all(['li'])

paragraphs = [para.get_text() for para in soup.find_all('p')]
headings = [heading.get_text() for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
lists = [item.get_text() for item in soup.find_all('li')]


extracted_data = {
    'paragraphs': paragraphs,
    'headings': headings,
    'lists': lists
}

# Generae the prompt for series
generate_prompts_with_url(user_series["user_prompt"], extracted_data, user_series["total_output"], user_series, "videos")

generate_audio_scripts(user_series["videos"])

generate_bgm_prompts(user_series["videos"])

audio_prompts.split_transcript(user_series["audios"][0], user_series, "audios")

# Effects and Images

api_key = STABILITY_SECRET_KEY
prompts = user_series["audios"]
negative_prompt = 'fire'
aspect_ratio = '9:16'
seed = 12345
output_format = 'jpeg'
output_folder = '/Users/toheed/PanduAI/backend/Images'
input_video_folder = '/Users/toheed/PanduAI/backend/Videos'
# output_video_path = f"/Users/toheed/PanduAI/backend/workflow/video/stitched_video{filenumber}.mp4"
output_video_path = '/Users/toheed/PanduAI/backend/workflow/video/'
input_image_folder = '/Users/toheed/PanduAI/backend/Images'
output_video_folder = '/Users/toheed/PanduAI/backend/Videos'
output_filename = f"stitched_video{filenumber}.mp4"

folder_path = "/Users/toheed/PanduAI/backend/output"
durations = audio_prompts.get_durations_from_folder(folder_path)

totalimages = [round(i / 5) for i in durations]

audio_prompts.generate_multiple_images(api_key, prompts, totalimages, output_dir=output_folder)

# retrive the effect function using name
effect_function_name = user_series["image_effects"]["name"]
effect_function = getattr(image_effects, effect_function_name)
print("Applying effects:", effect_function)
image_effects.process_multiple_images(input_image_folder, output_video_folder, effect_function)

# for image_path in os.listdir(output_folder):
#     print(image_path)
#     effect_function_name = random.choice(list(effect_images.values()))
#     effect_function = getattr(image_effects, effect_function_name)
#     print(f"Applying effect {effect_function_name} to image {image_path}")
#     image_effects.process_image(image_path, output_video_folder, effect_function)

# retrive the transition function using name
transition_function_name = user_series["video_transiton"]["name"]
transition_function = getattr(image_effects, transition_function_name)
print("Applying transitions:", transition_function)
# image_effects.stitch_videos_with_transition(input_video_folder, output_video_path, transition_function, transition_params={'frames': 30, 'fps': 10, 'frame_repeat': 1})
image_effects.stitch_videos(input_video_folder, output_video_path, output_filename)

# Audio and BGM
bgm_path = f"/Users/toheed/PanduAI/backend/workflow/bgm/bgm{filenumber}.mp3"
audio_path = f"/Users/toheed/PanduAI/backend/workflow/audio/audio{filenumber}.mp3"

# generate_audio(user_series["audios"][0], audio_path)
subtitle_data = speech_synthesis.generate_tts_with_timestamps(user_series["audios"][0], audio_path)
bgm_data = speech_synthesis.generate_and_save_sound(user_series["background_musics"][0], bgm_path)

# subtitle_data = {
#     'words': ['Born', 'and', 'raised', 'in', 'the', 'charming', 'south,', 'I', 'can', 'add', 'a', 'touch', 'of', 'sweet', 'southern', 'hospitality', 'to', 'your', 'audiobooks', 'and', 'podcasts'],
#     'word_start_times': [0.0, 0.383, 0.499, 0.801, 0.894, 0.998, 1.405, 1.869, 1.997, 2.148, 2.334, 2.403, 2.659, 2.775, 3.123, 3.483, 4.226, 4.342, 4.516, 5.074, 5.19],
#     'word_end_times': [0.348, 0.453, 0.766, 0.859, 0.964, 1.335, 1.811, 1.916, 2.101, 2.299, 2.357, 2.612, 2.705, 3.053, 3.437, 4.168, 4.296, 4.458, 5.039, 5.143, 6.223]
# }
    
duration = int(get_audio_duration(audio_path))
video_fps = 30
frames = duration * video_fps

# Compile everything
create_video_with_audio_and_text(
        video_path= f"/Users/toheed/PanduAI/backend/workflow/video/stitched_video{filenumber}.mp4",
        audio_path=audio_path,
        bgm_path=bgm_path,
        output_path= f"/Users/toheed/PanduAI/backend/workflow/result/result_final_{filenumber}.mp4",
        text=user_series["audios"][0],
        text_speed_factor=0.5,
        subtitle_data=subtitle_data,
        bgm_volume=0.15
    )











# chunk_size = 2000  # Define a suitable chunk size
# chunks = [scraped_data[i:i + chunk_size] for i in range(0, len(scraped_data), chunk_size)]

# # Sending each chunk separately
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}/{len(chunks)}: {chunk}")

# # Create the prompt
# prompt = f"""
# The following is the scraped HTML data:

# {scraped_data}

# Extract the following details:
# 1. Title of the page
# 2. Main headings
# 3. Any lists of items
# 4. Key information and summaries
# 5. Any specific data points like dates, numbers, or locations
# 6. Any other relevant information

# Please provide the extracted information in a clear and structured format.
# """
