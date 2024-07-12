# import utils

# data = {'characters': ['B', 'o', 'r', 'n', ' ', 'a', 'n', 'd', ' ', 'r', 'a', 'i', 's', 'e', 'd', ' ', 'i', 'n', ' ', 't', 'h', 'e', ' ', 'c', 'h', 'a', 'r', 'm', 'i', 'n', 'g', ' ', 's', 'o', 'u', 't', 'h', ',', ' ', 'I', ' ', 'c', 'a', 'n', ' ', 'a', 'd', 'd', ' ', 'a', ' ', 't', 'o', 'u', 'c', 'h', ' ', 'o', 'f', ' ', 's', 'w', 'e', 'e', 't', ' ', 's', 'o', 'u', 't', 'h', 'e', 'r', 'n', ' ', 'h', 'o', 's', 'p', 'i', 't', 'a', 'l', 'i', 't', 'y', ' ', 't', 'o', ' ', 'y', 'o', 'u', 'r', ' ', 'a', 'u', 'd', 'i', 'o', 'b', 'o', 'o', 'k', 's', ' ', 'a', 'n', 'd', ' ', 'p', 'o', 'd', 'c', 'a', 's', 't', 's'], 
#         'character_start_times_seconds': [0.0, 0.186, 0.244, 0.302, 0.36, 0.395, 0.418, 0.441, 0.464, 0.511, 0.557, 0.604, 0.662, 0.697, 0.755, 0.778, 0.813, 0.848, 0.871, 0.906, 0.929, 0.952, 0.975, 1.022, 1.057, 1.115, 1.161, 1.219, 1.265, 1.312, 1.335, 1.358, 1.428, 1.474, 1.591, 1.649, 1.718, 1.788, 1.939, 2.183, 2.241, 2.31, 2.345, 2.38, 2.415, 2.45, 2.496, 2.543, 2.577, 2.612, 2.647, 2.694, 2.728, 2.786, 2.833, 2.879, 2.914, 2.961, 2.984, 3.007, 3.077, 3.135, 3.193, 3.239, 3.297, 3.332, 3.39, 3.437, 3.495, 3.529, 3.564, 3.587, 3.611, 3.669, 3.704, 3.75, 3.785, 3.843, 3.913, 3.982, 4.04, 4.11, 4.226, 4.296, 4.354, 4.412, 4.458, 4.528, 4.551, 4.586, 4.644, 4.667, 4.69, 4.725, 4.76, 4.818, 4.853, 4.911, 4.957, 5.039, 5.12, 5.19, 5.248, 5.294, 5.352, 5.445, 5.503, 5.584, 5.631, 5.666, 5.735, 5.782, 5.909, 5.991, 6.06, 6.235, 6.304, 6.42], 
#         'character_end_times_seconds': [0.186, 0.244, 0.302, 0.36, 0.395, 0.418, 0.441, 0.464, 0.511, 0.557, 0.604, 0.662, 0.697, 0.755, 0.778, 0.813, 0.848, 0.871, 0.906, 0.929, 0.952, 0.975, 1.022, 1.057, 1.115, 1.161, 1.219, 1.265, 1.312, 1.335, 1.358, 1.428, 1.474, 1.591, 1.649, 1.718, 1.788, 1.939, 2.183, 2.241, 2.31, 2.345, 2.38, 2.415, 2.45, 2.496, 2.543, 2.577, 2.612, 2.647, 2.694, 2.728, 2.786, 2.833, 2.879, 2.914, 2.961, 2.984, 3.007, 3.077, 3.135, 3.193, 3.239, 3.297, 3.332, 3.39, 3.437, 3.495, 3.529, 3.564, 3.587, 3.611, 3.669, 3.704, 3.75, 3.785, 3.843, 3.913, 3.982, 4.04, 4.11, 4.226, 4.296, 4.354, 4.412, 4.458, 4.528, 4.551, 4.586, 4.644, 4.667, 4.69, 4.725, 4.76, 4.818, 4.853, 4.911, 4.957, 5.039, 5.12, 5.19, 5.248, 5.294, 5.352, 5.445, 5.503, 5.584, 5.631, 5.666, 5.735, 5.782, 5.909, 5.991, 6.06, 6.235, 6.304, 6.42, 6.827]}

# result = utils.characters_to_words(data['characters'], data['character_start_times_seconds'], data['character_end_times_seconds'])

# print(result)

# from pathlib import Path
# from openai import OpenAI
# client = OpenAI()

# speech_file_path = Path(__file__).parent / "speech.mp3"
# response = client.audio.speech.create(
#   model="tts-1",
#   voice="alloy",
#   input="Today is a wonderful day to build something people love!"
# )

# response.stream_to_file(speech_file_path)


# ElevenLabs API practice
# voice = client.clone(
#     name="Alex",
#     description="An old American male voice with a slight hoarseness in his throat. Perfect for news", # Optional
#     files=["./output.mp3"],
# )

# audio = client.generate(text="Hi! I'm a cloned voice!", voice=voice)

# audio = client.generate(
#     text="नमस्ते, क्या मुझे एक गिलास पानी मिल सकता है",
#     # text="Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!",
#     voice=Voice(
#         voice_id='EXAVITQu4vr4xnSDxMaL',
#         settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, speaking_rate=0.8)
#     )
# )

# play(audio)

# OpenAI API practice

# import os
# from pathlib import Path
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv() 
# SECRET_KEY = os.getenv("OPEN_AI_SECRET_KEY")

# client = OpenAI(api_key=SECRET_KEY)

# speech_file_path = Path(__file__).parent / "speech.mp3"
# response = client.audio.speech.create(
#   model="tts-1",
#   voice="alloy",
#   input="Today is a wonderful day to build something people love!"
# )

# response.stream_to_file(speech_file_path)



# import openai
# import os
# from dotenv import load_dotenv

# load_dotenv() 
# OPEN_AI_SECRET_KEY = os.getenv('OPEN_AI_SECRET_KEY')


# client = openai.OpenAI(api_key=OPEN_AI_SECRET_KEY)

# # User input and series data
# user_series = {
#     "user_prompt": "Create a video montage of various natural landscapes, including forests, mountains, rivers, and wildlife.",
#     "videos": [],      
#     "audios": [],                   
#     "background_musics": [],          
#     "subtitles": [],                
#     "duration": "",                  
#     "font": "",                        
#     "style": "",                      
#     "theme": "",                 
#     "audience": "",                
#     "platform": [],  
#     "total_output": 1,
# }

# def generate_prompts(prompt, number_of_prompts, user_series, prompt_type):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": f"Create {number_of_prompts} similar but not identical prompts for {prompt_type}, each enhanced with more detail and creativity."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     prompts = response.choices[0].message.content.strip().split('\n\n')
#     user_series[prompt_type].extend(prompts)

# def generate_audio_scripts(video_prompts):
#     for prompt in video_prompts:
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": f"Create a voiceover script for a video about: {prompt}. Ensure the output is just a paragraph."},
#                 {"role": "user", "content": prompt}
#             ]
#         )

#         audio_script = response.choices[0].message.content.strip()
#         user_series["audios"].append(audio_script)

# def generate_bgm_prompts(video_prompts):
#     for prompt in video_prompts:
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": f"Create a background music prompt for a video about: {prompt}. Provide creative and suitable prompt that matched this video theme. Generate a prompt to pass to a background music generation tool. The prompt can include instruments and suitable music and starting and. Make it loopable. Ensure the output is just a prompt."},
#                 {"role": "user", "content": prompt}
#             ]
#         )

#         bgm_prompt = response.choices[0].message.content.strip()
#         user_series["background_musics"].append(bgm_prompt)

# # Generate prompts for videos
# generate_prompts(user_series["user_prompt"], user_series["total_output"], user_series, "videos")

# # Generate audio scripts based on video prompts
# generate_audio_scripts(user_series["videos"])

# # Generate background music prompts based on video prompts
# generate_bgm_prompts(user_series["videos"])

# for i, prompt in enumerate(user_series["background_musics"]):
#     print(f"BGM Prompt {i+1}: {prompt}")

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip

# # Load video and audio
# video = VideoFileClip('./stitched_video.mp4')
# audio = AudioFileClip('./born.mp3')

# # Sample JSON data
# subtitle_data = {
#     'words': ['Born', 'and', 'raised', 'in', 'the', 'charming', 'south,', 'I', 'can', 'add', 'a', 'touch', 'of', 'sweet', 'southern', 'hospitality', 'to', 'your', 'audiobooks', 'and', 'podcasts'],
#     'word_start_times': [0.0, 0.383, 0.499, 0.801, 0.894, 0.998, 1.405, 1.869, 1.997, 2.148, 2.334, 2.403, 2.659, 2.775, 3.123, 3.483, 4.226, 4.342, 4.516, 5.074, 5.19],
#     'word_end_times': [0.348, 0.453, 0.766, 0.859, 0.964, 1.335, 1.811, 1.916, 2.101, 2.299, 2.357, 2.612, 2.705, 3.053, 3.437, 4.168, 4.296, 4.458, 5.039, 5.143, 6.223]
# }

# # Function to generate styled text clips
# def generate_subtitle_clips(data, video):
#     clips = []
#     num_words = len(data['words'])
#     for i in range(num_words):
#         if i > 0 and i < num_words - 1:
#             # Middle word in different color
#             text_clip = TextClip(
#                 txt=f"{data['words'][i-1]} {data['words'][i]} {data['words'][i+1]}",
#                 fontsize=24,
#                 color='white',
#                 bg_color='black',
#                 method='caption',
#                 align='center',
#                 size=video.size
#             ).set_position('bottom').set_start(data['word_start_times'][i-1]).set_duration(data['word_end_times'][i+1] - data['word_start_times'][i-1])

#             middle_word_duration = data['word_end_times'][i] - data['word_start_times'][i]
#             # Change color of middle word
#             middle_text_clip = TextClip(
#                 txt=data['words'][i],
#                 fontsize=24,
#                 color='yellow',
#                 bg_color='black',
#                 method='caption',
#                 align='center',
#                 size=video.size
#             ).set_position('bottom').set_start(data['word_start_times'][i]).set_duration(middle_word_duration)

#             clips.append(text_clip)
#             clips.append(middle_text_clip)

#     return clips

# Generate subtitle clips
# subtitle_clips = generate_subtitle_clips(subtitle_data, video)

# # Composite video with subtitles
# final_video = CompositeVideoClip([video] + subtitle_clips).set_audio(audio)

# # Output final video
# final_video.write_videofile('pop_video_1.mp4', codec='libx264', audio_codec='aac')

print(TextClip.list('color'))
# print(TextClip.list('font'))

# subtitle_data = {
#     'words': ['Born', 'and', 'raised', 'in', 'the', 'charming', 'south,', 'I', 'can', 'add', 'a', 'touch', 'of', 'sweet', 'southern', 'hospitality', 'to', 'your', 'audiobooks', 'and', 'podcasts'],
#     'word_start_times': [0.0, 0.383, 0.499, 0.801, 0.894, 0.998, 1.405, 1.869, 1.997, 2.148, 2.334, 2.403, 2.659, 2.775, 3.123, 3.483, 4.226, 4.342, 4.516, 5.074, 5.19],
#     'word_end_times': [0.348, 0.453, 0.766, 0.859, 0.964, 1.335, 1.811, 1.916, 2.101, 2.299, 2.357, 2.612, 2.705, 3.053, 3.437, 4.168, 4.296, 4.458, 5.039, 5.143, 6.223]
# }
 

# Load your video clip
video_path = "/Users/toheed/PanduAI/backend/workflow/result/result_final_with_watermark_38.mp4"
clip = VideoFileClip(video_path)

# Get video dimensions
width, height = clip.size

# Calculate the aspect ratio
aspect_ratio = width / height

# Print the aspect ratio
print(f"Video Dimensions: {width}x{height}")
print(f"Aspect Ratio: {aspect_ratio:.2f}")



# # voiceover_prompt = enhanced_prompts["voiceover_prompt"]
# # video_prompt = enhanced_prompts["video_prompt"]
# # background_music = enhance_prompts['background_music']

# # generate the prompts
# # generate_audio(voiceover_prompt, "/Users/toheed/PanduAI/backend/workflow/audio/test1.mp3")

# # audio_path = "/Users/toheed/PanduAI/backend/workflow/audio/test1.mp3"
# # duration = int(get_audio_duration(audio_path))

# # video_fps = 30
# # frames = duration * video_fps

# # prompt = video_prompt
# # negative_prompt = 'sky'
# # aspect_ratio = '9:16'
# # seed = 42
# # output_format = 'png'
# # output_video_path = '/Users/toheed/PanduAI/backend/workflow/video/generate_pan_left_to_right_video.mp4'

# # # Compile everything
# # create_video_with_audio_and_text(
# #     video_path="/Users/toheed/PanduAI/backend/workflow/video/generate_pan_left_to_right_video.mp4",
# #     audio_path="/Users/toheed/PanduAI/backend/workflow/audio/test1.mp3",
# #     output_path="/Users/toheed/PanduAI/backend/workflow/result/result1.mp4",
# #     text=voiceover_prompt,
# #     text_speed_factor=0.5
# # )

# @app.post("/video_tasks/", response_model=VideoTask)
# async def create_video_task(video_task: VideoTask, session: SessionContainer = Depends(verify_session())):
#     video_task_data = video_task.dict(by_alias=True)
#     video_task_data["created_at"] = datetime.now()
#     result = await db.video_tasks_collection.insert_one(video_task_data)
#     new_video_task = await db.video_tasks_collection.find_one({"_id": result.inserted_id})
#     return VideoTask(**new_video_task)

# @app.get("/video_tasks/", response_model=List[VideoTask])
# async def read_video_tasks(skip: int = 0, limit: int = 10, session: SessionContainer = Depends(verify_session())):
#     video_tasks_cursor = db.video_tasks_collection.find().skip(skip).limit(limit)
#     video_tasks = await video_tasks_cursor.to_list(length=limit)
#     return video_tasks

# @app.get("/video_tasks/{video_task_id}", response_model=VideoTask)
# async def read_video_task(video_task_id: str, session: SessionContainer = Depends(verify_session())):
#     video_task = await db.video_tasks_collection.find_one({"_id": ObjectId(video_task_id)})
#     if video_task is None:
#         raise HTTPException(status_code=404, detail="VideoTask not found")
#     return VideoTask(**video_task)

# @app.put("/video_tasks/{video_task_id}", response_model=VideoTask)
# async def update_video_task(video_task_id: str, video_task: VideoTask, session: SessionContainer = Depends(verify_session())):
#     video_task_data = video_task.dict(by_alias=True, exclude_unset=True)
#     video_task_data["updated_at"] = datetime.now()
#     result = await db.video_tasks_collection.update_one({"_id": ObjectId(video_task_id)}, {"$set": video_task_data})
#     if result.matched_count == 0:
#         raise HTTPException(status_code=404, detail="VideoTask not found")
#     updated_video_task = await db.video_tasks_collection.find_one({"_id": ObjectId(video_task_id)})
#     return VideoTask(**updated_video_task)

# @app.delete("/video_tasks/{video_task_id}", response_model=VideoTask)
# async def delete_video_task(video_task_id: str, session: SessionContainer = Depends(verify_session())):
#     video_task = await db.video_tasks_collection.find_one({"_id": ObjectId(video_task_id)})

#     if video_task is None:
#         raise HTTPException(status_code=404, detail="VideoTask not found")
    
#     await db.video_tasks_collection.delete_one({"_id": ObjectId(video_task_id)})
    
#     return VideoTask(**video_task)



# FROM python:3.11-slim

# WORKDIR /app

# COPY . /app

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
