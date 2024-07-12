from openai import OpenAI
from pathlib import Path
import os
from pydub import AudioSegment
import requests
import numpy as np
import cv2
import base64
from dotenv import load_dotenv
from config import secret_config

load_dotenv() 

OPEN_AI_SECRET_KEY = secret_config.OPEN_AI_SECRET_KEY
STABILITY_SECRET_KEY = secret_config.STABILITY_SECRET_KEY

client = OpenAI(api_key=OPEN_AI_SECRET_KEY)
stability_api = STABILITY_SECRET_KEY

def split_transcript(transcript, user_series, series_key):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Split the following transcript into multiple logical paragraphs."},
            {"role": "user", "content": transcript}
        ]
    )
    parts = response.choices[0].message.content.strip().split('\n\n')
    user_series[series_key] = parts

def get_split_transcripts(user_series):
    split_transcripts = []
    for key in user_series:
        split_transcripts.extend(user_series[key])
    return split_transcripts

def generate_audio(text, output_file_path, model="tts-1", voice="alloy"):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    with open(output_file_path, 'wb') as audio_file:
        audio_file.write(response.content)

def process_transcript_to_audio(transcript, base_output_path, user_series_key):
    user_series = {}
    split_transcript(transcript, user_series, user_series_key)
    parts = user_series[user_series_key]

    output_dir = os.path.dirname(base_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, part in enumerate(parts):
        output_file_path = f"{base_output_path}_part_{i+1}.mp3"
        generate_audio(part, output_file_path)

# Example usage
transcript = """
Welcome, heroes! Today, we're diving into the incredible world of "My Hero Academia" to explore the top 10 strongest quirks. Starting off at number 10, we have Dark Shadow! Fumikage Tokoyami's quirk allows him to control a sentient shadow-like being. It's incredibly powerful in dark environments, but it requires careful handling to prevent it from going berserk. At number 9, it's Explosion! Katsuki Bakugo's quirk enables him to create powerful explosions from his sweat. This quirk grants him incredible offensive capabilities and unmatched mobility in battle. Coming in at number 8, we have Half-Cold Half-Hot! Shoto Todoroki's dual quirk lets him control both ice and fire. This versatility makes him a formidable opponent, able to adapt to any situation. Number 7 goes to Hellflame! Endeavor's quirk allows him to generate and control intense flames. His mastery over this power makes him one of the most feared pro heroes. In the sixth spot, we have Rewind! Eri's quirk grants her the ability to reverse a person's state, effectively healing injuries or undoing transformations. It's an incredibly potent ability, but it's also very difficult to control. Halfway through at number 5, it's Permeation! Mirio Togata can phase through solid objects, making him nearly untouchable in combat. With his training and skill, this quirk makes him a top-tier hero. At number 4, we have Decay! Tomura Shigaraki's terrifying quirk causes anything he touches to disintegrate. This deadly ability makes him a formidable villain and a significant threat to all heroes. Taking the third spot, it's One For All! This legendary quirk passed down to All Might grants its user immense strength, speed, and agility. It's the ultimate power-up for any hero. Runner-up at number 2 is One For All, wielded by Izuku Midoriya! As the current holder of One For All, Deku continues to unlock its full potential, making him one of the strongest heroes in training. And finally, the number 1 strongest quirk in "My Hero Academia" is All For One! This villainous quirk allows its user to steal and combine multiple quirks, creating an arsenal of abilities. All For One's power and versatility make him the most formidable opponent in the series. There you have it, folks! The top 10 strongest quirks in "My Hero Academia." Did your favorite make the list? Let us know in the comments below. Don't forget to like, subscribe, and hit that notification bell for more awesome content. Until next time, keep aspiring to be the best hero you can be!
"""

# process_transcript_to_audio(transcript, "output/audio_file", "video_transcript")

def get_audio_duration(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_in_seconds = len(audio) / 1000  # Length of audio in seconds
        return duration_in_seconds
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_durations_from_folder(folder_path):
    durations = []
    print(folder_path)  
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".mp3") or filename.endswith(".wav"):
                audio_path = os.path.join(folder_path, filename)
                duration = get_audio_duration(audio_path)
                if duration is not None:
                    durations.append(duration)
    except Exception as e:
        print(f"Error processing folder: {e}")
    return durations

def generate_image_stability(api_key, prompt, negative_prompt, aspect_ratio, seed, output_format):
    url = 'https://api.stability.ai/v2beta/stable-image/generate/ultra'
    
    # Define your parameters
    data = {
        "prompt": (None, prompt),
        "negative_prompt": (None, negative_prompt),
        "aspect_ratio": (None, aspect_ratio),
        "seed": (None, str(seed)),  # Ensure seed is a string
        "output_format": (None, output_format)
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'image/*'  # Specify the expected response type
    }

    # Make the request to the Stability API
    response = requests.post(url, headers=headers, files=data)

    # Check the response
    if response.status_code == 200:
        # Convert response content to a NumPy array
        np_arr = np.frombuffer(response.content, np.uint8)
        # Decode the image array into an OpenCV image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Failed to generate image. Status code: {response.status_code}")
        print(response.text)
        return None

def generate_image_open_ai(prompt, size="1024x1792", quality="standard", n=1):

    try:

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=n
        )

        # Extract the URL of the generated image
        image_url = response.data[0].url

        # Download the image
        image_response = requests.get(image_url)
        
        if image_response.status_code == 200:
            # Convert response content to a NumPy array
            np_arr = np.frombuffer(image_response.content, np.uint8)
            # Decode the image array into an OpenCV image
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Failed to download image. Status code: {image_response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def generate_image_engine_stability(prompt, style_preset, api_key = stability_api, cfg_scale=7, engine_id="stable-diffusion-xl-1024-v1-0", height=1344, width=768, samples=1, steps=30, api_host='https://api.stability.ai'):
    if api_key is None:
        raise Exception("Missing Stability API key.")
    print(prompt)
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": prompt
                }
            ],
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
            "style_preset": style_preset,

        },
    )

    if response.status_code != 200:
        print(str(response.text))
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    os.makedirs("./out", exist_ok=True)
    for i, image in enumerate(data["artifacts"]):
        with open(f"./out/v1_txt2img_{i}.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))

    # Check the response
    if response.status_code == 200:
        # Convert the base64 string to a byte array
        image_data = base64.b64decode(data["artifacts"][0]["base64"])
        # Convert byte array to a NumPy array
        np_arr = np.frombuffer(image_data, np.uint8)
        # Decode the NumPy array into an OpenCV image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Failed to generate image. Status code: {response.status_code}")
        print(response.text)
        return None
    
def generate_multiple_images_with_audio(api_key, prompts, total_images, negative_prompt="", aspect_ratio="1:1", seed=42, output_format="jpeg", output_dir="generated_images"):
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if total_images is at least as long as prompts
    print(prompts)
    print(total_images)
    # if len(prompts) > len(total_images):
    #     raise ValueError("Number of total_images should be at least equal to the number of prompts.")

    # Iterate over each prompt and generate the specified number of images
    for i, prompt in enumerate(prompts):
        num_images = total_images[i]
        for j in range(num_images):
            image = generate_image_engine_stability(prompt)
            if image is not None:
                output_file_path = Path(output_dir) / f"image_{i+1}_{j+1}.{output_format}"
                cv2.imwrite(str(output_file_path), image)
                print(f"Generated image for prompt: {prompt}")
            else:
                print(f"Failed to generate image for prompt: {prompt}")

def generate_multiple_images(api_key, prompts, negative_prompt="", aspect_ratio="1:1", seed=42, output_format="jpeg", output_dir="generated_images"): 
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate over each prompt and generate one image per prompt
    for i, prompt in enumerate(prompts):
        image = generate_image_open_ai(prompt)
        if image is not None:
            output_file_path = Path(output_dir) / f"image_{i+1}.{output_format}"
            cv2.imwrite(str(output_file_path), image)
            print(f"Generated image for prompt: {prompt}")
        else:
            print(f"Failed to generate image for prompt: {prompt}")

def generate_multiple_images_with_stability_engine(prompt, style_preset, image_name, output_format="jpeg", output_dir="generated_images"):
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image = generate_image_engine_stability(prompt, style_preset)
    if image is not None:
        output_file_path = Path(output_dir) / f"image_{image_name}.{output_format}"
        cv2.imwrite(str(output_file_path), image)
        print(f"Generated image for prompt: {prompt}")
    else:
        print(f"Failed to generate image for prompt: {prompt}")


# transcript = """
# Welcome, heroes! Today, we're diving into the incredible world of "My Hero Academia" to explore the top 10 strongest quirks. Starting off at number 10, we have Dark Shadow! Fumikage Tokoyami's quirk allows him to control a sentient shadow-like being. It's incredibly powerful in dark environments, but it requires careful handling to prevent it from going berserk. At number 9, it's Explosion! Katsuki Bakugo's quirk enables him to create powerful explosions from his sweat. This quirk grants him incredible offensive capabilities and unmatched mobility in battle. Coming in at number 8, we have Half-Cold Half-Hot! Shoto Todoroki's dual quirk lets him control both ice and fire. This versatility makes him a formidable opponent, able to adapt to any situation. Number 7 goes to Hellflame! Endeavor's quirk allows him to generate and control intense flames. His mastery over this power makes him one of the most feared pro heroes. In the sixth spot, we have Rewind! Eri's quirk grants her the ability to reverse a person's state, effectively healing injuries or undoing transformations. It's an incredibly potent ability, but it's also very difficult to control. Halfway through at number 5, it's Permeation! Mirio Togata can phase through solid objects, making him nearly untouchable in combat. With his training and skill, this quirk makes him a top-tier hero. At number 4, we have Decay! Tomura Shigaraki's terrifying quirk causes anything he touches to disintegrate. This deadly ability makes him a formidable villain and a significant threat to all heroes. Taking the third spot, it's One For All! This legendary quirk passed down to All Might grants its user immense strength, speed, and agility. It's the ultimate power-up for any hero. Runner-up at number 2 is One For All, wielded by Izuku Midoriya! As the current holder of One For All, Deku continues to unlock its full potential, making him one of the strongest heroes in training. And finally, the number 1 strongest quirk in "My Hero Academia" is All For One! This villainous quirk allows its user to steal and combine multiple quirks, creating an arsenal of abilities. All For One's power and versatility make him the most formidable opponent in the series. There you have it, folks! The top 10 strongest quirks in "My Hero Academia." Did your favorite make the list? Let us know in the comments below. Don't forget to like, subscribe, and hit that notification bell for more awesome content. Until next time, keep aspiring to be the best hero you can be!
# """
# user_series = {}
# series_key = "example_transcript"
# split_transcript(transcript, user_series, series_key) 1

# prompts = get_split_transcripts(user_series) 2

#generate_multiple_images(api_key, prompts, total_images) 3