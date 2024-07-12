import requests
from database.models import StabilityGenerateImageRequest, StabilityImageToVideoRequest, HeygenVideoGenerateRequest
from fastapi import HTTPException,  File, UploadFile, Form
from openai import OpenAI
import os
from dotenv import load_dotenv
from config import secret_config

load_dotenv() 

OPEN_AI_SECRET_KEY = secret_config.OPEN_AI_SECRET_KEY

client = OpenAI(api_key=OPEN_AI_SECRET_KEY)

response = client.images.generate(
  model="dall-e-3",
  prompt="a realistic man smokiing cuban cigar in italy",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)


def stability_generate_image(api_key, data: StabilityGenerateImageRequest):
    url = f"https://api.stability.ai/v2beta/stable-image/generate/{data.model}"
    headers = {
        # "authorization": f"Bearer sk-MYAPIKEY",
        "authorization": api_key,
        "accept": "image/*"
    }
    request_data = {
        "prompt": data.prompt,
        "output_format": data.output_format
    }

    if data.negative_prompt:
        request_data["negative_prompt"] = data.negative_prompt
    if data.aspect_ratio:
        request_data["aspect_ratio"] = data.aspect_ratio
    if data.seed:
        request_data["seed"] = data.seed

    response = requests.post(url, headers=headers, files={"none": ''}, data=request_data)

    if response.status_code == 200:
        return response.content
    else:
        raise HTTPException(status_code=response.status_code, detail=str(response.json()))
    
def stability_image_to_video_start(api_key, image_file: UploadFile, data: StabilityImageToVideoRequest):
    url = f"https://api.stability.ai/v2beta/image-to-video"
    headers = {
        # "authorization": f"Bearer sk-MYAPIKEY",
        "authorization": api_key,

    }
    files = {
        "image": (image_file.filename, image_file.file, image_file.content_type)
    }
    data_dict = {
        "seed": data.seed,
        "cfg_scale": data.cfg_scale,
        "motion_bucket_id": data.motion_bucket_id
    }

    response = requests.post(url, headers=headers, files=files, data=data_dict)

    if response.status_code == 200:
        return response.json().get('id')
    else:
        raise HTTPException(status_code=response.status_code, detail=str(response.json()))

def stability_fetch_video_result(api_key, generation_id: str):
    url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
    headers = {
        'accept': "video/*",  # Use 'application/json' to receive base64 encoded JSON
        # 'authorization': f"Bearer sk-MYAPIKEY",
        "authorization": api_key,
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 202:
        return "Generation in-progress, try again in 10 seconds."
    elif response.status_code == 200:
        with open("video.mp4", 'wb') as file:
            file.write(response.content)
        return "Generation complete! Video saved as 'video.mp4'."
    else:
        raise HTTPException(status_code=response.status_code, detail=str(response.json()))

def segmind_image_generate(api_key, data, model_name="face-to-sticker"):
    url = f"https://api.segmind.com/v1/{model_name}"
    
    response = requests.post(url, json=data, headers={'x-api-key': api_key})
    
    return response

def heygen_video_generate(api_key, request: HeygenVideoGenerateRequest):
    url = "https://api.heygen.com/v2/video/generate"
    
    payload = request.dict(exclude_unset=True)
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    return response

def get_heygen_video_status(api_key, video_id):
    url = "https://api.heygen.com/v1/video_status.get"
    
    headers = {
        "accept": "application/json",
        "x-api-key": api_key
    }
    
    params = {
        "video_id": video_id
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    return response





