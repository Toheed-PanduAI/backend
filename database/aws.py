import boto3
import os
import json
import uuid
from botocore.exceptions import NoCredentialsError
from pathlib import Path
from dotenv import load_dotenv
from config import secret_config

load_dotenv()

# Get AWS credentials and bucket name from environment variables
aws_access_key_id = secret_config.AWS_ACCESS_KEY_ID 
aws_secret_access_key = secret_config.AWS_SECRET_ACCESS_KEY
bucket_name = secret_config.AWS_BUCKET_NAME
distribution_domain_name = secret_config.DISTRIBUTION_DOMAIN_NAME

if not aws_access_key_id or not aws_secret_access_key or not bucket_name:
    raise ValueError("Please set the AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_BUCKET_NAME environment variables")

# Define local directories for images and audio files
images_dir = secret_config.Image_folder_path
audio_dir = secret_config.Audio_folder_path
bgm_path = secret_config.BGM_folder_path
final_video_path = secret_config.Final_video_path
project_data_path = secret_config.Project_data_path

# Load the main JSON data
# with open(project_data_path, 'r') as f:
#     project_data = json.load(f)

# Create a session
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='eu-north-1'
)

# Get the S3 client
s3 = session.client('s3')

series_uuid = str(uuid.uuid1())

# TEMP_DIR = Path("/app/assets")
# TEMP_DIR.mkdir(parents=True, exist_ok=True)

# temp_file_path = TEMP_DIR / file.filename
# with open(temp_file_path, 'wb') as temp_file:
#     temp_file.write(await file.read())
  
#     session.upload_file(str(temp_file_path), 'your-s3-bucket-name', file.filename)
# # to delete the local file
# temp_file_path.unlink()


def upload_file(local_path, s3_file_path):
    try:
        s3.upload_file(local_path, bucket_name, s3_file_path)
        print(f"Upload Successful: {s3_file_path} to {bucket_name}")
    except FileNotFoundError:
        print("The file was not found")
    except NoCredentialsError:
        print("Credentials not available")

def upload_to_s3(project_data, series_uuid):
    prompt_uuid = str(uuid.uuid1())

    # List image and audio files
    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
    audio_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))])

    # Ensure both directories have the same number of files
    if len(image_files) != len(audio_files):
        raise ValueError("The number of image files and audio files must be the same")

    # Upload scenes
    for scene_id, (image_path, audio_path) in enumerate(zip(image_files, audio_files), start=1):
        # Upload image
        image_uuid = str(uuid.uuid1())
        s3_image_path = f'{series_uuid}/scenes/scene{scene_id}/images/{image_uuid}.png'
        upload_file(image_path, s3_image_path)

        # Upload audio
        audio_uuid = str(uuid.uuid1())
        s3_audio_path = f'{series_uuid}/scenes/scene{scene_id}/audio/{audio_uuid}.mp3'
        upload_file(audio_path, s3_audio_path)

    # Upload BGM
    bgm_uuid = str(uuid.uuid1())
    upload_file(bgm_path, f'{series_uuid}/bgm/{bgm_uuid}.mp3')

    # Upload main project data JSON
    project_data_s3_path = f'{series_uuid}/prompt/{prompt_uuid}.json'
    with open('temp_project_data.json', 'w+') as f:
        json.dump(project_data, f)
    upload_file('temp_project_data.json', project_data_s3_path)
    os.remove('temp_project_data.json')

    return series_uuid

def upload_final_video_to_s3():
    # Upload final video
    video_uuid = str(uuid.uuid1())
    series_uuid = str(uuid.uuid1())
    upload_file(final_video_path, f'{series_uuid}/final_video/{video_uuid}.mp4')
    return video_uuid, series_uuid

def get_cdn_url_video(series_uuid, video_uuid):
    def get_cdn_url(distribution_domain_name, s3_file_path):
        return f'{distribution_domain_name}/{s3_file_path}'

    # Get CDN URL for the video file
    cdn_url = get_cdn_url(distribution_domain_name, f'{series_uuid}/final_video/{video_uuid}.mp4')
    print(f"CDN URL: {cdn_url}")
    return cdn_url

# Main process
# series_uuid, video_uuid = upload_to_s3()
# video_uuid = upload_final_video_to_s3
# get_cdn_url_video(series_uuid, video_uuid)
