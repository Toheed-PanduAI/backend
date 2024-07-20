import os
from dotenv import load_dotenv

load_dotenv() 

OPEN_AI_SECRET_KEY = os.getenv("OPEN_AI_SECRET_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLIC_KEY = os.getenv("STRIPE_PUBLIC_KEY")
ELEVEN_LABS_SECRET_KEY = os.getenv("ELEVEN_LABS_SECRET_KEY")
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
STABILITY_SECRET_KEY = os.getenv("STABILITY_SECRET_KEY")

# Docker Compose file variables
MONGODB_CONTAINER_NAME = os.getenv("MONGODB_CONTAINER_NAME")
FASTAPI_CONTAINER_NAME = os.getenv("FASTAPI_CONTAINER_NAME")
# MONGO_DB_URL
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
DISTRIBUTION_DOMAIN_NAME = os.getenv("DISTRIBUTION_DOMAIN_NAME")

# Paths
Image_folder_path = os.getenv("Image_folder_path")
Audio_folder_path = os.getenv("Audio_folder_path")
BGM_folder_path = os.getenv("BGM_folder_path")
Final_video_path = os.getenv("Final_video_path")
Effect_folder_path = os.getenv("Effect_folder_path")
Scenes_folder_path = os.getenv("Scenes_folder_path")
Transition_folder_path = os.getenv("Transition_folder_path")
Project_data_path = os.getenv("Project_data_path")
Output_filename_stiched = os.getenv("Output_filename_stiched")

# URLs
CLIENT_SECRETS_FILE = os.getenv("CLIENT_SECRETS_FILE")
SCOPES = os.getenv("SCOPES")
redirect_url = os.getenv("redirect_url")
frontend_url = os.getenv("frontend_url")
cancel_url = os.getenv("cancel_url")
success_url = os.getenv("success_url")















