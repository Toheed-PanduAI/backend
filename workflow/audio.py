import requests
import numpy as np
import cv2
import os
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  
from dotenv import load_dotenv

load_dotenv() 
OPEN_AI_SECRET_KEY = os.getenv('OPEN_AI_SECRET_KEY')



client = OpenAI(api_key=OPEN_AI_SECRET_KEY)

def enhance_prompts(prompt):
    response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "This is a script for an entire video, I want you to divide into different number of scenes, make sure you do not change words. Number of scenes should depend upon the script."},
                {"role": "user", "content": prompt}
            ]
        )

