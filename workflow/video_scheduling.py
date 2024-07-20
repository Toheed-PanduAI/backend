
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import secrets
import logging
import os
import uuid
import json
import logging
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from fastapi import FastAPI, Request, HTTPException, Depends, Response
from database import db, aws
import tempfile
import requests


def schedule_video(video_data):
    if not video_data or "user_id" not in video_data or "video_id" not in video_data:
        raise HTTPException(status_code=400, detail="Invalid or missing video data")
    
    user_data = db.youtube_credentials_collection.find_one({"channel_id": video_data["channel_id"]})
    if not user_data:
        raise HTTPException(status_code=404, detail="User credentials not found")
    
    video_task = db.video_tasks_collection.find_one({"video_task_id": video_data['video_id']})
    if not video_task:
        raise HTTPException(status_code=404, detail="Video task not found")
    
    youtube_data = {}
    # if not youtube_data:
    #     raise HTTPException(status_code=400, detail="YouTube data not found in video task")
    
    video_path = video_task.get("video_url")
    
    try:
        credentials = google.oauth2.credentials.Credentials(
            token=user_data["token"],
            refresh_token=user_data["refresh_token"],
            token_uri=user_data["token_uri"],
            client_id=user_data["client_id"],
            client_secret=user_data["client_secret"],
            scopes=user_data["scopes"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing credentials: {e}")
    
    try:
        youtube = googleapiclient.discovery.build('youtube', 'v3', credentials=credentials)

        body = {
            "snippet": {
                "title": youtube_data.get("title", "Default Title"),
                "description": youtube_data.get("description", "This is a test description"),
                "tags": youtube_data.get("tags", ["test", "video"]),
                "categoryId": youtube_data.get("categoryId", "22"),
                "defaultLanguage": youtube_data.get("defaultLanguage", "en"),
                "defaultAudioLanguage": youtube_data.get("defaultAudioLanguage", "en")
            },
            "status": {
                "privacyStatus": youtube_data.get("privacyStatus", "private"),
                "embeddable": youtube_data.get("embeddable", True),
                "license": youtube_data.get("license", "youtube"),
                "publicStatsViewable": youtube_data.get("publicStatsViewable", True),
                "publishAt": youtube_data.get("publishAt", "2024-07-04T20:10:00Z"),
                "selfDeclaredMadeForKids": youtube_data.get("selfDeclaredMadeForKids", True)
            },
            "notifySubscribers": youtube_data.get("notifySubscribers", True),
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            response = requests.get(video_path, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        try:
            media_body = googleapiclient.http.MediaFileUpload(temp_file_path, chunksize=-1, resumable=True)

            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media_body
            )

            response = request.execute()
            return response
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling video: {e}")
    

# video_data = {
#     "user_id": "e199d2e3-bbaf-491f-a631-44f97fa4a8b4",
#     "video_id": "52a9e6b4-13ee-4f30-93e8-40b4511d33a0"
# }
# schedule_video(video_data)