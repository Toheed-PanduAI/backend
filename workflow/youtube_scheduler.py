from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import json
import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import uuid
import secrets
import logging

app = FastAPI()
filenumber = 34
output_path= f"/Users/toheed/PanduAI/backend/workflow/result/result_final_with_watermark_{filenumber}.mp4"

# Set up logging
logging.basicConfig(level=logging.INFO)

# In-memory storage for session data (for debugging purposes only)
session_store = {}

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
REDIRECT_URI = "http://localhost:8000/oauth2callback"

class VideoData(BaseModel):
    title: str
    description: str
    tags: Optional[List[str]] = []
    category_id: Optional[str] = "22"
    privacy_status: Optional[str] = "private"
    publish_at: Optional[str] = None

# Dependency to get session data
def get_session(request: Request):
    session_id = request.cookies.get("session_id")
    print(f"Session ID from cookies: {session_id}")

    if session_id and session_id in session_store:
        logging.info(f"Session found for session_id: {session_id}")
        return session_store[session_id]

    logging.info(f"No session found for session_id: {session_id}")
    return {}

# Middleware to set up session
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = secrets.token_hex(16)
        response = await call_next(request)
        response.set_cookie(key="session_id", value=session_id)
        session_store[session_id] = {}
    else:
        response = await call_next(request)
    return response

# Test session storage independently
@app.get("/test-session")
async def test_session(session: dict = Depends(get_session)):
    if "test_key" not in session:
        session["test_key"] = "test_value"
        return {"message": "Set session value"}
    return {"message": f"Session value: {session['test_key']}"}

@app.get("/auth")
async def auth(request: Request, session: dict = Depends(get_session)):
    logging.info("Redirecting to Google authorization URL")

    state = str(uuid.uuid4())
    session["state"] = state  # Store state in session
    session_id = request.cookies.get("session_id")
    session_store[session_id] = session  # Ensure session is stored correctly

    logging.info(f"Generated state: {state}")
    logging.info(f"Session data before redirect: {session}")

    try:
        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES)
        flow.redirect_uri = REDIRECT_URI

        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=state)

        logging.info(f"Authorization URL: {authorization_url}")
        logging.info(f"Updated session data with state: {session}")

        return {"authorization_url": authorization_url}
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found error: {fnf_error}")
        raise HTTPException(status_code=500, detail="Client secrets file not found.")
    except json.JSONDecodeError as json_error:
        logging.error(f"JSON decode error: {json_error}")
        raise HTTPException(status_code=500, detail="Error decoding client secrets file.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/oauth2callback")
async def oauth2callback(request: Request, response: Response, session: dict = Depends(get_session)):
    try:
        state = request.query_params.get('state')
        stored_state = session.get("state")  # Retrieve state from session
        print(f"State received: {state}")
        logging.info(f"State received: {state}")
        logging.info(f"State stored in session: {stored_state}")
        logging.info(f"Session data during callback: {session}")

        if state != stored_state:
            logging.error(f"State mismatch: received {state}, expected {stored_state}")
            raise HTTPException(status_code=400, detail="State mismatch")

        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
        flow.redirect_uri = REDIRECT_URI

        authorization_response = str(request.url)
        logging.info(f"Authorization response URL: {authorization_response}")

        flow.fetch_token(authorization_response=authorization_response)

        credentials = flow.credentials
        user_id = str(uuid.uuid4())  # Generate a unique user ID
        credentials_path = f"credentials/{user_id}.json"

        # Ensure the credentials directory exists
        os.makedirs(os.path.dirname(credentials_path), exist_ok=True)

        # Save the credentials for later use
        with open(credentials_path, "w") as creds_file:
            creds_file.write(json.dumps({
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }))

        session["user_id"] = user_id
        logging.info(f"Credentials saved for user_id: {user_id}")

        # Set user_id in cookies
        response.set_cookie(key="user_id", value=user_id)

        session_id = request.cookies.get("session_id")
        if session_id:
            # Save the updated session data back to the session_store
            session_store[session_id] = session
            logging.info(f"Session updated with user_id: {user_id} for session_id: {session_id}")
        else:
            logging.error("No session_id found in cookies")

        return {"message": "Authentication successful. You can now use the /upload endpoint to upload videos.", "user_id": user_id}
    except Exception as e:
        logging.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during OAuth callback")

@app.post("/upload")
async def upload_video(request: Request, video_data: VideoData, session: dict = Depends(get_session)):
    user_id = request.cookies.get("user_id")
    print(f"user_id received upload: {user_id}")

    if not user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    credentials_path = f"credentials/{user_id}.json"
    if not os.path.exists(credentials_path):
        raise HTTPException(status_code=401, detail="User credentials not found")

    with open(credentials_path, "r") as creds_file:
        credentials_info = json.load(creds_file)

    credentials = google.oauth2.credentials.Credentials(**credentials_info)
    youtube = googleapiclient.discovery.build('youtube', 'v3', credentials=credentials)

    body = {
        "snippet": {
            "title": "My Test Video",
            "description": "This is a test description",
            "tags": ["test", "video"],
            "categoryId": "22",
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en"
        },
        "status": {
            "privacyStatus": "private",
            "embeddable": True,
            "license": "youtube",
            "publicStatsViewable": True,
            "publishAt": "2024-07-04T20:10:00Z",
            "selfDeclaredMadeForKids": True
        },
        "notifySubscribers": True,
        "onBehalfOfContentOwner": "contentOwnerId",
        "onBehalfOfContentOwnerChannel": "contentOwnerChannelId"
    }

    media_body = googleapiclient.http.MediaFileUpload(output_path, chunksize=-1, resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media_body
    )

    response = request.execute()
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
