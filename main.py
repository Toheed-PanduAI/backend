import uvicorn
import httpx
from fastapi import FastAPI, Response, Query, Request, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from pymongo import MongoClient, DESCENDING, ASCENDING
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from io import BytesIO  
from starlette.middleware.cors import CORSMiddleware
import json
import stripe
import os
import requests
from datetime import datetime
from typing import List, Dict
from typing import Optional, List
from bson import ObjectId
from math import ceil
from supertokens_python import init, get_all_cors_headers
from supertokens_python.framework.fastapi import get_middleware
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session
from supertokens_python.recipe.multitenancy.asyncio import list_all_tenants
from supertokens_python.recipe.userroles import UserRoleClaim
from elevenlabs import Voice, VoiceSettings, play
import database.db as db
import utils.util as util
import scripts.speech_synthesis as speech_synthesis
from database.models import Item, PaginatedInvoiceResponse, PaginatedVideoTaskResponse,  VoiceResponse, SubscriptionItem, PriceResponse, CancelItem, UpdateItem, User, Permission, Payment, Plan, Subscription, VideoTask, TranscriptionResponse, ImageGenerationResponse, Message, ChatCompletionResponse, Invoice, CreditTransaction, ThirdPartyAPICost, SocialAccount
from dotenv import load_dotenv
from googleapiclient.discovery import build
# from gmail_oauth import get_credentials
import config.supertoken_config as supertoken_config
from config import secret_config, credits_config
from uuid import uuid4 
from workflow import video_app, youtube_app
import asyncio
import sys
import logging
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import secrets

load_dotenv() 

# Set up logging
logging.basicConfig(level=logging.INFO)

# Setup Stripe python client library
stripe.api_key = secret_config.STRIPE_SECRET_KEY
stripe_publishable_key = secret_config.STRIPE_PUBLIC_KEY 
# stripe.api_key =  "sk_test_51PQnqhP3fxV3o3WtOlLEclN5cK0FolvRFevDW0l9gkydYC89cR8KXV7CxS5051wbxk4eHjY11DU61G3XN1E9zu9s00YqAmKQXN"
# stripe_publishable_key = "pk_test_51PQnqhP3fxV3o3WtbvtjGmdVksLrTdMTKEpwS29TVLjz3En9cQK4XUbyO1X3UNlbVdBJgolhXidxaaQZiETR9bgE00fY8LeOYm"

# MathPix API credentials
mathpix_api_id = secret_config.MATHPIX_APP_ID 
mathpix_api_key =secret_config.MATHPIX_APP_KEY 

# ElevenLabs URL
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/voices"

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

CLIENT_SECRETS_FILE = "./workflow/client_secrets.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly", "https://www.googleapis.com/auth/youtube"]
REDIRECT_URI = "http://localhost:8000/oauth2callback"
frontend_url = "http://localhost:3000/create-series"
cancel_url = "http://localhost:3000/cancel"
success_url = "http://localhost:3000/success"

# To store session for youtube API
session_store = {}


init(
    supertokens_config=supertoken_config.supertokens_config,
    app_info=supertoken_config.app_info,
    framework=supertoken_config.framework,
    recipe_list=supertoken_config.recipe_list,
    mode="asgi",
)

app = FastAPI(title="PanduAI Backend", version="0.1.0")

app.add_middleware(get_middleware())


@app.get("/sessioninfo")    
async def secure_api(session: SessionContainer = Depends(verify_session())):
    return {
        "sessionHandle": session.get_handle(),
        "userId": session.get_user_id(),
        "accessTokenPayload": session.get_access_token_payload(),
    }

# Permissions and Roles API
@app.get('/create_role')  
async def create_role(role_data: str, permissions: List[str], session: SessionContainer = Depends(
        verify_session()
)):
    try:
        # Add the role to the session
        await supertoken_config.create_role(role_data, permissions)
        return {"status": "OK"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get('/add_permissions')  
async def create_role(role_data: str, permissions: List[str], session: SessionContainer = Depends(
        verify_session()
)):
    try:
        # Add the role to the session
        await supertoken_config.add_permission_for_role(role_data, permissions)
        return {"status": "OK"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get('/remove_permissions')  
async def create_role(role_data: str, permissions: List[str], session: SessionContainer = Depends(
        verify_session()
)):
    try:
        # Add the role to the session
        await supertoken_config.remove_permission_from_role(role_data, permissions)
        return {"status": "OK"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get('/delete_role')      
async def create_role(role_data: str, permissions: List[str], session: SessionContainer = Depends(
        verify_session()
)):
    try:
        # Add the role to the session
        await supertoken_config.delete_role_function(role_data, permissions)
        return {"status": "OK"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get('/delete_all')  
async def delete_all(session: SessionContainer = Depends(
        verify_session(
            # We add the UserRoleClaim's includes validator
            override_global_claim_validators=lambda global_validators, session, user_context: global_validators + \
            [UserRoleClaim.validators.includes("admin")]
        )
)):
    return {
        "status": "OK",
    }

@app.get('/update_user')  
async def update_user(session: SessionContainer = Depends(
        verify_session(
            # We add the UserRoleClaim's includes validator
            override_global_claim_validators=lambda global_validators, session, user_context: global_validators + \
            [UserRoleClaim.validators.includes("user")]
        )
)):
    return {
        "status": "OK",
    }


# Users API
@app.get("/user/{user_id}/videos-generated")
async def get_videos_generated(user_id: str):
    count = await credits_config.get_videos_generated_count(user_id)
    return {"total_videos_generated": count}

@app.get("/user/{user_id}/monthly-credits-used")
async def get_monthly_credits_used_endpoint(user_id: str, year: int, month: int):
    credits_used = await credits_config.get_monthly_credits_used(user_id, year, month)
    return {"year": year, "month": month, "credits_used": credits_used}

@app.get("/users/{user_id}", response_model=User)
async def read_user(user_id: str, session: SessionContainer = Depends(verify_session())):

    user =  db.users_collection.find_one({"user_id": user_id})

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user)

@app.post("/users/", response_model=User)
async def create_user(user : dict = Depends(User), session: SessionContainer = Depends(verify_session())):

    if db.users_collection is None:
        raise HTTPException(status_code=500, detail="Database connection not initialized")
    
    user_data = user.dict()

    # # Check if email is already registered
    user_exists = db.users_collection.find_one({"email": user_data["email"]})
    if user_exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    
    user_data["created_at"] = datetime.utcnow()
    user_data["updated_at"] = datetime.utcnow()

    result = db.users_collection.insert_one(user_data)
    new_user = db.users_collection.find_one({"_id": result.inserted_id})

    if new_user:
        return User(**new_user)
    
    raise HTTPException(status_code=500, detail="User creation failed")

@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: str, user_data: dict, session: SessionContainer = Depends(verify_session())):
    
    # user_data = user_update.dict(exclude_unset=True)

    if not user_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = db.users_collection.update_one(
        {"user_id": user_id}, {"$set": user_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    updated_user = db.users_collection.find_one({"user_id": user_id})
    return User(**updated_user)

@app.delete("/users/{user_id}", response_model=User)
async def delete_user(user_id: str, session: SessionContainer = Depends(verify_session())):
    user = await db.users_collection.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    await db.users_collection.delete_one({"_id": ObjectId(user_id)})
    return User(**user)

# User Permissons API
@app.post("/users/{user_id}/permissions/", response_model=User)
async def update_permissions(
    user_id: str,
    permissions: List[Permission]
):
    user_data = db.users_collection.find_one({"_id": ObjectId(user_id)})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    db.users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"permissions": [perm.dict() for perm in permissions]}}
    )

    updated_user = db.users_collection.find_one({"_id": ObjectId(user_id)})
    return User(**updated_user)

# Payments API
@app.get("/payments/{payment_id}", response_model=Payment)
async def read_payment(payment_id: str, session: SessionContainer = Depends(verify_session())):
    payment = await db.payments_collection.find_one({"_id": ObjectId(payment_id)})
    if payment is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    return Payment(**payment)

@app.post("/payments/", response_model=Payment)
async def create_payment(payment: Payment, session: SessionContainer = Depends(verify_session())):
    payment_data = payment.dict(by_alias=True)
    payment_data["payment_date"] = datetime.now()
    result = await db.payments_collection.insert_one(payment_data)
    new_payment = await db.payments_collection.find_one({"_id": result.inserted_id})
    return Payment(**new_payment)

@app.put("/payments/{payment_id}", response_model=Payment)
async def update_payment(payment_id: str, payment: Payment, session: SessionContainer = Depends(verify_session())):
    payment_data = payment.dict(by_alias=True, exclude_unset=True)
    result = await db.payments_collection.update_one({"_id": ObjectId(payment_id)}, {"$set": payment_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Payment not found")
    updated_payment = await db.payments_collection.find_one({"_id": ObjectId(payment_id)})
    return Payment(**updated_payment)

@app.delete("/payments/{payment_id}", response_model=Payment)
async def delete_payment(payment_id: str, session: SessionContainer = Depends(verify_session())):
    payment = await db.payments_collection.find_one({"_id": ObjectId(payment_id)})
    if payment is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    await db.payments_collection.delete_one({"_id": ObjectId(payment_id)})
    return Payment(**payment)

# Plans API
@app.post("/plans/", response_model=Plan)
async def create_plan(plan: Plan, session: SessionContainer = Depends(verify_session())):
    plan_data = plan.dict(by_alias=True)
    result = await db.plans_collection.insert_one(plan_data)
    new_plan = await db.plans_collection.find_one({"_id": result.inserted_id})
    return Plan(**new_plan)

@app.get("/plans/", response_model=List[Plan])
async def read_plans(skip: int = 0, limit: int = 10, session: SessionContainer = Depends(verify_session())):
    plans_cursor = db.plans_collection.find().skip(skip).limit(limit)
    plans = await plans_cursor.to_list(length=limit)
    return plans

@app.get("/plans/{plan_id}", response_model=Plan)
async def read_plan(plan_id: str, session: SessionContainer = Depends(verify_session())):
    plan = await db.plans_collection.find_one({"_id": ObjectId(plan_id)})
    if plan is None:
        raise HTTPException(status_code=404, detail="Plan not found")
    return Plan(**plan)

@app.put("/plans/{plan_id}", response_model=Plan)
async def update_plan(plan_id: str, plan: Plan, session: SessionContainer = Depends(verify_session())):
    plan_data = plan.dict(by_alias=True, exclude_unset=True)
    result = await db.plans_collection.update_one({"_id": ObjectId(plan_id)}, {"$set": plan_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Plan not found")
    updated_plan = await db.plans_collection.find_one({"_id": ObjectId(plan_id)})
    return Plan(**updated_plan)

@app.delete("/plans/{plan_id}", response_model=Plan)
async def delete_plan(plan_id: str, session: SessionContainer = Depends(verify_session())):
    plan = await db.plans_collection.find_one({"_id": ObjectId(plan_id)})
    if plan is None:
        raise HTTPException(status_code=404, detail="Plan not found")
    await db.plans_collection.delete_one({"_id": ObjectId(plan_id)})
    return Plan(**plan)

# Subscriptions API
@app.post("/subscriptions/", response_model=Subscription)
async def create_subscription(subscription: Subscription, session: SessionContainer = Depends(verify_session())):
    subscription_data = subscription.dict(by_alias=True)
    result = await db.subscriptions_collection.insert_one(subscription_data)
    new_subscription = await db.subscriptions_collection.find_one({"_id": result.inserted_id})
    return Subscription(**new_subscription)

@app.get("/subscriptions/", response_model=List[Subscription])
async def read_subscriptions(skip: int = 0, limit: int = 10, session: SessionContainer = Depends(verify_session())):
    subscriptions_cursor = db.subscriptions_collection.find().skip(skip).limit(limit)
    subscriptions = await subscriptions_cursor.to_list(length=limit)
    return subscriptions

@app.get("/subscriptions/{subscription_id}", response_model=Subscription)
async def read_subscription(subscription_id: str, session: SessionContainer = Depends(verify_session())):
    subscription = await db.subscriptions_collection.find_one({"_id": ObjectId(subscription_id)})
    if subscription is None:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return Subscription(**subscription)

@app.put("/subscriptions/{subscription_id}", response_model=Subscription)
async def update_subscription(subscription_id: str, subscription: Subscription, session: SessionContainer = Depends(verify_session())):
    subscription_data = subscription.dict(by_alias=True, exclude_unset=True)
    result = await db.subscriptions_collection.update_one({"_id": ObjectId(subscription_id)}, {"$set": subscription_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Subscription not found")
    updated_subscription = await db.subscriptions_collection.find_one({"_id": ObjectId(subscription_id)})
    return Subscription(**updated_subscription)

@app.delete("/subscriptions/{subscription_id}", response_model=Subscription)
async def delete_subscription(subscription_id: str, session: SessionContainer = Depends(verify_session())):
    subscription = await db.subscriptions_collection.find_one({"_id": ObjectId(subscription_id)})
    if subscription is None:
        raise HTTPException(status_code=404, detail="Subscription not found")
    await db.subscriptions_collection.delete_one({"_id": ObjectId(subscription_id)})
    return Subscription(**subscription)

# Video Tasks API

@app.get("/videos/{user_id}", response_model=PaginatedVideoTaskResponse)
async def get_video_task(
    user_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    # Count total videos for the user
    total_videos = db.video_tasks_collection.count_documents({"user_id": user_id})

    if total_videos == 0:
        raise HTTPException(status_code=404, detail="No videos found for this user")

    # Calculate total pages
    total_pages = ceil(total_videos / per_page)

    # Ensure the requested page is valid
    if page > total_pages:
        raise HTTPException(status_code=404, detail="Page not found")

    # Calculate skip for pagination
    skip = (page - 1) * per_page

    projection = {
        "video_url": 1,
        "video_task_id": 1,
        "created_at": 1,
        "user_prompt": 1,
        "video_metadata_details": 1,
        "is_active": 1, 
        "_id": 0  # Exclude the _id field
    }
    
    videos = list(db.video_tasks_collection.find(
        {"user_id": user_id},
        projection
    ).sort("created_at", DESCENDING).skip(skip).limit(per_page))
    
    return PaginatedVideoTaskResponse(
        videos=[VideoTask(**video) for video in videos],
        total_videos=total_videos,
        total_pages=total_pages,
        current_page=page
    )


@app.get("/videos_dummmy/{user_id}")
async def get_video_task(user_id: str):
    projection = {
        "video_url": 1,
        "video_task_id": 1,
        "created_at": 1,
        "user_prompt": 1,
        "video_metadata_details": 1,
        "is_active": 1, 
         "_id": 0  # Exclude the _id field if not needed
    }
    
    videos = list(db.video_tasks_collection.find({"user_id": user_id}, projection))
    
    if not videos:
        raise HTTPException(status_code=404, detail="No videos found for this user")
    
    return videos

@app.get("/video_tasks/{video_task_id}", response_model=VideoTask)
async def get_video_task(video_task_id: str):

    video_task = db.video_tasks_collection.find_one({"video_task_id": video_task_id})

    if video_task is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return VideoTask(**video_task)

@app.post("/video_tasks")
async def create_video_task(video_task: VideoTask, background_tasks: BackgroundTasks):
    
    video_task_data = video_task.dict()
    video_task_data["video_task_id"] = str(uuid4())
    video_task_data["created_at"] = datetime.utcnow()
    
    result = db.video_tasks_collection.insert_one(video_task_data)
    new_video_task = db.video_tasks_collection.find_one({"_id": result.inserted_id})
    
    user_prompt = video_task_data["user_prompt"]
    video_task_id = video_task_data["video_task_id"]
    user_id = video_task_data["user_id"]
    video_flow_type = video_task_data["video_flow_type"]
    channel_id = video_task_data["youtube"]["channel_id"]
    video_bgm_prompt = video_task_data["video_bgm_prompt"]
    video_data = {
        "prompt": user_prompt,
        "query": video_bgm_prompt,
    }

    # Check if user has sufficient credits
    user = db.users_collection.find_one({"user_id": user_id})

    if user["remaining_credits"] < credits_config.CREDIT_COSTS["video_generation"]:
        return HTTPException(status_code=400, detail="Insufficient credits")

    if video_flow_type == "default":
        background_tasks.add_task(video_app.setVideoID, video_task_id)
        background_tasks.add_task(video_app.setUserID, user_id)
        background_tasks.add_task(video_app.setChannelID, channel_id)
        background_tasks.add_task(video_app.main, user_prompt)
    elif video_flow_type == "youtube_bgm":
        background_tasks.add_task(youtube_app.setVideoID, video_task_id)
        background_tasks.add_task(youtube_app.setUserID, user_id)
        background_tasks.add_task(youtube_app.setChannelID, channel_id)
        background_tasks.add_task(youtube_app.main, video_data)

    if new_video_task:
        return {"video_task_id": str(new_video_task['video_task_id'])}
    
    raise HTTPException(status_code=500, detail="Video creation failed")

@app.put("/video_tasks/{video_task_id}", response_model=VideoTask)
async def update_video_task(video_task_id: str, video_task: VideoTask):
    
    video_task_data = video_task.dict()
    video_task_data["updated_at"] = datetime.utcnow()

    result = db.video_tasks_collection.update_one({"video_task_id": video_task_id}, {"$set": video_task_data})
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="VideoTask not found")
    
    updated_video_task = db.video_tasks_collection.find_one({"video_task_id": video_task_id})
    
    cost = credits_config.CREDIT_COSTS["video_editing"]
    credits_config.deduct_credits(updated_video_task["user_id"], cost, "video_editing")

    return VideoTask(**updated_video_task)

@app.delete("/video_tasks/{video_task_id}", response_model=VideoTask)
async def delete_video_task(video_task_id: str):

    video_task = db.video_tasks_collection.find_one({"video_task_id": video_task_id})

    if video_task is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Update the document to set is_active to false instead of deleting it
    result = db.video_tasks_collection.update_one(
        {"video_task_id": video_task_id},
        {"$set": {"is_active": False}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Video not found")
    
    new_video_task = db.video_tasks_collection.find_one({"video_task_id": video_task_id})

    return VideoTask(**new_video_task)

# Social acoounts API
@app.get("/social_accounts/{user_id}", response_model=List[SocialAccount])
async def get_social_accounts(user_id: str):
    social_accounts = db.social_accounts_collection.find({"user_id": user_id})

    if social_accounts is None:
        raise HTTPException(status_code=404, detail="Social accounts not found")
    
    return social_accounts


# Youtube API
def get_session(request: Request):
    youtube_session_id = request.cookies.get("youtube_session_id")
    print(f"Session ID from cookies: {youtube_session_id}")

    if youtube_session_id and youtube_session_id in session_store:
        logging.info(f"Session found for youtube_session_id: {youtube_session_id}")
        return session_store[youtube_session_id]

    logging.info(f"No session found for youtube_session_id: {youtube_session_id}")
    return {}

@app.get("/youtube_credentials/{user_id}")
async def get_youtube_credentials(user_id: str):
    credentials = db.youtube_credentials_collection.find({"user_id": user_id})
    if credentials is None:
        raise HTTPException(status_code=404, detail="Credentials not found")
    return credentials

@app.get("/youtube_auth/{user_id}")
async def auth(user_id: str, request: Request, response: Response, session: dict = Depends(get_session)):
    logging.info("Redirecting to Google authorization URL")
    if not user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")
    

    state = str(uuid4())
    session["state"] = state
    youtube_session_id = str(uuid4())
    session_store[youtube_session_id] = session
    response.set_cookie(key="user_id", value=user_id)
    response.set_cookie(key="youtube_session_id", value=youtube_session_id)

    logging.info(f"Generated state: {state}")
    logging.info(f"Session data before redirect: {session}")

    try:
        flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
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
        stored_state = session.get("state")
        logging.info(f"State received: {state}")
        logging.info(f"State stored in session: {stored_state}")
        logging.info(f"Session data during callback: {session}")

        if state != stored_state:
            logging.error(f"State mismatch: received {state}, expected {stored_state}")
            raise HTTPException(status_code=400, detail="State mismatch")

        flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
        flow.redirect_uri = REDIRECT_URI

        authorization_response = str(request.url)
        logging.info(f"Authorization response URL: {authorization_response}")

        flow.fetch_token(authorization_response=authorization_response)

        credentials = flow.credentials
        user_id = request.cookies.get("user_id")

        # Save the credentials in the database
        credentials_data = {
            'user_id': user_id,
            'channel_id': str(uuid4()),
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }

        db.youtube_credentials_collection.insert_one(credentials_data)
        logging.info(f"Credentials saved for user_id: {user_id}")

        session["user_id"] = user_id

        response.set_cookie(key="user_id", value=user_id)

        youtube_session_id = request.cookies.get("youtube_session_id")
        if youtube_session_id:
            session_store[youtube_session_id] = session
            logging.info(f"Session updated with user_id: {user_id} for youtube_session_id: {youtube_session_id}")
        else:
            logging.error("No youtube_session_id found in cookies")
        return RedirectResponse(url=frontend_url, status_code=303)
        
    except Exception as e:
        logging.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during OAuth callback")

@app.get("/youtube_channels/{channel_id}")
async def get_channels(channel_id: str, request: Request, session: dict = Depends(get_session)):

    if not channel_id:
        raise HTTPException(status_code=401, detail="Channel ID not provided")
    
    channel_data = db.youtube_credentials_collection.find_one({"channel_id": channel_id})

    try:
        credentials = google.oauth2.credentials.Credentials(
            token=channel_data["token"],
            refresh_token=channel_data["refresh_token"],
            token_uri=channel_data["token_uri"],
            client_id=channel_data["client_id"],
            client_secret=channel_data["client_secret"],
            scopes=channel_data["scopes"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing credentials: {e}")
    

    try:
        youtube = build('youtube', 'v3', credentials=credentials)
        request = youtube.channels().list(
            part="snippet,contentDetails,statistics",
            mine=True
        )
        response = request.execute()

        logging.info(f"Channels fetched for user_id: {channel_id}")
        account_id = response["items"][0]["id"]
        account_type = response["items"][0]["kind"]
        account_name = response["items"][0]["snippet"]["title"]
        account_thumbnail = response["items"][0]["snippet"]["thumbnails"]["default"]["url"]

        social_accounts = {
            "account_id": account_id,
            "account_type": account_type,
            "account_name": account_name,
            "account_thumbnail": account_thumbnail,
            "channel_id": channel_id,
            "user_id": channel_data["user_id"]
        }
        
        channel_exits = db.social_accounts_collection.find_one({"account_id": account_id})
        
        if channel_exits:
            return  {"message": "Channel already connected", "channel_id": channel_id}

        result = db.social_accounts_collection.insert_one(social_accounts)
        new_channel = db.social_accounts_collection.find_one({"_id": result.inserted_id})

        return jsonable_encoder(new_channel)

    except Exception as e:
        logging.error(f"Error fetching channels: {e}")
        raise HTTPException(status_code=500, detail="Error fetching channels")


# Usage API
@app.get("/credit_transactions/{user_id}")
def get_user_credit_transactions(user_id: int, skip: int = 0, limit: int = 100):
    user_transactions = [t for t in db.credits_transaction_collection.values() if t.user_id == user_id]
    return sorted(user_transactions, key=lambda x: x.timestamp, reverse=True)[skip:skip+limit]

@app.get("/video_credit_usage/{video_id}")
def get_video_credit_usage(video_id: int):
    video_transactions = [t for t in db.credits_transaction_collection.values() if t.video_id == video_id]
    return sorted(video_transactions, key=lambda x: x.timestamp)

@app.get("/api_call_credit_usage/{api_call_id}")
def get_api_call_credit_usage(api_call_id: int):
    api_call_transactions = [t for t in db.credits_transaction_collection.values() if t.api_call_id == api_call_id]
    return api_call_transactions[0] if api_call_transactions else None


@app.get("/credit_usage_by_data", response_model=List)
async def get_transactions(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    user_id: str = Query(...),
):
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not provided")

    query = {
        "timestamp": {
            "$gte": start_date,
            "$lte": end_date
        },
        "user_id": user_id
    }
    
    try:
        projection = {"_id": 0}
        transactions = list(db.credits_transaction_collection.find(query, projection))
        return transactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
# Polling API
@app.get("/progress/{video_task_id}")
async def progress(video_task_id: str, request: Request):

    if (video_app.video_data["video_id"] != video_task_id):
                # If video_task_id is not valid or video generation hasn't started, raise HTTP 404
        raise HTTPException(status_code=404, detail="Video generation not started")
    
    video_task_data = db.video_tasks_collection.find_one({"video_task_id": video_task_id})

    if video_task_data is None:
        raise HTTPException(status_code=404, detail="Video not found")
    elif video_task_data["task_status"] == "completed":
        return {"status": "complete", "video_url": video_task_data["video_url"]}
    elif video_task_data["task_status"] == "failed":
        raise HTTPException(status_code=404, detail={"status": "failed"})
    
    async def event_generator():
        while True:
            progress = video_app.get_progress(video_task_id)
            if progress:
                print(f"Progress for video {video_task_id}: {progress['percentage']}%")
                sys.stdout.flush()  # Force output to be written to the terminal
                yield f"data: {json.dumps(progress)}\n\n"
                if progress["percentage"] >= 100:
                    yield f"data: {json.dumps({'status': 'complete'})}\n\n"
                    break
            await asyncio.sleep(1)
            if await request.is_disconnected():
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Stripe API for payments
@app.get("/price_config")
async def get_config():
    try:
        prices = stripe.Price.list(
            lookup_keys=['sample_free', 'sample_basic', 'sample_premium']
        )
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PriceResponse(
        publishableKey=stripe_publishable_key,
        prices=prices.data,
    )

@app.post("/stripe_checkout")
async def checkout(request: Request):
    try:
        products = await request.json()
        # user_id = products[0]["user_id"]

        line_items = [
            {
                "price_data": {  
                    "currency": "inr",
                    "product_data": {
                        "name": item["lookup_key"],
                    },
                    "unit_amount": int(item["unit_amount"] * 100), 
                },
                "quantity": item.get("quantity", 1)
            }
            for item in products
        ]

        total_amount = sum(item["unit_amount"] * item.get("quantity", 1) for item in products)


        session = stripe.checkout.Session.create(  # Changed from sessions.create to Session.create
            payment_method_types=["card"],
            line_items=line_items,
            mode="payment",
            success_url= success_url + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=cancel_url,
            metadata={
                # 'user_id': user_id,
                'amount': int(total_amount * 100)
            }
        )

        return JSONResponse(content={"id": session.id})  # Changed from session["id"] to session.id
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))  # Return the actual error message
    
@app.post("/create_customer")
async def create_customer(item: Item, response: Response):
    try:
        # Create a new customer object
        customer = stripe.Customer.create(email=item.email)

        # At this point, associate the ID of the Customer object with your
        # own internal representation of a customer, if you have one.
        response.set_cookie(key="customer", value=customer.id)

        return {"customer": customer}
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.post("/create_subscription")
async def create_subscription(item: SubscriptionItem, request: Request):

    customer_id = item.customerId

    if not customer_id:
        raise HTTPException(status_code=403, detail="Customer ID not found")
    
    price_id = item.priceId

    try:
        subscription = stripe.Subscription.create(
            customer=item.customerId,
            items=[{
                'price': price_id,
            }],
            payment_behavior='default_incomplete',
            expand=['latest_invoice.payment_intent'],
            discounts=[{"coupon": "free-period"}],
        )
        return {"subscriptionId": subscription.id, "clientSecret": subscription.latest_invoice.payment_intent.client_secret}

    except Exception as e:
        raise HTTPException(status_code=400, detail=e.user_message)

@app.post("/cancel_subscription")
async def cancel_subscription(item: CancelItem):
    try:
        # Cancel the subscription by deleting it
        deletedSubscription = stripe.Subscription.delete(item.subscriptionId)
        return {"subscription": deletedSubscription}
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.post("/update_subscription")
async def update_subscription(item: UpdateItem):
    try:
        subscription = stripe.Subscription.retrieve(item.subscriptionId)

        update_subscription = stripe.Subscription.modify(
            item.subscriptionId,
            items=[{
                'id': subscription['items']['data'][0].id,
                'price': item.newPriceLookupKey.upper(),
                # 'price': os.getenv(item.newPriceLookupKey.upper()),
            }]
        )

        return {"update_subscription": update_subscription}
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.get("/subscriptions")
async def list_subscriptions(request: Request):
    # Simulating authenticated user. Lookup the logged in user in your
    # database, and set customer_id to the Stripe Customer ID of that user.
    customer_id = request.cookies.get('customer')

    try:
        # Retrieve all subscriptions for given customer
        subscriptions = stripe.Subscription.list(
            customer=customer_id,
            status='all',
            expand=['data.default_payment_method']
        )
        return {"subscriptions": subscriptions}
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.post("/invoices/", response_model=Invoice)
def create_invoice(invoice : dict = Depends(Invoice)):

    invoice_data = invoice.dict()
    invoice_data["created_at"] = datetime.now()

    result = db.invoices_collection.insert_one(invoice_data)
    new_invoice = db.invoices_collection.find_one({"_id": result.inserted_id})
    if new_invoice:
        return Invoice(**new_invoice)
    
    raise HTTPException(status_code=500, detail="Invoice creation failed")

@app.get("/invoices/{user_id}", response_model=PaginatedInvoiceResponse)
def read_invoices(
    user_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    # Count total invoices for the user
    total_invoices = db.invoices_collection.count_documents({"user_id": user_id})

    if total_invoices == 0:
        raise HTTPException(status_code=404, detail="No invoices found for this user")

    # Calculate total pages
    total_pages = ceil(total_invoices / per_page)

    # Ensure the requested page is valid
    if page > total_pages:
        raise HTTPException(status_code=404, detail="Page not found")

    # Calculate skip and limit for pagination
    skip = (page - 1) * per_page

    # Fetch invoices with pagination
    invoices = db.invoices_collection.find({"user_id": user_id}) \
                  .sort("created_at", ASCENDING) \
                  .skip(skip) \
                  .limit(per_page)

    # Convert to list of Invoice objects
    invoice_list = [Invoice(**invoice) for invoice in invoices]

    return PaginatedInvoiceResponse(
        invoices=invoice_list,
        total_invoices=total_invoices,
        total_pages=total_pages,
        current_page=page
    )

@app.get("/invoice_preview")
async def preview_invoice(request: Request, subscriptionId: Optional[str] = None, newPriceLookupKey: Optional[str] = None):
    # Simulating authenticated user. Lookup the logged in user in your
    # database, and set customer_id to the Stripe Customer ID of that user.
    customer_id = request.cookies.get('customer')

    try:
        # Retrieve the subscription
        subscription = stripe.Subscription.retrieve(subscriptionId)

        # Retrieve the Invoice
        invoice = stripe.Invoice.upcoming(
            customer=customer_id,
            subscription=subscriptionId,
            subscription_items=[{
                'id': subscription['items']['data'][0].id,
                'price': newPriceLookupKey,
                # 'price': os.getenv(newPriceLookupKey),

            }],
        )
        return {"invoice": invoice}
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.post("/recharge_credit22s")
async def recharge_credits(request: Request):
    try:
        data = await request.json()
        user_id = data["user_id"]
        amount = data["amount"]

        user = db.users_collection.find_one({"user_id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        payment_intent = stripe.PaymentIntent.create(
            amount=amount * 100, 
            currency='inr',
            customer=user["stripe_customer_id"],
            description=f"Recharge credits for user {user_id}"
        )

        # Update user's credits
        new_credits = user.get("total_credits", 0) + amount
        db.users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"total_credits": new_credits, "updated_at": datetime.utcnow()}}
        )

        # Record the credit transaction
        credit_transaction = CreditTransaction(
            credit_transaction_id=str(uuid4()),
            user_id=user_id,
            amount=amount,
            api_call_id=None,
            video_id=None,
            transaction_type="addition",
            timestamp=datetime.now(),
            description=f"Recharge of {amount} credits"
        )
        db.credits_transaction_collection.insert_one(credit_transaction.dict())

        return JSONResponse(content={"payment_intent": payment_intent["id"], "new_credits": new_credits})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")

@app.post("/recharge_credits")
async def recharge_credits(request: Request):
    try:
        data = await request.json()
        user_id = data["user_id"]
        amount = int(data["amount"])

        user = db.users_collection.find_one({"user_id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Create a Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'inr',
                    'unit_amount': amount * 100,
                    'product_data': {
                        'name': 'Credit Recharge',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url= success_url + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url= cancel_url,
            customer=user["stripe_customer_id"],
            metadata={
                'user_id': user_id,
                'amount': amount
            }
        )

        return JSONResponse(content={"id": checkout_session.id})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")

@app.post("/payment-success")
async def payment_success(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")

        # Retrieve the session to verify the payment status
        session = stripe.checkout.Session.retrieve(session_id)

        if session.payment_status == "paid":
            user_id = session.metadata.user_id
            amount = int(session.metadata.amount)

            user = db.users_collection.find_one({"user_id": user_id})
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            # Update user's credits
            new_credits = user.get("total_credits", 0) + amount
            db.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"total_credits": new_credits, "updated_at": datetime.utcnow()}}
            )

            # Record the credit transaction
            credit_transaction = CreditTransaction(
                credit_transaction_id=str(uuid4()),
                user_id=user_id,
                credits=amount,
                api_call_id=None,
                video_id=None,
                transaction_type="addition",
                timestamp=datetime.now(),
                description=f"Recharge of {amount} credits"
            )
            db.credits_transaction_collection.insert_one(credit_transaction.dict())

            return JSONResponse(content={"success": True, "new_credits": new_credits})
        else:
            return JSONResponse(content={"success": False, "message": "Payment not successful"})

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")

# ElevenLabs API
@app.get("/elevenlabs/voices")
async def get_external_voices():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(ELEVENLABS_API_URL)
            response.raise_for_status()  # Raise an HTTPError for bad responses
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return response.json()

# generate voice
@app.post("/elevenlabs_generate_voice/", response_class=StreamingResponse)
async def elevenlabs_generate_voice(
    text: str = Form(...),
    voice_id: str = Form(...),
):
    try:
        
        audio_generator = speech_synthesis.elevan_labs_client.generate(
            text=text,
            voice=Voice(
        voice_id=voice_id,
        settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, speaking_rate=0.8)
    )
        )
             
         # Convert the generator to bytes
        audio = b"".join(list(audio_generator))

        # Create a BytesIO stream from the audio bytes
        audio_stream = BytesIO(audio)

        headers = {
            'Content-Disposition': f'attachment; filename="{voice_id}.mp3"'
        }

        return StreamingResponse(audio_stream, media_type="audio/mpeg", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Cloned voice
@app.post("/elevenlabs_clone_voice/", response_class=StreamingResponse)
async def elevenlabs_clone_voice(
    name: str = Form(...),
    description: str = Form(...),
    text: str = Form(...),
    voice_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:

        file_paths = []
        for file in files:
            file_location = f"/tmp/{file.filename}"
            with open(file_location, "wb") as buffer:
                buffer.write(file.file.read())
            file_paths.append(file_location)
        
        audio_generator = speech_synthesis.elevan_labs_client.clone(
            name=name,
            description=description,
            files=file_paths
        )

        # Clean up temporary files
        for file_path in file_paths:
            os.remove(file_path)

         # Convert the generator to bytes
        audio = b"".join(list(audio_generator))

        # Create a BytesIO stream from the audio bytes
        audio_stream = BytesIO(audio)

        headers = {
            'Content-Disposition': f'attachment; filename="{voice_id}.mp3"'
        }

        return StreamingResponse(audio_stream, media_type="audio/mpeg", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OpenAI API
# Text to Speech
@app.post("/open_ai_generate_voice/", response_class=StreamingResponse)
async def open_ai_generate_voice(
    text: str = Form(...),
    voice: str = Form(...),
):
    try:
        
        audio_generator = speech_synthesis.open_ai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

             
         # Convert the generator to bytes
        audio = b"".join(list(audio_generator))

        # Create a BytesIO stream from the audio bytes
        audio_stream = BytesIO(audio)

        headers = {
            'Content-Disposition': f'attachment; filename="{voice}.mp3"'
        }

        return StreamingResponse(audio_stream, media_type="audio/mpeg", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Speech to Text
@app.post("/open_ai_generate_text/")
async def open_ai_clone_voice(
    file: UploadFile = File(...)
):
    try:

        # Read the uploaded file
        audio_data = await file.read()

        # Save the audio file temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_data)
        
        # Open the audio file for reading
        with open(temp_file_path, "rb") as audio_file:
            # Use OpenAI's Whisper model to transcribe the audio
            transcription = speech_synthesis.open_ai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                timestamp_granularities=["word"]
            )

        # Remove the temporary file
        os.remove(temp_file_path)

      

        return TranscriptionResponse(
            status="success",
            message="Transcription successful.",
            text=transcription['text']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image Generation
@app.post("/generate_image/", response_model=ImageGenerationResponse)
async def generate_image(
    prompt: str = Form(...),
    size: str = Form("1024x1024"), 
    quality: str = Form("standard"), 
    n: int = Form(1)
):
    try:
        response = speech_synthesis.open_ai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )

        image_url = response.data[0].url

        return ImageGenerationResponse(
            status="success",
            message="Image generation successful.",
            image_url=image_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat Completion
@app.post("/chat_completion/", response_model=ChatCompletionResponse)
async def chat_completion(messages: List[Message], model: Optional[str] = "gpt-4-turbo"):
    try:

        response = speech_synthesis.open_ai_client.chat.completions.create(
            model=model,
            messages=[message.dict() for message in messages]
        )


        assistant_message = response.choices[0].message

        return ChatCompletionResponse(
            status="success",
            message="Chat completion successful.",
            response=assistant_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# MathPix API
@app.post("/mathpix_process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()

        # Make a request to Mathpix API
        response = requests.post(
            "https://api.mathpix.com/v3/text",
            files={"file": (file.filename, contents, file.content_type)},
            data={
                "options_json": json.dumps({
                    "math_inline_delimiters": ["$", "$"],
                    "rm_spaces": True
                })
            },
            headers={
                "app_id": mathpix_api_id,
                "app_key": mathpix_api_key
            }
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Return the JSON response from Mathpix API
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mathpix_process_pdf/")
async def process_pdf(request: str):
    try:
        response = requests.post(
            "https://api.mathpix.com/v3/pdf",
            json={
                "url": request.url,
                "conversion_formats": {
                    "docx": True,
                    "tex.zip": True
                }
            },
            headers={
                "app_id": mathpix_api_id,
                "app_key": mathpix_api_key,
                "Content-type": "application/json"
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))



# @app.get("/gmail/messages/")
# async def list_gmail_messages(email: str):
#     # if email not in users_db:
#     #     raise HTTPException(status_code=400, detail="User not registered.")
#     # token_file = users_db[email]
#     token_file = email
#     try:
#         creds = get_credentials(token_file)
#         service = build('gmail', 'v1', credentials=creds)
#         results = service.users().messages().list(userId='me').execute()
#         messages = results.get('messages', [])

#         if not messages:
#             return {"message": "No messages found."}
        
#         return {"messages": messages}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



# @app.middleware("http")
# async def log_request(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     duration = time.time() - start_time

#     log_data = {
#         "user_id": request.headers.get("X-User-ID"),
#         "endpoint": request.url.path,
#         "method": request.method,
#         "status_code": response.status_code,
#         "duration": duration,
#         "timestamp": datetime.utcnow()
#     }
#     db.logs.insert_one(log_data)
#     return response
 

# @app.middleware("http")
# async def session_middleware(request: Request, call_next):
#     youtube_session_id = request.cookies.get("youtube_session_id")
#     if not youtube_session_id:
#         youtube_session_id = secrets.token_hex(16)
#         response = await call_next(request)
#         response.set_cookie(key="youtube_session_id", value=youtube_session_id)
#         session_store[youtube_session_id] = {}
#     else:
#         response = await call_next(request)
#     return response

# CORS Middleware
app = CORSMiddleware(
    app=app,
      allow_origins=[
        supertoken_config.app_info.website_domain, 
        "https://app.pandu.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000"
        "http://0.0.0.0:5500"
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type"] + get_all_cors_headers(),
)

if __name__  == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
