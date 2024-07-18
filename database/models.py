from pydantic import BaseModel, Field, EmailStr, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

def default_publishing_time():
    return datetime.utcnow() + timedelta(hours=1)

class Permission(BaseModel):
    permissions: List[str]
    granted: bool
    user_id: str

class UserMetadataDetail(BaseModel):
    who_are_you: Optional[str] = None
    how_do_you_intend_to_use_our_tool: Optional[str] = None
    how_did_you_hear_about_us: Optional[str] = None
    username: Optional[str] = None

class SocialAccountType(str, Enum):
    YOUTUBE = "youtube#channel"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"

class SocialAccount(BaseModel):
    user_id: Optional[str] = None
    account_id: Optional[str] = None
    account_type: SocialAccountType
    account_name: Optional[str] = None
    account_thumbnail: Optional[str] = None
    channel_id: Optional[str] = None

class User(BaseModel):
    user_id: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    total_credits: Optional[int] = 100
    remaining_credits: Optional[int] = 100
    stripe_customer_id: str
    user_metadata_details: Optional[UserMetadataDetail] = None
    subscription_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool

class ThirdPartyAPICost(BaseModel):
    api_call_id: str
    video_id: str
    user_id: str
    api_name: str
    credit_cost: int
    status: str
    created_at: Optional[datetime] = None

class CreditCost(BaseModel):
    video_generation: int
    video_editing: int
    open_ai: int
    stability: int
    eleven_labs: int
    serp: int

class CreditTransaction(BaseModel):
    credit_transaction_id: str
    user_id: str
    video_id: Optional[str] = None
    api_call_id: Optional[str] = None
    credits: int
    transaction_type: str  # "deduction" or "addition"
    timestamp: datetime
    description: Optional[str]

class SoundEffect(BaseModel):
    sound_effect: str

class ImageDetail(BaseModel):
    sound_effects: List[SoundEffect]
    prompt: str
    effects_animation: str
    image_url: str

class Transition(BaseModel):
    sound_effects: List[SoundEffect]
    transition: str

class Scene(BaseModel):
    script: str
    script_audio: str
    images: List[ImageDetail]
    transition: List[Transition]
    sound_effects: List[SoundEffect]

class SubtitleStyle(BaseModel):
    size: Optional[str] = None
    color: str
    fontsize: int
    bg_color: str
    font: str
    stroke_color: str
    stroke_width: int
    method: str
    kerning: Optional[str] = None
    align: str
    interline: Optional[str] = None
    transparent: bool
    remove_temp: bool
    print_cmd: Optional[str] = None

class YoutubeCredentials(BaseModel):
    token: str
    refresh_token: str
    token_uri: str
    client_id: str
    client_secret: str
    scopes: List[str]
    user_id: str
    channel_id: str

class Youtube(BaseModel):
    channel_id: Optional[str] = None
    title: Optional[str] = "Untitled Video"
    description: Optional[str] = "This is a default video description."
    tags: Optional[List[str]] = ["default", "video"]
    privacyStatus: Optional[str] = "private"
    category: Optional[str] = "22"  # "22" is the category ID for "People & Blogs"
    playlist: Optional[str] = None
    thumbnail: Optional[str] = None
    publishing_time: Optional[datetime] = Field(default_factory=default_publishing_time)    
    is_active: Optional[bool] = True
 
class VideoMetadataDetail(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


class VideoTask(BaseModel):
    user_id: str
    user_prompt: str
    video_task_id: Optional[str] = None
    series_id: Optional[str] = None
    task_status: Optional[str] = "pending"
    video_flow_type: Optional[str] = "default"
    credit_cost: Optional[int] = None
    video_metadata_details: Optional[VideoMetadataDetail] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    video_url: Optional[str] = None
    video_hashtag: Optional[str] = None
    scenes: Optional[List[Scene]] = None
    video_bgm_prompt: Optional[str] = None
    voice_id: Optional[str] = None
    bgm_prompt: Optional[str] = None
    bgm_audio_url: Optional[str] = None
    style_preset: Optional[str] = None
    subtitle_styles: Optional[SubtitleStyle] = None
    platform_selection: Optional[str] = None
    target_audience: Optional[str] = None
    duration: Optional[str] = None
    youtube: Optional[Youtube] = Field(default_factory=Youtube)
    is_active: bool

class PaginatedVideoTaskResponse(BaseModel):
    videos: List[VideoTask]
    total_videos: int
    total_pages: int
    current_page: int

class Invoice(BaseModel):
    invoice_id: str
    user_id: str
    stripe_customer_id: str
    created_at: Optional[datetime] = None
    subscription_id: Optional[str] = None
    amount_paid: str = None
    amount_due: str = None
    amount_remaining: str = None
    currency: str = None
    invoice_pdf: str = None
    invoice_status: str = None
    invoice_url: str = None
    payment_intent_id: str = None

class PaginatedInvoiceResponse(BaseModel):
    invoices: List[Invoice]
    total_invoices: int
    total_pages: int
    current_page: int

class ImageTask(BaseModel):
    image_task_id: str
    user_id: str
    task_status: str  # e.g., "pending", "in_progress", "completed", "failed"
    task_details: Dict[str, str]  # Detailed task information
    created_at: datetime
    updated_at: Optional[datetime] = None
    image_url: Optional[HttpUrl] = None  # URL of the generated video
    image_hashtag: str = None

class AudioTask(BaseModel):
    audio_task_id: str
    user_id: str
    task_status: str  # e.g., "pending", "in_progress", "completed", "failed"
    task_details: Dict[str, str]  # Detailed task information
    created_at: datetime
    updated_at: Optional[datetime] = None
    audio_url: Optional[HttpUrl] = None  # URL of the generated video
    audio_hashtag: str = None

class ApiUsage(BaseModel):
    user_id: str
    api_endpoint: str
    method: str
    status_code: str
    duration: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Payment(BaseModel):
    payment_id: str
    user_id: str
    subscription_id: str
    amount: float
    currency: str  # e.g., "USD"
    payment_date: datetime
    payment_status: str  # e.g., "completed", "failed"

class Plan(BaseModel):
    plan_id: str
    name: str  # e.g., "basic", "premium"
    price: float
    description: str
    price_id: str

class Subscription(BaseModel):
    subscription_id: str
    user_id: str
    plan: str  # e.g., "basic", "premium"
    start_date: datetime
    end_date: Optional[datetime] = None
    status: str  # e.g., "active", "inactive", "canceled"

class PriceResponse(BaseModel):
    publishableKey: str
    prices: list

class Item(BaseModel):
    email: str

class SubscriptionItem(BaseModel):
    priceId: str
    customerId: str

class CancelItem(BaseModel):
    subscriptionId: str

class UpdateItem(BaseModel):
    subscriptionId: str
    newPriceLookupKey: str

class VoiceResponse(BaseModel):
    status: str
    voice_id: str
    message: str
    audio_file: Optional[bytes] = None

class TranscriptionResponse(BaseModel):
    status: str
    message: str
    text: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    status: str
    message: str
    image_urls: Optional[List[str]] = None

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionResponse(BaseModel):
    status: str
    message: str
    response: Optional[str] = None

class StabilityGenerateImageRequest(BaseModel):
    model: str
    prompt: str
    negative_prompt: str = None
    aspect_ratio: str = None
    seed: int = None
    output_format: str = "webp"

class StabilityImageToVideoRequest(BaseModel):
    seed: int
    cfg_scale: float
    motion_bucket_id: int

class SegmindImageGenerateRequest(BaseModel):
    data: dict
    # model_name: str = "face-to-sticker"

class Dimension(BaseModel):
    width: int
    height: int

class VideoInput(BaseModel):
    character: Dict[str, Any]
    voice: Optional[Dict[str, Any]] = None
    background: Optional[Dict[str, Any]] = None

class HeygenVideoGenerateRequest(BaseModel):
    test: bool = True
    caption: bool = False
    dimension: Dimension = Field(default_factory=lambda: Dimension(width=1920, height=1080))
    video_inputs: List[VideoInput]
    title: Optional[str] = None
    callback_id: Optional[str] = None
