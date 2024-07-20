import pymongo
import os
from config import secret_config
# MONGO_URL = secret_config.MONGO_DB_URL
# MONGO_URL = "mongodb://localhost:27017" # for local development
MONGO_URL ="mongodb+srv://toheed:toheed123@cluster0.by2qgls.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
myclient = pymongo.MongoClient(MONGO_URL)
db = myclient["panduAI_db"]
users_collection = db["users"]
credits_transaction_collection = db["credits_transaction"]
third_party_api_cost_collection = db["third_party_api_cost"]
youtube_credentials_collection = db["youtube_credential"]
social_accounts_collection = db["social_accounts"]
payments_transaction_collection = db["payments_transaction"]
plans_collection = db["plans"]
subscriptions_collection = db["subscriptions"]
video_tasks_collection = db["video_task"]
invoices_collection = db["invoice"]
# api_usage_collection = db["api_usage"]


