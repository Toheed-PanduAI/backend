import pymongo
import os
from config import secret_config
MONGO_URL = secret_config.MONGO_DB_URL
# MONGO_URL = "mongodb://localhost:27017" # for local development
# MONGO_URL ="mongodb+srv://toheed:toheed123@cluster0.by2qgls.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
myclient = pymongo.MongoClient(MONGO_URL)
db = myclient["panduAI_db"]
users_collection = db["users"]
credits_transactions = db["credits_transactions"]
payments_collection = db["payments"]
plans_collection = db["plans"]
subscriptions_collection = db["subscriptions"]
video_tasks_collection = db["video_tasks"]
invoices_collection = db["invoices"]
api_usage_collection = db["api_usage"]


